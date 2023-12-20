import os
import torch
import shutil
import random
#from tensorboardX import SummaryWriter
from torch.optim import Adam

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import time

# --------------------------------
# helper functions
# --------------------------------


def check_dir(save_dirs):
    """
    creates the directory to save model
    """
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def get_dataloader(traindataset, testdataset, batch_size, seed=100):
    """
    splits the dataset into train/val/test loaders

    Args
       dataset: pytorch-geometric Dataset generated after parsing
       batch_size (int)
       data_split_ratio (list): training, validation and testing ratio
       seed: random seed to split the dataset randomly

    Returns
       a dictionary of training, validation, and testing dataLoader
    """
    random.seed(seed)
    """
    benign_dataset, malware_dataset = dataset

    def split_data(dataset, data_split_ratio, seed):
        # shuffle the data once for good measure
        random.shuffle(dataset)

        # data is seperated to train eval == 0.8, 0.2
        num_train = int(data_split_ratio[0] * len(dataset))
        num_test = len(dataset) - num_train
    
        train, _eval = random_split(dataset, lengths=[num_train, num_test], generator=torch.Generator().manual_seed(seed))
        return train, _eval

    # for benign data
    _benign_train, _benign_eval = split_data(benign_dataset, data_split_ratio, seed)
    print(f"Benign: #train:{len(_benign_train)}, #eval:{len(_benign_eval)}")
    # for malware data
    _malware_train, _malware_eval = split_data(malware_dataset, data_split_ratio, seed)
    print(f"Malware: #train:{len(_malware_train)}, #eval:{len(_malware_eval)}")
    
    train_dataset = _benign_train + _malware_train
    eval_dataset = _benign_eval + _malware_eval
    """
    train_dataset = traindataset
    test_dataset = testdataset

    dataloader = dict()
    dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #dataloader['eval'] = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def calculate_scores(prediction, truth, if_test=False):
    """
    claculates the confusion matrix and F1 score (code only works for binary classification tasks)
    Args:
       prediction (torch): the model output (binary labels)
       truth (torch): the ground truth
    Returns:
       true_positives_rate, false_positives_rate , F1 score

    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()  # TP
    false_positives = torch.sum(confusion_vector == float('inf')).item()  # FP
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()  # TN
    false_negatives = torch.sum(confusion_vector == 0).item()  # FN

    # print(true_positives)
    # print(false_positives)
    # print(true_negatives)
    # print(false_negatives)

    # calculating the rates
    if if_test:
        print(f"\n #True Positives: {true_positives}, #False_Positives: {false_positives}, #True Negatives: {true_negatives}, #False Negatives: {false_negatives}")
    # TPr = true_positives / (true_positives + false_negatives)
    # FPr = false_positives / (false_positives + true_negatives)
    # F1 = (2 * true_positives) / ((2 * true_positives) + (false_positives + false_negatives))
    # print(TPr, FPr, F1)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0  # of all predicted as malware, whats the % that is actually malware
    recall = true_positives / (true_positives + false_negatives)  if (true_positives + false_negatives) > 0 else 0  # of all malware samples in data, whats the % that was identified by the model
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # harmonic mean of precision, recall
    
    return precision, recall, F1

# -------------------------------
# main trainer class
# -------------------------------


class TrainModel(object):

    """
    classs for trainging GNN model
    """
    
    def __init__(self, model, traindataset, testdataset, device, verbose=0, save_dir=None, save_name='model', ** kwargs):
        """
        Args
        model (torch.model): the GNN model
        device (str): set to cpu or gpu:0
        save_dir (str): path to save model
        verbose (int): flag to print on screen
        save_name (str): model name
        ** kwargs: will be trainer_params, optimizer_params and dataloader_params
        """
        self.model = model
        self.device = device

        self.verbose = verbose
        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        dataloader_params = kwargs.get('dataloader_params')
        self.loader = get_dataloader(traindataset, testdataset, **dataloader_params)
        self.writer = None  # for logging
        if verbose < 0:
            self.writer = SummaryWriter('./logs/' + save_name)
        return

    # ------------------------
    # internal functions used
    # ------------------------
    
    def __loss__(self, logits, labels):
        """
        cross entropy loss for classification
        """
        # print(logits, labels)
        return F.cross_entropy(logits, labels)

    def _train_batch(self, data, labels):
        """
        runs single forward pass for mini-batch of data

        Args
           data (torch.Data): the data mini-batch
           labels (torch.Data): class-labels

        Returns
           loss (torch.tensor)
        """
        # 1. forward pass
        logits = self.model(data=data)

        # 2. get loss
        loss = self.__loss__(logits, labels)

        # 2.1. get preds
        preds = logits.argmax(-1)
        
        # 3. optimize weights
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        
        return loss.item(), preds

    def _eval_batch(self, data, labels, **kwargs):
        """
        similar to _train_batch above, but has no optimizer step
        
        Args
           data (torch.Data): the data mini-batch
           labels (torch.Data): class-labels

        Returns
           loss (torch.tensor)
           preds (torch.tensor)
        """
        self.model.eval() # JY 2 2023-0729
        print(data,flush= True)
        logits = self.model(data)
        loss = self.__loss__(logits, labels)
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds

    # -------------------------
    # main functions
    # -------------------------
    
    def eval(self):
        """
        runs the _eval_batch on eval-dataset
        """
        self.model.to(self.device)
        self.model.eval()  # stops gradient computation

        with torch.no_grad():
            losses, accs = [], []
            all_preds, all_truth = [], []
            for batch in self.loader['eval']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                accs.append(batch_preds == batch.y)
                all_preds.append(batch_preds)
                all_truth.append(batch.y)

            eval_loss = torch.tensor(losses).mean().item()
            eval_acc = torch.cat(accs, dim=-1).float().mean().item()

            # calculate scores
            all_preds = torch.cat(all_preds, dim=-1)  # .tolist()
            all_truth = torch.cat(all_truth, dim=-1)  # .tolist()
            eval_precision, eval_recall, eval_F1 = calculate_scores(all_preds, all_truth)
            
        self.model.train()  # reset gradient computation
        return eval_loss, eval_acc, eval_precision, eval_recall, eval_F1

    def test(self, ep=None):
        """
        runs _eval_batch() on test-dataset

        Args
           ep (int): epoch when test() was run
        """
        state_dict = torch.load(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))['net']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            losses, preds, accs, truths = [], [], [], []
            for batch in self.loader['test']:
                batch = batch.to(self.device)
                # print(f"batch: {batch}")
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                truths.append(batch.y)
                accs.append(batch_preds == batch.y)
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)  # .tolist()
            truths = torch.cat(truths, dim=-1)  # .tolist()

            # calculate scores
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
            test_precision, test_recall, test_F1 = calculate_scores(preds, truths, if_test=True)

            if self.verbose >= -1:
                print(f"\n preds: {preds}\n  truths: {truths}", flush=True)
                print(f"\nTest loss: {test_loss:.4f}, Test acc {test_acc:.4f}, Test Precision {test_precision:.4f}, Test Recall {test_recall:.4f}, Test F1 {test_F1:.4f}", flush=True)
                # if ep > 0 and self.writer is not None:
                #     self.writer.add_scalar('accuracy/test_acc', test_acc, ep + 1)
                #     self.writer.add_scalar('loss/test_loss', test_loss, ep + 1)
                #     self.writer.add_scalar('precision/test_precision', test_precision, ep + 1)
                #     self.writer.add_scalar('recall/test_recall', test_recall, ep + 1)
                #     self.writer.add_scalar('F1_score/test_F1', test_F1, ep + 1)

        self.model.train()  # reset gradiant computationn
        
        # Modified by JY for grid-search result saving @ 2022-06-02
        return {
                "test_loss": test_loss, 
                "test_acc": test_acc, "test_precision": test_precision, "test_recall": test_recall, "test_F1": test_F1,
                "preds":preds, "truths": truths,
               }

    def train(self, train_params=None, optimizer_params=None):
        """
        runs training iteration
        
        Args
           train_params (dict): the train_params set when initializing the trainer class
           optimizer_params (dict): the optimizer params set when initializing the trainer class
        """
        num_epochs = train_params['num_epochs']
        num_early_stop = train_params['num_early_stop']
        milestones = train_params['milestones']
        gamma = train_params['gamma']

        # intialize optimizer
        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)

        # set the learning rate, fixed or as a schedule
        if milestones is not None and gamma is not None:
            lr_schedule = MultiStepLR(self.optimizer,
                                      milestones=milestones,
                                      gamma=gamma)
        else:
            lr_schedule = None

        # train model
        self.model.to(self.device)
        #best_eval_acc = -1.0
        #best_eval_F1 = -1.0
        #best_eval_loss = 0.0
        best_train_loss = 0.0
        early_stop_counter = 0

        # Added by JY to see how model evolved in predicting during epochs @ 2022-06-23
        #trainingepochs_preds_truth = dict()

        # loop through for all epochs
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            is_best = False
            self.model.train()
            losses, accs = [], []

            all_preds = []
            all_truth = []

            #Added by JY to see how model evolved in predicting during epochs @ 2022-06-23
            #batches_preds_truth = dict()

            # for each epoch, loop through all batches
            for idx, batch in enumerate(self.loader['train']):
                batch = batch.to(self.device)
                loss, preds = self._train_batch(batch, batch.y)
                losses.append(loss)
                accs.append(preds == batch.y)
                all_preds.append(preds)
                all_truth.append(batch.y)

                #Added by JY to see how model evolved in predicting during epochs @ 2022-06-23
                #batches_preds_truth['batch_'+str(idx+1)] = {"batch": batch, "preds": preds,"truths":batch.y}


            #Added by JY to see how model evolved in predicting during epochs @ 2022-06-23
            #trainingepochs_preds_truth['epoch_'+str(epoch+1)] = batches_preds_truth
            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time


            # compute loss and stop early if needed
            train_loss = torch.FloatTensor(losses).mean().item()
            train_acc = torch.cat(accs, dim=-1).float().mean().item()
            # calculate TPr, FPr, F1
            all_preds = torch.cat(all_preds, dim=-1)  # .tolist()
            all_truth = torch.cat(all_truth, dim=-1)  # .tolist()
            # print(all_preds)
            # print(all_truth)
            train_precision, train_recall, train_F1 = calculate_scores(all_preds, all_truth)
            
           # eval_loss, eval_acc, eval_precision, eval_recall, eval_F1 = self.eval()

            if self.verbose >= 0:
                print(f'Epoch:{epoch} | Elapsed-Time: {epoch_elapsed_time:.2f} || Train_loss:{train_loss:.4f}, Train_acc:{train_acc:.4f}, Train_Precision:{train_precision:.4f}, Train_Recall:{train_recall:.4f}, Train_F1:{train_F1:.4f}', flush = True)
                # print(f'Epoch:{epoch}, Train_loss:{train_loss:.4f}, Train_acc:{train_acc:.4f}, Train_Precision:{train_precision:.4f}, Train_Recall:{train_recall:.4f}, Train_F1:{train_F1:.4f}')
                '''print(f'Epoch:{epoch}, Train_loss:{train_loss:.4f}, Train_acc:{train_acc:.4f}, Train_Precision:{train_precision:.4f}, Train_Recall:{train_recall:.4f}, Train_F1:{train_F1:.4f} || Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}, Eval_Precision:{eval_precision:.4f}, Eval_Recall:{eval_recall:.4f}, Eval_F1:{eval_F1:.4f}')'''
            elif self.writer is not None:
                # for training steps
                self.writer.add_scalar('loss/train_loss', train_loss, epoch + 1)
                self.writer.add_scalar('accuracy/train_acc', train_acc, epoch + 1)
                self.writer.add_scalar('precision/train_precision', train_precision, epoch + 1)
                self.writer.add_scalar('recall/train_recall', train_recall, epoch + 1)
                self.writer.add_scalar('F1_score/train_F1', train_F1, epoch + 1)
                """
                # for eval steps
                self.writer.add_scalar('loss/eval_loss', eval_loss, epoch + 1)
                self.writer.add_scalar('accuracy/eval_acc', eval_acc, epoch + 1)
                self.writer.add_scalar('precision/eval_precision', eval_precision, epoch + 1)
                self.writer.add_scalar('recall/eval_recall', eval_recall, epoch + 1)
                self.writer.add_scalar('F1_score/eval_F1', eval_F1, epoch + 1)
                

            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break
                """
            # change learning rate if there is a schedule
            if lr_schedule:
                lr_schedule.step()

            # if best_eval_acc < eval_acc:
            #    is_best = True
            #    best_eval_acc = eval_acc
            """
            if best_eval_F1 < eval_F1:
                is_best = True
                best_eval_F1 = eval_F1
                # Added by JY to store training scores associated with the bestmodel @ 2022-06-23
                bestmodel_trainingloss = train_loss
                bestmodel_train_acc = train_acc
                bestmodel_train_precision = train_precision
                bestmodel_train_recall = train_recall
                bestmodel_train_F1 = train_F1
            """
            if epoch == 0:
                best_train_loss = train_loss

            if train_loss <= best_train_loss:
                is_best = True
                best_train_loss = train_loss
                bestmodel_trainingloss = train_loss
                bestmodel_train_acc = train_acc
                bestmodel_train_precision = train_precision
                bestmodel_train_recall = train_recall
                bestmodel_train_F1 = train_F1

                if num_early_stop > 0:
                    early_stop_counter = 0
            else:
                if num_early_stop > 0:
                    early_stop_counter += 1
            if num_early_stop > 0 and epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                break

            recording = {'epoch': epoch, 'is_best': str(is_best)}
            if self.save:
                self.save_model(is_best, recording=recording, train_loss=train_loss, train_accuracy=train_acc, train_F1=train_F1, ep=epoch)

        # Added by JY for grid-search result saving @ 2022-06-22
        return {
                "bestmodel_train_loss": bestmodel_trainingloss, 
                "bestmodel_train_acc": bestmodel_train_acc, "bestmodel_train_precision": bestmodel_train_precision,
                "bestmodel_train_recall": bestmodel_train_recall, "bestmodel_train_F1": bestmodel_train_F1,

                #"trainingepochs_preds_truth": trainingepochs_preds_truth
               }


    def save_model(self, is_best=False, recording=None, train_loss=None, train_accuracy=None, train_F1=None, ep=None):
        """
        saves model to file, defaults to save best version based on eval.
        """
        self.model.to('cpu')
        state = {'net': self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f'{self.save_name}_best.pth'
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            if self.verbose >= 0:
                print('saving best model version...')
            else:
                self.writer.add_scalar('model_save_log(loss)', train_loss, ep + 1)
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        """
        loads model from file
        """
        state_dict = torch.load(os.path.join(self.save_dir, f"{self.save_name}_best.pth"))['net']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
