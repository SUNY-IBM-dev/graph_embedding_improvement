import os
import torch
import shutil
import random
from tensorboardX import SummaryWriter
from torch.optim import Adam

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from joblib.externals.loky.backend.context import get_context

import pandas as pd
pd.set_option('display.width', None)


import pprint
import numpy as np
import copy
import time


# [ Added by JY @ 2022-06-27 ]:
# For Reproducibility from PyTorch (Not that there is a PyTorch operation which )
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#debug_mode (str or int) – If “default” or 0, don’t error or warn on nondeterministic operations. 
#If “warn” or 1, warn on nondeterministic operations. If “error” or 2, error on nondeterministic operations.
torch.set_deterministic_debug_mode(debug_mode="default")



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

def get_dataloader(dataset, batch_size, data_split_ratio, split_shuffle_seed):
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
    # [Added by JY @ 2022-07-13]: for compatibility w/ sklearn-GridSearchCV
    # shuffle the data once for good measure
    dataset_deepcopy = list(dataset)                            # For reproducibility in "non-single-experiment context", deep-copy and don't touch original dataset 
    random.Random(split_shuffle_seed).shuffle(dataset_deepcopy) # For reproducibility in "non-single-experiment context", 
                                                                # make local-RNG w/ seed then shuffle with that. 

    if len(dataset) > 2:
        benign_dataset, malware_dataset = [data for data in dataset_deepcopy if data.y == 0] , [data for data in dataset_deepcopy if data.y == 1] 
    else:
        benign_dataset, malware_dataset = dataset_deepcopy

    # [ Added by JY @ 2022-07-13 ]: 
    if data_split_ratio == [1,0,0]:
        #print(f"Benign: #train:{len(benign_dataset)}")
        #print(f"Malware: #train:{len(malware_dataset)}")
        train_dataset = benign_dataset + malware_dataset
        dataloader = dict()

        # "pin_memory": https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
        # "num_workers"
        dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, 
                                         shuffle=True, generator=torch.Generator().manual_seed(split_shuffle_seed),
                                         #multiprocessing_context=get_context('loky'), 
                                         num_workers = 4, 
                                         pin_memory = True
                                         )
        return dataloader
    
    # If we need to split-data
    def split_data(dataset, data_split_ratio, split_seed):
        """
        helper function to split data
        """

        # if data is only seperated to train, eval == 0.8, 0.2
        # where eval == test sets
        if data_split_ratio[-1] == 0.0:
            num_train = int(data_split_ratio[0] * len(dataset))
            num_eval = len(dataset) - num_train
            # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
            train, _eval = random_split(dataset, lengths=[num_train, num_eval], generator=torch.Generator().manual_seed(split_seed))
            return train, _eval, None

        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval
    
        train, _eval, test = random_split(dataset,
                                          lengths=[num_train, num_eval, num_test],
                                          generator=torch.Generator().manual_seed(split_seed))
        return train, _eval, test

    # Note: Already doing stratified-split, despite data is random-splitted, it is done separately for each class (benign, malware)
    # for benign data
    _benign_train, _benign_eval, _benign_test = split_data(benign_dataset, data_split_ratio, split_shuffle_seed)
    print(f"Benign: #train:{len(_benign_train)}, #test:{len(_benign_eval)}")
    # for malware data
    _malware_train, _malware_eval, _malware_test = split_data(malware_dataset, data_split_ratio, split_shuffle_seed)
    print(f"Malware: #train:{len(_malware_train)}, #test:{len(_malware_eval)}")
    
    train_dataset = _benign_train + _malware_train
    eval_dataset = _benign_eval + _malware_eval

    # in case we have set eval==test set
    dataloader = dict()
    if _benign_test is not None and _malware_test is not None:
        test_dataset = _benign_test + _malware_test
        dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, 
                                         shuffle=True, generator=torch.Generator().manual_seed(split_shuffle_seed),
                                         #multiprocessing_context=get_context('loky'),
                                         num_workers = 4, pin_memory = True)
                                         
        dataloader['eval'] = DataLoader(eval_dataset, batch_size=batch_size,
                                        shuffle=True, generator=torch.Generator().manual_seed(split_shuffle_seed),
                                        #multiprocessing_context=get_context('loky'),
                                        num_workers = 4, pin_memory = True)

        dataloader['test'] = DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=True, generator=torch.Generator().manual_seed(split_shuffle_seed),
                                        #multiprocessing_context=get_context('loky'),
                                        num_workers = 4, pin_memory = True)
   
    else:
        dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, generator=torch.Generator().manual_seed(split_shuffle_seed),
                                         #multiprocessing_context=get_context('loky'),
                                         num_workers = 4, pin_memory = True)

        dataloader['eval'] = DataLoader(eval_dataset, batch_size=batch_size, 
                                        shuffle=True, generator=torch.Generator().manual_seed(split_shuffle_seed),
                                        #multiprocessing_context=get_context('loky'),
                                        num_workers = 4, pin_memory = True)
                                         # eval == test
        
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

    def __init__(self, model, dataset, device, 
                 grad_clip_bool = True, grad_clip_value = 2.0, # Added by JY @ 2022-08-18 to enable turning-off gradient-clipping. 
                 best_criteria = 'train_loss', verbose=0, save_dir=None, save_name='model', ** kwargs):
        
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
        self.loader = get_dataloader(dataset, **dataloader_params)

        # [ Added @ 2022-07-16 by JY ] : "best_criteria" can be either "eval_f1" or "train_loss"
        if best_criteria not in ["eval_f1", "train_loss"]:
            raise ValueError("Choices for 'best_criteria' are 'eval_f1' or 'train_loss.'")
        if best_criteria == 'eval_f1' and 'eval' not in self.loader.keys():
            raise ValueError("Invalid 'best_criteria' input of 'eval_f1' as no data allocated for 'eval' set.")
        self.best_criteria = best_criteria

        # [ Added @ 2022-08-18 by JY ] : Set gradient-clipping bool and value.
        #                                This can be useful when searching the optimal learning-rate.
        #                                
        #                                Refer to: "You used an incorect learning rate" section in
        #                                           https://theorangeduck.com/page/neural-network-not-working 
        self.grad_clip_bool = grad_clip_bool
        self.grad_clip_value = grad_clip_value

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
    
        if self.grad_clip_bool: # [Added by JY @ 2022-08-18]
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.grad_clip_value)    
        
        self.optimizer.step()
        
        #print( "train_batch fit: {}".format( list(self.model.parameters())[0] ) )
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
        self.model.eval()
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
        
        # check if test set exists
        # otherwise eval == test, when data is split as 80%-20%
        _key = "test"
        if _key not in self.loader.keys():
            _key = "eval"

        with torch.no_grad():
            losses, preds, accs, truths = [], [], [], []
            # [ Added/Modified @ 2022-06-29 by JY ]: to demonstrate that different seeds lead to ifferent test (eval) datasets.
            for idx, batch in enumerate( self.loader[_key] ):
                #print("idx: {} | batch: {}".format(idx, batch))
                batch = batch.to( self.device )
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

            if self.verbose >= 0:
                print(f"\n preds: {preds}\n  truths: {truths}")
                print(f"\nTest loss: {test_loss:.4f}, Test acc {test_acc:.4f}, Test Precision {test_precision:.4f}, Test Recall {test_recall:.4f}, Test F1 {test_F1:.4f}")
                if ep > 0 and self.writer is not None:
                    self.writer.add_scalar('accuracy/test_acc', test_acc, ep + 1)
                    self.writer.add_scalar('loss/test_loss', test_loss, ep + 1)
                    self.writer.add_scalar('precision/test_precision', test_precision, ep + 1)
                    self.writer.add_scalar('recall/test_recall', test_recall, ep + 1)
                    self.writer.add_scalar('F1_score/test_F1', test_F1, ep + 1)

        self.model.train()  # reset gradiant computation
        
        # Modified by JY for grid-search result saving @ 2022-06-02
        return { "test_loss": test_loss, "test_acc": test_acc, "test_precision": test_precision, "test_recall": test_recall, "test_F1": test_F1,
                 "preds":preds, "truths": truths }

    def train(self, train_params=None, optimizer_params=None ) -> object:  
        """
        runs training iteration 

        [ Added by JY @ 2022-07-16 ] 
        returns the best-model based on best_criteria
        
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
        best_eval_F1 = -1.0
        # best_eval_loss = 0.0
        best_eval_loss = float('inf')   # [ Comment Added @ 2022-07-16 by JY ] Shouldn't 'best_eval_loss' initially be inf instead of 0?
        # [ Added by JY @ 2022-07-16 ]
        best_train_loss = float('inf')

        # [ Comment Added @ 2022-07-16 by JY ] 'early_stop' will depend on 'best_criteria'
        early_stop_counter = 0

        # [ Added by JY @ 2022-07-16 ]: 'best_model' depends on 'best_criteria' 
        best_model = None

        print("\n\n")
        # loop through for all epochs

        epoch_cnt = 0 # Added by JY @ 2023-05-17


        for epoch in range(num_epochs):
            epoch_start_time = time.time()


            is_best = False
            self.model.train()
            losses, accs = [], []

            all_preds = []
            all_truth = []

            for idx, batch in enumerate(self.loader['train']):
                #print("idx: {} | batch: {}".format(idx, batch))
                batch = batch.to(self.device)
                loss, preds = self._train_batch(batch, batch.y)
                losses.append(loss)
                accs.append(preds == batch.y)
                all_preds.append(preds)
                all_truth.append(batch.y)


                ###########################################################################################
                # Added by JY @ 2023-05-17 (GAT Performance Debugging)
                
                # print("-"*100)
                # print(f"epoch {epoch_cnt} | batch {idx}", flush=True)
                # print(f"batch-accuracy: {np.mean(accs[0].tolist())} | batch-loss: {loss}", flush=True)
                # batch_info_df = pd.DataFrame( sorted( list(zip([x.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") for x in batch.name], 
                #                                                 batch.y.tolist(), 
                #                                                 preds.tolist()))),
                #                              columns=["NAME","Y","PRED"] )
                # print(batch_info_df, flush= True)
                ###########################################################################################


                # [Added by JY @ 2022-08-22]
                # for Sanity-Checking : Investiage why training-loss is not decreasing or not leraning
                #  Refer to: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/batch.html
                #            https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
                '''
                if self.verbose >= 0:    
                    #print("="*60)
                    #print("\nEPOCH {} | TRAINING-BATCH INDEX {}".format(epoch, idx))
                    #print("\n*[batch train-loss]: {}".format(loss))
                    #print("*[batch_preds]: {}".format([pred.item() for pred in preds]))
                    #print("*[batch_y]: {}\n".format(batch.y))

                    
                    #For data (== batch.get_example(dataidx))
                    #    x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
                    #    edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
                    #                     > https://courses.engr.illinois.edu/cs357/fa2019/references/ref-10-sparse/
                    #                     > https://www.researchgate.net/figure/Coordinate-COO-format-illustration_fig3_334536999
                    #    edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
                    #    y (Tensor, optional) – Graph-level or node-level ground-truth labels with arbitrary shape. (default: None)
                    

                    subgraphs_node_features = [ pd.DataFrame(batch.get_example(i).x.cpu() ) for i in range(batch._num_graphs) ]
                    subgraphs_nonzero_node_features = [ subgraphs_node_features[i].loc[:, (subgraphs_node_features[i] != 0).any(axis=0)] for i in range(batch._num_graphs) ]
                    subgraphs_nonzero_node_features_descriptive_stats = [ [ {"feature_"+str(colname): {'mean': subgraph_nznf[colname].mean(),'std':subgraph_nznf[colname].std() }} for colname in subgraph_nznf.columns] for subgraph_nznf in subgraphs_nonzero_node_features]                    

                    # Perhaps also prepare descriptive-stats for both edge_indices and edge_attrs
                    # subgraphs_edge_indices = [ pd.DataFrame(batch.get_example(i).edge_index ) for i in range(batch._num_graphs) ]
                    # subgraphs_edge_attrs = [ pd.DataFrame(batch.get_example(i).edge_attr) for i in range(batch._num_graphs) ]


                    subgraphs_num_nodes = [batch.get_example(i).num_nodes for i in range(batch._num_graphs)]
                    subgraphs_num_edges = [batch.get_example(i).num_edges for i in range(batch._num_graphs)]
                    subgraphs_labels = [batch.get_example(i).y.cpu().item() for i in range(batch._num_graphs)]
                    subgraph_preds = [pred.item() for pred in preds]

                    zipped_list = list(zip(subgraphs_labels, subgraph_preds, subgraphs_num_nodes, subgraphs_num_edges, subgraphs_nonzero_node_features_descriptive_stats))
                    batch_subgraphs_summaries = [{"subgraph_label":tup[0],"subgraph_pred":tup[1], "subgraph_num_nodes":tup[2], "subgraph_num_edges":tup[3], "subgraph_NonZero_NodeFeatures_DescStats":tup[4]} for tup in zipped_list ]

                    print("-"*60)
                    for i in range(len(batch_subgraphs_summaries)):
                        print("epoch: {} | batch: {} | subraph#: {}".format(epoch, idx, i))
                        pprint.pprint( batch_subgraphs_summaries[i] , sort_dicts = False )
                    '''

            epoch_cnt += 1 # Added by JY @ 2023-05-17
            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time

            # compute loss and stop early if needed
            train_loss = torch.FloatTensor(losses).mean().item()
            train_acc = torch.cat(accs, dim=-1).float().mean().item()
            # calculate TPr, FPr, F1
            all_preds = torch.cat(all_preds, dim=-1)  # .tolist()
            all_truth = torch.cat(all_truth, dim=-1)  # .tolist()
            #print(all_preds)
            #print(all_truth)
            train_precision, train_recall, train_F1 = calculate_scores(all_preds, all_truth)
            
            # [ Added by JY @ 2022-07-13 ]
            eval_loss = eval_acc = eval_precision = eval_recall = eval_F1 = -1
            if self.best_criteria == "eval_f1":
               eval_loss, eval_acc, eval_precision, eval_recall, eval_F1 = self.eval() 

            if self.verbose >= 0:

                if self.best_criteria == "eval_f1": 
                    print(f'Epoch:{epoch} | Elapsed-Time: {epoch_elapsed_time:.2f} || Train_loss:{train_loss:.4f}, Train_acc:{train_acc:.4f}, Train_Precision:{train_precision:.4f}, Train_Recall:{train_recall:.4f}, Train_F1:{train_F1:.4f} || Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}, Eval_Precision:{eval_precision:.4f}, Eval_Recall:{eval_recall:.4f}, Eval_F1:{eval_F1:.4f}', flush = True)
                else:
                    print(f'Epoch:{epoch} | Elapsed-Time: {epoch_elapsed_time:.2f} || Train_loss:{train_loss:.4f}, Train_acc:{train_acc:.4f}, Train_Precision:{train_precision:.4f}, Train_Recall:{train_recall:.4f}, Train_F1:{train_F1:.4f}', flush = True)
                
            elif self.writer is not None:
                # for training steps
                self.writer.add_scalar('loss/train_loss', train_loss, epoch + 1)
                self.writer.add_scalar('accuracy/train_acc', train_acc, epoch + 1)
                self.writer.add_scalar('precision/train_precision', train_precision, epoch + 1)
                self.writer.add_scalar('recall/train_recall', train_recall, epoch + 1)
                self.writer.add_scalar('F1_score/train_F1', train_F1, epoch + 1)
                # for eval steps
                self.writer.add_scalar('loss/eval_loss', eval_loss, epoch + 1)
                self.writer.add_scalar('accuracy/eval_acc', eval_acc, epoch + 1)
                self.writer.add_scalar('precision/eval_precision', eval_precision, epoch + 1)
                self.writer.add_scalar('recall/eval_recall', eval_recall, epoch + 1)
                self.writer.add_scalar('F1_score/eval_F1', eval_F1, epoch + 1)
                
            
            #  # [ Comment Added @ 2022-07-16 by JY ] : Didn't touch the following 'early_stop' part.
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break

            # change learning rate if there is a schedule
            if lr_schedule:
                lr_schedule.step()
            

            # [Added/Modified by JY @ 2022-07-16 ]: Update 'best_model' based on 'best_criteria'
            if self.best_criteria == 'eval_f1':

                # Following is Dinal's code except one line.
                if best_eval_F1 < eval_F1:
                    is_best = True
                    best_eval_F1 = eval_F1
                    best_model = copy.deepcopy(self.model)  # [Added by JY @ 2022-07-15 ]
                                                            # References regarding deepcopy:
                                                            # https://medium.com/@thawsitt/assignment-vs-shallow-copy-vs-deep-copy-in-python-f70c2f0ebd86
                                                            # https://pytorch.org/docs/stable/generated/torch.nn.Module.html                    
                recording = {'epoch': epoch, 'is_best': str(is_best)}
                if self.save:
                    self.save_model(is_best, recording=recording, eval_accuracy=eval_acc, eval_F1=eval_F1, ep=epoch)
   
            else:
                if train_loss <= best_train_loss:
                    best_train_loss = train_loss        
                    best_model = copy.deepcopy(self.model)

        return best_model   # returns the best model


    def save_model(self, is_best=False, recording=None, eval_accuracy=None, eval_F1=None, ep=None):
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
                self.writer.add_scalar('model_save_log(F1)', eval_F1, ep + 1)
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)


    def load_model(self):
        """
        loads model from file
        """
        state_dict = torch.load(os.path.join(self.save_dir, f"{self.save_name}_best.pth"))['net']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
