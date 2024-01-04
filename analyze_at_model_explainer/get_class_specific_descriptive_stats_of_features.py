import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

''' JY @ 2024-1-4: '''

if __name__ == "__main__":

   # SET PARAMETERS

   # Global_SHAP__Important_FeatureNames__Train_Dataset__csvpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_3__Standard_Message_Passing/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__Best_RF__Dataset_Case_1__1hops__sum_aggr__sum_pool__2023_12_29_060125__final_test__standard_message_passing_graph_embedding__1hops__sum_aggr__sum_pool__2024-01-01_213240/sklearn.ensemble._forest.RandomForestClassifier 1-gram Global-SHAP Important FeatureNames Train-Dataset.csv"
   Global_SHAP__Important_FeatureNames__Test_Dataset__csvpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-04_121751/graph_embedding__GlobalSHAP_TestDataset_df.csv"
      # "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-04_121751/no_graph__GlobalSHAP_TestDataset_csv__df.csv"

   SAVE_PARENT_DIRPATH = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-04_121751"
   
   # -----------------------------------------------------------------------------------------------------------------------------------------

   # Read in dataframes
   # Global_SHAP__Important_FeatureNames__Train_Dataset = pd.read_csv(Global_SHAP__Important_FeatureNames__Train_Dataset__csvpath)
   Global_SHAP__Important_FeatureNames__Test_Dataset = pd.read_csv(Global_SHAP__Important_FeatureNames__Test_Dataset__csvpath)


   # Identify malware rows and benign rows for each dataset
   # * check whether data_name value contains 'malware' or not, since rows in Priti's dataframes resulted in all rows containing 'benign'
   # Global_SHAP__Important_FeatureNames__Train_Dataset__benign_rows = \
   #    Global_SHAP__Important_FeatureNames__Train_Dataset[ ~ Global_SHAP__Important_FeatureNames__Train_Dataset['data_name'].str.contains('malware', case=False, na=False)]

   # Global_SHAP__Important_FeatureNames__Train_Dataset__malware_rows = \
   #    Global_SHAP__Important_FeatureNames__Train_Dataset[ Global_SHAP__Important_FeatureNames__Train_Dataset['data_name'].str.contains('malware', case=False, na=False)]


   Global_SHAP__Important_FeatureNames__Test_Dataset__benign_rows = \
      Global_SHAP__Important_FeatureNames__Test_Dataset[ ~ Global_SHAP__Important_FeatureNames__Test_Dataset['data_name'].str.contains('malware', case=False, na=False)]

   Global_SHAP__Important_FeatureNames__Test_Dataset__malware_rows = \
      Global_SHAP__Important_FeatureNames__Test_Dataset[ Global_SHAP__Important_FeatureNames__Test_Dataset['data_name'].str.contains('malware', case=False, na=False)]


   # Now get the descriptive statistics dataframes

   def get_descriptive_stats_df(dataframe: pd.DataFrame, class_ : str):
      dataframe.set_index('data_name', inplace = True)

      columns_to_drop = ['predict_proba', 'Unnamed: 0']
      dataframe.drop(list( set(dataframe.columns).intersection(set(columns_to_drop)) ), axis = 1, inplace= True)

      mean_df = pd.DataFrame( dataframe.mean(), columns = [f'{class_}_mean'] )
      std_df = pd.DataFrame( dataframe.std(), columns = [f'{class_}_std'] )
      median_df = pd.DataFrame( dataframe.median(), columns = [f'{class_}_median'] )

      return pd.concat([mean_df, std_df, median_df], axis = 1)


   # Global_SHAP__Important_FeatureNames__Train_Dataset__benign_rows__Descriptive_Stats = get_descriptive_stats_df( Global_SHAP__Important_FeatureNames__Train_Dataset__benign_rows, "benign" )
   # Global_SHAP__Important_FeatureNames__Train_Dataset__malware_rows__Descriptive_Stats = get_descriptive_stats_df( Global_SHAP__Important_FeatureNames__Train_Dataset__malware_rows, "malware" )
   # Global_SHAP__Important_FeatureNames__Train_Dataset__Descriptive_Stats_Comparison = \
   #    pd.concat([Global_SHAP__Important_FeatureNames__Train_Dataset__benign_rows__Descriptive_Stats,
   #               Global_SHAP__Important_FeatureNames__Train_Dataset__malware_rows__Descriptive_Stats], axis = 1)


   Global_SHAP__Important_FeatureNames__Test_Dataset__benign_rows__Descriptive_Stats = get_descriptive_stats_df( Global_SHAP__Important_FeatureNames__Test_Dataset__benign_rows, "benign" )
   Global_SHAP__Important_FeatureNames__Test_Dataset__malware_rows__Descriptive_Stats = get_descriptive_stats_df( Global_SHAP__Important_FeatureNames__Test_Dataset__malware_rows, "malware" )
   Global_SHAP__Important_FeatureNames__Test_Dataset__Descriptive_Stats_Comparison = \
      pd.concat([Global_SHAP__Important_FeatureNames__Test_Dataset__benign_rows__Descriptive_Stats,
                 Global_SHAP__Important_FeatureNames__Test_Dataset__malware_rows__Descriptive_Stats], axis = 1) 
   

   # Now save out the descriptive statistics comparison dataframes

   
   SAVE_DIRPATH = os.path.join(SAVE_PARENT_DIRPATH,
                              f"Descriptive_Stats_Comparison__of__Global_SHAP__Important_FeatureNames__@{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}")
   
   if not os.path.exists(SAVE_DIRPATH):
      os.makedirs(SAVE_DIRPATH)

   with open(os.path.join(SAVE_DIRPATH, "description.txt"), "w") as f:
      # f.write("Global_SHAP__Important_FeatureNames__Train_Dataset__csvpath:\n")
      # f.write(f"   {Global_SHAP__Important_FeatureNames__Train_Dataset__csvpath}")
      # f.write("\n")
      f.write("Global_SHAP__Important_FeatureNames__Test_Dataset__csvpath:\n")
      f.write(f"   {Global_SHAP__Important_FeatureNames__Test_Dataset__csvpath}")

   # Global_SHAP__Important_FeatureNames__Train_Dataset__Descriptive_Stats_Comparison.to_csv( os.path.join(SAVE_DIRPATH,
   #                                                                                                       f"Global_SHAP__Important_FeatureNames__Train_Dataset__Descriptive_Stats_Comparison.csv"))

   Global_SHAP__Important_FeatureNames__Test_Dataset__Descriptive_Stats_Comparison.to_csv( os.path.join(SAVE_DIRPATH,
                                                                                                         f"Global_SHAP__Important_FeatureNames__Test_Dataset__Descriptive_Stats_Comparison.csv"))

