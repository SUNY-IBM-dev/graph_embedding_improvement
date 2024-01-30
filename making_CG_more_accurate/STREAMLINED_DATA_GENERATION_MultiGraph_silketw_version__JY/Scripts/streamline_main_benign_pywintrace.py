"""
[Goal]
Streameline the steps from "Before-format elastic search index" to "train-test splitted processed pickle files"
(Starting with elastic-search-indices obtained by general-log-collection)

[Steps]
(Step-1) Given a set of "elastic-search indices (before format)", read in those, format those, and re-upload to elastic-search.
(Step-2) With formated elastic-sarch indices, generate CG and Project SGs.
(Step-3) Train-test split the subgraphs and data-preprocess those into pickle files. 


[Referred Dirs/Files]
# step-1
/data/d2/etw-logs_JY/benign/user-4/Code/format_logentry_on_elasticsearch.py (ocelot)
# step-2 (benign)
/data/d1/jgwak1/BENIGN_USER_ACTIVITY_DATA_GENERATION/NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs (ocelot)
# step-2 (malware)
/data/d1/jgwak1/BENIGN_USER_ACTIVITY_DATA_GENERATION/TARGETTED_SUBGRAPH_GENERATION__UserActivityBenign_OR_MALWARE (ocelot)
# step-3
/data/d1/jgwak1/tabby/GENERAL_LOG_COLLECTION_SUBGRAPHS_20230203/make_data_processable_for_general_log_collection.py
/data/d1/jgwak1/STREAMLINED_DATA_GENERATION/STEP_3_processdata_traintestsplit/1_data_preprocessing
/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/TrainTest_Split_helper.py
/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/TrainTest_Split_with_FullControl.py (*)
/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/TrainTest_Splitter.py
/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/UserActivity_TrainTest_Splitter.py
"""

###################################################################################################################################################
from argparse import ArgumentParser
from distutils.util import strtobool
import datetime
import os
import shutil
import numpy as np
from itertools import islice
from multiprocessing import Process, process


import sys
sys.path.append("/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_1_FORMATTING_ES_INDICES_FILES")
from STEP_1_FORMATTING_ES_INDICES_FILES.format_logentry_on_elasticsearch import run_format_logentry_on_elasticsearch
from STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs.main import run_NonTargetted_Subgraph_Generation
from STEP_2_malware_TARGETTED_SUBGRAPH_GENERATION__UserActivityBenign_OR_MALWARE.main import run_Targetted_Subgraph_Generation

from STEP_3_processdata_traintestsplit.make_data_processable import organize_and_rename_subgraphs_into_Benign_dir, organize_and_rename_subgraphs_into_Malware_dir
sys.path.append("/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_3_processdata_traintestsplit/data_preprocessing")
# print(sys.path)
from STEP_3_processdata_traintestsplit.data_preprocessing.dp_v2_ONLY_TASKNAME_EDGE_ATTR.run_data_processor_MultiEdge import run_data_processor as run_data_processor_MultiEdge_only_taskname_edgeattr_ver
from STEP_3_processdata_traintestsplit.data_preprocessing.dp_v2_ONLY_TASKNAME_EDGE_ATTR.run_data_processor_MultiEdge_5BitNodeAttr import run_data_processor as run_data_processor_MultiEdge_5BitNodeAttr_only_taskname_edgeattr_ver



###################################################################################################################################################
# SET PARAMETERS
 
benign_es_indices =[
   # collected 4 @ 2023-02-05 Sunday ----------------------------------------------------------------------------------------
   "general_log_collection_60mins_started_20230205_20_31_01",
   "general_log_collection_60mins_started_20230205_20_02_31",
   "general_log_collection_60mins_started_20230205_23_12_03",
   "general_log_collection_60mins_started_20230205_21_44_27",
   # collected 17 @ 2023-02-06 Monday ----------------------------------------------------------------------------------------
   "general_log_collection_60mins_started_20230206_14_37_43",
   "general_log_collection_60mins_started_20230206_12_33_28",
   "general_log_collection_60mins_started_20230206_19_31_50",
   "general_log_collection_60mins_started_20230206_15_26_07",
   "general_log_collection_60mins_started_20230206_22_54_16",
   "general_log_collection_60mins_started_20230206_08_39_00",
   "general_log_collection_60mins_started_20230206_18_18_13",
   "general_log_collection_60mins_started_20230206_16_48_54",
   "general_log_collection_60mins_started_20230206_00_12_38",
   "general_log_collection_60mins_started_20230206_10_21_08",
   "general_log_collection_60mins_started_20230206_13_54_39",
   "general_log_collection_60mins_started_20230206_16_33_40",
   "general_log_collection_60mins_started_20230206_15_41_56",
   "general_log_collection_60mins_started_20230206_21_46_07",
   "general_log_collection_60mins_started_20230206_20_35_19",
   "general_log_collection_60mins_started_20230206_09_43_10",
   "general_log_collection_60mins_started_20230206_11_30_29",
   # collected 15 @ 2023-02-07 Tuesday ----------------------------------------------------------------------------------------
   "general_log_collection_60mins_started_20230207_13_54_03",
   "general_log_collection_60mins_started_20230207_14_58_19",
   "general_log_collection_60mins_started_20230207_00_15_05",
   "general_log_collection_60mins_started_20230207_21_46_17",
   "general_log_collection_60mins_started_20230207_10_44_10",
   "general_log_collection_60mins_started_20230207_16_49_54",
   "general_log_collection_60mins_started_20230207_19_37_13",
   "general_log_collection_60mins_started_20230207_18_13_33",
   "general_log_collection_60mins_started_20230207_07_50_52",
   "general_log_collection_60mins_started_20230207_16_10_25",
   "general_log_collection_60mins_started_20230207_09_31_07",
   "general_log_collection_60mins_started_20230207_09_10_19",
   "general_log_collection_60mins_started_20230207_12_38_46",
   "general_log_collection_60mins_started_20230207_11_51_11",
   "general_log_collection_60mins_started_20230207_17_12_56",
   # collected 13 @ 2023-02-08 Wednesday ----------------------------------------------------------------------------------------
   "general_log_collection_60mins_started_20230208_17_22_56",
   "etw0_general_log_collection_60mins_started_20230208_01_00_56", # drop the "etw0_" later.
   "general_log_collection_60mins_started_20230208_21_28_25",
   "general_log_collection_60mins_started_20230208_11_39_32",
   "general_log_collection_60mins_started_20230208_09_53_52_meng",
   "general_log_collection_60mins_started_20230208_10_22_24",
   "general_log_collection_60mins_started_20230208_11_00_13_meng",
   "general_log_collection_60mins_started_20230208_15_34_53",
   "general_log_collection_60mins_started_20230208_11_43_02",
   "general_log_collection_60mins_started_20230208_13_25_03_meng",
   "general_log_collection_60mins_started_20230208_14_29_37_meng",
   "general_log_collection_60mins_started_20230208_16_11_37_meng",
   "general_log_collection_60mins_started_20230208_12_09_40_meng",

]

#benign_es_indices = [ "general_log_collection_60mins_started_20230205_20_31_01" ]

benign_es_indices = \
["whale_chrome_edge_021320231207_recordtime30min_collectedtimespan10min_441565entries", 
 "whale_chrome_edge_021320230249_recordtime45min_collectedtimespan45min_341610entries",
 "whale_chrome_edge_021320130345_recordtime45min_collectedtimespan45min_366682entries"]


old_malware_dict = \
{'mal0': 4196, 'mal1':6860, 'mal2': 1004, 'mal3': 4024, 'mal4': 5876, 'mal5': 3280, 
'mal6': 8832, 'mal7': 4024, 'mal8': 3824, 'mal9': 5876, 'mal10': 5876, 'mal11': 5876,'mal12': 4024,
'mal13': 7492,'mal14': 4196,'mal15': 6860,'mal16': 3824,'mal17': 4024,'mal18': 5876,'mal19': 4820, 
'mal20':3824,'mal21': 8680, 'mal22': 3112, 'mal23': 3112, 'mal24': 3112, 'mal25': 3880, 'mal26': 8340,
'mal27': 3528, 'mal28': 5876, 'mal29': 3824, 'mal30': 3436, 'mal31': 4584, 'mal32': 6860, 'mal33': 3112,
'mal34': 3112, 'mal35': 7492, 'mal36': 4196, 'mal37': 3824, 'mal38': 3112, 'mal39': 5876, 'mal40': 3112, 
'mal41': 3824,'mal42': 5876, 'mal43': 5940, 'mal44': 4024, 'mal45': 5876, 'mal46': 4024, 'mal47': 5876, 
'mal48': 7632,'mal49': 3824, 'mal50': 3280, 'mal51': 4024, 'mal52': 3280, 'mal53': 8360, 'mal54': 8340,
'mal55': 3112,'mal56': 4024, 'mal57': 3824, 'mal58': 3280, 'mal59': 8336, 'mal60': 5940, 'mal61': 4196,
'mal62': 5876, 'mal63': 3112, 'mal64': 3824, 'mal65': 8360, 'mal66': 6860, 'mal67': 5940, 'mal68': 6860,
'mal69': 7060, 'mal70': 4024, 'mal71': 7492, 'mal72': 3824, 'mal73': 5876, 'mal74': 5876, 'mal75': 3824,
'mal76': 4196, 'mal77': 4196, 'mal78': 3112, 'mal79': 5876, 'mal80': 7800, 'mal81': 4196, 'mal82': 3824,
'mal83': 3880, 'mal84': 3112, 'mal85': 7896, 'mal86': 3824, 'mal87': 4196, 'mal88': 3280, 'mal89': 3824,
'mal90': 4024, 'mal91': 8232, 'mal92': 8360, 'mal93': 5876, 'mal94': 8680, 'mal95': 8340, 'mal96': 2764,
'mal97': 5876, 'mal98': 5876, 'mal99': 7632, 'mal100': 4024, 'mal101': 4196, 'mal102': 8832, 'mal103': 3956,
'mal104': 2992, 'mal105': 4864, 'mal106': 5876, 'mal107': 4024, 'mal108': 5876, 'mal109': 5876, 'mal110': 7796,
'mal111': 3824, 'mal112': 8232, 'mal113': 5876, 'mal114': 4196, 'mal115': 3112, 'mal116': 3112, 'mal117': 4024,
'mal118': 3112, 'mal119': 5876, 'mal120': 3112, 'mal121': 5876, 'mal122': 3112, 'mal123': 4196, 'mal124': 8340,
'mal125': 5876, 'mal126': 3824, 'mal127': 7344, 'mal128': 8340, 'mal129': 2764, 'mal130': 6860, 'mal131': 4820,
'mal132': 299, 'mal133': 7632, 'mal134': 7632, 'mal135': 7800, 'mal136': 5876, 'mal137': 8340, 'mal138': 4820,
'mal139': 8360, 'mal140': 3824, 'mal141': 8680, 'mal142': 4024, 'mal143': 3112, 'mal144': 4196, 'mal145': 6088,
'mal146': 8340, 'mal147': 4196, 'mal148': 4196, 'mal149': 3824, 'mal150': 4024, 'mal151': 4196, 'mal152': 7496,
'mal153': 3284, 'mal154': 5940, 'mal155': 5876, 'mal156': 4196, 'mal157': 5876, 'mal158': 5876, 'mal159': 4196,
'mal160': 8340, 'mal161': 2764, 'mal162': 3112, 'mal163': 5876, 'mal164': 1748, 'mal165': 6860, 'mal166': 4024,
'mal167': 3824, 'mal168': 4196, 'mal169': 1748, 'mal170': 3112, 'mal171': 4864, 'mal172': 3956, 'mal173': 8680,
'mal174': 4024, 'mal175': 7492, 'mal176': 7828, 'mal177':8360, 'mal178': 5876, 'mal179': 3112, 'mal180': 8680,
'mal181': 3956, 'mal182': 4584, 'mal183': 8340, 'mal184': 1748, 'mal185': 6860, 'mal186': 4584, 'mal187': 4024,
'mal188': 7632, 'mal189': 3280, 'mal190': 1748, 'mal191': 4024, 'mal192': 5876, 'mal193': 4024, 'mal194': 4864,
'mal195': 3112, 'mal196': 5876, 'mal197': 5876, 'mal198': 5876, 'mal199': 3280, 'mal200': 7492, 'mal201': 4024,
'mal202': 4196, 'mal203': 3112, 'mal204': 3824, 'mal205': 5876, 'mal206': 6860, 'mal207': 5876, 'mal208': 4024,
'mal209': 3956, 'mal210': 4820, 'mal211': 4024, 'mal212': 5876, 'mal213': 3824, 'mal214': 4044, 'mal215': 4024,
'mal216': 8340, 'mal217': 4196, 'mal218': 8360, 'mal219': 3280, 'mal220': 4024, 'mal221': 4196, 'mal222': 3824,
'mal223': 7088, 'mal224': 3824, 'mal225': 4584, 'mal226': 3824, 'mal227': 4196, 'mal228': 4024, 'mal229': 4024,
'mal230': 6860, 'mal231': 8680, 'malware0': 2764, 'malware1': 3280, 'malware2': 5876, 'malware3': 5876, 'malware4': 6636,
'malware5': 3112, 'malware6': 3824, 'malware7': 4024, 'malware8': 7060, 'malware9': 7492, 'malware10': 4820,
'malware11': 6636, 'malware12': 8680, 'malware13': 7492, 'malware14': 5940, 'malware15': 3824, 'malware16': 8680,
'malware17': 4024, 'malware18': 5876, 'malware19': 5876, 'malware20': 8360, 'malware21': 3112, 'malware22': 6636,
'malware23': 5876, 'malware24': 7632, 'malware25': 8360, 'malware26': 7632, 'malware27': 8340, 'malware28': 5940,
'malware29': 3880, 'malware30': 3956, 'malware31': 3956, 'malware32': 5940, 'malware33': 3880, 'malware34': 5940,
'malware35': 7492, 'malware36': 5876, 'malware37': 7880, 'malware38': 3112, 'malware39': 3824, 'malware40': 2204, 
'malware41': 4196, 'malware42': 7632, 'malware43': 3956, 'malware44': 1004, 'malware45': 7044, 'malware46': 7632,
'malware47': 3280, 'malware48': 7044, 'malware49': 4024, 'malware50': 7044, 'malware51': 5940, 'malware52': 4864,
'malware53': 8360, 'malware54': 4024, 'malware55': 5876, 'malware56': 5876, 'malware57': 8360, 'malware58': 3880,
'malware59': 3824, 'malware60': 6632, 'malware61': 8680, 'malware62': 4196, 'malware63': 5876, 'malware64': 7492,
'malware65': 5876, 'malware66': 5940, 'malware67': 8232, 'malware68': 7632, 'malware69': 4024, 'malware70': 5876,
'malware71': 3112, 'malware72': 3112, 'malware73': 7632, 'malware74': 4024, 'malware75': 6860, 'malware76': 7492,
'malware77': 7492, 'malware78': 1748, 'malware79': 5876, 'malware80': 5876, 'malware81': 5876, 'malware82': 4024,
'malware83': 3880, 'malware84': 4196, 'malware85': 5940, 'malware86': 4196, 'malware87': 8680, 'malware88': 7044,
'malware89': 6860, 'malware90': 3824, 'malware91': 5876, 'malware92': 5876, 'malware93': 8360, 'malware94': 8224,
'malware95': 5876, 'malware96': 3112, 'malware97': 8360, 'malware98': 8360, 'malware99': 8832, 'malware100': 4196,
'malware101': 5876, 'malware102': 5124, 'malware103': 7800, 'malware104': 4024, 'malware105': 3112, 'malware106': 7344,
'malware107': 3280, 'malware108': 4864, 'malware109': 4024, 'malware110': 3112, 'malware111': 3280, 'malware112': 3800,
'malware113': 5876, 'malware114': 5876, 'malware115': 4024, 'malware116': 8832, 'malware117': 7492, 'malware118': 2764,
'malware119': 1988, 'malware120': 6860, 'malware121': 9208, 'malware122': 5940, 'malware123': 2764, 'malware124': 8340,
'malware125': 3824, 'malware126': 7044, 'malware127': 4864, 'malware128': 7044, 'malware129': 6636, 'malware130': 7632,
'malware131': 5940, 'malware132': 6636, 'malware133': 3880, 'malware134': 5876, 'malware135': 3112, 'malware136': 2992,
'malware137': 4024, 'malware138': 3956, 'malware139': 3824, 'malware140': 4864, 'malware141': 3824, 'malware142': 5876,
'malware143': 8832, 'malware144': 6860, 'malware145': 3280, 'malware146': 7632, 'malware147': 7632, 'malware148': 4024,
'malware149': 3112, 'malware150': 5876, 'malware151': 3956, 'malware152': 5876, 'malware153': 7492, 'malware154': 4196, 
'malware155': 3824, 'malware156': 5876, 'malware157': 3824, 'malware158': 7044, 'malware159': 4584, 'malware160': 5940,
'malware161': 3280, 'malware162': 3112, 'malware163': 4196, 'malware164': 5876, 'malware165': 7044, 'malware166': 5876,
'malware167': 4820, 'malware168': 5876, 'malware169': 4196, 'malware170': 5940, 'malware171': 7060, 'malware172': 6636,
'malware173': 4024, 'malware174': 3824, 'malware175': 3956, 'malware176': 1748, 'malware177': 5940, 'malware178': 5940, 
'malware179': 8680, 'malware180': 6852, 'malware181': 7508, 'malware182': 7044, 'malware183': 2764, 'malware184': 5876, 
'malware185': 7632, 'malware186': 8224, 'malware187': 8340, 'malware188': 2764, 'malware189': 8680, 'malware190': 7044, 
'malware191': 3112, 'malware192': 7632, 'malware193': 5876, 'malware194': 8680, 'malware195': 4196, 'malware196': 4024, 
'malware197': 7472, 'malware198': 7044, 'malware199': 7632, 'malware200': 5876, 'malware201': 8680, 'malware202': 2204, 
'malware203': 3112, 'malware204': 8680, 'malware205': 3112, 'malware206': 4584, 'malware207': 7044, 'malware208': 3112, 
'malware209': 8360, 'malware210': 3280, 'malware211': 7344, 'malware212': 7044, 'malware213': 6860, 'malware214': 4024, 
'malware215': 6860, 'malware216': 4820, 'malware217': 5876, 'malware218': 7044, 'malware219': 8232, 'malware220': 4196, 
'malware221': 4024, 'malware222': 8832, 'malware223': 4024, 'malware224': 1748, 'malware225': 4196, 'malware226': 3880, 
'malware227': 4024, 'malware228': 8336, 'malware229': 3112, 'malware230': 5876, 'malware231': 4024, 'malware232': 3112}


non_trojan_malware_dict = {
'non_trojan_malware5': '2848',
'non_trojan_malware6': '2848',
'non_trojan_malware7': '2724',
'non_trojan_malware8': '4688',
'non_trojan_malware9': '3800',
'non_trojan_malware10': '3288',
'non_trojan_malware11': '100',
'non_trojan_malware12': '2600',
'non_trojan_malware13': '2848',
'non_trojan_malware14': '2848',
'non_trojan_malware15': '2144',
'non_trojan_malware16': '2252',
'non_trojan_malware17': '3288',
'non_trojan_malware18': '2252',
'non_trojan_malware19': '2144',
'non_trojan_malware20': '2848',
'non_trojan_malware21': '2600',
'non_trojan_malware22': '3800',
'non_trojan_malware23': '2600',
'non_trojan_malware24': '8080',
'non_trojan_malware25': '3288',
'non_trojan_malware26': '100',
'non_trojan_malware27': '3800',
'non_trojan_malware28': '6416',
'non_trojan_malware29': '2668',
'non_trojan_malware30': '2252',
'non_trojan_malware31': '100',
'non_trojan_malware32': '2600',
'non_trojan_malware33': '3288',
'non_trojan_malware34': '3288',
'non_trojan_malware35': '2848',
'non_trojan_malware36': '3288',
'non_trojan_malware37': '3288',
'non_trojan_malware38': '4688',
'non_trojan_malware39': '2848',
'non_trojan_malware40': '8040',
'non_trojan_malware41': '6204',
'non_trojan_malware42': '2600',
'non_trojan_malware43': '3800',
'non_trojan_malware44':	'3288',
'non_trojan_malware45':	'3288',
'non_trojan_malware46':	'3288',
'non_trojan_malware47':	'4396',
'non_trojan_malware48':	'3800',
'non_trojan_malware49':	'3288',
#'non_trojan_malware50':	-
#'non_trojan_malware51':	-
'non_trojan_malware52':	'3800',
}


#>>> for x in range(0, 47):
#...     print(f"'zoo_malware{x}' : '',")
zoo_malware_dict = {
'zoo_malware0' : '2600',
'zoo_malware1' : '2848',
'zoo_malware2' : '3800',
'zoo_malware3' : '2600',
'zoo_malware4' : '100',
'zoo_malware5' : '100',
'zoo_malware6' : '100',
'zoo_malware7' : '2252',
'zoo_malware8' : '2144',
'zoo_malware9' : '1404',
'zoo_malware10' : '6204',
'zoo_malware11' : '100',
'zoo_malware12' : '2668',
'zoo_malware13' : '3676',
'zoo_malware14' : '100',
'zoo_malware15' : '1404',
'zoo_malware16' : '8096',
'zoo_malware17' : '2252',
'zoo_malware18' : '3800',
'zoo_malware19' : '100',
'zoo_malware20' : '2724',
'zoo_malware21' : '2848',
'zoo_malware22' : '100',
'zoo_malware23' : '1404',
'zoo_malware24' : '3068',
'zoo_malware25' : '3288',
'zoo_malware26' : '100',
'zoo_malware27' : '2600',
'zoo_malware28' : '100',
'zoo_malware29' : '3288',
#'zoo_malware30' : '',
'zoo_malware31' : '3288',
'zoo_malware32' : '1404',
'zoo_malware33' : '2848',
'zoo_malware34' : '2724',
'zoo_malware35' : '2600',
'zoo_malware36' : '3236',
'zoo_malware37' : '8040',
'zoo_malware38' : '8040',
'zoo_malware39' : '3768',
'zoo_malware40' : '8096',
'zoo_malware41' : '1404',
'zoo_malware42' : '5224',
'zoo_malware43' : '100',
#'zoo_malware44' : '',
'zoo_malware45' : '2252',
'zoo_malware46' : '2848',
}

zoo_malware_set2_dict =  { "zoo_malware_set2_1" : "3288", "zoo_malware_set2_2": "1404", "zoo_malware_set2_4" : "2848", "zoo_malware_set2_7": "3288", "zoo_malware_set2_8" : "4016", "zoo_malware_set2_9": "2144", "zoo_malware_set2_12" : "1404", "zoo_malware_set2_13": "2252", "zoo_malware_set2_14" : "100", "zoo_malware_set2_15": "2144", "zoo_malware_set2_17" : "3800", "zoo_malware_set2_19": "1404", "zoo_malware_set2_20" : "1404", "zoo_malware_set2_21": "1404", "zoo_malware_set2_22" : "6204", "zoo_malware_set2_23": "2904", "zoo_malware_set2_24" : "100", "zoo_malware_set2_25": "8040", "zoo_malware_set2_26" : "8040", "zoo_malware_set2_27": "1748", "zoo_malware_set2_28" : "5316"}




zoo_malware_set3_dict = {
'zoo_malware_set3_0': 5460,
'zoo_malware_set3_1': 384,
'zoo_malware_set3_2': 1404,
'zoo_malware_set3_3': 2252,
'zoo_malware_set3_4': 2252,
'zoo_malware_set3_5': 2848,
'zoo_malware_set3_6': 3288,
'zoo_malware_set3_7': 2600,
'zoo_malware_set3_8': 100,
'zoo_malware_set3_9': 2252,
'zoo_malware_set3_10': 328,
'zoo_malware_set3_11': 2848,
'zoo_malware_set3_12': 3800,
'zoo_malware_set3_13': 3768,
'zoo_malware_set3_14': 100,
'zoo_malware_set3_15': 3288,
'zoo_malware_set3_16': 2848,
'zoo_malware_set3_17': 8040,
'zoo_malware_set3_18': 3800,
'zoo_malware_set3_19': 100,
'zoo_malware_set3_20': 2848,
'zoo_malware_set3_21': 2252,
'zoo_malware_set3_22': 7072,
'zoo_malware_set3_23': 3288,
'zoo_malware_set3_24': 1404,
'zoo_malware_set3_25': 100,
'zoo_malware_set3_26': 1404,
'zoo_malware_set3_27': 3288,
'zoo_malware_set3_28': 5460,
'zoo_malware_set3_29': 3800,
'zoo_malware_set3_30': 3288,
'zoo_malware_set3_31': 2848,
'zoo_malware_set3_32': 2252,
'zoo_malware_set3_33': 2252,
'zoo_malware_set3_34': 100,
'zoo_malware_set3_35': 5316,
'zoo_malware_set3_36': 3288,
'zoo_malware_set3_37': 100,
'zoo_malware_set3_38': 3288,
'zoo_malware_set3_39': 100,
'zoo_malware_set3_40': 2600,
'zoo_malware_set3_41': 1404,
'zoo_malware_set3_42': 100,
'zoo_malware_set3_43': 1404,
'zoo_malware_set3_44': 8096,
'zoo_malware_set3_45': 100,
'zoo_malware_set3_46': 1404,
'zoo_malware_set3_47': 1404,
'zoo_malware_set3_48': 3288,
'zoo_malware_set3_49': 100,
'zoo_malware_set3_50': 2724,
'zoo_malware_set3_51': 100,
'zoo_malware_set3_52': 3800,
'zoo_malware_set3_53': 4616,
'zoo_malware_set3_54': 2600,
'zoo_malware_set3_55': 3288,
'zoo_malware_set3_56': 2252,
'zoo_malware_set3_57': 2252,
'zoo_malware_set3_58': 3288,
'zoo_malware_set3_59': 2252,
'zoo_malware_set3_60': 2600,
'zoo_malware_set3_61': 2724,
'zoo_malware_set3_62': 2252,
'zoo_malware_set3_63': 2600,
'zoo_malware_set3_64': 2848,
'zoo_malware_set3_65': 2600,
'zoo_malware_set3_66': 1864,
'zoo_malware_set3_67': 2252,
'zoo_malware_set3_68': 8040,
'zoo_malware_set3_69': 3288,
'zoo_malware_set3_70': 2848,
'zoo_malware_set3_71': 2252,
'zoo_malware_set3_72': 2600,
'zoo_malware_set3_73': 2252,
'zoo_malware_set3_74': 3800,
'zoo_malware_set3_75': 2848,
'zoo_malware_set3_76': 3288,
'zoo_malware_set3_77': 5224,
'zoo_malware_set3_78': 2600,
'zoo_malware_set3_79': 1404,
'zoo_malware_set3_80': 2600,
'zoo_malware_set3_81': 2144,
'zoo_malware_set3_82': 2600,
'zoo_malware_set3_83': 8040,
'zoo_malware_set3_84': 2848,
'zoo_malware_set3_85': 2848,
'zoo_malware_set3_86': 2600,
'zoo_malware_set3_87': 1748,
'zoo_malware_set3_88': 2252,
'zoo_malware_set3_89': 1404,
'zoo_malware_set3_90': 3288,
'zoo_malware_set3_91': 3768,
'zoo_malware_set3_92': 2252,
'zoo_malware_set3_93': 100,
'zoo_malware_set3_94': 2600,
'zoo_malware_set3_95': 1404,
'zoo_malware_set3_96': 2848,
'zoo_malware_set3_97': 100,
'zoo_malware_set3_98': 2600,
'zoo_malware_set3_99': 2600,
'zoo_malware_set3_100': 3768,
'zoo_malware_set3_101': 2848,
'zoo_malware_set3_102': 1404,
'zoo_malware_set3_103': 2252,
'zoo_malware_set3_104': 2848,
'zoo_malware_set3_105': 100,
'zoo_malware_set3_106': 5292,
'zoo_malware_set3_107': 2848,
'zoo_malware_set3_108': 8040,
'zoo_malware_set3_109': 3288,
'zoo_malware_set3_110': 2252,
'zoo_malware_set3_111': 2252,
'zoo_malware_set3_112': 2252,
'zoo_malware_set3_113': 5316,
'zoo_malware_set3_114': 2252,
'zoo_malware_set3_115': 100,
'zoo_malware_set3_116': 8040,
'zoo_malware_set3_117': 2252,
'zoo_malware_set3_118': 6204,
'zoo_malware_set3_119': 2252,
'zoo_malware_set3_120': 100,
'zoo_malware_set3_121': 2668,
'zoo_malware_set3_122': 8096,
'zoo_malware_set3_123': 100,
'zoo_malware_set3_124': 2600,
'zoo_malware_set3_125': 2668,
'zoo_malware_set3_127': 2848,
'zoo_malware_set3_128': 100,
'zoo_malware_set3_129': 3768,
'zoo_malware_set3_130': 3288,
'zoo_malware_set3_131': 8040,
'zoo_malware_set3_132': 1404,
'zoo_malware_set3_133': 8040,
'zoo_malware_set3_134': 1404,
'zoo_malware_set3_135': 3288,
'zoo_malware_set3_136': 8080,
'zoo_malware_set3_137': 3288,
'zoo_malware_set3_138': 100,
'zoo_malware_set3_139': 3288,
'zoo_malware_set3_140': 3288,
'zoo_malware_set3_141': 8040,
}
zoo_malware_set3_dict = {k: str(v) for k,v in zoo_malware_set3_dict.items()}


# powershell_benign_priti_meng_set1_set2_set3 = {
#    # powershell.exe is the ImageName for the PROCESSSTART event, not cmd.exe

# 	# /data/d1/jgwak1/tabby/Powershell_benign_datasets_Priti_set1
# 	"benign_powershell__add-memo":"616",  
# 	"benign_powershell__alert":"5444",  
# 	"benign_powershell__cd-autostart":"5916",  
# 	"benign_powershell__change-wallpaper":"8848",  
# 	"benign_powershell__play-cat-sound":"5492",

# 	# /data/d1/jgwak1/tabby/Powershell_benign_dataset_Meng_set2 ( I have updated the index on elastic search now
# 	#                                                             install-github-cli belongs to Meng_set2)

# 	"benign_powershell__check-cpu":"9080",
# 	"benign_powershell__check-dns":"6632",          
# 	"benign_powershell__check-drive-space":"9168",  
# 	"benign_powershell__check-file-system":"3000",
# 	"benign_powershell__check-health":"2916",                
# 	"benign_powershell__check-windows-system-files":"1640",  
# 	"benign_powershell__install-chrome-browser":"8180",  
# 	"benign_powershell__install-github-cli":"8728",      
# 	"benign_powershell__play-jingle-bells":"4008",       
# 	"benign_powershell__speak-test":"5640",
# 	"benign_powershell__speak-text":"2872",
# 	"benign_powershell__tell-joke":"8108",

# 	# /data/d1/jgwak1/tabby/Powershell_benign_datasets_Priti_Meng_set3 (Priti @ 2023-03-30: please ignore the sample install-github-cli from (Priti_Meng_set3))

# 	"benign_powershell__check-day":"5352",
# 	"benign_powershell__clear-recycle-bin":"1472",  
# 	"benign_powershell__convert-ps2bat":"9076",     
# 	"benign_powershell__download-dir":"9020",       
# 	"benign_powershell__edit":"8428",               
# 	"benign_powershell__enable-god-mode":"8668",    
# 	"benign_powershell__export-to-manuals":"8720",  
# 	"benign_powershell__get-md5":"8896",                
# 	"benign_powershell__install-knot-resolver":"9044",  
# 	"benign_powershell__install-ssh-client":"3588",     
# 	"benign_powershell__install-ssh-server":"8736",     
# 	"benign_powershell__install-updates":"4304",        
# 	"benign_powershell__introduce-powershell":"8560",   
# 	"benign_powershell__list-cli-tools":"6852",         
# 	"benign_powershell__list-empty-dirs":"9024",  
# 	"benign_powershell__locate-city":"4328",      
# 	"benign_powershell__make-install":"9020",     
# 	"benign_powershell__merry-christmas":"9200",  
# 	"benign_powershell__moon":"5728",             
# 	"benign_powershell__new-qrcode":"8772",
# 	"benign_powershell__open-chrome":"680",
# 	"benign_powershell__remove-empty-dirs":"3616",
# 	"benign_powershell__replace-in-files":"8836",
# 	"benign_powershell__save-screenshot":"9088",
# 	"benign_powershell__write-quote":"8276",
# 	"benign_powershell__write-vertical":"8056",

# }

# prof scott-epx 2023-04-27
queryea_experiment_dict = \
{
"query_experiment_check-powershell_checkpoint":"10372",
"query_experiment_check-powershell_checkpoint1":"9368",
"query_experiment_check-powershell_checkpoint2":"10724",
"query_experiment_check-powershell_checkpoint3":"7908",
"query_experiment_check-powershell_checkpoint4":"8464",
"query_experiment_check-powershell_checkpoint5":"7588",
}


caldera_stockpile_builtin_attack_dict = {

# In elastic-search (Kibana):
   #   Task Name: PROCESSSTART and ImageName : *splunkd.exe*
   #   Task Name: PROCESSSTART and ProcessId: <caldera-agent-process-pid>

'caldera_builtin_adversary_01c96671-afd3-47d4-8d31-8c116cc0221a_20230308_06_47_55':'7772',
'caldera_builtin_adversary_01d77744-2515-401a-a497-d9f7241aac3c_20230308_03_43_17':'3536',
'caldera_builtin_adversary_09ad625e-6cba-490f-afe3-5417e7edb9c6_20230308_08_37_45':'4764',
'caldera_builtin_adversary_0b5636cf-f019-4ec9-aa7c-6e4f55505374_20230308_10_28_07':'168',
'caldera_builtin_adversary_0b73bf34-fc5b-48f7-9194-dce993b915b1_20230308_04_19_48':'7772',
'caldera_builtin_adversary_0f4c3c67-845e-49a0-927e-90ed33c044e0_20230308_09_51_29':'168',
'caldera_builtin_adversary_1a98b8e6-18ce-4617-8cc5-e65a1a9d490e_20230308_13_32_50':'4764',
'caldera_builtin_adversary_4c28c132-d7d7-4a04-8908-d643b7cb1d58_20230308_11_05_06':'7596',
'caldera_builtin_adversary_50855e29-3b4e-4562-aa55-b3d7f93c26b8_20230308_02_29_54':'2092',
'caldera_builtin_adversary_564ae20d-778d-4965-93dc-b523be2e2ab4_20230308_00_04_09':'2248',
'caldera_builtin_adversary_5d3e170e-f1b8-49f9-9ee1-c51605552a08_20230308_12_56_05':'4660',
'caldera_builtin_adversary_725226e0-45b8-4432-84ee-144d3f37ff8d_20230308_09_14_31':'7608',
'caldera_builtin_adversary_78e7504d-968f-477d-8806-4d6c04b94431_20230308_03_06_30':'4764',
'caldera_builtin_adversary_bcdbf6b9-14c5-495c-be84-37bce32c312b_20230308_05_33_27':'4636',
'caldera_builtin_adversary_c724545d-a4cc-492e-8075-2ab9a699c847_20230308_08_00_43':'7772',
'caldera_builtin_adversary_d6ea4c1e-7959-4eb1-a292-b6fd2b06c73e_20230308_01_16_42':'184',
'caldera_builtin_adversary_dbd49a4a-ba2d-40d0-9348-2db24fc4b0b6_20230308_00_41_04':'1380',
'caldera_builtin_adversary_ddbd1850-5fd7-41d5-a7a1-1b15dac49090_20230308_14_09_17':'168',
'caldera_builtin_adversary_de07f52d-9928-4071-9142-cb1d3bd851e8_20230308_06_11_05':'4660',
'caldera_builtin_adversary_e4324b88-8836-4803-b6b7-09b3c6cd4e94_20230308_07_24_09':'7596',
'caldera_builtin_adversary_e89a10d3-004f-4c15-b0eb-d1ba76a4b67f_20230308_01_53_01':'168',
'caldera_builtin_adversary_eddc8f03-f930-41e7-95ba-33fb87bfed74_20230308_11_42_00':'4636',
'caldera_builtin_adversary_f98193a0-8b5b-4b5e-a5aa-e8c3adfcd4e6_20230308_12_18_58':'168',
'caldera_builtin_adversary_fbc41624-1052-490c-b5ec-4fd718e2501d_20230308_04_56_43':'2248',
}



caldera_custom_manual_attack_set_1_dict = {

"caldera_custom_manual_attack_adversaryid__custom_adversary_1__in_vm_20230402_21_26_54": "168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_2__in_vm_20230402_22_00_37": "168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_3__in_vm_20230402_22_34_41": "168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_4__in_vm_20230402_23_08_37": "280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_5__in_vm_20230402_23_41_42": "280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_6__in_vm_20230403_00_09_08": "168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_7__in_vm_20230403_00_47_18": "168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_8__in_vm_20230403_01_22_06": "168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_9__in_vm_20230403_01_56_50": "7124",
"caldera_custom_manual_attack_adversaryid__custom_adversary_10__in_vm_20230403_02_33_13": "280",                          
"caldera_custom_manual_attack_adversaryid__custom_adversary_11__in_vm_20230403_03_08_09": "168",                          
"caldera_custom_manual_attack_adversaryid__custom_adversary_12__in_vm_20230403_03_42_45": "280",                          
"caldera_custom_manual_attack_adversaryid__custom_adversary_13__in_vm_20230403_04_17_49": "168",                          
"caldera_custom_manual_attack_adversaryid__custom_adversary_14__in_vm_20230403_04_52_48": "168",                          
"caldera_custom_manual_attack_adversaryid__custom_adversary_15__in_vm_20230403_05_27_57": "280",                          
"caldera_custom_manual_attack_adversaryid__custom_adversary_16__in_vm_20230403_06_01_03": "168",                          
                      
}


caldera_custom_manual_attack_set_2_dict = {

"caldera_custom_manual_attack_adversaryid__custom_adversary_17__in_vm_20230405_17_09_38":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_18__in_vm_20230405_17_48_17":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_19__in_vm_20230405_18_22_33":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_20__in_vm_20230405_18_56_39":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_21__in_vm_20230405_19_32_21":"7488",
"caldera_custom_manual_attack_adversaryid__custom_adversary_22__in_vm_20230405_20_05_18":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_23__in_vm_20230405_20_41_39":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_24__in_vm_20230405_21_18_27":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_25__in_vm_20230405_21_57_46":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_26__in_vm_20230405_22_33_39":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_27__in_vm_20230405_23_09_31":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_28__in_vm_20230405_23_47_22":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_29__in_vm_20230406_00_25_16":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_30__in_vm_20230406_01_03_41":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_31__in_vm_20230406_01_41_03":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_32__in_vm_20230406_02_17_30":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_33__in_vm_20230406_02_53_55":"5840",
"caldera_custom_manual_attack_adversaryid__custom_adversary_34__in_vm_20230406_03_35_56":"7488",
"caldera_custom_manual_attack_adversaryid__custom_adversary_35__in_vm_20230406_04_12_06":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_36__in_vm_20230406_04_47_32":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_37__in_vm_20230406_05_24_00":"7488",
"caldera_custom_manual_attack_adversaryid__custom_adversary_38__in_vm_20230406_06_03_58":"7488",
"caldera_custom_manual_attack_adversaryid__custom_adversary_39__in_vm_20230406_06_46_39":"7488",
"caldera_custom_manual_attack_adversaryid__custom_adversary_40__in_vm_20230406_07_28_00":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_41__in_vm_20230406_08_02_31":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_42__in_vm_20230406_08_38_17":"7488",
"caldera_custom_manual_attack_adversaryid__custom_adversary_43__in_vm_20230406_09_16_41":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_44__in_vm_20230406_09_46_14":"7488",
"caldera_custom_manual_attack_adversaryid__custom_adversary_45__in_vm_20230406_10_23_34":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_46__in_vm_20230406_11_05_09":"168",
"caldera_custom_manual_attack_adversaryid__custom_adversary_47__in_vm_20230406_11_40_24":"280",
"caldera_custom_manual_attack_adversaryid__custom_adversary_48__in_vm_20230406_12_15_05":"280",
}





caldera_stockpile_builtin_attack_removed_unrelated_artifacts_dict = \
{ f"{k}__unrelated_artifact_events_dropped": v for k,v in caldera_stockpile_builtin_attack_dict.items() }



for_debugging_developing_dict = { "zoo_malware_set2_1" : "3288" }


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
# Added to 
#           suffix_to_add = "preprocessed_method_1" 

benign_powershell_set1_20230413_dict ={
# from logs-priti
'benign_powershell_install-github-cli':'3748', 
'benign_powershell_list-cli-tool':'4540', 
'benign_powershell_enable-god-mode':'6212', 
'benign_powershell_save-screenshot':'4724', 
'benign_powershell_moon':'1264', 
'benign_powershell_new-qrcode':'1808', 
'benign_powershell_write-vertical':'4724', 
'benign_powershell_list-voices':'5984', 
'benign_powershell_check-battery':'2588', 
'benign_powershell_check-cpu':'1644', 
'benign_powershell_check-gpu':'3128', 
'benign_powershell_check-ipv4-address':'5984', 
'benign_powershell_check-mac-address':'3828', 
'benign_powershell_check-midnight':'6276', 
'benign_powershell_check-moon-phase':'3952', 
'benign_powershell_check-outlook':'4420', 
'benign_powershell_check-ping':'3956', 
'benign_powershell_check-powershell':'1644', 
'benign_powershell_check-ram':'1644', 
'benign_powershell_check-subnet-mask':'4592', 
'benign_powershell_check-swap-space':'2160', 
'benign_powershell_check-time-zone':'2752', 
'benign_powershell_check-uptime':'6008', 
'benign_powershell_check-windows-system-files':'6992', 
'benign_powershell_install-git-extensions':'2252', 
'benign_powershell_install-git-for-windows':'1812', 
'benign_powershell_install-microsoft-teams':'1644', 
'benign_powershell_install-obs-studio':'1808', 
'benign_powershell_install-opera-gx':'1644', 
'benign_powershell_uninstall-opera-gx':'3984',
# from logs_Meng 
'benign_powershell_check-new-year':'5536', 
'benign_powershell_check-noon':'2568', 
'benign_powershell_install-audacity':'264', 
'benign_powershell_list-aliases':'2408', 
'benign_powershell_list-anagrams':'6992', 
'benign_powershell_list-battery-status':'2336', 
'benign_powershell_new-script':'4932', 
'benign_powershell_new-shortcut':'2408', 
'benign_powershell_new-tag':'1528', 
'benign_powershell_new-user':'6992', 
'benign_powershell_open-microsoft-paint':'4104', 
'benign_powershell_open-microsoft-solitaire':'2408', 
'benign_powershell_open-microsoft-store':'4120', 
'benign_powershell_open-microsoft-teams':'528', 
'benign_powershell_open-music-folder':'4104', 
'benign_powershell_open-note-pad':'6992', 
'benign_powershell_open-pictures-folder':'4104', 
'benign_powershell_open-recycle-bin-folder':'4104', 
'benign_powershell_open-screen-clip':'1528', 
'benign_powershell_open-snipping-tool':'1528', 
'benign_powershell_open-stack-overflow':'4120', 
'benign_powershell_open-task-manager':'1528', 
'benign_powershell_open-videos-folders':'2408', 
'benign_powershell_open-windows-defender':'2336', 
'benign_powershell_search-files':'4104', 
'benign_powershell_set-profile':'6992', 
'benign_powershell_set-wallpaper':'2336', 
'benign_powershell_show-notification':'3848',
}


benign_powershell_set2_20230420_dict ={
"benign_powershell_check-health.ps1":"2660",                                                      
"benign_powershell_list-installed-scripts.ps1":"2432",
"benign_powershell_install-windows-terminal.ps1":"4424",                            
"benign_powershell_check-iss-position.ps1":"1940",                                 
"benign_powershell_list-installed-software.ps1":"2432",
"benign_powershell_install-zoom.ps1":"2432",                                       
"benign_powershell_check-ram.ps1":"932",                                          
"benign_powershell_list-network-connections.ps1":"6888",
"benign_powershell_introduce-powershell.ps1":"788",                                
"benign_powershell_clear-dns-cache.ps1":"3380",                                    
"benign_powershell_list-network-routes.ps1":"2660",
"benign_powershell_list-calendar.ps1":"2432",                                      
"benign_powershell_close-file-explorer.ps1":"932",                                
"benign_powershell_list-network-shares.ps1":"6888",
"benign_powershell_list-clipboard.ps1":"2660",                                     
"benign_powershell_connect-vpn.ps1":"932",                                        
"benign_powershell_list-os.ps1":"1940",
"benign_powershell_list-cmdlets.ps1":"2432",                                       
"benign_powershell_install-chrome-browser.ps1":"6856",                             
"benign_powershell_list-os-releases.ps1":"1952",
"benign_powershell_list-cpu.ps1":"1940",                                            
"benign_powershell_install-crystal-disk-info.ps1":"1328",                          
"benign_powershell_list-os-updates.ps1":"932",
"benign_powershell_list-dns-servers.ps1":"932",                                    
"benign_powershell_install-edge.ps1":"6356",                                       
"benign_powershell_list-passwords.ps1":"2432",
"benign_powershell_list-drives.ps1":"1940",                                         
"benign_powershell_install-power-toys.ps1":"1940",                                 
"benign_powershell_list-pins.ps1":"2432",
"benign_powershell_list-empty-dirs.ps1":"2432",                                     
"benign_powershell_install-rufus.ps1":"2660",                                      
"benign_powershell_list-processes.ps1":"2660",
"benign_powershell_list-empty-files.ps1":"932",                                    
"benign_powershell_install-skype.ps1":"2432",                                      
"benign_powershell_scan-ports.ps1":"2660",
"benign_powershell_list-environment-variables.ps1":"2432",                          
"benign_powershell_install-updates.ps1":"1940",                                    
"benign_powershell_uninstall-visual-studio-code.ps1":"3104",
"benign_powershell_list-folder.ps1":"1952",                                         
"benign_powershell_install-vivaldi.ps1":"6356",                                    
"benign_powershell_uninstall-windows-terminal.ps1":"4488",
"benign_powershell_list-hidden-files.ps1":"1940",                                   
"benign_powershell_install-vlc.ps1":"3856"
}                                        



suffix_to_add = "__preprocessed_method_1__dropped_queryea"
benign_powershell_set1_20230413_dict__preprocessed_method_1_dropped_queryea =  { key+suffix_to_add : value for key, value in benign_powershell_set1_20230413_dict.items() }

benign_powershell_set2_20230420_dict__preprocessed_method_1_dropped_queryea =  { key+suffix_to_add : value for key, value in benign_powershell_set2_20230420_dict.items() }


# /data/d1/jgwak1/tabby/MALWARE_POWERSHELL_SET1_20230413_LOGS
# Nishang
malware_powershell_set1_20230413_dict ={
"malware_powershell_check-vm":"824",
"malware_powershell_copy-vss":"7456",
"malware_powershell_download-execute-ps":"3820",
"malware_powershell_enable-duplicatetoken":"6872",
"malware_powershell_execute-dnstxt-code":"3504",
"malware_powershell_execute-ontime":"3420",
"malware_powershell_get-information":"1000",
"malware_powershell_get-passhashes":"5992",
"malware_powershell_get-passhints":"3060",
"malware_powershell_get-webcredentials":"5520",
"malware_powershell_get-wlan-keys":"6196",
"malware_powershell_gupt-backdoor":"3420",
"malware_powershell_http-backdoor":"1844",
"malware_powershell_invoke-adsbackdoor":"1084",
"malware_powershell_invoke-amsibypass":"1844",
"malware_powershell_invoke-mimikat":"1740",
"malware_powershell_invoke-mimikitten":"4904",
"malware_powershell_invoke-sessiongopher":"7456",
"malware_powershell_remove-update":"3504",

"malware_powershell_add-scrnsavebackdoor": "4024",
}

suffix_to_add = "__preprocessed_method_1__dropped_queryea"   
malware_powershell_set1_20230413_dict__preprocessed_method_1_dropped_queryea = { key+suffix_to_add : value for key, value in malware_powershell_set1_20230413_dict.items() }



# /data/d1/jgwak1/tabby/MALWARE_MAFIA_LOGS_SET1_UNEVALUATED_LOGS_20230423
# malware_powershell_mafia_unevaluated_set1_20230423_dict ={
# "malware_powershell_mafia_Get-TimedScreenshot":"6052",               
# "malware_powershell_mafia_Invoke-TokenManipulation":"1600",
# "malware_powershell_mafia_Find-AVSignature":"7932",
# "malware_powershell_mafia_Get-VaultCredential":"6052",               
# "malware_powershell_mafia_Invoke-WmiCommand":"7712",
# "malware_powershell_mafia_Invoke-CompareAttributesForClass":"3972",
# "malware_powershell_mafia_Out-CompressedDll":"340",
# "malware_powershell_mafia_Get-ComputerDetail":"2532",                        
# "malware_powershell_mafia_Invoke-CredentialInjection":"5716",        
# "malware_powershell_mafia_Out-EncodedCommand":"6644",
# "malware_powershell_mafia_Get-GPPAutologon":"8148",                          
# "malware_powershell_mafia_Invoke-DllInjection":"6052",               
# "malware_powershell_mafia_Out-Minidump":"6052",
# "malware_powershell_mafia_Get-GPPPassword":"1600",                           
# "malware_powershell_mafia_Invoke-Portscan":"3972",                   
# "malware_powershell_mafia_PowerUp":"340",
# "malware_powershell_mafia_Get-HttpStatus":"340",      
# "malware_powershell_mafia_Invoke-ReflectivePEInjection":"2236",   
# "malware_powershell_mafia_PowerView":"340",
# "malware_powershell_mafia_Get-Keystrokes":"8148",                            
# "malware_powershell_mafia_Invoke-ReverseDnsLookup":"6644",            
# "malware_powershell_mafia_Remove-Comment":"4556",
# "malware_powershell_mafia_Get-System":"9160",                                
# "malware_powershell_mafia_Invoke-Shellcode":"7712",                  
# "malware_powershell_mafia_VolumeShadowCopyTools":"6052",
# }



# benign_powershell_jhochwald_powershell_collection_set1_20230424_dict ={
#    "benign_powershell_jhochwald_powershell_collection_new-computercheckpoint": "9988",
#    "benign_powershell_jhochwald_powershell_collection_set-officeinsider": "9676",
#    "benign_powershell_jhochwald_powershell_collection_set-ipv6inwindows": "9988",
#    "benign_powershell_jhochwald_powershell_collection_get-ipinfo": "9988",
#    "benign_powershell_jhochwald_powershell_collection_get-dateintervalhumanreadable": "9988",
#    "benign_powershell_jhochwald_powershell_collection_forcetimeresync": "7196",
#    "benign_powershell_jhochwald_powershell_collection_resolve-dnshost": "4256",
#    "benign_powershell_jhochwald_powershell_collection_get-deviceautopilotinfo": "4448",
#    "benign_powershell_jhochwald_powershell_collection_installpendingwindowsupdates": "5700",
#    "benign_powershell_jhochwald_powershell_collection_invoke-exportdrivers": "9676",
# }

# benign_powershell_ledragox_win_debloats_tools_set1_20230424_dict ={
#    "benign_powershell_ledragox_win_debloats_tools_optimize-performance": "3092",
#    "benign_powershell_ledragox_win_debloats_tools_remove-msedge": "6848",
#    "benign_powershell_ledragox_win_debloats_tools_update-allpackage": "1512",
#    "benign_powershell_ledragox_win_debloats_tools_remove-capabilitieslist": "2220",
#    "benign_powershell_ledragox_win_debloats_tools_install-defaultappslist": "636",
#    "benign_powershell_ledragox_win_debloats_tools_repair-windowssystem": "10004",
#    "benign_powershell_ledragox_win_debloats_tools_register-personaltweakslist": "7980",
#    "benign_powershell_ledragox_win_debloats_tools_show-debloatinfo": "8248",
#    "benign_powershell_ledragox_win_debloats_tools_install-nerdfont": "5136",
#    "benign_powershell_ledragox_win_debloats_tools_install-archwsl": "4584",

# }


benign_psh_stevencohnwindowspowershell_20230426 = {
	
# /data/d1/jgwak1/tabby/BENIGN_PSH_LOGS_WITH_UNOPTIMIZED_AGENT__20230426/BENIGN_POWERSHELL_stevencohnWindowsPowerShell_MengVM_20230426_LOGS

"benign_powershell_stevencohnwindowspowershell_convertto-hex":"3108",   
"benign_powershell_stevencohnwindowspowershell_get-network":"3736",
"benign_powershell_stevencohnwindowspowershell_enable-multird":"11428",  
"benign_powershell_stevencohnwindowspowershell_install-chocolatey":"11428",
"benign_powershell_stevencohnwindowspowershell_get-account":"12376",     
"benign_powershell_stevencohnwindowspowershell_install-programs":"11396",
"benign_powershell_stevencohnwindowspowershell_get-installed":"7436",   


# /data/d1/jgwak1/tabby/BENIGN_PSH_LOGS_WITH_UNOPTIMIZED_AGENT__20230426/BENIGN_POWERSHELL_stevencohnWindowsPowerShell_JYVM_20230426_LOGS	
"benign_powershell_stevencohnwindowspowershell_convertfrom-hex":"4500",         
"benign_powershell_stevencohnwindowspowershell_get-productkey":"9676",
"benign_powershell_stevencohnwindowspowershell_enable-trustedremoting":"5700",  
"benign_powershell_stevencohnwindowspowershell_get-specialfolder":"4256",
"benign_powershell_stevencohnwindowspowershell_get-path":"5700",                
"benign_powershell_stevencohnwindowspowershell_test-rebootpending":"4256",

}


benign_psh_jhochwaldpowershellcollection_20230426 = {
	
# /data/d1/jgwak1/tabby/BENIGN_PSH_LOGS_WITH_UNOPTIMIZED_AGENT__20230426/BENIGN_POWERSHELL_jhochwaldPowerShellcollection_JYVM_20230424_LOGS

"benign_powershell_jhochwaldpowershellcollection_forcetimeresync":"7196",
"benign_powershell_jhochwaldpowershellcollection_get-dateintervalhumanreadable":"9988",
"benign_powershell_jhochwaldpowershellcollection_get-ipinfo":"9988",
"benign_powershell_jhochwaldpowershellcollection_new-computercheckpoint":"9988",
"benign_powershell_jhochwaldpowershellcollection_resolve-dnshost":"4256",
"benign_powershell_jhochwaldpowershellcollection_set-ipv6inwindows":"9988",
"benign_powershell_jhochwaldpowershellcollection_set-officeinsider":"9676",



# /data/d1/jgwak1/tabby/BENIGN_PSH_LOGS_WITH_UNOPTIMIZED_AGENT__20230426/BENIGN_POWERSHELL_jhochwaldPowerShellcollection_MengVM_20230425_LOGS

"benign_powershell_jhochwaldpowershellcollection_get-localgroupmembersh":"13272", # check
"benign_powershell_jhochwaldpowershellcollection_hosts_helper":"11396",
"benign_powershell_jhochwaldpowershellcollection_get-runasadmin":"7880",          
"benign_powershell_jhochwaldpowershellcollection_install-sysinternalssuite":"9588",
"benign_powershell_jhochwaldpowershellcollection_grant-logonasservice":"11396",

}



benign_psh_ledragoxwindebloattools_20230427 = {
	
# /data/d1/jgwak1/tabby/BENIGN_PSH_LOGS_WITH_OPTIMIZED_AGENT__20230427/BENIGN_POWERSHELL_LedragoX_Win-Debloat-Tools_JYVM_20230427_LOGS

"benign_powershell_ledragoxwindebloattools_install-defaultappslist":"10100",
"benign_powershell_ledragoxwindebloattools_optimize-performance":"1704",
"benign_powershell_ledragoxwindebloattools_register-personaltweakslist":"5372",




# /data/d1/jgwak1/tabby/BENIGN_PSH_LOGS_WITH_OPTIMIZED_AGENT__20230427/BENIGN_POWERSHELL_LedragoX_Win-Debloat-Tools_MengVM_20230427_LOGS

"benign_powershell_ledragoxwindebloattools_update-allpackage":"8020", # check


}


#/data/d1/jgwak1/tabby/MASTERSNAPSHOT_Benign_logs_JY_Machine/Benign_logs_JY_Machine_4th_May_2023_set1

Benign_logs_JY_Machine_4th_May_2023_set1_dict = {
	
	"benign_fleschutz_powershell_check-drives": "2680",
	"benign_fleschutz_powershell_install-wsl": "2648",
	"benign_fleschutz_powershell_scan-ports": "4856",	
	"benign_fleschutz_powershell_uninstall-paint-3d": "2680",
	
	"benign_jhochwald_powershell_set-powerplantoauto": "4856",

	"benign_stevencohn_windowspowershell_get-network": "4856",
	"benign_stevencohn_windowspowershell_get-path": "4856",
	"benign_stevencohn_windowspowershell_install-buildtools": "3524",
	"benign_stevencohn_windowspowershell_update-environment": "4856",

	"benign_win-debloat_ledrago_install-wsl": "2680",
	"benign_win-debloat_ledrago_new-systemcolor": "2680",
	"benign_win-debloat_ledrago_show-debloatinfo": "1348",
}


# /data/d1/jgwak1/tabby/MASTERSNAPSHOT_Benign_logs_JY_Machine/Benign_logs_Machine_4th_May_2023_alittle_more
Benign_logs_JY_Machine_4th_May_2023_alittle_more_dict = { 
  "benign_fleschutz_powershell_check-battery": "11200",
  "benign_fleschutz_powershell_list-hidden-files": "11052",
  "benign_fleschutz_powershell_remove-empty-dirs": "4924",
  "benign_jhochwald_powershell_Test-ValidEmail":"7984",
  "benign_win-debloat_ledrago_Update-AllPackage":"4856",
}

#############################################################################################################################3
eventseqrepodtest_trials_1_2_Benign_dict_20230430 =\
{ 
 # trial_1   
 "eventseqrepodtest_trial_1_invoke-exportdrivers":"3896", # Benign
 "eventseqrepodtest_trial_1_list-cmdlets":"6120",  # Benign
 "eventseqrepodtest_trial_1_new-host":"6120",  # Benign
 # trial_2
 "eventseqrepodtest_trial_2_invoke-exportdrivers":"3896",   # Benign
 "eventseqrepodtest_trial_2_list-cmdlets":"3896",  # Benign
 "eventseqrepodtest_trial_2_new-host":"6120",   # Benign

}

eventseqrepodtest_trials_1_2_Malware_dict_20230430 = \
{  
 # trial_1
 "eventseqrepodtest_trial_1_cred-popper":"5420",   # Malware
 "eventseqrepodtest_trial_1_get-browserdata":"3896",  # Malware
 "eventseqrepodtest_trial_1_get-webcredentials": "5136", # Malware
 # trial_2
 "eventseqrepodtest_trial_2_cred-popper":"5420",   # Malware
 "eventseqrepodtest_trial_2_get-browserdata":"6120",  # Malware
 "eventseqrepodtest_trial_2_get-webcredentials": "3896", # Malware
 }
#############################################################################################################################3


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

# Added by JY @ 2023-03-14:
#
# Following "index" is for checking correctness of 
# handlings made for newly identfied edge-cases in event-types of 
# ["CPUPRIORITYCHANGE", "CPUBASEPRIORITYCHANGE", "IOPRIORITYCHANGE", "PAGEPRIORITYCHANGE"] + ["IMAGELOAD", "IMAGEUNLOAD"].
# i.e., Above 'index' contains all above event-types.
# 
# > For correctness check, 
#   (1) Generate the CG by setting first-step-hash-generation as debugging mode.
#       > (Only perform Malware-Step 1)
#   (2) Check the Edge-driection and Proc / Thread Info for 
#       ["CPUPRIORITYCHANGE", "CPUBASEPRIORITYCHANGE", "IOPRIORITYCHANGE", "PAGEPRIORITYCHANGE"] + ["IMAGELOAD", "IMAGEUNLOAD"]
#       by comparing several log-entries with its corresponding-representation in the CG.
CG_modification_for_edge_cases_correctness_check_2023_03_14 = \
{'caldera_builtin_adversary_01d77744-2515-401a-a497-d9f7241aac3c_20230308_03_43_17':'3536'}

caldera_custom_attack_file_registry_abilities_20230319_20_13_21 =\
{"caldera_custom_attack_file_registry_abilities_20230319_20_13_21": "2872"}



caldera_manual_attack_in_vm_20230323_21_32_52 =\
{"caldera_manual_attack_in_vm_20230323_21_32_52": "4444"}




# 2023-03-29/30 produced logs
# > /data/d1/jgwak1/tabby/CALDERA_BUILTIN_MANUAL_ATTACK_LOGS_BASED_ON_OPTIMIZED_CODE
caldera_builtin_manual_attack_eventlogs__generated_by_optimized_code = \
{"caldera_builtin_manual_attack_adversaryid__01c96671-afd3-47d4-8d31-8c116cc0221a__in_vm_20230329_16_33_01":"6876",
"caldera_builtin_manual_attack_adversaryid__01d77744-2515-401a-a497-d9f7241aac3c__in_vm_20230329_22_32_34":"6904",
"caldera_builtin_manual_attack_adversaryid__09ad625e-6cba-490f-afe3-5417e7edb9c6__in_vm_20230329_17_22_42":"4100",
"caldera_builtin_manual_attack_adversaryid__0b73bf34-fc5b-48f7-9194-dce993b915b1__in_vm_20230329_23_12_05":"5556",
"caldera_builtin_manual_attack_adversaryid__50855e29-3b4e-4562-aa55-b3d7f93c26b8__in_vm_20230329_20_35_50":"6300",
"caldera_builtin_manual_attack_adversaryid__564ae20d-778d-4965-93dc-b523be2e2ab4__in_vm_20230329_18_03_41":"5756",
"caldera_builtin_manual_attack_adversaryid__78e7504d-968f-477d-8806-4d6c04b94431__in_vm_20230329_21_34_26":"7476",
"caldera_builtin_manual_attack_adversaryid__bcdbf6b9-14c5-495c-be84-37bce32c312b__in_vm_20230330_00_14_55":"6904",
"caldera_builtin_manual_attack_adversaryid__c724545d-a4cc-492e-8075-2ab9a699c847__in_vm_20230329_16_49_06":"6904",
"caldera_builtin_manual_attack_adversaryid__d6ea4c1e-7959-4eb1-a292-b6fd2b06c73e__in_vm_20230329_19_49_08":"6848",
"caldera_builtin_manual_attack_adversaryid__dbd49a4a-ba2d-40d0-9348-2db24fc4b0b6__in_vm_20230329_19_27_53":"5028",
"caldera_builtin_manual_attack_adversaryid__de07f52d-9928-4071-9142-cb1d3bd851e8__in_vm_20230329_14_57_56":"4740",
"caldera_builtin_manual_attack_adversaryid__e89a10d3-004f-4c15-b0eb-d1ba76a4b67f__in_vm_20230329_20_21_52":"7468",
"caldera_builtin_manual_attack_adversaryid__fbc41624-1052-490c-b5ec-4fd718e2501d__in_vm_20230329_23_59_06":"6944"}



###################################################################################################

Benign_logs_Meng_Priti_Machine_5th_May_2023_dict = {
	# /data/d1/jgwak1/tabby/Benign_logs_Meng_Machine_5th_May_2023
	# /data/d1/jgwak1/tabby/Benign_logs_Priti_Machine_5th_May_2023


	# /data/d1/jgwak1/tabby/Benign_logs_Meng_Machine_5th_May_2023/benign_fleschutz_powershell
	"benign_fleschutz_powershell_install-irfanview": "5796",

	# /data/d1/jgwak1/tabby/Benign_logs_Meng_Machine_5th_May_2023/benign_stevencohn_windowspowershell
	"benign_stevencohn_windowspowershell_get-account": "5356",
	"benign_stevencohn_windowspowershell_invoke-normaluser": "8772",

	# /data/d1/jgwak1/tabby/Benign_logs_Priti_Machine_5th_May_2023/benign_fleschutz_powershell
	"benign_fleschutz_powershell_install-ssh-server": "3452",

	# /data/d1/jgwak1/tabby/Benign_logs_Priti_Machine_5th_May_2023/benign_jhochwald_powershell_collection
	"benign_jhochwald_powershell_collection_get-localipaddresses": "10364",
	"benign_jhochwald_powershell_collection_out-ziparchive": "10188",
	"benign_jhochwald_powershell_collection_resolve-dnshost": "6792",
	"benign_jhochwald_powershell_collection_set-ipv6inwindows": "11056",
	"benign_jhochwald_powershell_collection_test-port": "6964",

	# /data/d1/jgwak1/tabby/Benign_logs_Priti_Machine_5th_May_2023/benign_stevencohn_windowspowershell
	"benign_stevencohn_windowspowershell_remove-browserhijack": "6648",

	# /data/d1/jgwak1/tabby/Benign_logs_Priti_Machine_5th_May_2023/benign_win-debloat_ledrago
	"benign_win-debloat_ledrago_install-nerdfont": "6892",

}



Malware_logs_JY_Priti_Meng_Machine_7th_May_2023_dict = {
	
	# /data/d1/jgwak1/tabby/Malware_logs_JY+Priti+Meng_Machine_7th_May_2023


	# /data/d1/jgwak1/tabby/Malware_logs_JY+Priti+Meng_Machine_7th_May_2023/malware_empire
	"malware_empire_exploit-eternalblue": "4856",
	"malware_empire_exploit-jenkins": "4856",
	"malware_empire_get-system": "4856",	
	"malware_empire_get-usbkeystrokes": "10636",
	"malware_empire_invoke-bypassuactokenmanipulation": "2648",
	"malware_empire_invoke-runas": "7984",	
	"malware_empire_invoke-sqloscmd": "2648",
	"malware_empire_invoke-thunderstruck": "8772",

	# /data/d1/jgwak1/tabby/Malware_logs_JY+Priti+Meng_Machine_7th_May_2023/malware_mafia
	"malware_mafia_remove-comment": "7984",
	"malware_mafia_volumeshadowcopytools": "7980",
	# /data/d1/jgwak1/tabby/Malware_logs_JY+Priti+Meng_Machine_7th_May_2023/malware_nishang
	"malware_nishang_get-passhashes": "8700",
	"malware_nishang_invoke-mimikitten": "11052",	
	# /data/d1/jgwak1/tabby/Malware_logs_JY+Priti+Meng_Machine_7th_May_2023/malware_poshc2
	"malware_poshc2_get-firewallrules": "9380",
	"malware_poshc2_get-idletime": "9384",
	"malware_poshc2_get-serviceperms": "4924",
	"malware_poshc2_get-system": "2648",
	"malware_poshc2_get-tokenelevationtype": "4856",
	"malware_poshc2_get-userinfo": "9368",
	"malware_poshc2_get-userlogons": "7984",	
	"malware_poshc2_inveigh": "3260",
	"malware_poshc2_invoke-kerberoast": "4856",
	"malware_poshc2_invoke-mimikat": "2680",
	"malware_poshc2_invoke-ms16-032": "2648",
	"malware_poshc2_invoke-ms16-032-proxy": "2680",
	"malware_poshc2_invoke-pipekat": "5796",
	"malware_poshc2_invoke-portscan": "7984",

}


Benign_logs_JY_9th_May_2023_dict = {
   # /data/d1/jgwak1/tabby/Benign_logs_JY_Machine_9th_May_2023
   "benign_fleschutz_powershell_check-firewall": "2680",
   "benign_fleschutz_powershell_check-gpu": "4272",
   "benign_fleschutz_powershell_check-swap-space": "2680",
   "benign_fleschutz_powershell_list-pins": "4856",
   "benign_fleschutz_powershell_new-shortcut": "2680",
   "benign_fleschutz_powershell_replace-in-files": "2680",
   "benign_fleschutz_powershell_search-filename": "4924",

   "benign_jhochwald_powershell_collection_forcetimeresync": "4924",
   "benign_jhochwald_powershell_collection_get-myipaddress": "4856",
   "benign_jhochwald_powershell_collection_get-outdatedpowershellmodules": "2648",
   "benign_jhochwald_powershell_collection_invoke-dscperfreqconfigcheck": "4272",
   "benign_jhochwald_powershell_collection_optimize-microsoftdefenderexclusions": "4272",
   "benign_jhochwald_powershell_collection_remove-fileendingblanklines": "3524",
   "benign_jhochwald_powershell_collection_update-modulefrompsgallery": "4856",

   "benign_stevencohn_windowspowershell_get-commandline": "2680",
   "benign_stevencohn_windowspowershell_new-runasshortcut": "2680",

}

Benign_logs_Meng_Machine_10th_May_2023_dict = {
	
	# /data/d1/jgwak1/tabby/Benign_logs_Meng_Machine_10th_May_2023

	"benign_fleschutz_powershell_check-drive-space": "5572",
	"benign_fleschutz_powershell_list-dns-servers": "5356",	
	"benign_fleschutz_powershell_list-empty-dirs": "9368",
	"benign_fleschutz_powershell_ping-weather": "3260",

	"benign_jhochwald_powershell_collection_get-ipinfo": "5796",
	"benign_jhochwald_powershell_collection_set-ipv6inwindows": "9368",
}


Benign_logs_Priti_Machine_10th_May_2023_dict = {
	
	# /data/d1/jgwak1/tabby/Benign_logs_Priti_Machine_10th_May_2023

	"benign_fleschutz_powershell_check-iss-position": "10952",
	"benign_fleschutz_powershell_check-os": "11016",
	"benign_fleschutz_powershell_check-pending-reboot": "10596",
	"benign_fleschutz_powershell_list-drives": "8024",
	"benign_fleschutz_powershell_list-empty-files": "9524",
	"benign_fleschutz_powershell_list-environment-variables": "1764",
	"benign_fleschutz_powershell_locate-ipaddress": "10540",
	"benign_fleschutz_powershell_open-c-drive": "5712",

	"benign_jhochwald_powershell_collection_get-directorysize": "11056",
	"benign_jhochwald_powershell_collection_get-localgroupmembership": "10704",
	"benign_jhochwald_powershell_collection_hosts_helper": "4204",
	"benign_jhochwald_powershell_collection_invoke-mitigatecve202230190": "9928",
	"benign_jhochwald_powershell_collection_test-isadmin": "9380",

	"benign_stevencohn_windowspowershell_get-productkey": "8648",

	"benign_win-debloat_ledrago_git-gnupgsshkeyssetup": "1952",

}



# This for 2023-05-20 subgraph generation.
from all_benign_malware_psh_samples_dicts_asof_20230520 import all_benign_psh_samples_asof_20230520_dict, all_malware_psh_samples_asof_20230520_dict
Apache_benign_logs = {
"benign_powershell_devblackops_convertfrom-colorescapesequence" : "4468",
"benign_powershell_devblackops_get-terminaliconsglyphs" : "356",
"benign_powershell_devblackops_get-terminaliconstheme" : "3872",
"benign_powershell_farag2_acrobat_pro_dc_x64" : "780",
"benign_powershell_farag2_acrobat_pro_dc_x86" : "4236",
"benign_powershell_farag2_acrobat_reader_dc_x64" : "6052",
"benign_powershell_farag2_add_custom_folder_to_file_explorer" : "4504",
"benign_powershell_farag2_add_open_with_powershell_as_admin_in_context_menu" : "3028",
"benign_powershell_farag2_associate_extension" : "5796",
"benign_powershell_farag2_chrome_setup_exe" : "6052",
"benign_powershell_farag2_chrome_setup_msi" : "7560",
"benign_powershell_farag2_compare_jsons_and_merge_them_into_one" : "1512",
"benign_powershell_farag2_configure_apps_and_the_start_menu_shortcuts" : "5388",
"benign_powershell_farag2_convert_path_into_escape_sequence" : "5292",
"benign_powershell_farag2_create_scheduled_task_triggered_by_eventid" : "1224",
"benign_powershell_farag2_create_sha256sum_file" : "6692",
"benign_powershell_farag2_download_install_updates_for_drivers" : "5068",
"benign_powershell_farag2_edge" : "3328",
"benign_powershell_farag2_firefox_setup_exe" : "8900",
"benign_powershell_farag2_firefox_setup_msi" : "3956",
"benign_powershell_farag2_get-attribute" : "7372",
"benign_powershell_farag2_get_focus_assist_status" : "6692",
"benign_powershell_farag2_get_locking_file_process" : "6356",
"benign_powershell_farag2_get_scheduled_tasks_created_for_the_previous_period" : "5068",
"benign_powershell_farag2_get_wi-fi_passwords" : "5068",
"benign_powershell_farag2_get_windows_10_settings_uri_pages" : "5068",
"benign_powershell_farag2_ivalidatesetvaluesgenerator" : "7908",
"benign_powershell_farag2_json" : "6052",
"benign_powershell_farag2_other" : "2688",
"benign_powershell_farag2_parse_site" : "6692",
"benign_powershell_farag2_powershell_errors" : "4964",
"benign_powershell_farag2_remove-attribute" : "7060",
"benign_powershell_farag2_restore_windows_photo_viewer" : "4964",
"benign_powershell_farag2_search_uwp_package_in_store" : "6356",
"benign_powershell_farag2_set-association2" : "5292",
"benign_powershell_farag2_set-attribute" : "7908",
"benign_powershell_farag2_split_sophiapp_translations_into_separate_files" : "5116",
"benign_powershell_farag2_taskbarx" : "7560",
"benign_powershell_farag2_teamviewer" : "3624",
"benign_powershell_farag2_visual_studio" : "3528",
"benign_powershell_farag2_vscode" : "6184",
"benign_powershell_farag2_winget" : "5328",
"benign_powershell_farag2_yt-dlp" : "2572",
"benign_powershell_fleschutz_list-aliases" : "5292",
"benign_powershell_fleschutz_list-anagrams" : "6152",
"benign_powershell_fleschutz_list-apps" : "5768",
"benign_powershell_fleschutz_list-battery-status" : "8436",
"benign_powershell_fleschutz_list-bluetooth-devices" : "7372",
"benign_powershell_fleschutz_list-cheat-sheet" : "4084",
"benign_powershell_fleschutz_list-clipboard" : "1052",
"benign_powershell_fleschutz_list-cpu" : "6752",
"benign_powershell_fleschutz_list-earthquakes" : "4140",
"benign_powershell_fleschutz_list-emojis" : "4892",
"benign_powershell_fleschutz_list-fibonacci" : "5388",
"benign_powershell_fleschutz_list-folder" : "5328",
"benign_powershell_fleschutz_list-headlines" : "5232",
"benign_powershell_fleschutz_list-installed-software" : "4300",
"benign_powershell_fleschutz_list-modules" : "4300",
"benign_powershell_fleschutz_list-motherboard" : "5768",
"benign_powershell_fleschutz_list-network-connections" : "2700",
"benign_powershell_fleschutz_list-network-routes" : "5768",
"benign_powershell_fleschutz_list-network-shares" : "5336",
"benign_powershell_fleschutz_list-nic" : "6052",
"benign_powershell_fleschutz_list-os" : "6052",
"benign_powershell_fleschutz_list-passwords" : "3192",
"benign_powershell_fleschutz_list-printers" : "2952",
"benign_powershell_fleschutz_list-profiles" : "4504",
"benign_powershell_fleschutz_list-ram" : "5556",
"benign_powershell_fleschutz_list-recycle-bin" : "6956",
"benign_powershell_fleschutz_list-services" : "3884",
"benign_powershell_fleschutz_list-suggestions" : "5292",
"benign_powershell_fleschutz_list-system-info" : "3360",
"benign_powershell_fleschutz_list-tasks" : "6692",
"benign_powershell_fleschutz_list-timezones" : "6356",
"benign_powershell_fleschutz_list-timezone" : "2572",
"benign_powershell_fleschutz_list-user-accounts" : "6376",
"benign_powershell_fleschutz_list-user-groups" : "6956",
"benign_powershell_fleschutz_list-voices" : "4892",
"benign_powershell_fleschutz_list-weather" : "1752",
"benign_powershell_fleschutz_list-wifi" : "4140",
"benign_powershell_fleschutz_locate-city" : "5292",
"benign_powershell_fleschutz_locate-my-phone" : "3416",
"benign_powershell_fleschutz_locate-zip-code" : "7560",
"benign_powershell_fleschutz_lock-desktop" : "5292",
"benign_powershell_fleschutz_merry-christmas" : "5328",
"benign_powershell_fleschutz_minimize-all-windows" : "7908",
"benign_powershell_fleschutz_moon" : "3356",
"benign_powershell_fleschutz_my-profile" : "3356",
"benign_powershell_fleschutz_new-email" : "4564",
"benign_powershell_fleschutz_new-qrcode" : "7372",
"benign_powershell_fleschutz_new-script" : "3028",
"benign_powershell_fleschutz_remind-me" : "6752",
"benign_powershell_fleschutz_remove-print-jobs" : "6928",
"benign_powershell_fleschutz_save-screenshot" : "5016",
"benign_powershell_fleschutz_set-volume" : "8324",
"benign_powershell_fleschutz_set-wallpaper" : "3968",
"benign_powershell_fleschutz_show-lightnings" : "372",
"benign_powershell_fleschutz_show-traffic" : "2436",
"benign_powershell_fleschutz_spell-word" : "3768",
"benign_powershell_fleschutz_tell-joke" : "2952",
"benign_powershell_fleschutz_tell-quote" : "1224",
"benign_powershell_fleschutz_toggle-caps-lock" : "6692",
"benign_powershell_fleschutz_toggle-num-lock" : "5292",
"benign_powershell_fleschutz_toggle-scroll-lock" : "5380",
"benign_powershell_fleschutz_translate-file" : "6956",
"benign_powershell_fleschutz_translate-text" : "4892",
"benign_powershell_fleschutz_uninstall-all-apps" : "2684",
"benign_powershell_fleschutz_uninstall-crystal-disk-info" : "7908",
"benign_powershell_fleschutz_uninstall-crystal-disk-mark" : "6928",
"benign_powershell_fleschutz_uninstall-discord" : "7560",
"benign_powershell_fleschutz_uninstall-edge" : "3512",
"benign_powershell_fleschutz_uninstall-firefox" : "3028",
"benign_powershell_fleschutz_uninstall-git-extensions" : "2332",
"benign_powershell_fleschutz_uninstall-irfanview" : "1752",
"benign_powershell_fleschutz_uninstall-microsoft-teams" : "1224",
"benign_powershell_fleschutz_uninstall-netflix" : "6752",
"benign_powershell_fleschutz_uninstall-nine-zip" : "5388",
"benign_powershell_fleschutz_uninstall-opera-gx" : "1196",
"benign_powershell_fleschutz_uninstall-rufus" : "736",
"benign_powershell_fleschutz_uninstall-skype" : "1252",
"benign_powershell_fleschutz_uninstall-spotify" : "5772",
"benign_powershell_fleschutz_uninstall-twitter" : "4176",
"benign_powershell_fleschutz_uninstall-visual-studio-code" : "5336",
"benign_powershell_fleschutz_uninstall-vlc" : "3192",
"benign_powershell_fleschutz_weather-report" : "9020",
"benign_powershell_fleschutz_weather" : "2436",
"benign_powershell_fleschutz_write-animated" : "3744",
"benign_powershell_fleschutz_write-blue" : "6052",
"benign_powershell_fleschutz_write-fractal" : "5388",
"benign_powershell_fleschutz_write-green" : "1224",
"benign_powershell_fleschutz_write-joke" : "4892",
"benign_powershell_fleschutz_write-lowercase" : "6376",
"benign_powershell_fleschutz_write-red" : "2700",
"benign_powershell_fleschutz_write-time" : "5336",
"benign_powershell_fleschutz_write-uppercase" : "4116",
"benign_powershell_fleschutz_write-vertical" : "7372",
"benign_powershell_jimbrig_remove-olddrivers" : "2812",
"benign_powershell_jimbrig_trace-networkadapter" : "6852",
"benign_powershell_jrussellfreelance_clean-system-local" : "7180",
"benign_powershell_jrussellfreelance_list-top-ten-processes" : "6172",
"benign_powershell_jrussellfreelance_ping-server" : "2572",
"benign_powershell_jrussellfreelance_powershellhelp" : "8992",
"benign_powershell_jrussellfreelance_remove-all-print-jobs" : "3624",
"benign_powershell_jrussellfreelance_setup-domain-controller" : "3428",
"benign_powershell_jrussellfreelance_silent-chrome-install" : "4712",
"benign_powershell_jrussellfreelance_system-info-html-report-local" : "4252",
"benign_powershell_jrussellfreelance_system-info-local-json" : "184",
"benign_powershell_nickrod518_archive-duplicatefile" : "4252",
"benign_powershell_nickrod518_convert-robocopyexitcode" : "6852",
"benign_powershell_nickrod518_convert-tabletohtml" : "5956",
"benign_powershell_nickrod518_get-backupstatus" : "7816",
"benign_powershell_nickrod518_get-bitcoinpricechart" : "184",
"benign_powershell_nickrod518_get-certificate" : "5328",
"benign_powershell_nickrod518_get-citrixclientversionlist" : "8604",
"benign_powershell_nickrod518_get-computernetinfo" : "4252",
"benign_powershell_nickrod518_get-diskinfo" : "3268",
"benign_powershell_nickrod518_get-microsoftupdates" : "184",
"benign_powershell_nickrod518_get-program" : "6852",
"benign_powershell_nickrod518_get-serverload" : "5616",
"benign_powershell_nickrod518_get-wanspeed" : "2444",
"benign_powershell_nickrod518_install-chrome" : "4252",
"benign_powershell_nickrod518_install-firefoxesr" : "4260",
"benign_powershell_nickrod518_install-powershellget" : "4492",
"benign_powershell_nickrod518_move-datatoonedrive" : "6852",
"benign_powershell_nickrod518_new-cwsyncserver" : "3624",
"benign_powershell_nickrod518_new-encryptedpasswordfile" : "7800",
"benign_powershell_nickrod518_test-isise" : "6852",
"benign_powershell_nickrod518_upgrade-app" : "4260",
"benign_powershell_nickrod518_userclass" : "3688",
"benign_powershell_redttr_createrestorepoint" : "6052",
"benign_powershell_redttr_installfoxitreader" : "1260",
"benign_powershell_redttr_installgoogleearth" : "8992",
"benign_powershell_redttr_installpowershell" : "3192",
"benign_powershell_redttr_installsplashtoprmmviewer" : "1116",
"benign_powershell_redttr_installsplashtopsos" : "7372",
"benign_powershell_redttr_installteams" : "5956",
"benign_powershell_redttr_installzoom" : "6068",
"benign_powershell_redttr_managelocaladmins" : "988",
"benign_powershell_redttr_removelibreoffice" : "6068",
"benign_powershell_redttr_removemerakism" : "6520",
"benign_powershell_redttr_removequicktime" : "5176",
"benign_powershell_redttr_removesilverlight" : "4252",
"benign_powershell_redttr_removewpsoffice" : "5616",
"benign_powershell_redttr_removezerotier" : "4252",
"benign_powershell_redttr_removezoomuserinstalls" : "5720",
"benign_powershell_redttr_runinpwsh" : "3268",
"benign_powershell_redttr_scandefender" : "5520",
"benign_powershell_redttr_startprogram" : "5520",
"benign_powershell_redttr_updategrouppolicy" : "5232",
"benign_powershell_redttr_updatestoreapps" : "184",
"benign_powershell_redttr_updatezerotier" : "3224",
"benign_powershell_sysadmin-survival-kit_backupgui" : "7800",
"benign_powershell_sysadmin-survival-kit_backup" : "4492",
"benign_powershell_sysadmin-survival-kit_create-multiples-azstorages" : "6016",
"benign_powershell_sysadmin-survival-kit_desligamentogui" : "988",
"benign_powershell_sysadmin-survival-kit_disable-memory-compression" : "5176",
"benign_powershell_sysadmin-survival-kit_disable-prefetch-prelaunch" : "184",
"benign_powershell_sysadmin-survival-kit_disable-services" : "4252",
"benign_powershell_sysadmin-survival-kit_enable-god-mode" : "4252",
"benign_powershell_sysadmin-survival-kit_gerararquivosenhacriptografado" : "5000",
"benign_powershell_sysadmin-survival-kit_install-ubuntu-wsl" : "6852",
"benign_powershell_sysadmin-survival-kit_mensagem-acessivel" : "6852",
"benign_powershell_sysadmin-survival-kit_narrador" : "7120",
"benign_powershell_sysadmin-survival-kit_ssd-tune" : "6852",
"benign_powershell_sysadmin-survival-kit_trabalho" : "2444",
"benign_powershell_sysadmin-survival-kit_tratandoerros" : "5400",
"benign_powershell_sysadmin-survival-kit_validandocaminhos" : "2316",
"benign_powershell_sysadmin-survival-kit_wsl" : "2684",

}



###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################


# "main_offline_train_data_dirpath" (does not include e.g. "Processed_Benign")
# > "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/OFFLINE_TRAINING__BenignTrain666_BenignUser57_MalTrain416__edgeattr_only_TaskName"
# "main_offline_test_data_main_dirpath" (does not include e.g. "Processed_Benign")
# > e.g. /data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NOT_USED_FOR_OFFLINE_TRAINING__BenignTest167_BenignUser15_MalTest104__edgeattr_only_TaskName
## SET THESE MANUALLY (Main parameters)
#subgraphs_savedirpath = \
#f"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/BASED_ON_STREAMLINEMAIN_EXEC_AT_20230208_6:17pm"   # Do this manually

# SET
subgraphs_savedirpath = \
"/home/pwakodi1/tabby/Apache_benign_train_test_multigraph_data/Indices"
# "/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/all_malware_psh_samples_asof_20230520_MULTI_GRAPH"


#"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/Benign_logs_Meng_Machine_10th_May_2023"
#f"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/Benign_logs_Priti_Machine_10th_May_2023" (check)
#"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/Benign_logs_Meng_Machine_10th_May_2023"
#f"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/Benign_logs_Priti_Machine_10th_May_2023" (check)

malware_es_indices_and_ProcessIDs = Apache_benign_logs
# dir_start_pattern doesn't have to be exact.
dir_start_pattern = "benign_powershell" 

# SET
main_offline_train_data_dirpath=\
"/home/pwakodi1/tabby/Apache_benign_train_test_multigraph_data/offline_train"
#"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/MAIN_OFFLINE_TRAIN_DATA_PSH__PREPROCESSED_METHOD_1_DROPPED_QUERYEA"

# SET
main_offline_test_data_main_dirpath=\
"/home/pwakodi1/tabby/Apache_benign_train_test_multigraph_data/offline_test"
#"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/MAIN_OFFLINE_TEST_DATA_PSH__PREPROCESSED_METHOD_1_DROPPED_QUERYEA"

# SET non_target and/or target
non_target = False
target = True # typically malware
label = "Benign"
if label not in ["Benign", "Malware"]:
   raise ValueError("label must be either 'Benign' or 'Malware' ; case-matters b/c of compatibiltiy with existing code")

# SET whether it is debugging-mode 
firststep_hash_debugging_mode = False

#f"/data/d1/jgwak1/STREAMLINED_DATA_GENERATION_JY/DEVELOP_DROPPING_EVENTTYPES/MAIN_OFFLINE_TEST_DATA"
#"/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/MAIN_OFFLINE_TEST_DATA"

# SET Train data vs Test data ratio (set as 80% vs 20%)
Train_ratio = 0.8
Test_ratio = 1 - Train_ratio 



# SET benign steps
non_target__step_1 = False
non_target__step_2 = True
non_target__step_3 = False
non_target__step_4 = False

# SET malware steps
target__step_1 = False   # JY @ 2023-05-21 : currently first_step is commented out as want to start from edge-direction
target__step_2 = True
target__step_3 = False




# SET Number of processes to run in parallel for benign_step_1 & benign_step_2
# IMPORTANT: IF N_parallel is TOO HIGH (SAY OVER 5), IT IS POSSIBLE THAT ELASTIC-SEARCH dies
N_parallel = 5 # (JUST FIX IT TO 5, since we don't want to end up having issue)
if N_parallel > 5:
   raise ValueError(f"N_parallel ({N_parallel}) > 5 has risk to server-down of Elastic-Search")


# Added by JY @ 2023-03-02: Added 3rd argument "EventTypes_to_Exclude_set"
            #  Elements in 'EventTypes_to_Exclude_set' will be in format of 
            #  f"{Task Name in original-format as in raw event-log}{Opcode (optional; only for Registry/Network events)}" 
            #  ^ Notice that no space between TaskName and Opcode.
            #  > To check "Task Name in original-format as in raw event-log" ---> Go to ElasticSearch
            # >  To check "Opcode (optional; only for Registry/Network events)" ---> Go to ~/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RFSVM_ngram_flattened_subgraph.py
EventTypes_to_Exclude_set = {}
# EventTypes_to_Exclude_set = { 
#    # PROCESS Event-Types --------------------

#    # FILE Event-Types -----------------------

#    # REGISTRY Event-Types -------------------
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY32", # CreateKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY33", # OpenKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY34", # DeleteKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY35", # QueryKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY36", # SetValueKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY37", # DeleteValueKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY38", # QueryValueKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY39", # EnumerateKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY40", # EnumerateValueKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY41", # QueryMultipleValueKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY42", # SetInformationKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY43", # FlushKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY44", # CloseKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY45", # QuerySecurityKey
#    "MICROSOFT-WINDOWS-KERNEL-REGISTRY46", # SetSecurityKey

#    # NETWORK Event-Types --------------------
#    ## TCP --
#    # "KERNEL_NETWORK_TASK_TCPIP10", # TCPIPDatasent
#    # "KERNEL_NETWORK_TASK_TCPIP11", # TCPIPDatareceived
#    # "KERNEL_NETWORK_TASK_TCPIP12", # TCPIPConnectionattempted
#    # "KERNEL_NETWORK_TASK_TCPIP13", # TCPIPDisconnectissued
#    # "KERNEL_NETWORK_TASK_TCPIP14", # TCPIPDataretransmitted
#    # "KERNEL_NETWORK_TASK_TCPIP15", # TCPIPConnectionaccepted
#    # "KERNEL_NETWORK_TASK_TCPIP16", # TCPIPReconnectattempted
#    # "KERNEL_NETWORK_TASK_TCPIP17", # TCPIPTCPconnectionattemptfailed
#    # "KERNEL_NETWORK_TASK_TCPIP18", # TCPIPProtocolcopieddataonbehalfofuser

#    ## UDP --
#    # "KERNEL_NETWORK_TASK_UDPIP42", # DatasentoverUDPprotocol
#    # "KERNEL_NETWORK_TASK_UDPIP43", # DatareceivedoverUDPprotocol
#    # "KERNEL_NETWORK_TASK_UDPIP49", # UDPconnectionattemptfailed


# }

###################################################################################################################################################

if __name__ == "__main__":
   

   if non_target:

      # (Step-0) In case of step_1 & step_2, prepare for multi-processing by splitting the "benign_es_indices" into N parts.
      benign_es_indices_segments = [ segment.tolist() for segment in np.array_split(benign_es_indices, N_parallel) if len(segment) != 0 ]
      print(f"'benign_es_indices' divided into {N_parallel} segments (note: empty segments are dropped)\n",flush=True)
      print(benign_es_indices_segments, sep="\n", flush= True)
      #-------------------------------------------------------------------------------------------------------------------------------------      


      # (Step-1) Given a set of "elastic-search indices (before format)", read in those, format those, and re-upload to elastic-search.
      #         TODO: This part can take quite a while. Better to parallelize.
      #         TODO: Also measure time

      if non_target__step_1:
         step1_start = datetime.datetime.now()

         # run_format_logentry_on_elasticsearch( unformatted_elasticsearch_indices = benign_es_indices )
         #----------------------------------------------------------------------------------------------------
         # [ Multi-processing for 'run_format_logentry_on_elasticsearch' ] 
         processes=[]
         for benign_es_indices_segment in benign_es_indices_segments:
            proc = Process(target = run_format_logentry_on_elasticsearch, kwargs = { "unformatted_elasticsearch_indices": benign_es_indices_segment })
            proc.start()
            processes.append(proc)
         for proc in processes:
            proc.join()
         #----------------------------------------------------------------------------------------------------
         step1_done = datetime.datetime.now()
         print(f"\nBenign Step-1 elapsed time (#parallel-processes: {N_parallel}): {str(step1_done - step1_start)}", flush=True)

      # (Step-2) With formated elastic-sarch indices, generate CG and Project SGs.
      #         TODO: This part can take quite a while. Better to parallelize.
      #         TODO: Also measure time
      
      if non_target__step_2:
         step2_start = datetime.datetime.now()

         #run_nonTargetted_Subgraph_Generation(Log_Collection_Indices = benign_es_indices, 
         #                                     subgraphs_to_save_dirpath = subgraphs_savedirpath)
         #----------------------------------------------------------------------------------------------------
         # [ Multi-processing for 'run_NonTargetted_Subgraph_Generation' ] 
         processes=[]
         for benign_es_indices_segment in benign_es_indices_segments:
            proc = Process(target = run_NonTargetted_Subgraph_Generation, 
                           kwargs = {"Log_Collection_Indices": benign_es_indices_segment,
                                     "subgraphs_to_save_dirpath": subgraphs_savedirpath,
                                     # Added by JY @ 2023-03-02: Added 3rd argument "EventTypes_to_Exclude_set"
                                     "EventTypes_to_Exclude_set": EventTypes_to_Exclude_set,
                                     "firststep_hash_debugging_mode": firststep_hash_debugging_mode,
                                     })
            proc.start()
            processes.append(proc)
         for proc in processes:
            proc.join()
         #----------------------------------------------------------------------------------------------------    
         
         print(f"\nSubgraphs saved at: {subgraphs_savedirpath}", flush=True)
         
         step2_done = datetime.datetime.now()
         print(f"\nBenign Step-2 elapsed time (#parallel-processes: {N_parallel}): {str(step2_done - step2_start)}", flush=True)


      # (Step-3) Data-preprocess subgraphs into pickle files.
      if non_target__step_3:
         step3_start = datetime.datetime.now()
         # 3-1: organize and rename subgraphs for compatibility with the data-processor code
         #      > Referring to /data/d1/jgwak1/tabby/GENERAL_LOG_COLLECTION_SUBGRAPHS_20230203/make_data_processable_for_general_log_collection.py
         organize_and_rename_subgraphs_into_Benign_dir( main_dirpath = subgraphs_savedirpath ) 
         print(f"\nOrganized/Renamed and copied subgraphs to: {subgraphs_savedirpath}/Benign", flush=True)

         # 3-2: run the data-processor which will generate + save processed-pickle-files to 
         #      "subgraphs_savedirpath/Processed_Benign" directory.       
         # run_data_processor_MultiEdge_only_taskname_edgeattr_ver(data_type = "Benign", load_and_save_path = subgraphs_savedirpath)   
         run_data_processor_MultiEdge_5BitNodeAttr_only_taskname_edgeattr_ver(data_type = "Benign", load_and_save_path = subgraphs_savedirpath)   


         Processed_picklefiles_dirpath = f"{subgraphs_savedirpath}/Processed_Benign_ONLY_TaskName_edgeattr"
         print(f"\nProcessed-data (pickle files) are saved to : {Processed_picklefiles_dirpath}", flush=True)
         step3_done = datetime.datetime.now()
         print(f"\nBenign Step-3 elapsed time: {str(step3_done - step3_start)}", flush=True)


      # (Step-4) Train & Test split the newly created Processed-data,
      #          distribute the "train-set" and "test-set" to 
      #          "offline_training_subgraphs_dirpath" and "offline_testing_subgraphs_dirpath", respectively.
      #          (Even if we are adding new train and test-data to existing train and test-dirs,
      #          the 80% vs 20% ratio will be preserved, as we are also distributing new train and test data 
      #          based on that ratio. )
      #          + For future refernce, add a "txt file that records these increments of data."
      if non_target__step_4:
         step4_start = datetime.datetime.now()

         Processed_picklefiles_dirpath = f"{subgraphs_savedirpath}/Processed_Benign_ONLY_TaskName_edgeattr"
         source_dirpath = Processed_picklefiles_dirpath
         saved_dirname = os.path.split(source_dirpath)[1] # e.g. 'Processed_Benign_ONLY_TaskName_edgeattr'
         
         # prepare the destination dirpaths for both train-pickle and test-pickles (if non-existent, create dest-dir)
         train_dest_dirpath = os.path.join(main_offline_train_data_dirpath, saved_dirname)
         test_dest_dirpath = os.path.join(main_offline_test_data_main_dirpath, saved_dirname)
         if not os.path.exists(train_dest_dirpath):
            os.makedirs(train_dest_dirpath)
            print(f"Created A New Destination-Directory for New Train-Subgraphs {train_dest_dirpath}",flush=True)
         if not os.path.exists(test_dest_dirpath):
            os.makedirs(test_dest_dirpath)
            print(f"Created A New Destination-Directory for New Test-Subgraphs {test_dest_dirpath}",flush=True)

         # split pickles into train and test
         source_pickle_filenames = [ pklfname for pklfname in os.listdir(source_dirpath) if pklfname.startswith("Processed")] 
         num_of_picklefiles = len(source_pickle_filenames)
         first_X_for_train = int(num_of_picklefiles * Train_ratio)

         train_pickles = source_pickle_filenames[:first_X_for_train]
         test_pickles = source_pickle_filenames[first_X_for_train:]

         # pointer to a train-pickle addition-record file
         train_pickles_addition_record_fp = open(os.path.join(train_dest_dirpath, "train_pickles_addition_record.txt"), "a") # "a" flag: Open for appending at the end of the file without truncating it. 
         print(f"="*150, flush= True, file= train_pickles_addition_record_fp)
         print(f"Added at {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush= True, file= train_pickles_addition_record_fp)
         print(f"> copy {len(train_pickles)} train-pickles from '{source_dirpath}' to '{train_dest_dirpath}'", flush= True, file= train_pickles_addition_record_fp)
         print(f"> corresponds to {Train_ratio*100}% of total {len(source_pickle_filenames)} pickle files.", flush= True, file= train_pickles_addition_record_fp)
                                                                    
                                                                                                                           #           Creates a new file if it does not exist.
         # pointer to a test-pickle addition-record file
         test_pickles_addition_record_fp = open(os.path.join(test_dest_dirpath, "test_pickles_addition_record.txt"), "a")
         print(f"="*150, flush= True, file= test_pickles_addition_record_fp)
         print(f"Added at {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush= True, file= test_pickles_addition_record_fp)
         print(f"> copy {len(test_pickles)} train-pickles from '{source_dirpath}' to '{test_dest_dirpath}'", flush= True, file= test_pickles_addition_record_fp)
         print(f"> corresponds to {Test_ratio*100}% of total {len(source_pickle_filenames)} pickle files.", flush= True, file= test_pickles_addition_record_fp)


         train_pickle_cnt = 1
         for train_pkl in train_pickles:
            shutil.copyfile(src= os.path.join(source_dirpath, train_pkl),
                              dst= os.path.join(train_dest_dirpath, train_pkl))
            print(f"{train_pickle_cnt}.copied '{train_pkl}' from '{source_dirpath}' to '{train_dest_dirpath}'", flush= True)
            print(f"{train_pickle_cnt}.copied '{train_pkl}' from '{source_dirpath}' to '{train_dest_dirpath}'", flush= True, file= train_pickles_addition_record_fp)
            train_pickle_cnt += 1

         test_pickle_cnt = 1
         for test_pkl in test_pickles:
            shutil.copyfile(src= os.path.join(source_dirpath, test_pkl),
                              dst= os.path.join(test_dest_dirpath, test_pkl))
            print(f"{test_pickle_cnt}. copied '{test_pkl}' from '{source_dirpath}' to '{test_dest_dirpath}'", flush= True)
            print(f"{test_pickle_cnt}. copied '{test_pkl}' from '{source_dirpath}' to '{test_dest_dirpath}'", flush= True, file= test_pickles_addition_record_fp)
            test_pickle_cnt += 1

         step4_done = datetime.datetime.now()

   #######################################################################################################################################################################
   #######################################################################################################################################################################
   #######################################################################################################################################################################

   if target:
      # malware_es_indices_and_ProcessIDs

      # (Step-0) In case of step_1 & step_2, prepare for multi-processing by splitting the "benign_es_indices" into N parts.
      def divide_dict(dictionary, n):
         # Calculate the number of items per partition
         items_per_partition = len(dictionary) // n
         # Split the keys of the dictionary into partitions
         keys_partitions = [list(dictionary.keys())[i:i+items_per_partition] for i in range(0, len(dictionary), items_per_partition)]
         # Convert partitions to separate dictionaries
         return [{k: dictionary[k] for k in partition} for partition in keys_partitions]

      malware_es_indices_and_ProcessIDs = {k:str(v) for k,v in malware_es_indices_and_ProcessIDs.items()} # ensure all PIDs are str format
      malware_es_indices_and_ProcessIDs_segments = divide_dict( malware_es_indices_and_ProcessIDs , N_parallel )
      print(f"'malware_es_indices_and_ProcessIDs' divided into {N_parallel} segments (note: empty segments are dropped)\n",flush=True)
      print(malware_es_indices_and_ProcessIDs_segments, sep="\n", flush= True)
      #-------------------------------------------------------------------------------------------------------------------------------------      

      # (Step-1) With formated elastic-sarch indices, generate CG and Project SGs.
      #         TODO: This part can take quite a while. Better to parallelize.
      #         TODO: Also measure time
      
      if target__step_1:
         step1_start = datetime.datetime.now()

         #run_NonTargetted_Subgraph_Generation(Log_Collection_Indices = benign_es_indices, 
         #                                     subgraphs_to_save_dirpath = subgraphs_savedirpath)
         #----------------------------------------------------------------------------------------------------
         # [ Multi-processing for 'run_NonTargetted_Subgraph_Generation' ] 
         processes=[]
         for malware_es_indices_and_ProcessIDs_segment in malware_es_indices_and_ProcessIDs_segments:
            proc = Process(target = run_Targetted_Subgraph_Generation, kwargs = {"ESIndex_ProcessID_dict": malware_es_indices_and_ProcessIDs_segment,
                                                                                 "subgraphs_to_save_dirpath": subgraphs_savedirpath,
                                                                                 "EventTypes_to_Exclude_set": EventTypes_to_Exclude_set,
                                                                                 "firststep_hash_debugging_mode": firststep_hash_debugging_mode })
        
            
            proc.start()
            processes.append(proc)
         for proc in processes:
            proc.join()
         #----------------------------------------------------------------------------------------------------    
         
         print(f"\nSubgraphs saved at: {subgraphs_savedirpath}", flush=True)
         
         step1_done = datetime.datetime.now()
         print(f"\Targetted Subgraph Generation - {label} Step-1 elapsed time (#parallel-processes: {N_parallel}): {str(step1_done - step1_start)}", flush=True)


      # (Step-2) Data-preprocess subgraphs into pickle files.
      if target__step_2:
         step2_start = datetime.datetime.now()
         # 3-1: organize and rename subgraphs for compatibility with the data-processor code
         #      > Referring to /data/d1/jgwak1/tabby/GENERAL_LOG_COLLECTION_SUBGRAPHS_20230203/make_data_processable_for_general_log_collection.py

         organize_and_rename_subgraphs_into_Malware_dir( main_dirpath = subgraphs_savedirpath, 
                                                         dir_start_pattern= dir_start_pattern , 
                                                         label = label) 
         print(f"\nOrganized/Renamed and copied subgraphs to: {subgraphs_savedirpath}/Malware", flush=True)
         # print(f"\nOrganized/Renamed and copied subgraphs to: {subgraphs_savedirpath}/{label}", flush=True)

         # # 3-2: run the data-processor which will generate + save processed-pickle-files to 
         # #      "subgraphs_savedirpath/Processed_Benign" directory.       
         # run_data_processor_MultiEdge_only_taskname_edgeattr_ver(data_type = label, load_and_save_path = subgraphs_savedirpath)   
         run_data_processor_MultiEdge_5BitNodeAttr_only_taskname_edgeattr_ver(data_type = label, load_and_save_path = subgraphs_savedirpath)  

         Processed_picklefiles_dirpath = f"{subgraphs_savedirpath}/Processed_{label}_ONLY_TaskName_edgeattr"
         print(f"\nProcessed-data (pickle files) are saved to : {Processed_picklefiles_dirpath}", flush=True)
         step2_done = datetime.datetime.now()
         # print(f"\nMalware Step-2 elapsed time: {str(step2_done - step2_start)}", flush=True)


      # (Step-3) Train & Test split the newly created Processed-data,
      #          distribute the "train-set" and "test-set" to 
      #          "offline_training_subgraphs_dirpath" and "offline_testing_subgraphs_dirpath", respectively.
      #          (Even if we are adding new train and test-data to existing train and test-dirs,
      #          the 80% vs 20% ratio will be preserved, as we are also distributing new train and test data 
      #          based on that ratio. )
      #          + For future refernce, add a "txt file that records these increments of data."
      if target__step_3:
         step3_start = datetime.datetime.now()

         Processed_picklefiles_dirpath = f"{subgraphs_savedirpath}/Processed_{label}_ONLY_TaskName_edgeattr"
         source_dirpath = Processed_picklefiles_dirpath
         saved_dirname = os.path.split(source_dirpath)[1] # e.g. 'Processed_Benign_ONLY_TaskName_edgeattr'
         
         # prepare the destination dirpaths for both train-pickle and test-pickles (if non-existent, create dest-dir)
         train_dest_dirpath = os.path.join(main_offline_train_data_dirpath, saved_dirname)
         test_dest_dirpath = os.path.join(main_offline_test_data_main_dirpath, saved_dirname)
         if not os.path.exists(train_dest_dirpath):
            os.makedirs(train_dest_dirpath)
            print(f"Created A New Destination-Directory for New Train-Subgraphs {train_dest_dirpath}",flush=True)
         if not os.path.exists(test_dest_dirpath):
            os.makedirs(test_dest_dirpath)
            print(f"Created A New Destination-Directory for New Test-Subgraphs {test_dest_dirpath}",flush=True)

         # split pickles into train and test
         source_pickle_filenames = [ pklfname for pklfname in os.listdir(source_dirpath) if pklfname.startswith("Processed")] 
         num_of_picklefiles = len(source_pickle_filenames)
         first_X_for_train = int(num_of_picklefiles * Train_ratio)

         train_pickles = source_pickle_filenames[:first_X_for_train]
         test_pickles = source_pickle_filenames[first_X_for_train:]

         # pointer to a train-pickle addition-record file
         train_pickles_addition_record_fp = open(os.path.join(train_dest_dirpath, "train_pickles_addition_record.txt"), "a") # "a" flag: Open for appending at the end of the file without truncating it. 
         print(f"="*150, flush= True, file= train_pickles_addition_record_fp)
         print(f"Added at {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush= True, file= train_pickles_addition_record_fp)
         print(f"> copy {len(train_pickles)} train-pickles from '{source_dirpath}' to '{train_dest_dirpath}'", flush= True, file= train_pickles_addition_record_fp)
         print(f"> corresponds to {Train_ratio*100}% of total {len(source_pickle_filenames)} pickle files.", flush= True, file= train_pickles_addition_record_fp)
                                                                    
                                                                                                                           #           Creates a new file if it does not exist.
         # pointer to a test-pickle addition-record file
         test_pickles_addition_record_fp = open(os.path.join(test_dest_dirpath, "test_pickles_addition_record.txt"), "a")
         print(f"="*150, flush= True, file= test_pickles_addition_record_fp)
         print(f"Added at {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush= True, file= test_pickles_addition_record_fp)
         print(f"> copy {len(test_pickles)} train-pickles from '{source_dirpath}' to '{test_dest_dirpath}'", flush= True, file= test_pickles_addition_record_fp)
         print(f"> corresponds to {Test_ratio*100}% of total {len(source_pickle_filenames)} pickle files.", flush= True, file= test_pickles_addition_record_fp)


         train_pickle_cnt = 1
         for train_pkl in train_pickles:
            shutil.copyfile(src= os.path.join(source_dirpath, train_pkl),
                              dst= os.path.join(train_dest_dirpath, train_pkl))
            print(f"{train_pickle_cnt}.copied '{train_pkl}' from '{source_dirpath}' to '{train_dest_dirpath}'", flush= True)
            print(f"{train_pickle_cnt}.copied '{train_pkl}' from '{source_dirpath}' to '{train_dest_dirpath}'", flush= True, file= train_pickles_addition_record_fp)
            train_pickle_cnt += 1

         test_pickle_cnt = 1
         for test_pkl in test_pickles:
            shutil.copyfile(src= os.path.join(source_dirpath, test_pkl),
                              dst= os.path.join(test_dest_dirpath, test_pkl))
            print(f"{test_pickle_cnt}. copied '{test_pkl}' from '{source_dirpath}' to '{test_dest_dirpath}'", flush= True)
            print(f"{test_pickle_cnt}. copied '{test_pkl}' from '{source_dirpath}' to '{test_dest_dirpath}'", flush= True, file= test_pickles_addition_record_fp)
            test_pickle_cnt += 1

         step3_done = datetime.datetime.now()

         print(f"\nMalware Step-3 elapsed time: {str(step3_done - step3_start)}", flush=True)
