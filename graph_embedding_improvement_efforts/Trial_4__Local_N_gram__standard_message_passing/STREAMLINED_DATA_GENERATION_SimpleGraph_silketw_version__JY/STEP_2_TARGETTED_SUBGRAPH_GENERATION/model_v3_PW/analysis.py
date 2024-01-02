import os
import json

def extract_info_from_json(json_file_path):
    #print(json_file_path)
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        # print(data.get('RelativeName',''))
        for key, value in data.items():
            return value.get('RelativeName',''), value.get('FileName',''), value.get('daddr','')
    return "", "", ""

def process_directory(directory_path, output_file_path,output_reg_path,output_net_path):
    with open(output_file_path, 'w') as output_file:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('file_node.json'):
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                            # print(data.get('RelativeName',''))
                        for key, value in data.items():
                            
                            file_name = value.get('FileName','')
                           
 
                            output_line = f"{file_name}\n"
                            output_file.write(output_line)

    with open(output_reg_path, 'w') as output_file:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('reg_node.json'):
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                            # print(data.get('RelativeName',''))
                        for key, value in data.items():
                            relative_name = value.get('RelativeName','')
                            
                            
 
                            output_line = f"{relative_name}\n"
                            output_file.write(output_line)


    with open(output_net_path, 'w') as output_file:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('net_node.json'):
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                            # print(data.get('RelativeName',''))
                        for key, value in data.items():
                            daddr = value.get('daddr','')
 
                            output_line = f"{daddr}\n"
                            output_file.write(output_line)

if __name__ == "__main__":
    input_directory = "/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs/model_v3_PW/Subgraphs/malware_poshc2_powershell_logs_brute-ad"
    output_file_path = "/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs/model_v3_PW/Subgraphs/malware_poshc2_powershell_logs_brute-ad/file.txt"
    output_reg_path = "/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs/model_v3_PW/Subgraphs/malware_poshc2_powershell_logs_brute-ad/reg.txt"
    output_net_path = "/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs/model_v3_PW/Subgraphs/malware_poshc2_powershell_logs_brute-ad/net.txt"
    process_directory(input_directory, output_file_path,output_reg_path,output_net_path)
