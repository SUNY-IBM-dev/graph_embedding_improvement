import os

def find_directories_without_subgraph(parent_directory, subgraph_directory_prefix):
    directories_without_subgraph = []

    for directory in os.listdir(parent_directory):
        full_path = os.path.join(parent_directory, directory)
        if os.path.isdir(full_path) and not any(d.startswith(subgraph_directory_prefix) for d in os.listdir(full_path)):
            directories_without_subgraph.append(directory)

    return directories_without_subgraph

# Replace "/path/to/directory_A" with the actual path to your directory A
# Replace "SUBGRAPH" with the prefix of the subdirectory you're looking for
parent_directory = "/home/pwakodi1/tabby/SILKETW_benign_train_test_data_case2/Indices"
subgraph_directory_prefix = "SUBGRAPH"

missing_directories = find_directories_without_subgraph(parent_directory, subgraph_directory_prefix)

if missing_directories:
    print("Directories without a subdirectory starting with 'SUBGRAPH':")
    for directory in missing_directories:
        print(directory)
else:
    print("All directories contain a subdirectory starting with 'SUBGRAPH'.")
