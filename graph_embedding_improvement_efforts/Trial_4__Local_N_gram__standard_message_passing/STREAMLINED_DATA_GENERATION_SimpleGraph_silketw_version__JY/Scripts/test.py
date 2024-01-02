import os
import shutil

def copy_subgraph_directories(src_parent_directory, dest_directory, subgraph_directory_prefix):
    for root, dirs, files in os.walk(src_parent_directory):
        dirs[:] = [d for d in dirs]  # Exclude the specified directory
        for dir in dirs:
            if dir.startswith(subgraph_directory_prefix):
                src_path = os.path.join(root, dir)
                dest_path = os.path.join(dest_directory, dir)

                if not os.path.exists(dest_path):
                    shutil.copytree(src_path, dest_path)
                else:
                    print(f"Destination directory '{dest_path}' already exists. Skipping.")# Replace "/path/to/src_directory" with the actual path to your source directory
# Replace "/path/to/destination_directory" with the actual path to your destination directory
# Replace "SUBGRAPH" with the prefix of the subdirectory you want to copy
# Replace "ExcludeThisDirectory" with the name of the directory you want to exclude
src_parent_directory = "/home/pwakodi1/tabby/SILKETW_benign_train_test_data_case2/Indices"
dest_directory = "/home/pwakodi1/tabby/SILKETW_benign_train_test_data_case2/Indices/Benign"
subgraph_directory_prefix = "SUBGRAPH"
# exclude_directory = "/home/pwakodi1/tabby/Silketw_train_test_benign_data/Indices/Benign"

copy_subgraph_directories(src_parent_directory, dest_directory, subgraph_directory_prefix)
