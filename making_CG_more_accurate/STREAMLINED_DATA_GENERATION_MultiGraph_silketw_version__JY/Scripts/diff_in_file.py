# Function to read prefixes from a file and return a set of prefixes
def read_prefixes(file_path):
    with open(file_path, 'r') as file:
        prefixes = set(line.strip() for line in file)
    return prefixes

# Paths to your two text files
file1_path = '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/silketw_indices_list.txt'
file2_path = '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/silketw_new_indices_list.txt'

# Read prefixes from the two files
prefixes_file1 = read_prefixes(file1_path)
prefixes_file2 = read_prefixes(file2_path)

# Find the missing prefixes
missing_prefixes = prefixes_file1 - prefixes_file2

# Print or do whatever you want with the missing prefixes
print("Missing Prefixes:")
for prefix in missing_prefixes:
    print(prefix)
