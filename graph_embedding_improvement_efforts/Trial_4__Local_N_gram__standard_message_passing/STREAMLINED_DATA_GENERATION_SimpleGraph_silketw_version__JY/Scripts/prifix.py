def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    return lines

def remove_suffix(line):
    # Ignore specific suffixes (e.g., "--case1--1234")
    if " :" in line:
        line = line.split(" :")[0]
    return line

file1_path = '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/malware_prefixes.txt'
# Open the file for reading
with open(file1_path, 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

processed_lines = ['"{0}"'.format(line.strip()) for line in lines]

# Open the file for writing
with open(file1_path, 'w') as file:
    # Write the processed lines back to the file
    file.write('\n'.join(processed_lines))

file2_path = '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/benign_case2.txt'

lines_file1 = set(map(remove_suffix, read_lines(file1_path)))
lines_file2 = set(map(remove_suffix, read_lines(file2_path)))

missing_lines = lines_file2 - lines_file1

print("Missing Lines:")
for line in missing_lines:
    print(line)
