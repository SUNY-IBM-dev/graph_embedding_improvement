def filter_lines(input_file, prefix, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith(prefix):
                outfile.write(line)

def filter_non_matching_lines(input_file, prefixes, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not any(line.startswith(prefix) for prefix in prefixes):
                outfile.write("\""+line+"\""+" : "+""+","+"\n")

# Replace 'input.txt' with your input file name
# Replace 'malware_output.txt' with your desired output file for lines starting with "malware_"
# Replace 'trial_output.txt' with your desired output file for lines starting with "trial_"
# Replace 'benign_idx.txt' with your desired output file for lines not starting with "malware_" or "trial_"

# filter_lines('/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/silketw_indices_list.txt', 'malware_', '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/malware_idx.txt')
# filter_lines('/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/silketw_indices_list.txt', 'atomic__', '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/caldera_atomic_idx.txt')
# filter_lines('/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/silketw_indices_list.txt', 'custom_', '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/caldera_custom_idx.txt')
#filter_non_matching_lines('/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/silketw_indices_list.txt', ['malware_', 'atomic__', 'custom_'], '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/benign_idx.txt')

def format_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(infile, start=1):
            if i % 2 != 0:  # Keep only odd-numbered lines
                cleaned_line = line.strip().strip('"')  # Remove leading/trailing whitespaces and double quotes
                formatted_line = f'"{cleaned_line}" : "",\n'
                outfile.write(formatted_line)# Replace 'input.txt' with your input file name
# Replace 'output.txt' with your desired output file
format_lines('/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/benign_idx.txt', '/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/Scripts/benign_idx_dict.txt')
