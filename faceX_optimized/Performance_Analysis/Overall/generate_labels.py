import sys
import random

def sample_lines(file_path, num_lines):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if num_lines > len(lines):
        raise ValueError("Number of lines to sample exceeds the total number of lines in the file.")

    sampled_lines = random.sample(lines, num_lines)

    return sampled_lines

if __name__ == "__main__":
    # Check if the user provided the required command-line argument
    if len(sys.argv) < 2:
        print("Please provide the number of lines to sample as a command-line argument.")
        sys.exit(1)

    # Get the number of lines to sample from the command-line argument
    num_lines_to_sample = int(sys.argv[1])

    # Example usage:
    file_path = "labels.txt"

    sampled_lines = sample_lines(file_path, num_lines_to_sample)

    # Write the sampled lines to a new file
    output_file_path =  f"train_large/labels.txt"
    with open(output_file_path, 'w') as file:
        file.writelines(sampled_lines)

    print(f"Sampled lines written to {output_file_path}")