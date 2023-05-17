import os
import re
import glob
from collections import defaultdict
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')  # render without X server running
import matplotlib.pyplot as plt

def plot_by_group(group_path_list):

    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)
    #print(group_path_list)
    log_files = group_path_list
    function_name_list =[]
    min_node_list =[]
    data_size_list =[]
    fig, axes = plt.subplots(figsize=(10, 10))

    loc='center left'
    bbox_to_anchor=(0.8, 0.75)

    to_sort_log_files = []
    number_regex = re.compile(r'\d+')
    for file in log_files:
        match = number_regex.search(file)
        if match:
            # If a number is found, extract it and add it to the numbers list
            number = int(match.group())
            data_size_list.append(number)
            to_sort_log_files.append(file)
    sorted_pairs = sorted(zip(data_size_list, to_sort_log_files))
    data_size_list, sorted_log_files = zip(*sorted_pairs)

    function_name_set = set()

    for idx, file in enumerate(sorted_log_files):
        file1 = open(file, "r")
        lines = file1.readlines()
        function_name = lines[1].strip()
        if function_name not in function_name_set:
            function_name_set.add(function_name)
            function_name_list.append(function_name)
        process_count_list = [int(x) for x in lines[2].strip().split(",")]
        mean_wall_time_list = [float(x) for x in lines[3].strip().split(",")]
        standard_deviation_list = [float(x) for x in lines[4].strip().split(",")]
        n = len(process_count_list)
        Speedup_list = [mean_wall_time_list[0]/mean_wall_time_list[idx] for idx in range(n)]

        min_node_list.append(process_count_list[mean_wall_time_list.index(min(mean_wall_time_list))])

        # plot strong scaling

        plt.plot(process_count_list, Speedup_list, label=str(data_size_list[idx])+" images", marker='o')


    axes.set_xlabel("Number of parallel processors p")
    axes.set_ylabel("Strong Speedup $S_p$")
    lims = [
        np.min([axes.get_xlim(), axes.get_ylim()]),  # min of both axes
        np.max([axes.get_xlim(), axes.get_ylim()]),  # maxes of both axes
    ]

    # now plot both limits against each other
    axes.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    axes.set_aspect('equal')
    plt.legend(loc = loc, bbox_to_anchor = bbox_to_anchor)
    function_names = ", ".join(function_name_list)
    #plotname = 'Strong Scaling Plot for '+ function_names + ''
    data_sizes = ", ".join([str(x) for x in data_size_list])
    plotname = 'Strong Scaling Plot for Optimized ' + function_names + ' Training over Different Number of Images'
    plt.title(plotname)
    filepath = save_dir + plotname + '.png'

    fig.savefig(filepath)
    plt.close()

    fig, axes = plt.subplots(figsize=(10, 10))
    plt.plot(data_size_list, min_node_list, color = "blue", marker='o')
    plotname = 'Optimal Node for Optimized ' + function_names + ' Training over Different Number of Images'
    plt.title(plotname)
    filepath = save_dir + plotname + '.png'
    axes.set_ylabel("Number of parallel processors p")
    axes.set_xlabel("Number of Training Images")
    fig.savefig(filepath)
    plt.close()

    fig, axes = plt.subplots(figsize=(10, 10))
    function_name_list =[]
    for idx, file in enumerate(sorted_log_files):
        file1 = open(file, "r")
        lines = file1.readlines()
        function_name = lines[1].strip()
        function_name_list.append(function_name)
        process_count_list = [int(x) for x in lines[2].strip().split(",")]
        mean_wall_time_list = [float(x) for x in lines[3].strip().split(",")]
        standard_deviation_list = [float(x) for x in lines[4].strip().split(",")]
        n = len(process_count_list)
        Speedup_list = [mean_wall_time_list[0]/mean_wall_time_list[idx] for idx in range(n)]

        plt.errorbar(process_count_list, mean_wall_time_list, yerr=standard_deviation_list, label=str(data_size_list[idx])+" images", marker='o', markersize=8, capsize=20)
    plt.legend(loc = loc, bbox_to_anchor = bbox_to_anchor)
    axes.set_xlabel("Number of parallel processors p")
    axes.set_ylabel("Time in seconds")

    #plotname = 'Time Measurement for '+ function_names + ''
    plotname = 'Time Measurement for Optimized ' + function_names + ' Training over Different Number of Images'
    plt.title(plotname)
    filepath = save_dir + plotname + '.png'

    fig.savefig(filepath)
    plt.close()
def clean_logs_and_plot():

    # 1. Go over a given directory and find all log files
    given_directory = './logs'
    log_files = glob.glob(given_directory+'/*.log')

    # 2. Group log files based on the part of their filenames before "images"
    file_groups = defaultdict(list)
    pattern = r'(.*\s+images).*\.log'
    for log_file in log_files:
        filename = os.path.basename(log_file)
        match = re.match(pattern, filename)
        if match:
            group_key = match.group(1)
            file_groups[group_key].append(log_file)

    # 3. For each group, read their file content line by line and concatenate different lines
    # 4. Store the resulting concatenated lines in the "combined logs" folder
    combined_logs_folder = './combined_logs'
    os.makedirs(combined_logs_folder, exist_ok=True)

    for group_key, group_files in file_groups.items():
        combined_lines = defaultdict(set)

        for log_file in group_files:
            with open(log_file, 'r') as file:
                for line_idx, line in enumerate(file):
                    combined_lines[line_idx].add(line.strip())

        combined_log_file = os.path.join(combined_logs_folder, f'{group_key}.log')

        with open(combined_log_file, 'w') as file:
            for line_idx in range(5):
                for line in combined_lines[line_idx]:
                    file.write(line + '\n')

        #print(f"Combined log files for group '{group_key}' are stored in '{combined_log_file}'")


    # Set the directory for combined logs
    combined_logs_directory = './combined_logs'

    # find each kernel's runtime logs for different image sizes
    log_files = glob.glob(combined_logs_directory+"/*.log")

    # Create groups based on the part of file names prior to "using" in the filename
    file_groups = defaultdict(list)
    pattern = r'(.*\s+using).*\.log'

    for log_file in log_files:
        filename = os.path.basename(log_file)
        match = re.match(pattern, filename)
        if match:
            group_key = match.group(1)
            file_groups[group_key].append(filename)

    # For each group, call plot_by_group(group_path_list)
    for group_key, group_files in file_groups.items():
        group_path_list = [os.path.join(combined_logs_directory, filename) for filename in group_files]
        plot_by_group(group_path_list)

if __name__ == "__main__":
    clean_logs_and_plot()