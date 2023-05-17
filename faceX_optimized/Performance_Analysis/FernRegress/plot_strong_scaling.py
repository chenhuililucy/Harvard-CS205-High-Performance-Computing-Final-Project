import os
import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # render without X server running
import matplotlib.pyplot as plt

def main():
    directory = './plots'
    log_files = glob.glob(directory+"/*.log")
    for file in log_files:
        file1 = open(file, "r")
        lines = file1.readlines()
        function_name = lines[1].strip()
        process_count_list = [int(x) for x in lines[2].strip().split(",")]
        mean_wall_time_list = [float(x) for x in lines[3].strip().split(",")]
        standard_deviation_list = [float(x) for x in lines[4].strip().split(",")]
        n = len(process_count_list)
        Speedup_list = [mean_wall_time_list[0]/mean_wall_time_list[idx] for idx in range(n)]
        
        # plot strong scaling
        fig, axes = plt.subplots(figsize=(10, 10))
        plt.plot(process_count_list, Speedup_list, color = "blue", marker='o')
        axes.set_xlabel("Number of parallel processors p")
        axes.set_ylabel("Strong Speedup $S_p$")
        lims = [
            np.min([axes.get_xlim(), axes.get_ylim()]),  # min of both axes
            np.max([axes.get_xlim(), axes.get_ylim()]),  # maxes of both axes
        ]

        # now plot both limits against each other
        axes.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        axes.set_aspect('equal')
        plotname = 'Strong Scaling Plot for '+ function_name + ' function'
        plt.title(plotname)
        filepath = "./" + plotname + '.png'

        fig.savefig(filepath)
        plt.close()


        # plot time measurement
        fig, axes = plt.subplots(figsize=(10, 10))

        plt.errorbar(process_count_list, mean_wall_time_list, yerr=standard_deviation_list, label='both limits (default)', marker='o', markersize=8, capsize=20)

        axes.set_xlabel("Number of parallel processors p")
        axes.set_ylabel("Time in seconds")

        plotname = 'Time Measurement for '+ function_name + ' function'
        plt.title(plotname)
        filepath = "./" + plotname + '.png'

        fig.savefig(filepath)
        plt.close()


if __name__ == "__main__":
    main()
