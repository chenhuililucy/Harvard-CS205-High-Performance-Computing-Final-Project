import os
import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # render without X server running
import matplotlib.pyplot as plt

def main():
    data_factor = 10**3
    time_factor = 10**-6

    directory = './'
    log_files = glob.glob(directory+"/*.log")


    # Intel Xeon E5-2683 v4 CPU specs
    ridge_x = 7.0
    ridge_y = 537.6

    # Covariance Operational Intensity
    intensity = 0.25
    peak_mem_bandwidth = 76.8
    
    # for visual
    dot_displacement = 0.05

    for file in log_files:
        file1 = open(file, "r")
        lines = file1.readlines()
        label_list = lines[0].strip().split("; ")
        function_name = lines[1].strip()
        data_size_list = [int(x)*data_factor for x in lines[2].strip().split(",")]
        sequential_wall_time_list = [float(x)*time_factor for x in lines[3].strip().split(",")]
        ispc_wall_time_list = [float(x)*time_factor for x in lines[4].strip().split(",")]

        n = len(data_size_list)
        ispc_Speedup_list = [sequential_wall_time_list[idx]/ispc_wall_time_list[idx] for idx in range(n)]

        dot_ispc_y = round(data_size_list[-1]*4/ispc_wall_time_list[-1]/(10**9),2)

        # Plot the roofline ceiling graph
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the line connecting the origin to the ridge point (logarithmic scale)

        ax.loglog([0, ridge_x], [0, ridge_y], 'b-', linewidth=2)
        ax.hlines(ridge_y, ridge_x, 100, color='blue', linewidth=2, label='Hardware Ceiling')
        ax.vlines(intensity, -1, 1000, color='black', linestyle = 'dashed', linewidth=1, label='Operational Intensity')

        # Set the axis labels and title
        ax.set_xlabel('Operational Intensity (FLOPs/Byte)')
        ax.set_ylabel('Performance (GFLOPs)')
        ax.set_title('Roofline Analysis of the ' + function_name)
        ax.scatter(ridge_x, ridge_y, color = "red", s = 100, label = "Ridge Point")
        ax.scatter(intensity, peak_mem_bandwidth*intensity, color = "blue", s = 100, label = "Peak Attainable at "+str(intensity)+" flop/byte")
        ax.scatter(intensity, data_size_list[-1]*4/sequential_wall_time_list[-1]/(10**9), color = "green", s = 100, label = "Sequential Performance")
        ax.scatter(intensity, dot_ispc_y, color = "orange", s = 100, label = "ISPC Performance")

        ax.annotate("("+str(ridge_x)+", "+str(ridge_y)+")", (6.5, 320))
        ax.annotate("(0.25, "+str(dot_ispc_y)+")", (intensity+dot_displacement, dot_ispc_y))



        # Show the plot
        plt.legend(loc = "lower right")
        plt.grid(":", alpha = 0.4)
        plotname = 'Roofline analysis for '+ function_name + ''
        filepath = "./" + plotname + '.png'

        fig.savefig(filepath)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(data_size_list, sequential_wall_time_list, 'o-', color = "green", label = "Sequential")

        plt.plot(data_size_list, ispc_wall_time_list, 'o-', color = "orange", label = "ISPC")
        plt.xlabel("Data Size (thousands)")
        plt.ylabel("Run Time (microseconds)")
        plt.title("Time Measurement of "+function_name)
        plt.legend()
        plt.grid(":", alpha = 0.4)
        plotname = 'Time Measurement of '+ function_name + ''
        filepath = "./" + plotname + '.png'

        fig.savefig(filepath)
        plt.close()

if __name__ == "__main__":
    main()