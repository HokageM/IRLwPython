import matplotlib.pyplot as plt
import os


class OutputHandler:
    def __int__(self):
        pass

    def save_heatmap_as_png(self, data, output_path, title=None, xlabel="Position", ylabel="Velocity"):
        """
        Create a heatmap from a numpy array and save it as a PNG file.
        :param data: 2D numpy array containing the heatmap data.
        :param output_path: Output path for saving the PNG file.
        :param xlabel: Label for the x-axis (optional).
        :param ylabel: Label for the y-axis (optional).
        :param title: Title for the plot (optional).
        """
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='viridis', interpolation='nearest')
        plt.colorbar(im)

        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        target_dir = os.path.basename(output_path)
        if not os.path.isdir(target_dir):
            print(f"Creating directory {target_dir}")
            os.mkdir(target_dir)

        plt.savefig(output_path, format='png')
        plt.close(fig)

    def save_plot_as_png(self, x, y, output_path, title=None, xlabel="Episodes", ylabel="Scores"):
        """
        Create a line plot from x and y data and save it as a PNG file.
        :param x: 1D numpy array or list representing the x-axis values.
        :param y: 1D numpy array or list representing the y-axis values.
        :param output_path: Output path for saving the plot as a PNG file.
        :param xlabel: Label for the x-axis (optional).
        :param ylabel: Label for the y-axis (optional).
        :param title: Title for the plot (optional).
        """
        fig, ax = plt.subplots()
        ax.plot(x, y)

        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        target_dir = os.path.basename(output_path)
        if not os.path.isdir(target_dir):
            print(f"Creating directory {target_dir}")
            os.mkdir(target_dir)

        plt.savefig(output_path, format='png')
        plt.close(fig)

    def save_network(self, network, output_path):
        target_dir = os.path.basename(output_path)
        if not os.path.isdir(target_dir):
            print(f"Creating directory {target_dir}")
            os.mkdir(target_dir)

    def save_qtable(self, qtable, output_path):
        target_dir = os.path.basename(output_path)
        if not os.path.isdir(target_dir):
            print(f"Creating directory {target_dir}")
            os.mkdir(target_dir)
