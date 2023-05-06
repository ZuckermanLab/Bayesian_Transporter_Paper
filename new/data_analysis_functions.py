import numpy as np 
import matplotlib.pyplot as plt


def plot_histograms(data_files, colors, bins, ranges=None, references=None, x_labels=None, titles=None, fname=None):
    """
    Plot histograms of multiple datasets with the same column structure.

    Args:
        data_files (list of str): List of file names containing data to plot.
        colors (list of str): List of colors to use for each data set.
        bins (int): Number of bins to use for the histograms.
        ranges (list of tuple, optional): List of (min, max) range tuples for each subplot, or None to use automatic range.
        references (list of float, optional): List of reference values for vertical lines, or None to not plot references.
        x_labels (list of str, optional): List of x-axis labels for each subplot, or None to use default values.
        titles (list of str, optional): List of titles for each subplot, or None to use default values.

    Returns:
        None.
    """
    # Create a figure with subplots for each column of data.
    num_cols = len(ranges) if ranges is not None else len(np.loadtxt(data_files[0], delimiter=',').T)
    fig, axs = plt.subplots(nrows=1, ncols=num_cols, figsize=(16, 5))

    # Load and plot data for each file.
    for i, file_name in enumerate(data_files):
        # Load the data file into a numpy array.
        data = np.loadtxt(file_name, delimiter=',')

        # Plot each column of data as a histogram on the corresponding subplot.
        for j in range(num_cols):
            range_j = ranges[j] if ranges is not None else None
            axs[j].hist(data[:, j], bins=bins, range=range_j, density=True, color=colors[i], alpha=0.5, histtype='step')

    # Add vertical lines for reference values to each subplot.
    if references is not None:
        for j in range(num_cols):
            for ref in references:
                axs[j].axvline(x=ref, color='black', linestyle='--', linewidth=1)

    # Set labels and titles for each subplot.
    for j in range(num_cols):
        if x_labels is not None:
            axs[j].set_xlabel(x_labels[j])
        axs[j].set_ylabel("Density")
        if titles is not None:
            axs[j].set_title(titles[j])

    # Add a legend for the different data sets.
    fig.legend(labels=data_files, loc="upper right")

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
        return fig, axs
    else:
        return fig, axs
