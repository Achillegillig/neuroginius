from matplotlib import pyplot as plt
import numpy as np

def plot_fdr(pvalues, pvalues_corr, alpha=0.05):
    fig, ax = plt.subplots()
    x = np.linspace(0, alpha, 100)
    ax.plot(x, x, color="black", label="Reference")
    ax.plot(np.sort(pvalues), np.sort(pvalues_corr), label="Corrected")
    ax.set_xlim(0, alpha)
    ax.hlines(
        y=alpha,
        xmin=0,
        xmax=alpha,
        colors="red", 
        label=f"Threshold $\\alpha = {alpha}$",
        linestyles="--"
    )
    ax.set_xlabel("pvalues_raw")
    ax.set_ylabel("pvalues_corrected")
    ax.set_title(f"False discovery correction")
    ax.legend()
    return fig