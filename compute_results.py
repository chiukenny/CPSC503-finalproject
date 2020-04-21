#!/usr/bin/python

import getopt
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sys


# Directories
result_dir = "results/"
plot_dir = "plots/"

# Expected metric column names in result CSV file:
col_human = "Human"
col_coh = "Coherence"
col_cohep = "Coherence+eps"
col_pmi = "PMI"
col_npmi = "NPMI"
col_cos = "Cosine"
col_kl = "KL"
col_euc = "Euclidean"
col_jac = "Jaccard"
col_vari = "Variability"
col_pvari = "Post Variability"
metrics = [col_coh, col_cohep, col_pmi, col_npmi, col_cos, col_kl, col_euc, col_jac, col_vari, col_pvari]


# Compute metric averages
def compute_average(file_name):
	results = pd.read_csv(result_dir + file_name)
	print(col_human, "average:", np.mean(results[col_human]))
	for metric in metrics:
		print(metric, "average:", round(np.mean(results[metric]),2))


# Compute correlation between metrics and human scores
def compute_correlation(file_name):
	results = pd.read_csv(result_dir + file_name)
	for metric in metrics:
		print(metric)
		print("\tPearson corr.:", round(pearsonr(results[col_human], results[metric])[0],3))
		print("\tSpearman's corr.:", round(spearmanr(results[[col_human, metric]])[0],3))
		
        
# Plot human scores against metrics
def plot_scores(file_name):
	results = pd.read_csv(result_dir + file_name)
	
	# Add jitter to human scores to avoid overlap
	human = results[col_human].copy()
	human += np.random.random(len(human)) * 0.25
	
    # Specify metrics to plot and number of plots per row
	plot_metrics = [cNPMI, cPVari]
	plots_per_row = 2
	point_size = 15
	point_alpha = 0.5
    
	fig, axs = plt.subplots(math.ceil(len(plot_metrics)/plots_per_row), plots_per_row, figsize=(4.5,2.5))
	fig.set_tight_layout(True)
	
	plot_num = 0
	row = 0
	for metric in plot_metrics:
		if plot_num == plots_per_row:
			plot_num = 0
			row += 1
		scores = results[metric]
		score_range = max(scores) - min(scores)
		if plots_per_row == 1:
			axs[row].scatter(scores,human, s=point_size, alpha=point_alpha)
			axs[row].set_xlim(min(scores)-0.1*score_range, max(scores)+0.1*score_range)
			axs[row].set_xlabel(metric)
			axs[row].set_ylabel("Human score")
		elif len(plot_metrics) <= plots_per_row:
			axs[plot_num].scatter(scores,human, s=point_size, alpha=point_alpha)
			axs[plot_num].set_xlim(min(scores)-0.1*score_range, max(scores)+0.1*score_range)
			axs[plot_num].set_xlabel(metric)
			if plot_num > 0:
				axs[plot_num].set_yticklabels([])
			else:
				axs[plot_num].set_ylabel("Human score")
			axs[plot_num].set_title(r"$r$: " + str(round(pearsonr(results[col_human], results[metric])[0],3)) + r", $\rho$: " + str(round(spearmanr(results[[col_human, metric]])[0],3)))
		else:
			axs[row,plot_num].scatter(scores,human, s=point_size, alpha=point_alpha)
			axs[row,plot_num].set_xlim(min(scores)-0.1*score_range, max(scores)+0.1*score_range)
			axs[plot_num].set_xlabel(metric)
			if plot_num > 0:
				axs[row,plot_num].set_yticklabels([])
			else:
				axs[row,plot_num].set_ylabel("Human score")
		plot_num += 1
	plt.savefig(plot_dir+"plot_"+file_name+".png", dpi=300)


# Path not needed in name of result file
def main(argv):
	try:
		opts, args = getopt.getopt(argv, "h:f:",["file="])
	except getopt.GetoptError:
		print("py compute_results.py -f <resultfile>")
		sys.exit(2)
        
	for opt, arg in opts:
		if opt == "-h":
			print("py compute_results.py -f <resultfile>")
			sys.exit()
		elif opt == "-f":
			f = arg
            
	compute_average(f)
	compute_correlation(f)
	plot_scores(f)
			
            
if __name__ == "__main__":
   main(sys.argv[1:])