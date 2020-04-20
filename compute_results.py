import sys, getopt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import math
import matplotlib.pyplot as plt

# Directories
result_dir = "results/"
plot_dir = "plots/"

# Expected column names in result CSV file:
cHuman = 'Human'
cCoh = 'Coherence'
cCohEps = 'Coherence+eps'
cPMI = 'PMI'
cNPMI = 'NPMI'
cCos = 'Cosine'
cKL = 'KL'
cEuc = 'Euclidean'
cJac = 'Jaccard'
cVari = 'Variability'
cPVari = 'Post Variability'

metrics = [cCoh, cCohEps, cPMI, cNPMI, cCos, cKL, cEuc, cJac, cVari, cPVari]


def compute_average(file_name):
	results = pd.read_csv(result_dir + file_name)
	
	print(cHuman, "average:", np.mean(results[cHuman]))
	for metric in metrics:
		print(metric, "average:", round(np.mean(results[metric]), 2))

def compute_correlation(file_name):
	results = pd.read_csv(result_dir + file_name)
	
	for metric in metrics:
		print(metric)
		print("\tPearson corr.:", round(pearsonr(results[cHuman], results[metric])[0], 3))
		print("\tSpearman's corr.:", round(spearmanr(results[[cHuman, metric]])[0], 3))
		
def plot_scores(file_name):
	results = pd.read_csv(result_dir + file_name)
	
	# Add jitter to human scores to avoid overlap
	human = results[cHuman].copy()
	human += np.random.random(len(human)) * 0.25
	
	plot_metrics = [cNPMI, cPVari]
	plots_per_row = 2
	fig, axs = plt.subplots(math.ceil(len(plot_metrics)/plots_per_row), plots_per_row, figsize=(4.5,2.5))
	fig.set_tight_layout(True)
	
	plot_num = 0
	row = 0
	point_size = 15
	point_alpha = 0.5
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
			axs[row].set_ylabel('Human score')
		elif len(plot_metrics) <= plots_per_row:
			axs[plot_num].scatter(scores,human, s=point_size, alpha=point_alpha)
			axs[plot_num].set_xlim(min(scores)-0.1*score_range, max(scores)+0.1*score_range)
			axs[plot_num].set_xlabel(metric)
			if plot_num > 0:
				axs[plot_num].set_yticklabels([])
			else:
				axs[plot_num].set_ylabel('Human score')
			axs[plot_num].set_title(r'$r$: ' + str(round(pearsonr(results[cHuman], results[metric])[0], 3)) + r", $\rho$: " + str(round(spearmanr(results[[cHuman, metric]])[0], 3)))
		else:
			axs[row,plot_num].scatter(scores,human, s=point_size, alpha=point_alpha)
			axs[row,plot_num].set_xlim(min(scores)-0.1*score_range, max(scores)+0.1*score_range)
			axs[plot_num].set_xlabel(metric)
			if plot_num > 0:
				axs[row,plot_num].set_yticklabels([])
			else:
				axs[row,plot_num].set_ylabel('Human score')
		plot_num += 1
	plt.savefig(plot_dir + "plot_" + file_name + ".png", dpi = 300)

def main(argv):
	f = ''
	try:
		opts, args = getopt.getopt(argv, 'h:f:',['file='])
	except getopt.GetoptError:
		print('compute_metric_correlation.py -f <resultfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('compute_metric_correlation.py -f <resultfile>')
			sys.exit()
		elif opt == '-f':
			f = arg
	# compute_average(f)
	# compute_correlation(f)
	plot_scores(f)
			
if __name__ == '__main__':
   main(sys.argv[1:])