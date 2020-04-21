# Evaluating topic stability and variability for variational autoencoder topic models

This repository contains the source code for Kenny Chiu's [CPSC 503](https://www.cs.ubc.ca/~carenini/TEACHING/CPSC503-20/503-20.html) final project at the University of British Columbia. The project report can be found [here](https://github.com/chiukenny/CPSC503-finalproject/blob/master/report/report.pdf).

---
#### NVLDA and ProdLDA

Requirements:
* Python 3.7.6
* TensorFlow 1.15.0
* CUDA 10.0
* cuDNN 7.6.5

Run ProdLDA model:
> `py run.py -m prodlda -c 20ng -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 200`

Run NVLDA model:
> `py run.py -m nvlda -c 20ng -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300`

---
#### Datasets

1. 20 Newsgroups: included with original ProdLDA code. To train on 20NG, run with argument
> `-c 20ng`

2. New York Times: download from [Kaggle](https://www.kaggle.com/nzalake52/new-york-times-articles). Place text file in data/nytimes and run the following script to preprocess it before use:
> `py nyt_to_numpy.py`

To train on NYT, run with argument
> `-c nyt`

---
#### Computing results

Running run.py produces CSVs in \\results by default. Run the following to calculate statistics and create plots:
> `py compute_results.py -f 20news_topics_prodlda.csv`

---
#### Acknowledgments

The ProdLDA/NVLDA code in this repository is a Python 3 adaptation of the original TensorFlow implementation by [@akashgit](https://github.com/akashgit).

* Original NVLDA/ProdLDA paper: [Autoencoding Variational Inference for Topic Models](https://arxiv.org/abs/1703.01488)
* Original NVLDA/ProdLDA TensorFlow implementation: [GitHub](https://github.com/akashgit/autoencoding_vi_for_topic_models)