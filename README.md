# CPSC 503 final project source code

WIP.

---
#### ProdLDA and NVLDA
---

ProdLDA/NVLDA code in this repository is a modified version of the original TensorFlow implementation by @akashgit.

* Original paper: [Autoencoding Variational Inference for Topic Models](https://arxiv.org/abs/1703.01488)
* Original TensorFlow implementation: [GitHub](https://github.com/akashgit/autoencoding_vi_for_topic_models)

Requirements:

* Python 3.7.6
* TensorFlow 1.15.0
* CUDA 10.0
* cuDNN 7.6.5

Run ProdLDA model:
> `py run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 200`

Run NVLDA model:
> `py run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300`

To change the target corpus, uncomment the corresponding corpus variable in run.py and comment out the other.

---
#### Data sets
---

1. 20 Newsgroups: included with original ProdLDA code
2. New York Times: download from [Kaggle](https://www.kaggle.com/nzalake52/new-york-times-articles). Put in data/nytimes and run corpus_to_numpy.py