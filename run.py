#!/usr/bin/python

import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle
import sys, getopt
from models import prodlda, nvlda

from scipy.special import comb
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import random
import math
import csv
'''-----------Data--------------'''

# Change to select corpus
corpus = "20news"
# corpus = "nytimes"

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

if corpus == "20news":
    dataset_tr = 'data/20news_clean/train.txt.npy'
    data_tr = np.load(dataset_tr, allow_pickle=True, encoding='latin1')
    dataset_te = 'data/20news_clean/test.txt.npy'
    data_te = np.load(dataset_te, allow_pickle=True, encoding='latin1')
    vocab = 'data/20news_clean/vocab_unix.pkl'
    vocab = pickle.load(open(vocab,'rb'))
    vocab_size=len(vocab)
    #--------------convert to one-hot representation------------------
    print('Converting data to one-hot representation')
    data_tr = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
else:
    dataset_tr = 'data/nytimes/train.txt.npy'
    data_tr = np.load(dataset_tr, allow_pickle = True, encoding = 'latin1')
    dataset_te = 'data/nytimes/test.txt.npy'
    data_te = np.load(dataset_te, allow_pickle = True, encoding = 'latin1')
    vocab = 'data/nytimes/vocab.pkl'
    vocab = pickle.load(open(vocab,'rb'))
    vocab_size = len(vocab)
#--------------print the data dimentions--------------------------
print('Data Loaded')
print('Dim Training Data',data_tr.shape)
print('Dim Test Data',data_te.shape)

# Co-occurrence frequency
data_tr_freq = (data_tr > 0.5).astype(int)
data_tr_doc_freq = np.sum(data_tr_freq, 0)
with tf.Session() as sess:
    data_tr_cofreq = tf.linalg.matmul(data_tr_freq, data_tr_freq, transpose_a=True).eval()
'''-----------------------------'''

'''--------------Global Params---------------'''
n_samples_tr = data_tr.shape[0]
n_samples_te = data_te.shape[0]
docs_tr = data_tr
docs_te = data_te
batch_size=200
learning_rate=0.002
network_architecture = \
    dict(n_hidden_recog_1=100, # 1st layer encoder neurons
         n_hidden_recog_2=100, # 2nd layer encoder neurons
         n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
         n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
         n_z=50)  # dimensionality of latent space

'''-----------------------------'''

'''--------------Netowrk Architecture and settings---------------'''

def make_network(layer1=100,layer2=100,num_topics=50,bs=200,eta=0.002):
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
             n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space
    batch_size=bs
    learning_rate=eta
    return network_architecture,batch_size,learning_rate



'''--------------Methods--------------'''
def create_minibatch(data, batch_size=200):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]


def train(network_architecture, minibatches, type='prodlda',learning_rate=0.001,
          batch_size=200, training_epochs=100, display_step=5):
    tf.reset_default_graph()
    vae=''
    if type=='prodlda':
        vae = prodlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size)
    elif type=='nvlda':
        vae = nvlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size)
    emb=0
    
    num_topics = network_architecture['n_z']
    # Intermediate topic-document distributions
    theta = np.zeros((n_samples_tr, num_topics, training_epochs))
    # Intermediate word-topic distributions
    phi = np.zeros((num_topics, vocab_size, training_epochs))
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples_tr / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = next(minibatches)
            # Fit training using batch data
            cost,emb = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples_tr * batch_size

            if np.isnan(avg_cost):
                print(epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()
                
        for i, doc in enumerate(docs_tr):
            theta[i,:,epoch] = vae.topic_prop(doc)
        theta[:,:,epoch] = tf.nn.softmax(theta[:,:,epoch],1).eval()
        phi[:,:,epoch] = tf.nn.softmax(emb,1).eval()

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
    return vae,emb,theta,phi

def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        print(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print('---------------End of Topics------------------')

def calcPerp(model):
    cost=[]
    for doc in docs_te:
        doc = doc.astype('float32')
        n_d = np.sum(doc)
        if n_d == 0:
            continue
        c=model.test(doc)
        cost.append(c/n_d)
    print('The approximated perplexity is: ',(np.exp(np.mean(np.array(cost)))))
   
# Semantic coherence   
# https://mimno.infosci.cornell.edu/papers/mimno-semantic-emnlp.pdf
# Epsilion instead of +1: https://www.aclweb.org/anthology/D12-1087.pdf
def calcCoherence(beta, n_top_words=10):
    print("Calculating coherence")
    
    num_topics = len(beta)
    coherence = np.zeros(num_topics)
    coherence_eps = np.zeros(num_topics)
    
    for k in range(num_topics):
        words = beta[k].argsort()[:-n_top_words - 1:-1]
        for i, word1 in enumerate(words[1:]):
            for j, word2 in enumerate(words[0:i-1]):
                coherence[k] += np.log(data_tr_cofreq[word1,word2]+1) - np.log(data_tr_doc_freq[word2])
                coherence_eps[k] += np.log(data_tr_cofreq[word1,word2]+1e-12) - np.log(data_tr_doc_freq[word2])
    # print("Coherence:", coherence)
    return coherence, coherence_eps
    
# Pointwise mutual information
# https://www.aclweb.org/anthology/E14-1056.pdf
# Epsilion: https://www.aclweb.org/anthology/D12-1087.pdf
def calcPMI(beta, n_top_words=10):
    print("Calculating PMI")
    data_tr_prob = np.sum(data_tr, 0) / np.sum(data_tr)
    data_tr_joint_prob = data_tr_cofreq / np.sum(data_tr_cofreq)
    num_topics = len(beta)
    pmi = np.zeros(num_topics)
    
    for k in range(num_topics):
        words = beta[k].argsort()[:-n_top_words - 1:-1]
        for i, word1 in enumerate(words[:-1]):
            for word2 in words[i+1:]:
                pmi[k] += np.log(data_tr_joint_prob[word1,word2]+1e-12) - np.log(data_tr_prob[word1]) - np.log(data_tr_prob[word2])
    # print("PMI:", pmi)
    return pmi
    
# Normalized pointwise mutual information
# https://www.aclweb.org/anthology/E14-1056.pdf
def calcNPMI(beta, n_top_words=10):
    print("Calculating NPMI")
    
    data_tr_prob = np.sum(data_tr, 0) / np.sum(data_tr)
    data_tr_joint_prob = data_tr_cofreq / np.sum(data_tr_cofreq)
    num_topics = len(beta)
    npmi = np.zeros(num_topics)
    
    for k in range(num_topics):
        words = beta[k].argsort()[:-n_top_words - 1:-1]
        for i, word1 in enumerate(words[:-1]):
            for word2 in words[i+1:]:
                p = 0 if data_tr_joint_prob[word1,word2] == 0 else (np.log(data_tr_prob[word1]) + np.log(data_tr_prob[word2])) / np.log(data_tr_joint_prob[word1,word2])
                npmi[k] += p - 1
    # Re-scale to [0,1]
    npmi = npmi / (2*comb(n_top_words,2)) + 0.5
    # print("NPMI:", npmi)
    return npmi
    
# Topic stability for optimization
# http://deanxing.net/assets/pdf/aaai18.pdf
def calcStability(phi):
    print("Calculating stability")
    
    # Mean of word-topic distribution over epochs
    means = np.mean(phi,2)
    num_topics = phi.shape[0]
    training_epochs = phi.shape[2]
    cosine_stability = np.zeros(num_topics)
    symkl_stability = np.zeros(num_topics)
    euclid_stability = np.zeros(num_topics)
    
    for k in range(num_topics):
        cosine_stability[k] = np.mean(cosine_similarity(np.transpose(phi[k,:,:]), np.reshape(means[k,:],(1,-1))))
        for i in range(training_epochs):
            symkl_stability[k] += entropy(phi[k,:,i],means[k,:]) + entropy(means[k,:],phi[k,:,i])
            euclid_stability[k] += np.linalg.norm(phi[k,:,i] - means[k,:])
    symkl_stability /= (2 * training_epochs)
    euclid_stability /= training_epochs
    # print("Topic cosine stability:", cosine_stability)
    # print("Topic KL stability:", symkl_stability)
    # print("Topic Euclidean stability:", euclid_stability)
    return cosine_stability, symkl_stability, euclid_stability
    
def calcJaccardStability(phi, n_top_words=10):
    print("Calculating Jaccard stability")
    
    # Mean of word-topic distribution over epochs
    means = np.mean(phi,2)
    num_topics = phi.shape[0]
    training_epochs = phi.shape[2]
    jaccard_stability = np.zeros(num_topics)
    
    for k in range(num_topics):
        top_words = means[k].argsort()[:-n_top_words - 1:-1]
        for i in range(training_epochs):
            # Jaccard: in terms of document frequencies
            top_words_ep = phi[k,:,i].argsort()[:-n_top_words - 1:-1]
            for word1 in top_words:
                for word2 in top_words_ep:
                    if word1 == word2 or (data_tr_doc_freq[word1] == 0 and data_tr_doc_freq[word2] == 0):
                        jaccard_stability[k] += 1
                    else:
                        jaccard_stability[k] += data_tr_cofreq[word1,word2] / (data_tr_doc_freq[word1] + data_tr_doc_freq[word2] - data_tr_cofreq[word1,word2])
    jaccard_stability /= (training_epochs*n_top_words**2)
    # print("Topic Jaccard stability:", jaccard_stability)
    return jaccard_stability
    
def calcVariability(theta):
    print("Calculating variability")
    
    means = np.mean(theta,2)
    sds = np.std(theta,2)
    cvs = sds / means
    variability = np.std(cvs,0)
    # print("Topic variability:", variability)
    return variability
    
def calcPosteriorVariability(model, num_samples=1000):
    print("Calculating posterior variability")
    random.seed(503)
    docs_tr_samples = random.sample(range(n_samples_tr), num_samples)
    pcvs = np.zeros((num_samples, network_architecture['n_z']))
    for i in range(num_samples):
        if i % 100 == 0:
            print("Documents processed:", i)
        doc = docs_tr[docs_tr_samples[i]]
        samples = tf.nn.softmax(model.topic_prop_samples(doc),1).eval()
        pcvs[i,:] = np.std(samples,0) / np.mean(samples,0)
    print("Total documents processed:", num_samples)
    pvariability = np.std(pcvs,0)
    # print("Topic posterior variability over", num_samples, "samples:", pvariability)
    return pvariability
    
def printDocument(doc, feature_names):
    print(" ".join([feature_names[i] for i in doc]))

def main(argv):
    m = ''
    f = ''
    s = ''
    t = ''
    b = ''
    r = ''
    e = ''
    try:
      opts, args = getopt.getopt(argv,"hpnm:f:s:t:b:r:,e:",["default=","model=","layer1=","layer2=","num_topics=","batch_size=","learning_rate=","training_epochs"])
    except getopt.GetoptError:
        print('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1] -e <training_epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1]> -e <training_epochs>')
            sys.exit()
        elif opt == '-p':
            print('Running with the Default settings for prodLDA...')
            print('CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 100')
            m='prodlda'
            f=100
            s=100
            t=50
            b=200
            r=0.002
            e=100
        elif opt == '-n':
            print('Running with the Default settings for NVLDA...')
            print('CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300')
            m='nvlda'
            f=100
            s=100
            t=50
            b=200
            r=0.01
            e=300
        elif opt == "-m":
            m=arg
        elif opt == "-f":
            f=int(arg)
        elif opt == "-s":
            s=int(arg)
        elif opt == "-t":
            t=int(arg)
        elif opt == "-b":
            b=int(arg)
        elif opt == "-r":
            r=float(arg)
        elif opt == "-e":
            e=int(arg)

    minibatches = create_minibatch(docs_tr.astype('float32'), b)
    network_architecture,batch_size,learning_rate=make_network(f,s,t,b,r)
    print(network_architecture)
    print(opts)
    vae,emb,theta,phi = train(network_architecture, minibatches,m, training_epochs=e,batch_size=batch_size,learning_rate=learning_rate)
    feature_names = next(zip(*sorted(vocab.items(), key=lambda x: x[1])))
    print_top_words(emb, feature_names)
    calcPerp(vae)
    coh, cohep = calcCoherence(emb)
    pmi = calcPMI(emb)
    npmi = calcNPMI(emb)
    cos, kl, euc = calcStability(phi)
    jac = calcJaccardStability(phi)
    vari = calcVariability(theta)
    pvari = calcPosteriorVariability(vae, math.floor(n_samples_tr*0.1))
    n_top_words = 10
    print("Writing results to CSV")
    with open('results/' + corpus + '_topics_' + m + '.csv', 'w', newline='') as csvf:
        csvw = csv.writer(csvf)
        csvw.writerow(["ID", "Topic", "Coherence", "Coherence+eps", "PMI", "NPMI", "Cosine", "KL", "Euclidean", "Jaccard", "Variability", "Post Variability"])
        for k in range(network_architecture['n_z']):
            csvw.writerow([k+1, " ".join([feature_names[i] for i in emb[k].argsort()[:-n_top_words - 1:-1]]), coh[k], cohep[k], pmi[k], npmi[k], cos[k], kl[k], euc[k], jac[k], vari[k], pvari[k]])

if __name__ == "__main__":
   main(sys.argv[1:])
