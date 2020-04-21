#!/usr/bin/python

import getopt
import math
import numpy as np
import pickle
import sys
import tensorflow as tf
from models import prodlda, nvlda
from metrics import calcPerp, calcCoherence, calcPMI, calcNPMI, calcStability, calcJaccardStability, calcVariability, calcPosteriorVariability
from utils import DataReader, get_top_words, write_to_csv


# Initialize training and test sets and return as a single object
def initialize_data(corpus, verbose=True):
    # Read preprocessed training and test sets
    if corpus == "20ng":
        dataset_tr = "data/20news_clean/train.txt.npy"
        dataset_te = "data/20news_clean/test.txt.npy"
        vocab = "data/20news_clean/vocab_unix.pkl"
    elif corpus == "nyt":
        dataset_tr = "data/nytimes/train.txt.npy"
        dataset_te = "data/nytimes/test.txt.npy"
        vocab = "data/nytimes/vocab.pkl"
    else:
        print("Corpus not recognized:", corpus)
        sys.exit(2)
    
    vocab = pickle.load(open(vocab, "rb"))
    data_tr = np.load(dataset_tr, allow_pickle=True, encoding="latin1")
    data_te = np.load(dataset_te, allow_pickle=True, encoding="latin1")
    
    # 20NG dataset not in bag-of-words representation
    if corpus == "20ng":
        data_tr = np.array([np.bincount(doc.astype("int"), minlength=len(vocab)) for doc in data_tr if np.sum(doc)!=0])
        data_te = np.array([np.bincount(doc.astype("int"), minlength=len(vocab)) for doc in data_te if np.sum(doc)!=0])
    
    if verbose:
        print("\nData loaded")
        print("\tTraining data dimensions:", data_tr.shape)
        print("\tTest data dimensions", data_te.shape)
        
    data = DataReader(data_tr, data_te, vocab)
    return data


# Bundle network architecture settings
def make_network(vocab_size, num_units_layer1=100, num_units_layer2=100, num_topics=50):
    network_architecture = \
        dict(n_hidden_recog_1 = num_units_layer1, # 1st layer encoder neurons
             n_hidden_recog_2 = num_units_layer2, # 2nd layer encoder neurons
             n_hidden_gener_1 = vocab_size,       # 1st layer decoder neurons
             n_input = vocab_size,                # Vocabulary size
             n_z = num_topics)                    # Dimension of latent space
    return network_architecture


# Create random data batches
def create_minibatch(data, batch_size=200):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size "minibatch_size" at each iteration
        ixs = rng.randint(data.n_tr, size=batch_size)
        yield data.tr[ixs]


# Create and train the network
def train(network_architecture, minibatches, data, type="prodlda",
          learning_rate=0.001, batch_size=200, num_epochs=100, display_step=5, verbose=True):
    print("\nStarting training")
    tf.compat.v1.reset_default_graph()
    if type == "prodlda":
        vae = prodlda.VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
    elif type == "nvlda":
        vae = nvlda.VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
    else:
        print("Model not recognized:", type)
        sys.exit(2)
    
    num_topics = network_architecture["n_z"]
    theta = np.zeros((data.n_tr, num_topics, num_epochs))     # Intermediate topic-document distributions
    phi = np.zeros((num_topics, data.vocab_size, num_epochs)) # Intermediate word-topic distributions
    
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0.
        total_batch = int(data.n_tr / batch_size)
        
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = next(minibatches)
            
            # Fit training using batch data
            cost, emb = vae.partial_fit(batch_xs)
            
            # Compute average loss
            avg_cost += cost / data.n_tr * batch_size

            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs,1).astype(np.int), batch_xs.shape)
                print("Training stopped due to NaN. Please check the learning_rate settings")
                sys.exit()
                
        # Get intermediate distributions
        for i, doc in enumerate(data.tr):
            theta[i,:,epoch] = vae.topic_prop(doc)
        theta[:,:,epoch] = tf.nn.softmax(theta[:,:,epoch],1).eval()
        phi[:,:,epoch] = tf.nn.softmax(emb,1).eval()

        # Display logs per epoch step
        if verbose and epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch+1), "\tCost =", "{:.9f}".format(avg_cost))
            
    return vae, emb, theta, phi


def main(argv):
    # Collect run settings
    try:
        opts, args = getopt.getopt(argv, "hpnm:c:f:s:t:b:r:e:", ["default=","model=","corpus=","num_units_layer1=","num_units_layer2=","num_topics=","batch_size=","learning_rate=","num_epochs"])
    except getopt.GetoptError:
        print("py run.py -m <model> -c <corpus> -f <#units layer1> -s <#units layer2> -t <#topics> -b <batch_size> -r <learning_rate [0,1] -e <num_epochs>")
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == "-h":
            print("py run.py -m <model> -c <corpus> -f <#units layer1> -s <#units layer2> -t <#topics> -b <batch_size> -r <learning_rate [0,1]> -e <num_epochs>")
            sys.exit()
        elif opt == "-p":
            print("Running with default settings for ProdLDA ...")
            print("py run.py -m prodlda -c 20ng -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 100")
            model = "prodlda"
            corpus = "20ng"
            num_units_layer1 = 100
            num_units_layer2 = 100
            num_topics = 50
            batch_size = 200
            learning_rate = 0.002
            num_epochs = 100
        elif opt == "-n":
            print("Running with default settings for NVLDA ...")
            print("py run.py -m nvlda -c 20ng -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300")
            model = "nvlda"
            corpus = "20ng"
            num_units_layer1 = 100
            num_units_layer2 = 100
            num_topics = 50
            batch_size = 200
            learning_rate = 0.01
            num_epochs = 300
        elif opt == "-m":
            model = arg
        elif opt == "-c":
            corpus = arg
        elif opt == "-f":
            num_units_layer1 = int(arg)
        elif opt == "-s":
            num_units_layer2 = int(arg)
        elif opt == "-t":
            num_topics = int(arg)
        elif opt == "-b":
            batch_size = int(arg)
        elif opt == "-r":
            learning_rate = float(arg)
        elif opt == "-e":
            num_epochs = int(arg)
            
    print("\nRunning with inputs")
    print("\tModel:", model)
    print("\tCorpus:", corpus)
    print("\tNumber of units (layer 1):", num_units_layer1)
    print("\tNumber of units (layer 2):", num_units_layer2)
    print("\tNumber of topics:", num_topics)
    print("\tBatch size:", batch_size)
    print("\tLearning rate:", learning_rate)
    print("\tNumber of epochs:", num_epochs)

    # Initialize data and train network
    data = initialize_data(corpus)
    minibatches = create_minibatch(data, batch_size)
    network_architecture = make_network(data.vocab_size, num_units_layer1, num_units_layer2, num_topics)
    vae, emb, theta, phi = train(network_architecture, minibatches, data, model, learning_rate, batch_size, num_epochs)
    
    # Show results
    top_words = get_top_words(emb, data.vocab_by_ind)
    calcPerp(vae, data)
    
    # Calculate metrics
    coh, cohep = calcCoherence(emb, data)
    pmi = calcPMI(emb, data)
    npmi = calcNPMI(emb, data)
    cos, kl, euc = calcStability(phi)
    jac = calcJaccardStability(phi, data)
    vari = calcVariability(theta)
    pvari = calcPosteriorVariability(vae, num_topics, data, math.floor(data.n_tr*0.1))
    
    # Write results to CSV
    # Specify columns to include, their order, and their column names
    col_id = "ID"
    col_topic = "Topic"
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
    
    csv_header = [col_id, col_topic, col_coh, col_cohep, col_pmi, col_npmi, col_cos, col_kl, col_euc, col_jac, col_vari, col_pvari]
    results = {col_id: [t+1 for t in range(num_topics)], col_topic: top_words,
        col_coh: coh,
        col_cohep: cohep,
        col_pmi: pmi,
        col_npmi: npmi,
        col_cos: cos,
        col_kl: kl,
        col_euc: euc,
        col_jac: jac,
        col_vari: vari,
        col_pvari: pvari
        }
    write_to_csv("results/"+corpus+"_topics_"+model+".csv", csv_header, results)


if __name__ == "__main__":
   main(sys.argv[1:])
