import math
import numpy as np
import random
from scipy.special import comb
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf


# Perplexity
def calcPerp(model, data, verbose=True):
    cost = []
    for doc in data.te:
        n_d = np.sum(doc)
        
        if n_d == 0: # Empty doc
            continue
            
        c = model.test(doc)
        cost.append(c / n_d)
    perp = np.exp(np.mean(np.array(cost)))
    
    if verbose:
        print("The approximated perplexity is:", perp)
        
    return perp
   
   
# Semantic coherence: https://mimno.infosci.cornell.edu/papers/mimno-semantic-emnlp.pdf
# With epsilon instead of +1: https://www.aclweb.org/anthology/D12-1087.pdf
def calcCoherence(beta, data, n_top_words=10, verbose=False):
    print("Calculating coherence")
    num_topics = len(beta)
    
    coherence = np.zeros(num_topics)
    coherence_eps = np.zeros(num_topics)
    for k in range(num_topics):
        words = beta[k].argsort()[:-n_top_words-1:-1]
        for i, word1 in enumerate(words[1:]):
            for j, word2 in enumerate(words[0:i-1]):
                coherence[k] += np.log(data.tr_doc_cofreq[word1,word2]+1) - np.log(data.tr_doc_freq[word2])
                coherence_eps[k] += np.log(data.tr_doc_cofreq[word1,word2]+1e-12) - np.log(data.tr_doc_freq[word2])
    
    if verbose:
        print("Coherence:", coherence)
        print("Coherence+eps:", coherence_eps)
        
    return coherence, coherence_eps
    
    
# Pointwise mutual information: https://www.aclweb.org/anthology/E14-1056.pdf
# With epsilon: https://www.aclweb.org/anthology/D12-1087.pdf
def calcPMI(beta, data, n_top_words=10, verbose=False):
    print("Calculating PMI")
    data_tr_prob = np.sum(data.tr, 0) / np.sum(data.tr)
    data_tr_joint_prob = data.tr_doc_cofreq / np.sum(data.tr_doc_cofreq)
    num_topics = len(beta)
    
    pmi = np.zeros(num_topics)
    for k in range(num_topics):
        words = beta[k].argsort()[:-n_top_words - 1:-1]
        for i, word1 in enumerate(words[:-1]):
            for word2 in words[i+1:]:
                pmi[k] += np.log(data_tr_joint_prob[word1,word2]+1e-12) - np.log(data_tr_prob[word1]) - np.log(data_tr_prob[word2])
    if verbose:
        print("PMI:", pmi)
        
    return pmi
    
    
# Normalized pointwise mutual information: https://www.aclweb.org/anthology/E14-1056.pdf
def calcNPMI(beta, data, n_top_words=10, verbose=False):
    print("Calculating NPMI")
    data_tr_prob = np.sum(data.tr, 0) / np.sum(data.tr)
    data_tr_joint_prob = data.tr_doc_cofreq / np.sum(data.tr_doc_cofreq)
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
    
    if verbose:
        print("NPMI:", npmi)
        
    return npmi


# Topic stability for optimization based on http://deanxing.net/assets/pdf/aaai18.pdf
def calcStability(phi, verbose=False):
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
    
    if verbose:
        print("Topic cosine stability:", cosine_stability)
        print("Topic KL stability:", symkl_stability)
        print("Topic Euclidean stability:", euclid_stability)
        
    return cosine_stability, symkl_stability, euclid_stability
    
def calcJaccardStability(phi, data, n_top_words=10, verbose=False):
    print("Calculating Jaccard stability")
    
    # Mean of word-topic distribution over epochs
    means = np.mean(phi,2)
    num_topics = phi.shape[0]
    training_epochs = phi.shape[2]
    
    jaccard_stability = np.zeros(num_topics)
    for k in range(num_topics):
        top_words = means[k].argsort()[:-n_top_words-1:-1]
        for i in range(training_epochs):
            # Jaccard in terms of document frequencies
            top_words_ep = phi[k,:,i].argsort()[:-n_top_words-1:-1]
            for word1 in top_words:
                for word2 in top_words_ep:
                    if word1 == word2 or (data.tr_doc_freq[word1] == 0 and data.tr_doc_freq[word2] == 0):
                        jaccard_stability[k] += 1
                    else:
                        jaccard_stability[k] += data.tr_doc_cofreq[word1,word2] / (data.tr_doc_freq[word1] + data.tr_doc_freq[word2] - data.tr_doc_cofreq[word1,word2])
    jaccard_stability /= (training_epochs*n_top_words**2)
    
    if verbose:
        print("Topic Jaccard stability:", jaccard_stability)
        
    return jaccard_stability
    

# Topic epoch variability for optimization based on http://deanxing.net/assets/pdf/emnlp19.pdf
def calcVariability(theta, verbose=False):
    print("Calculating variability")
    means = np.mean(theta, 2)
    sds = np.std(theta, 2)
    cvs = sds / means
    variability = np.std(cvs, 0)
    
    if verbose:
        print("Topic variability:", variability)
        
    return variability
    
# Topic posterior variability for optimization based on http://deanxing.net/assets/pdf/emnlp19.pdf    
def calcPosteriorVariability(model, num_topics, data, num_samples=1000, verbose=False):
    print("Calculating posterior variability")
    doc_samples = random.sample(range(data.n_tr), num_samples)
    
    pcvs = np.zeros((num_samples, num_topics))
    for i in range(num_samples):
        if i % 100 == 0:
            print("\tDocuments processed:", i)
            
        doc = data.tr[doc_samples[i]]
        samples = tf.nn.softmax(model.topic_prop_samples(doc),1).eval()
        pcvs[i,:] = np.std(samples,0) / np.mean(samples,0)
    pvariability = np.std(pcvs,0)
    
    print("\tTotal documents processed:", num_samples)
    if verbose:
        print("Topic posterior variability over", num_samples, "samples:", pvariability)
        
    return pvariability