import csv
import numpy as np
import tensorflow as tf


# Object for managing training and test data
class DataReader:
    def __init__(self, data_tr, data_te, vocab):
        # Training data
        self.tr = data_tr
        self.tr_freq = (data_tr > 0.5).astype(int)
        self.tr_doc_freq = np.sum(self.tr_freq, 0)
        self._tr_doc_cofreq = None
        self.n_tr = data_tr.shape[0]
        
        # Test data
        self.te = data_te
        
        # Vocabulary
        self.vocab = vocab
        self.vocab_by_ind = next(zip(*sorted(vocab.items(), key=lambda x: x[1])))
        self.vocab_size = len(vocab)
    
    # Training document cofrequency is computed only when needed for certain metrics
    @property
    def tr_doc_cofreq(self):
        if self._tr_doc_cofreq is None:
            with tf.compat.v1.Session() as sess:
                self._tr_doc_cofreq = tf.linalg.matmul(self.tr_freq, self.tr_freq, transpose_a=True).eval()
        return self._tr_doc_cofreq


# Get most probable words of each topic
def get_top_words(beta, feature_names, n_top_words=10, verbose=True):
    top_words = [" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]) for i in range(len(beta))]
    
    if verbose:
        print("\nMost probable", n_top_words, "words of each topic")
        for i in range(len(top_words)):
            print("Topic " + str(i+1) + ":", top_words[i])
            
    return top_words


# Write results to CSV
def write_to_csv(file_name, header, results):
    print("\nWriting results to", file_name)
    with open(file_name, "w", newline="") as csvf:
        csvw = csv.writer(csvf)
        csvw.writerow(header)
        for t in range(len(results[header[0]])):
            row = []
            for i in range(len(header)):
                col = header[i]
                row.append(results[col][t])
            csvw.writerow(row)