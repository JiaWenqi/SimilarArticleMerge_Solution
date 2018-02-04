# -coding:utf-8-
import gensim
import os
import collections
import numpy as np
import scipy.spatial.distance
from scipy import linalg, mat, dot
from progressbar import ProgressBar
dim = 50
min_count=0
min_word_count=0
max_word_count=1000
name='default'
def load_data(file_path):
    word_dict = collections.defaultdict(int)
    word_index = dict()
    data = []
    sentences_count = 0
    senlen = collections.defaultdict(int)
    with open(file_path,encoding='utf-8') as f:
        for line in f:
            words = line.split()
            if len(words) < min_word_count or len(words) > max_word_count:
                continue
            data.append(line)
            senlen[len(words)] = senlen[len(words)] + 1
            for word in words:
                if not word in word_dict:
                    word_index[word] = len(word_dict) - 1
                word_dict[word] = word_dict[word] + 1
    used_word_dict = collections.defaultdict(int)
    used_word_index = {}

    #senlen = sorted(senlen.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    # print('senlen: {}'.format(senlen))
    for w, c in word_dict.items():
        if c <= min_count:
            continue
        used_word_dict[w] = c
        used_word_index[w] = len(used_word_dict) - 1
        word_dict = used_word_dict
        word_index = used_word_index
    print('train data sentences size: {}'.format(len(data)))
    print('train data word size: {}'.format(len(word_index)))
    return data, word_dict, word_index
def load_w2v(w2v_path,word_dict):
    embeddings_index = {}
    f = open(w2v_path,encoding='utf-8')
    used_word_dict = collections.defaultdict(int)
    used_word_index = {}
    not_exist = []
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if word in word_dict:
            used_word_dict[word] = word_dict[word]
            used_word_index[word] = len(used_word_dict) - 1
            embeddings_index[word] = coefs
        else:
            not_exist.append(word)
    word_dict = used_word_dict
    word_index = used_word_index
    print(word_index)
    print('embeddings_index done!')
    f.close()
    embeddings = np.zeros((len(word_dict), dim))
    for word, index in word_index.items():
        embeddings[index] = embeddings_index[word]
    print('embeddings done!')
    print('not exist word size: {}'.format(len(not_exist)))
    #print '\tnot exist word: {}'.format(not_exist)
    return word_dict,word_index,embeddings
def cos_cdist(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector.
    """
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
def gen_sim_mat(word_index,embeddings):
    progress = ProgressBar()
    n = embeddings.shape[0]
    sim_mat = np.zeros((n, n))
    for i in progress(range(n)):
        a = embeddings[i]
        mat = 1.0 - cos_cdist(embeddings, a)
        mat[np.isnan(mat)] = 0.
        mat[i] = 0.
        sim_mat[i, ...] = mat[...]
    print('embedding sim mat: {}'.format(sim_mat.shape))
    np.savez('sim_mat_{}.npz'.format(name), sim_mat=sim_mat)
    word_max_sim = np.zeros((len(word_index),))
    for i in range(len(word_index)):
        word_max_sim[i] = max(sim_mat[i, ...])
    np.savez('word_max_sim_{}.npz'.format(name), word_max_sim=word_max_sim)
    return word_index,embeddings, sim_mat
def gen_sen2vec_q(sen,word_index,word_dict):
    vec = np.zeros((1, len(word_dict)))
    sen_vec = np.zeros((1, len(word_dict)))
    words = sen.split()
    for word in words:
        if word in word_dict:
            sen_vec[0, word_index[word]] = 1.0
    p = dot(sen_vec, sim_mat)
    vec[0, ...] = np.max(p, axis=0)[...]
    return vec
def gen_train_sens_vec(data,word_index,word_dict):
    train_sens_vec = np.zeros((len(data), len(word_index)))
    progress = ProgressBar()
    for i in progress(range(len(data))):
        train_sens_vec[i,...] = gen_sen2vec_q(data[i],word_index,word_dict)[...]
    np.savez('train_sens_vec_{}.npz'.format(name), train_sens_vec=train_sens_vec)
    return train_sens_vec
if __name__ == '__main__':
    w2v_path = 'zhwiki_2017_03.sg_50d.word2vec'
    file_path = ''
    data,word_dict, word_index = load_data('ceshi_zh2.txt')
    word_dict, word_index, embeddings = load_w2v(w2v_path,word_dict)
    word_index, embeddings, sim_mat = gen_sim_mat(word_index,embeddings)
    sentence_vec = gen_train_sens_vec(data,word_index,word_dict)
