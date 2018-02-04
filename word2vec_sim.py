# -*- coding: UTF-8 -*-
from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
import jieba.analyse

######生成每篇文章的关键词
file = open('article1.txt','r',encoding='utf-8')
lines = file.readlines()
tags_list = []
for line in lines:
    tags = jieba.analyse.extract_tags(lines[0],10)
    print(tags)
    tags_list.append(tags)
######将关键词转换成word2vec向量
n_dim = 300
imdb_w2v = Word2Vec.load('./weibo_model')
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
from sklearn.preprocessing import scale
tags_vecs = np.concatenate([buildWordVector(z, n_dim) for z in tags_list])
tags_vecs = scale(tags_vecs)
print(11)