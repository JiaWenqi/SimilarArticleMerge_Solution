from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import jieba
import xlsxwriter
import pandas as pd
import numpy as np
def splitFilterWords(content,stopwords):
    seg_list = jieba.cut(content, cut_all=False)
    seg_result = []
    for w in seg_list:
        if w not in stopwords:
            seg_result.append(w)
    content_result = ' '.join(seg_result).strip()
    return content_result
def tfidfTransform(corpus):
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 1))
    corpus_tfidf = tfidf_vec.fit_transform(corpus)
    return corpus_tfidf
def svdTransform(corpus_tfidf):
    n_comp = 40
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    corpus_svd = svd_obj.fit(corpus_tfidf)
    return corpus_svd
if __name__ == '__main__':
    filename = 'ceshi_zh1.txt'
    corpus = [];lines = []
    stopwords = [line.rstrip() for line in open('chinese_stopword.txt', 'r',encoding='utf-8')]
    with open(filename,'r',encoding='utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()  # 整行读取数据
            if not line:
                break
                pass
            lines.append(line)
            corpus.append(splitFilterWords(line,stopwords))
    print(corpus)
    corpus_tfidf = tfidfTransform(corpus)
    corpus_svd = svdTransform(corpus_tfidf)
    print(corpus_svd)
    y_pred = DBSCAN(eps=0.1, min_samples=2).fit_predict(corpus_tfidf)
    #y_pred = DBSCAN(eps=0.1, min_samples=2).fit_predict(corpus_svd)
    print (y_pred)
    y_pred = y_pred.reshape((-1,1))
    df = pd.DataFrame(y_pred,columns = ['label'])
    df['content'] = 'ColumnD'
    for i in range(len(df)):
        df.iloc[i,1] = lines[i]
    print(df)
    df.to_csv('result.csv')


