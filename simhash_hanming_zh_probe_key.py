#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from hashlib import md5
import sys
import jieba.analyse
from scipy.spatial import distance
import numpy as np
import time
import collections
import sys
class Token:

    def __init__(self, hash_list, weight):
        self.hash_list = hash_list
        self.weight = weight

def tokenize(doc):
    doc = filter(None, doc)
    return doc

def md5Hash(token):
    h = bin(int(md5(token.encode("utf-8")).hexdigest(), 16))
    return h[2:]

def hash_threshold(token_dict, fp_len):
    """
    Iterate through the token dictionary multiply the hash lists with the weights
    and apply the binary threshold
    """
    sum_hash = [0] * fp_len
    for _, token in token_dict.items():
        sum_hash = [ x + token.weight * y for x, y in zip(sum_hash, token.hash_list)]

    # apply binary threshold
    for i, ft in enumerate(sum_hash):
        if ft > 0:
            sum_hash[i] = 1
        else:
            sum_hash[i] = 0
    return sum_hash

def binconv(fp, fp_len):
    """
    Converts 0 to -1 in the tokens' hashes to facilitate
    merging of the tokens' hashes later on.
    input  : 1001...1
    output : [1,-1,-1, 1, ... , 1]
    """
    vec = [1] * fp_len
    for indx, b in enumerate(fp):
        if b == '0':
            vec[indx] = -1
    return vec


def calc_weights(terms, fp_len):
    """
    Calculates the weight of each one of the tokens. In this implementation
    these weights are equal to the term frequency within the document.

    :param tokens: A list of all the tokens (words) within the document
    :fp_len: The length of the Simhash values
    return dictionary "my_term": Token([-1,1,-1,1,..,-1], 5)
    """
    term_dict = {}
    for term in terms:
        # get weights
        if term not in term_dict:
            fp_hash = md5Hash(term).zfill(fp_len)
            fp_hash_list = binconv(fp_hash, fp_len)
            token = Token(fp_hash_list, 0)
            term_dict[term] = token
        term_dict[term].weight += 1
    return term_dict

def simhash(doc, fp_len=128):
    """
    :param doc: The document we want to generate the Simhash value
    :fp_len: The number of bits we want our hash to be consisted of.
                Since we are hashing each token of the document using
                md5 (which produces a 128 bit hash value) then this
                variable fp_len should be 128. Feel free to change
                this value if you use a different hash function for
                your tokens.
    :return The Simhash value of a document ex. '0000100001110'
    """
    tokens = tokenize(doc)
    token_dict = calc_weights(tokens, fp_len)
    fp_hash_list = hash_threshold(token_dict, fp_len)
    fp_hast_str =  ''.join(str(v) for v in fp_hash_list)
    return fp_hast_str
def distance_calculation(hash_code1,hash_code2):
    return 1 - distance.cdist(np.array([list(hash_code1)]), np.array([list(hash_code2)]), 'hamming')
if __name__ == '__main__':
    filename = 'hashcode.txt'
    probename = 'probe.txt'
    with open(filename, 'r',encoding='utf-8') as file_to_read:
        binary_hash = []
        while True:
            line = file_to_read.readline()  # 整行读取数据
            if not line:
                break
                pass
            binary_hash.append(line.strip())
    for k in range(7):
        binary_hash.extend(binary_hash)
####生成8个table，每个table以128位中16位为key,剩余bit位组成value
    bit_hashmap = [collections.defaultdict(list) for i in range(8)]
    for k in range(8):
        for i in range(len(binary_hash)):
            bit_hashmap[k][binary_hash[i][k*16:(k+1)*16]].append(binary_hash[i][:k*16] + binary_hash[i][(k+1)*16:])
    print(bit_hashmap.__sizeof__())
    for n in range(len(bit_hashmap)):
        print(sys.getsizeof(bit_hashmap[n]))
    start = time.time()
    with open(probename, 'r', encoding='utf-8') as file_to_read:
        probe_text = file_to_read.readline()
        probe_key = jieba.analyse.extract_tags(probe_text, 20)
        print(probe_key)
        probe_hash = simhash(probe_key)
    dis_list1 = []
    distance_dict = collections.defaultdict(list)
    ###将probe分割成8段
    subprobe = ['' for i in range(8)]
    remainprobe = ['' for i in range(8)]
    for i in range(8):
        subprobe[i] = probe_hash[(i*16):(i+1)*16]
    for i in range(8):
        remainprobe[i] = probe_hash[:i*16] + probe_hash[(i+1)*16:]
    for i in range(8):
        if subprobe[i] in bit_hashmap[i].keys():
            print(bit_hashmap[i][subprobe[i]])
            print(remainprobe[i])
            for j in range(len(bit_hashmap[i][subprobe[i]])):
                distance_dict['table'+str(i+1)].append(distance_calculation(bit_hashmap[i][subprobe[i]][j], remainprobe[i]))
    #if probe_hash[:16] in bit_hashmap_1.keys():
            #for j in range(len(exec('bit_hashmap_'+str(i))[probe_hash[:16]])):
                #dis = distance_calculation(bit_hashmap_1[probe_hash[:16]][i],probe_hash[16:])
                #dis_list1.append(dis)
    print(distance_dict)
    end = time.time()
    print(str(end - start))