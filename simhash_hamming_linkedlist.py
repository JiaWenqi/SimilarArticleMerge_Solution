# -*- coding: UTF-8 -*-
#导入pymysql的包
import pymysql
import pymysql.cursors
import gen_simhash
import jieba.analyse
from scipy.spatial import distance
import numpy as np
import collections
from hashlib import md5
import json
from time import strftime,gmtime
import time

def gen_probe_hash(probe_text):
    probe_key = jieba.analyse.extract_tags(probe_text, 20)
    #print(probe_key)
    #probe = ''.join(probe_key) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + str(time.time())
    probe = ''.join(probe_key) + str(time.time())
    probe_unique_id = md5(probe.encode("utf-8")).hexdigest()
    probe_simhash = gen_simhash.simhash(probe_key)

    return probe_unique_id,probe_simhash

def distance_calculation(hash_code1,hash_code2):
    return 1 - distance.cdist(np.array([list(hash_code1)]), np.array([list(hash_code2)]), 'hamming')

if __name__ == '__main__':

    start = time.time()

#****************连接mysql数据库*************
    connection=pymysql.connect(host='112.124.10.98',
                               user='root',
                               password='zhujsh1230',
                               db='simarticle',
                               port=3306,
                               charset='utf8')
    cursor = connection.cursor()
    library_remain_block_dict = [collections.OrderedDict() for i in range(8)]
    article_cluster_dict = {}
    sub_block_list = [[[] for _ in range(2**16)] for _ in range(8)]
    sql = 'select * from article_property'
    cursor.execute(sql)
    for row in cursor.fetchall():
        for i in range(8):
            library_remain_block_dict[i][row[0]] = row[1][:i*16] + row[1][(i+1)*16:]
            sub_block_list[i][int(row[1][(i*16):(i+1)*16], base=2)].append(row[0])
        article_cluster_dict[row[0]] = row[2]

# ****************读入数据流数据并生成对应的simhash*************
    #probename = '../probe.txt'
    #probename = '../probe_stream100.txt'
    probename = 'F:/word2vec所需语料/yiyuan16.txt'
    with open(probename, 'r', encoding='utf-8') as file_to_read:
        text = file_to_read.readlines()
    for i in range(len(text)):
        print('处理第%s篇文章： '%i)
        simhash_dictance = {}
        probe_unique_id, probe_simhash = gen_probe_hash(text[i])
        for i in range(8):
            probe_remain_block = probe_simhash[:i*16] + probe_simhash[(i+1)*16:]
            probe_sub_block = probe_simhash[(i*16):(i+1)*16]
            library_unique_id_list = sub_block_list[i][int(probe_sub_block, base=2)]
            for j in range(len(library_unique_id_list)):
                library_remain_block = library_remain_block_dict[i][library_unique_id_list[j]]

                dis = distance_calculation(probe_remain_block,library_remain_block)
                if dis > 0.93:
                    simhash_dictance[library_unique_id_list[j]] = dis
        if (simhash_dictance == {}):
            max_id = sorted(article_cluster_dict.items(), key=lambda d: d[1], reverse=True)[0][1]
            cur_id = max_id + 1
            insert_sql = "insert into article_property(article_unique_id,article_simhash,article_cluster_id)  \
                          values ('%s','%s',%d)" % (probe_unique_id, probe_simhash,cur_id)
            cursor.execute(insert_sql)
            # 加入内存
            article_cluster_dict[probe_unique_id] = cur_id  # 加入内存
            for i in range(8):
                library_remain_block_dict[i][probe_unique_id] = probe_simhash[:i*16] + probe_simhash[(i+1)*16:]
                sub_block_list[i][int(probe_simhash[(i*16):(i+1)*16], base=2)].append(probe_unique_id)
        else:
            max_article_id = sorted(simhash_dictance.items(), key=lambda d: d[1], reverse=True)[0][0]
            cur_id = article_cluster_dict[max_article_id]
            insert_sql = "insert into article_property(article_unique_id,article_simhash,article_cluster_id)  \
                          values ('%s','%s',%d)" % (probe_unique_id, probe_simhash,cur_id)
            cursor.execute(insert_sql)
            #加入内存
            article_cluster_dict[probe_unique_id] = cur_id  # 加入内存
            for i in range(8):
                library_remain_block_dict[i][probe_unique_id] = probe_simhash[:i*16] + probe_simhash[(i+1)*16:]
                sub_block_list[i][int(probe_simhash[(i*16):(i+1)*16], base=2)].append(probe_unique_id)
    connection.commit()
    connection.close()
    end = time.time()
    print(end-start)