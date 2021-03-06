import codecs

import numpy as np
import csv
import pickle
import logging
logger = logging.getLogger(__name__)
import os
import glob
import shutil
import random
from configs import *
import tensorflow as tf
from tensorflow.python.framework import ops

def load_data(file,vectors=None):
    """只读取第1列和最后1列"""
    model_config = ModelConfig()
    embed = ""
    if vectors:
        with open(vectors, 'rb') as fp:
            vectors = pickle.load(fp)
    with codecs.open(file, encoding='utf-8') as f:
        texts = []
        text = []
        labels = []
        label = []
        embed = []
        i=0
        for line in f:
            if "-DOCSTART-" not in line:
                line = line.strip()
                if len(line) == 0:  # 空白行，表示一句话已经结束
                    if len(label)+len(text)>1:
                        texts.append(text)
                        labels.append([lab for lab in label])
                        if vectors:
                            embed.append(vectors[i])
                        i+=1
                    text = []
                    label = []

                else:
                    line = line.split()
                    text.append(line[0])
                    label.append(line[-1])
            else:
                i+=1


    return {'texts': texts, 'labels': labels, 'embed': embed}


def get_embeding(texts_ids, labels_, embed=None):

    labels = labels_.copy()

    embeddings = [embed[id] for id in texts_ids]
    return embeddings, labels

def save_data(texts, labels, file, word_sep=' ', line_sep='\r\n', line_break='\r\n'):
    """
    :param texts:
    :param labels:
    :param file:
    :param word_sep: 字(词)与其标签之间的分隔符；
    :param line_sep: 在model为'ernie'时，对应文本和标签的分隔符；
    :param line_break: 不同文本间的分隔符；
    """
    assert len(texts) == len(labels)
    save_list = []
    for text, label in zip(texts, labels):
        for t, l in zip(text, label):
            save_list.append(word_sep.join([t, l]) + line_sep)
        save_list.append(line_break)
    with codecs.open(file, 'w', encoding='utf-8') as f:
        for l in save_list:
            f.write(l)


def load_all_texts(files):
    """
    将多个文件中的文本合并在一起
    """
    all_texts = []
    for file in files:
        data = load_data(file)
        all_texts.extend(data['texts'])
    return all_texts


def load_embedding(embedding_file):
    """
    加载词向量文件，并返回一个dict，其中key是词，val则是对应的向量
    """
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embedding_file, encoding='utf-8')))
    return embeddings_index



def choose_ids_by_price(idxs, budget, texts):
    selected_ids = []
    price = 0
    for id in idxs:
        cost = len(texts[id])
        if price + cost > budget:
            pass
        else:
            selected_ids.append(id)
            price += cost
    return selected_ids, budget - price, price

def init_distribution(labels):
    all_labels = []
    for label_n in labels:
        for label in label_n:
            all_labels.append(label)
    unic_labels = set(all_labels)
    distribution = {}
    for label in unic_labels:
        distribution.update({label: all_labels.count(label)})
    return sorted(distribution.items())

def compute_price(data):
    price = 0
    for sent in data:
        price+=len(sent)
    return price

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    ops.reset_default_graph()
    tf.set_random_seed(seed)

def find_new_number(directory):
    result = 0
    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in os.listdir(directory):
        try:
            num = int(filename[:2])
            result = num if num > result else result
        except Exception:
            pass

    if result+1<10:
        result = "0"+str(result+1)
    else:
        result = str(result+1)
    return result

