from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import time
from torchtext.datasets import AG_NEWS
import os
from collections import defaultdict
import numpy as np
import gzip
from tqdm import tqdm
import operator

output_dir = './runtime'
data_dir = './dataset'
dataset_pair = AG_NEWS(root=data_dir)
num_train = 10000
num_test = 1000
num_classes = 4

train_idx_fn = os.path.join(output_dir, 'train_indicies_{}per_class'.format(num_train))
test_idx_fn = os.path.join(output_dir, 'test_indicies_{}per_class'.format(num_test))

# 从数据集ds中的每个类别随机加载n个数据到output_fn文件地址
def pick_samples(ds, n, output_fn, index_only=False):
    label2text = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []
    for i, (label, text) in enumerate(ds):
        label2text[label].append(text)
        label2idx[label].append(i)
    for cl in label2text:
        class2count[cl] = len(label2text[cl])
    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n, replace=False)
        select_text = np.array(label2text[c])[select_idx]
        select_text_idx = np.array(label2idx[c])[select_idx]
        recorded_idx+=list(select_text_idx)
        result+=list(select_text)
        labels+=[c]*n
    if output_fn is not None:
        np.save(output_fn, np.array(recorded_idx))
    if index_only:
        return np.array(recorded_idx), labels
    return result, labels

# 加载全部数据
def load_all(ds):
    result = []
    labels = []
    for i, (label, line) in enumerate(ds):
        result.append(line)
        labels.append(label)
    return result, labels

train_data, train_labels = load_all(dataset_pair[0])
test_data, test_labels = load_all(dataset_pair[1])

def NCD(c1, c2, c12):
    dis = (c12-min(c1,c2))/max(c1, c2)
    return dis

def cal_dist(data, train_data):
    dis_matrix = []
    for i, t1 in tqdm(enumerate(data)):
        print('{}/{}'.format(i,len(data)))
        distance4i = []
        t1_compressed = len(gzip.compress(t1.encode('utf-8')))
        for j, t2 in tqdm(enumerate(train_data)):
            t2_compressed = len(gzip.compress(t2.encode('utf-8')))
            t1t2_compressed = len(gzip.compress((t1+' '+t2).encode('utf-8')))
            distance = NCD(t1_compressed, t2_compressed, t1t2_compressed)
            distance4i.append(distance)
        dis_matrix.append(distance4i)
    return dis_matrix

dis = cal_dist(test_data, train_data)
np.save(output_dir+'/dis.npy', np.array(dis))

def inference(dis, test_labels, train_labels):
    compare_label = train_labels
    start = 0
    end = num_classes
    correct = []
    pred = []
    for i in range(len(dis)):
        sorted_idx = np.argsort(np.array(dis[i]))
        pred_labels = defaultdict(int)
        for j in range(start, end):
            pred_l = compare_label[sorted_idx[j]]
            pred_labels[pred_l] += 1
        sorted_pred_lab = sorted(pred_labels.items(), key=operator.itemgetter(1), reverse=True)
        most_count = sorted_pred_lab[0][1]
        if_right = 0
        most_label = sorted_pred_lab[0][0]
        most_voted_labels = []
        for pair in sorted_pred_lab:
            if pair[1] < most_count:
                break
            if pair[0] == test_labels[i]:
                if_right = 1
                most_label = pair[0]
            else:
                most_voted_labels.append(pair[0])
        pred.append(most_label)
        correct.append(if_right)
    print("Accuracy is {}".format(sum(correct)/len(correct)))
    return pred, correct

pred, correct = inference(dis,test_labels, train_labels)