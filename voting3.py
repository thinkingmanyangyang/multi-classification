from sklearn.metrics import classification_report, f1_score
from data_utils import DataProcessor
from sklearn.model_selection import train_test_split
from assess import sentiment_f1_score
from constant import *
import pickle
import argparse
import numpy as np
import json
import os
from itertools import combinations
from collections import Counter

class Voting(object):
    def __init__(self, args):
        self.args = args
        self.X_train_dir = args.X_train_dir
        # self.data_dir = args.data_dir
        self.X_test_dir = args.X_test_dir

    def load_y_train(self):
        args = self.args
        processor = DataProcessor(args)
        all_train_examples = processor.get_train_examples()
        all_train_labels = processor.get_train_labels()
        train_examples, dev_examples, _, _ = train_test_split(all_train_examples, all_train_labels,
                                                              random_state=233)
        # dev_labels = [example.labels for example in dev_examples]
        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

        def convert_labels_to_id(labels):
            label_ids = [0] * 8
            for l in labels:
                label_ids[label_map[l]] = 1
            return label_ids
        targets = []
        for example in dev_examples:
            target = convert_labels_to_id(example.labels)
            targets.append(target)
        # targets = [label_map[example.label] for example in train_examples]
        return np.array(targets)

    def load_X_train(self, path):
        path = os.path.join(path, 'oof_train')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_X_test(self, path):
        path = os.path.join(path, 'oof_test')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def filter_path(self, paths):
        filter_names = ['new_model']
        result_paths = []
        # for path in paths:
        #     if not any(fn in path for fn in filter_names):
        #         result_paths.append(path)
        for path in paths:
            if any(fn in path for fn in filter_names):
                result_paths.append(path)
        result_paths = [
            # 'new_model1',
            # 'new_model2',
            # 'new_model3',
            # 'new_model4',
            'new_model5',
            # 'new_model6',
            'baseline',
        ]
        return result_paths

    def load_data(self):
        self.y_train = self.load_y_train()
        self.labels = LABEL_LIST
        self.n_classes = len(self.labels)
        pathes = os.listdir(self.X_train_dir)
        pathes = self.filter_path(pathes)
        # self.X_train = np.zeros((len(self.y_train), self.n_classes))
        # for i, path in enumerate(pathes):
        #     try:
        #         self.X_train[:, :] += self.load_X_train(os.path.join(self.X_train_dir, path))
        #     except:
        #         print(path)
        # self.X_train = self.X_train/len(pathes)
        # pathes = os.listdir(self.X_test_dir)
        # pathes = self.filter_path(pathes)
        n_test = len(self.load_X_test(os.path.join(self.X_test_dir,pathes[0])))
        # n_train = len(self.y_train)
        self.X_test = np.zeros((n_test, self.n_classes))

        for i, path in enumerate(pathes):
            self.X_test[:, :] += self.load_X_test(os.path.join(self.X_train_dir, path))
        self.X_test = self.X_test/len(pathes)
        # self.X_train = self.X_train.reshape(n_train, -1)
        self.X_test = self.X_test.reshape(n_test, -1)

    def get_result(self, preds, reals):
        f_score = sentiment_f1_score(y_true=reals, y_pred=preds, average='macro')
        return f_score

    def stack(self, reload=True):
        if reload:
            self.load_data()
        pred_labels = (self.X_train > 0.46).astype(np.int)
        # for i, p in enumerate(pred_labels):
        #     if np.sum(p) == 0:
        #         j = np.argmax(p)
        #         pred_labels[i, j] = 1
        real_labels = self.y_train
        f_score = self.get_result(pred_labels, real_labels)
        return f_score

    def predict(self):
        print(self.X_test.shape)
        preds = (self.X_test > 0.46).astype(np.int)
        return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mode = 'virus'
    parser.add_argument('--X_train_dir', type=str, default='sentiment_model',
                        help='模型的预测输出位置，应为一个examples nums * class nums的矩阵')
    parser.add_argument('--data_dir', type=str, default='data/no_split_word',
                        help='模型的真实数据，包含句子的真正标签')
    parser.add_argument('--X_test_dir', type=str, default='sentiment_model',
                        help='模型的test数据，生成的oof_test')
    parser.add_argument('--output_dir', type=str, default='voting_results',
                        help='输出的目录')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    st = Voting(args)
    st.load_data()
    # f_score = st.stack()
    # print(f_score)
    preds = st.predict()
    preds = preds.tolist()

    label_list = LABEL_LIST
    label_map = {label: id for id, label in enumerate(label_list)}

    import csv
    with open(os.path.join(args.output_dir, 'submit2.csv'), 'w', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        writer.writerow(["ID", "Labels"])
        for id, label in enumerate(preds):
            labels = [RAW_LABEL_LIST[i] for i, l in enumerate(label) if l != 0]
            writer.writerow([id, labels])


