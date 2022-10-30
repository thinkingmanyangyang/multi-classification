# 采用普通的机器学习分类方法
from sklearn.tree import DecisionTreeClassifier
from tokenizer import Tokenizer
from data_utils import DataProcessor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import logging
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/split_word')
parser.add_argument('--seed', default=233)
args = parser.parse_args()
tokenizer = Tokenizer(
        'data/split_word/keep_vocab.txt',
        do_lower_case=True,
        pre_tokenize=lambda s: s.split(' ')
)
processor = DataProcessor(args)
all_train_examples = processor.get_train_examples()
all_train_examples = np.array(all_train_examples)
logging.info("start training... ...")

all_train_examples = np.array(processor.get_train_examples())
all_train_labels = np.array(processor.get_train_labels())

train_examples, dev_examples, _, _ = train_test_split(all_train_examples, all_train_labels, random_state=args.seed)
from constant import *
def convert_labels_to_id(labels):
    label_list = LABEL_LIST
    label_map = dict(zip(label_list, range(len(label_list))))
    label_ids = [0] * 8
    for l in labels:
        label_ids[label_map[l]] = 1
    return label_ids

corpus = [example.content for example in all_train_examples]
corpus = [
    'word1 word2 word3',
    'word1 word2 word3',
    'word1 word2 word3',
          ]
from sklearn.multiclass import OneVsRestClassifier

OneVsRestClassifier()
vectoerizer = TfidfVectorizer()
vectoerizer.fit(corpus)
X_train = vectoerizer.transform([example.content for example in train_examples]).toarray()
y_train = [example.labels for example in train_examples]
y_train = np.array([convert_labels_to_id(y) for y in y_train])
X_test = vectoerizer.transform([example.content for example in dev_examples]).toarray()
y_test = [example.labels for example in dev_examples]
y_test = np.array([convert_labels_to_id(y) for y in y_test])

cls = DecisionTreeClassifier()

cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)
from assess import sentiment_f1_score
print(sentiment_f1_score(y_test, y_pred, average='macro'))



