import logging
import jieba
import os
import json
import torch
from tokenizer import Tokenizer
from torch.utils.data import Dataset
from constant import *

class InputExample(object):
    def __init__(self, id, content, labels=None):
        self.id = id
        self.content = content
        self.labels = labels

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

class SentimentDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, item):
        return self.features[item]
    def get_labels(self):
        return LABEL_LIST
    def get_raw_labels(self):
        return RAW_LABEL_LIST

class DataProcessor(object):
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.train_examples = None
        self.train_labels = None
        print(self.data_dir)

    def get_train_examples(self):
        logging.info("*" * 10 + 'train dataset' + "*" * 10)
        if self.train_examples is None:
            self.train_examples = self._create_examples(
                self._read_file(os.path.join(self.data_dir, 'train.json')))
        return self.train_examples

    def get_train_labels(self):
        if self.train_labels is None:
            self.train_labels = [example.labels for example in self.get_train_examples()]
        return self.train_labels

    def get_predict_examples(self):
        logging.info("*" * 10 + 'predict dataset' + "*" * 10)
        examples = self._create_examples(
            self._read_file(os.path.join(self.data_dir, 'test.json')),
            do_predict=True)
        return examples

    def get_pseudo_data(self):
        logging.info("*" * 10 + 'use pseudo' + "*" * 10)
        examples = self._create_examples(
            self._read_file(os.path.join(self.data_dir, 'pseudo_train.txt')))
        return examples

    def get_labels(self):
        return LABEL_LIST

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, encoding='utf-8') as f:
            data_list = json.load(f)
        return data_list

    @classmethod
    def _create_examples(cls, data_list, do_predict=False):
        examples = []
        for data in data_list:
            id = data['id']
            content = data['content']
            if do_predict:
                labels = []
            else:
                labels = data['labels']
            examples.append(
                InputExample(
                    id=id,
                    content=content,
                    labels=labels
                ))
        return examples

def convert_examples_to_features(examples,
                               tokenizer,
                               max_length=128,
                               label_list=None,
                               pad_token=0,
                               pad_token_segment_id=0,
                               mask_padding_with_zero = True):
    logging.info('***** converting to features *****')
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    def _truncate(content, max_length):
        while len(content) > max_length:
            content = list(content)
            content.pop(len(content)//2)
        return ''.join(content)
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode(
            first_text=example.content,
            maxlen=max_length,
        )
        input_ids, token_type_ids = inputs
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),max_length)

        labels = [0] * 8
        for label in example.labels:
            try:
                labels[int(label_map[label])] = 1
            except:
                print(example.labels)
        # labels = [int(label_map[label]) for label in example.labels]
        features.append(
            InputFeatures(input_ids, attention_mask, token_type_ids, labels)
        )
    return features

def convert_to_dataset(examples, tokenizer, max_length, label_list):

    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_length=max_length,
                                            label_list=label_list)
    return SentimentDataset(features)

def collate_batch(features):
    # In this method we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    first = features[0]
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    # 将features的label属性转换为labels, 以匹配模型的输入参数名称
    if hasattr(first, "label") and first.label is not None:
        if type(first.label) is int:
            labels = torch.tensor([f.label for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label for f in features], dtype=torch.float)
        batch = {"labels": labels}
        # print(labels)
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
    return batch

if __name__ == '__main__':
    tokenizer = Tokenizer(
        'data/split_word/keep_vocab.txt',
        do_lower_case=True,
        pre_tokenize=lambda s: s.split(' ')
    )

    processor = DataProcessor(args=None, data_dir='data/split_word')
    train_examples = processor.get_train_examples()
    convert_examples_to_features(train_examples, tokenizer,
                                 max_length=128,
                                 label_list=LABEL_LIST,
                                 pad_token_segment_id=1,
                                 )
