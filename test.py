from tokenizer import Tokenizer
from tokenizer import load_vocab
from transformers import BertModel
from transformers import BertConfig
import jieba
import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from transformers import load_tf_weights_in_bert
from transformers import BertForPreTraining

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

real = [[1, 1, 1, 0], [1, 1, 1, 1]]
pred = [[1, 0, 1, 0], [1, 0, 1, 1]]
print(f1_score(y_pred=pred, y_true=real, average='macro'))
print(accuracy_score(y_pred=pred, y_true=real))
print(classification_report(y_pred=pred, y_true=real))
# config = BertConfig.from_pretrained('chinese_wobert/bert_config.json')
# config.vocab_size = 33586
# model = BertForPreTraining(config)

# init_vars = tf.train.list_variables('chinese_wobert/bert_model.ckpt')
# for name, shape in init_vars:
#     print(name)
    # logger.info("Loading TF weight {} with shape {}".format(name, shape))
    # array = tf.train.load_variable(tf_path, name)
    # names.append(name)
    # arrays.append(array)

# model = load_tf_weights_in_bert(model, config, 'chinese_wobert/bert_model.ckpt')
# model.save_pretrained('pytorch_wobert')
#
# model = BertModel.from_pretrained('pytorch_wobert')
# X = np.ones((3,3))
# Y = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=7)
# res = sss.split(X, Y)
# for i in res:
#     print(i)
# train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3)
# print(train_X, train_y)

# from bert4keras.models import build_transformer_model
# vocab = load_vocab('data/split_word/keep_vocab.txt', simplified=False)
# print(len(vocab))
# config = BertConfig.from_pretrained('chinese_wobert/bert_config.json')
# config.num_hidden_layers = 3
# model = BertModel(config=config)
# print(model)

# import jieba
# print(load_vocab('chinese_wobert/vocab.txt', simplified=True))

#
# tokenizer = Tokenizer(
#     'pytorch_wobert/vocab.txt',
#     do_lower_case=True,
#     pre_tokenize=lambda s: jieba.cut(s, HMM=False)
# )
#
# res = tokenizer.encode(first_text='老师好，我叫何同学', second_text='财大牛逼')
# print(res)
# print(res)
# with open('data/split_word/train.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#     for d in data:
#         res = tokenizer.encode_plus(d['content'])
#         print(res)

# import numpy as np
# from scipy import optimize
#
# z = np.array([1, -1, -2])
# a = np.array([[-1, -3, -2], [1, 1, -1]])
# b = np.array([-12, 2])
# x1_b = x2_b = x3_b = (0, None)
# a_eq = np.array([[0, 2, 1]])
# b_eq = np.array([4])
# res = optimize.linprog(z, A_ub=-a, b_ub=-b, A_eq=a_eq, b_eq=b_eq, bounds=(x1_b, x2_b, x3_b))
# print(res)