import os
import torch
import jieba
import pickle
import logging
import argparse
import numpy as np
from data_utils import DataProcessor, convert_to_dataset
from utils import set_seed, set_logger
from tokenizer import Tokenizer
from multi_class_model import MultiClassification
from transformers import BertConfig
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedShuffleSplit
from constant import *
from train_and_eval import trains, predict, test, _predict
from net.combination_model import CombinationModel
jieba.initialize()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="roberta_wwm_sentiment.log", type=str, required=True,
                        help="设置日志的输出目录")
    parser.add_argument("--data_dir", default='data/train/usual', type=str, required=True,
                        help="数据文件目录，应当有train.text dev.text")
    parser.add_argument("--pretrain_path", default='pytorch_wobert', type=str, required=False,
                        help="预训练模型所在的路径，包括 pytorch_model.bin, vocab.txt, config.json")
    parser.add_argument("--model_name", default='RobertaBase')
    parser.add_argument("--output_dir", default='sentiment_model/usual2', type=str, required=True,
                        help="输出结果的文件")
    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="输入到bert的最大长度，通常不应该超过512")
    # 这里改成store_false 方便直接运行
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练")
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="训练集的batch_size")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="验证集的batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=3.0, type=float,
                        help="最大的梯度更新")
    parser.add_argument("--num_train_epochs", default=25, type=int,
                        help="epoch 数目")
    parser.add_argument('--seed', type=int, default=233,
                        help="random seed for initialization")
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="让学习增加到1的步数，在warmup_steps后，再衰减到0")
    parser.add_argument("--warmup_rate", default=0.00, type=float,
                        help="让学习增加到1的步数，在warmup_steps后，再衰减到0，这里设置一个小数，在总训练步数*rate步时开始增加到1")
    parser.add_argument("--attack", default=None,
                        help="是否进行对抗样本训练, 选择攻击方式或者不攻击")
    parser.add_argument("--label_smooth", default=0.0, type=float,
                        help="设置标签平滑参数")
    parser.add_argument("--use_pseudo_data", action='store_true',
                        help='是否使用生成的伪标签数据集')
    parser.add_argument("--k_fold", default=5, type=int,
                        help='k折交叉验证的划分数目')
    parser.add_argument("--cls_pooler", default=None)
    parser.add_argument("--hidden_pooler", default=None)
    parser.add_argument("--loss_fct", default='bce_loss')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.pretrain_path)
    assert os.path.exists(args.output_dir)

    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    # 设置日志
    log_dir = os.path.join(args.output_dir, args.log_dir)
    set_logger(log_dir)

    processor = DataProcessor(args)
    logging.info('loading model... ...')

    # 不用预训练模型
    tokenizer = Tokenizer(
        os.path.join(args.pretrain_path, 'vocab.txt'),
        do_lower_case=True,
        pre_tokenize=lambda s: jieba.cut(''.join(s.split(' ')), HMM=False)
        # pre_tokenize=lambda s: s.split(' ')
    )
    # 获取数据
    all_train_examples = processor.get_train_examples()
    all_train_examples = np.array(all_train_examples)
    logging.info("start training... ...")

    all_train_examples = np.array(processor.get_train_examples())
    all_train_labels = np.array(processor.get_train_labels())
    all_test_examples = processor.get_predict_examples()
    test_dataset = convert_to_dataset(all_test_examples, tokenizer, max_length=args.max_seq_length,
                                      label_list=LABEL_LIST)

    k_folder = KFold(n_splits=args.k_fold, random_state=args.seed, shuffle=False)

    oof_train = np.zeros((len(all_train_examples), len(processor.get_labels())))
    oof_test = np.zeros((len(all_test_examples), len(processor.get_labels())))
    for fold_num, (train_idx, dev_idx) in enumerate(k_folder.split(all_train_examples),
                                                    start=1):
        train_examples, dev_examples = all_train_examples[train_idx], all_train_examples[dev_idx]
        train_dataset = convert_to_dataset(train_examples, tokenizer, max_length=args.max_seq_length,
                                           label_list=LABEL_LIST)
        dev_dataset = convert_to_dataset(dev_examples, tokenizer, max_length=args.max_seq_length,
                                         label_list=LABEL_LIST)
        if args.do_train:
            config = BertConfig.from_pretrained(args.pretrain_path)
            config.num_labels = len(processor.get_labels())
            # config.num_hidden_layers = 3
            # config.vocab_size = tokenizer._vocab_size
            # 定义model 注意这里使用的是BertModel，如果没有足够算力建议换成rnn，cnn分类器
            # model = MultiClassification.from_pretrained(args.pretrain_path, config=config)
            model = CombinationModel.build_model(args, config, pretrain_path=args.pretrain_path)
            model = model.to(args.device)
            trains(args, train_dataset, dev_dataset, model, fold_num=str(fold_num))
        # 测试模型
        if args.do_test or args.do_train:
            print('predict model ...')
            # 获取数据
            config = BertConfig.from_pretrained(args.output_dir)
            config.num_labels = len(processor.get_labels())
            # model = MultiClassification.from_pretrained(args.output_dir)
            model = CombinationModel.load_model(args, config, os.path.join(args.output_dir, str(fold_num)))
            model = model.to(args.device)

            test(args, model, test_dataset=dev_dataset)
            oof_train[dev_idx] = _predict(args, model, dev_dataset)
            predict(args, model, predict_dataset=test_dataset, processor=processor)
            oof_test += _predict(args, model, test_dataset)
    oof_test = oof_test / args.k_fold
    def save_probs(probs, dir):
        with open(dir, 'wb') as fw:
            pickle.dump(probs, fw)
    save_probs(oof_train, os.path.join(args.output_dir, 'oof_train'))
    save_probs(oof_test, os.path.join(args.output_dir, 'oof_test'))
    print('test end ...')
        # wobert 原生分词 73.73












