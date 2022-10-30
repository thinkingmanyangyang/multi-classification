import os
import csv
import json
import jieba

split_words = True
is_test = True

train_data = []
source_labels = ['Expect', 'Hate', 'Sorrow', 'Surprise', 'Anxiety', 'Joy', 'Love', 'Anger']
target_labels = ['期待', '仇恨', '悲伤', '惊奇', '焦虑', '快乐', '爱', '愤怒']
max_length = 0
keep_vocab = []
labels_map = dict(zip(source_labels, target_labels))
graph = [[0] * 8] * 8
print(labels_map)
if is_test:
    # file_name = 'raw_test.csv'
    # file_name = 'test.csv'
    # file_name = 'ys_test.csv'
    file_name = 'lx_test.csv'
else:
    file_name = 'raw_train.csv'
with open(file_name, 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    for row in reader:
        # 跳过表头
        if row[0] == 'ID':
            continue
        if is_test:
            labels = []
        else:
            labels = [labels_map[label] for label in eval(row[2])]
        content = row[1].strip()
        if split_words:
            words = content.split(' ')
            max_length = max(max_length, len(words))
            keep_vocab += words
            keep_vocab += list(''.join(words))
            # keep_vocab.extend(list(jieba.cut(content)))
            train_data.append({
                'id': int(row[0]),
                'content': content,
                'labels': labels
            })
        else:
            content = ''.join(row[1].split(' '))
            train_data.append({
                'id': int(row[0]),
                'content': ''.join(content.split(' ')).strip(),
                'labels': labels
            })
print('data load end...')


if split_words:
    root_dir = 'split_word'
else:
    root_dir = 'no_split_word'

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

if is_test:
    file_name = 'test.json'
else:
    file_name = 'train.json'

save_file_name = os.path.join(root_dir, file_name)
with open(save_file_name, 'w', encoding='utf-8') as fw:
    json.dump(train_data, fw, ensure_ascii=False, indent=4)
    print('data to file to -> ' + save_file_name)

save_vocab_file = 'split_word/keep_vocab.txt'
keep_vocab += target_labels
keep_vocab += ["None"]
keep_vocab = list(set(keep_vocab))
keep_vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + keep_vocab
with open(save_vocab_file, 'w', encoding='utf-8') as fw:
    for word in keep_vocab:
        fw.write(word + '\n')
    print("total words num:", len(keep_vocab))

# print('labels:', list(set(labels)))
print('max length:', max_length)



