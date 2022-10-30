import json
from constant import *
label_map = dict(zip(LABEL_LIST, RAW_LABEL_LIST))
source_file = 'sentiment_model/new_model5/test_result.json'
target_file = 'sentiment_model/new_model5/submit.csv'

with open(source_file, 'r', encoding='utf-8') as f:
    result = json.load(f)

import csv
with open(target_file, 'w', encoding='utf-8') as fw:
    writer = csv.writer(fw)
    writer.writerow(["ID", "Labels"])
    count = 0
    for res in result:
        id = res['id']
        labels = res['labels']
        if count < 0:
            labels = []
            count += 1
        labels = [label_map[l] for l in labels]
        # if len(labels) == 0:
        #     labels = [RAW_LABEL_LIST[-2]]
        writer.writerow([id, labels])

