import json
import numpy as np
graph = [[0] * 8 for i in range(8)]

labels = ['期待', '仇恨', '悲伤', '惊奇', '焦虑', '快乐', '爱', '愤怒']
label_map = dict(zip(labels, range(len(labels))))
with open('split_word/train.json', encoding='utf8') as f:
    data = json.load(f)

print(graph)
for d in data:
    labels = d['labels']
    label_ids = [label_map[label] for label in labels]
    for label_id1 in label_ids:
        for label_id2 in label_ids:
            if label_id1 != label_id2:
                graph[label_id1][label_id2] += 1

print(np.array(graph))
graph = np.array(graph)
graph1 = np.array(graph) > np.mean(graph, axis=1, keepdims=True)
print(np.mean(graph, axis=1, keepdims=True))
graph2 = np.array(graph) > np.mean(graph, axis=0, keepdims=True)
print(np.mean(graph, axis=0, keepdims=True))
print('graph 1', graph1)
print('graph 2', graph2)


graph = graph1 | graph2
graph = graph.astype(np.int)
print(graph)
