# 计算f值
from sklearn.metrics import f1_score, classification_report, accuracy_score
import numpy as np

def sentiment_f1_score(y_true, y_pred, average='macro'):
    scores = []
    label_nums = y_true.shape[1]
    for i in range(label_nums):
        true_label = y_true[:, i]
        pred_label = y_pred[:, i]
        score = f1_score(y_true=true_label, y_pred=pred_label, average=average)
        scores.append(score)
    f_score = np.array(scores).mean()
    return f_score, scores

def sentiment_accuracy(y_true, y_pred):
    accuracys = []
    label_nums = y_true.shape[1]
    for i in range(label_nums):
        true_label = y_true[:, i]
        pred_label = y_pred[:, i]
        accuracys.append(accuracy_score(y_true=true_label, y_pred=pred_label))
    accuracy = np.array(accuracys).mean()
    return accuracy, accuracys

if __name__ == '__main__':
    true = np.array([[1,0,1,1], [1,0,1,1]])
    pred = np.array([[1,0,1,1], [1,0,0,1]])
    print(sentiment_f1_score(y_true=true, y_pred=pred))
    print(sentiment_accuracy(y_true=true, y_pred=pred))
    print(classification_report(y_true=true, y_pred=pred))
    # print(f1_score(['A', 'B', 'C'], ['A', 'B']))
    # print(accuracy_score(['A', 'B', 'C'], ['A', 'B']))
    # print(recall(['A', 'B', 'C'], ['A', 'B']))