import numpy as np


roc_points = [(0, 0), (0.2, 0.6), (0.5, 0.8), (1, 1)]


fpr = [x[0] for x in roc_points]
recall = [x[1] for x in roc_points]

auc = 0
for i in range(len(fpr) - 1):
    auc += (fpr[i + 1] - fpr[i]) * (recall[i + 1] + recall[i]) / 2

print(auc)