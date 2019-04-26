
from sklearn.metrics import f1_score
y_true = [0, 1, 3, 0, 1, -1]
y_pred = [0, 2, 1, 0, 0, 2]
res=f1_score(y_true, y_pred, average='macro')
res2 = f1_score(y_true, y_pred, average='micro')
print(res,res2)