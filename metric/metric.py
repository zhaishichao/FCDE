import numpy as np
from scipy.stats import gmean
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score

# 模型评估
def fit_pred(X_train, y_train, X_test=None, clf=None, soft_lable=False):
    # 在测试集上评估模型
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if soft_lable:
        y_prob = clf.predict_proba(X_test)
        return y_pred, y_prob
    else:
        return y_pred

def metric(y_test, y_pred, y_prob, scoring):
    result = {}
    # F1-score (多分类使用宏平均)
    if 'f1_macro' in scoring:
        f1 = f1_score(y_test, y_pred, average='macro')
        result[scoring['f1_macro']] = round(f1, 4)
    if 'g_mean' in scoring:
        # G-mean (计算每个类召回率的几何平均)
        cm = confusion_matrix(y_test, y_pred)
        # 遍历混淆矩阵，计算每个类的召回率 （每类正确预测个数 / 该类总数）
        recall_per_class = cm.diagonal() / cm.sum(axis=1)
        g_mean = gmean(recall_per_class)
        result[scoring['g_mean']] = round(g_mean, 4)
    if 'roc_auc_ovr' in scoring:
        # AUC
        num_classes = len(np.unique(y_test))
        if num_classes == 2:
            roc_auc_ovr = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        result[scoring['roc_auc_ovr']] = round(roc_auc_ovr, 4)
    if 'accuracy' in scoring:
        accuracy = accuracy_score(y_test, y_pred)
        result[scoring['accuracy']] = round(accuracy, 4)

    return result
