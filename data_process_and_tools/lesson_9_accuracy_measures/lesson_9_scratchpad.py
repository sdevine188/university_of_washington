#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

pred = [0, 0, 1, 1, 0, 1, 0, 0, 0, 0]
actual = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

pred_class2 = pd.DataFrame(pred)
pred_class2.columns = ["pred"]
original_zero_index = pred_class2.pred == 0
original_one_index = pred_class2.pred == 1
pred_class2.loc[original_zero_index, :] = 1
pred_class2.loc[original_one_index, :] = 0
pred_class2

actual_class2 = pd.DataFrame(actual)
actual_class2.columns = ["actual"]
original_zero_index = actual_class2.actual == 0
original_one_index = actual_class2.actual == 1
actual_class2.loc[original_zero_index, :] = 1
actual_class2.loc[original_one_index, :] = 0
actual_class2



confusion_matrix(y_true = actual, y_actual = pred)
(5 + 2) / (5 + 2 + 1 + 2)

accuracy_score(y_true = actual, y_pred = pred)
precision_score(y_true = actual, y_pred = pred)
recall_score(y_true = actual, y_pred = pred)
f1_score(y_true = actual_class2, y_pred = pred_class2)


CM = confusion_matrix(y_true = actual_class2, y_pred = pred_class2)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
tp / (tp + fn)
fp / (fp + tn)

pred_class2
actual_class2
class2 = pd.concat([actual_class2, pred_class2], axis = 1)
class2

class2.loc[(class2.actual == 1) & (class2.pred == 1), "true_positive"] = 1
class2.loc[~(class2.actual == 1) | ~(class2.pred == 1), "true_positive"] = 0
class2

class2.loc[(class2.actual == 1) & (class2.pred == 1), "true_positive"] = 1
class2.loc[~(class2.actual == 1) | ~(class2.pred == 1), "true_positive"] = 0
class2

class2