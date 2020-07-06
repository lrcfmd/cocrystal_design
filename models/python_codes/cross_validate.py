import pandas as pd
import numpy as np
from sklearn import metrics

def cross_val(clf, X_train):
  valid =[]
  X_train_val=pd.concat([pd.DataFrame(X_train.values), pd.DataFrame(np.zeros(len(X_train)))], axis=1)

  # Perform k-fold cross validation
  from sklearn.model_selection import KFold
  kf = KFold(n_splits = 5)
  kf.get_n_splits(X_train_val)
  for train, test in kf.split(X_train_val):
      train_data = np.array(X_train_val)[train]
      train_label = train_data[:,-1]
      test_data = np.array(X_train_val)[test]
      test_label = test_data[:, -1]
      train_data = np.vstack([train_data, np.hstack([train_data[:,24:], train_data[:,:24]])])
      train_label = np.concatenate([train_label, train_label])
      clf.fit(train_data[:, :-1],train_label )
      pred_train = clf.predict(train_data[:,:-1])
      pred_test = clf.predict(test_data[:,:-1])
      valid.append(metrics.accuracy_score(pred_test, test_label))
  return valid