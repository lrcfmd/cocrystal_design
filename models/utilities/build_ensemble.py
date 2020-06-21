from sklearn.base import BaseEstimator

"""  Builds an ensemble of all the selected classifiers
      by averaging the scores 
"""

class Ensemble(BaseEstimator):
  def __init__(self, classifiers):
    self.classifiers = classifiers

  def fit(self, X, y):
    for clf in classifiers.values():
      clf.fit(X, y)

  def predict(self, X):
    scores = []
    for clf in classifiers.values():
      scores.append(clf.predict(X))
     
    return (np.mean(scores, 0) >= 0.5).astype('int').ravel()