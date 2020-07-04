from sklearn.mixture import GaussianMixture
from numpy import percentile
"""
Builds the Gaussian Mixture Classifier to be used for anomaly detection
When points are scored under a pre-specified threshold, they are regarged as outliers

"""
contamination = 0.05

class GMM(GaussianMixture):
  def __init__(self, n_components, covariance_type, random_state):
    super().__init__(n_components=n_components , covariance_type=covariance_type, random_state=random_state)

  def fit(self, X, y):
    super().fit(X, y)
    self.prob = super().score_samples(X)
    self.c = percentile(self.prob, 100 * contamination)

  def predict(self, X):
    scores = []
    proba=super().score_samples(X)
    
    scores =(proba <= self.c).astype('int').ravel()

    return scores