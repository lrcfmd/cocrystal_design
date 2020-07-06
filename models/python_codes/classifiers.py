from GMM_classifier import GMM
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.cblof import CBLOF
from sklearn.mixture import GaussianMixture

classifiers = {
    'Gaussiann Mixture Model (GMM)': GMM(n_components= 4, covariance_type='diag', random_state=None), 
    'K Nearest Neighbors (KNN)': KNN(contamination=0.05, method='mean', n_neighbors= 20, metric='minkowski', algorithm='kd_tree'),
    'Histogram-base Outlier Detection (HBOS)':  HBOS(contamination=0.05, n_bins=16, alpha=0.7), 
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=8), contamination=0.05),
        'Isolation Forest': IForest(behaviour="new", bootstrap=False, contamination=0.05, n_estimators=400,  max_features=1.0, max_samples=1000, random_state=0),
    'One class SVM (OCSVM)': OCSVM(contamination=0.05, kernel='rbf' , nu= 0.5, degree=10, gamma=7), 
    'Local Outlier Factor (LOF)':
       LOF(n_neighbors=10, contamination=0.05), 
    'CBLOF':    CBLOF(contamination=0.05,  alpha=0.9, beta=4, n_clusters=12)
 }


 #'Gaussiann Mixture Model (GMM)': GMM(n_components= 4, covariance_type='full', random_state=0), 
  #    'K Nearest Neighbors (KNN)': KNN(contamination=0.05, method='mean', n_neighbors= 20, metric='minkowski', algorithm='kd_tree'),
   # 'Histogram-base Outlier Detection (HBOS)':  HBOS(contamination=0.05, n_bins=16, alpha=0.7), 
   # 'Feature Bagging':
    #    FeatureBagging(LOF(n_neighbors=8), contamination=0.05, random_state=0),
     #   'Isolation Forest': IForest(behaviour="new", bootstrap=False, contamination=0.05, n_estimators=400,  max_features=1.0, max_samples=1000, random_state=0), 
   # 'One class SVM (OCSVM)': OCSVM(contamination=0.05, kernel='rbf' , nu= 0.5, degree=10, gamma=6), 
   # 'Local Outlier Factor (LOF)':
   #    LOF(n_neighbors=10, contamination=0.05), 
   #  'CBLOF':    CBLOF(contamination=0.05,  alpha=0.9, beta=4, n_clusters=12, random_state=0)