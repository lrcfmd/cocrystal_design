Supporting code reproducing all the results for the paper: 
'One class classification as a practical approach for accelerating target specific co-crystal discovery'

The computational workflow for the co-crystal design involves the following steps:

1. Generating the datasets:\
a. Co_crystals extraction from Cambridge Crystalographic Structural Database (CSD)
(Requires the installation of the CCDC Python API)\
b. Designing the labeled (CSD) and unlabeled (ZINC15) dataset

2. Standard one class/novelty detection:\
-Involves feature engineering following the paper (Fábián, L. Cambridge Structural Database Analysis of Molecular 
Complementarity in Cocrystals. Cryst. Growth Des. 2009, 9 (3), 1436–1443)\
-2a part involves the hyperparameter tunning process using the hyperopt library (https://github.com/hyperopt/hyperopt) \
-The models were adapted and modified from https://github.com/yzhao062/pyod.git and https://scikit-learn.org/stable/modules/mixture.html

3. Deep one class\
-The deep one class model was adapted and modified from https://github.com/lukasruff/Deep-SVDD-PyTorch.git\
-The set trasformer autoencoder used from https://github.com/juho-lee/set_transformer.git to denote the permutation invariant property of the molecualar pairs

4. Models evaluation and comparison 

5. Ratios prediction

6. SHAP explanations for detecting the significant features

7. Experimental realization (Pareto Optimization)

8. Comparison with known co-crystals in CSD (Euclidean distance and Packing considerations)

