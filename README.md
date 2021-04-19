# Co-crystal prediction
The aim of this work is to introduce the concept of One Class Classification and its application in a co-crystal design problem as described in our paper [One class classification as a practical approach for accelerating π–π co-crystal discovery](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc04263c#!divAbstract). 
A very important consern that currrent data driven approaches for materials discovery face is the lack of negative data i.e., data for experiments that didn’t work or materials that cannot be formed. This  drives to the excistence of extremely biased towards one class materials datasets. Instead of trying to gather large amounts of negative data for binary classification, that might be an expensive or unreliable process, one class classification is focusing on extracting the trends that dominate the known and reliable data and try to predict similar materials that could be synthesizable. 

<img src="https://github.com/lrcfmd/cocrystal_design/blob/master/figures/main_fig.png" width="800" height="400">

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

6. SHAP explanations for detecting the significant features (https://github.com/slundberg/shap)

7. Experimental realization (Pareto Optimization)

8. Comparison with known co-crystals in CSD (Euclidean distance and Packing considerations)


# Citing
This work is described in detail in the following publication:
```

@article{10.1039/d0sc04263c,
    author = {Vriza, Aikaterini and Canaj, Angelos B. and Vismara, Rebecca and Kershaw Cook, Laurence J.and Manning, Troy D. and
              Gaultois, Michael W. and Wood, Peter A. and Kurlin, Vitaliy and Berry, Neil and Dyer, Matthew S. and Rosseinsky, Matthew J.},
    title = {One class classification as a practical approach for accelerating π–π co-crystal discovery},
    journal = {Chemical Science},
    volume = {12},
    number = {5},
    pages = {1702-1719},
    year = {2021},
    doi = {10.1039/d0sc04263c},
    URL = {https://doi.org/10.1039/D0SC04263C}
    }

```
