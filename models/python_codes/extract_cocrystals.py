# Import the libraries
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import os.path
import ccdc
from ccdc import search, io, molecule
from ccdc import search, io, molecule
from ccdc.search import SimilaritySearch, SubstructureSearch, MoleculeSubstructure
from ccdc.io import MoleculeReader, CrystalReader, EntryReader
import csv
from collections import Counter
from itertools import groupby

def remove_polymorphs(lst):
    res = []
    for g, l in groupby(sorted(lst), lambda x: x[:6]):
        res.append(next(l))
    return res

def solvents():
    df = pd.read_csv("../solvent_smiles.txt", header=None)
    lista = df.iloc[:, 0].values
    organic_solvents = ['c1ccccc1', 'Clc1ccccc1', 'Cc1ccccc1', 'C1CCOC1' ,'c1ccncc1', 'Cc1ccc(C)cc1', 'Cc1ccccc1C']
    solvents = [x for x in lista if x not in organic_solvents]
    return solvents

def search_cocrystals():
    '''
    Search the whole CSD for structures that contain two different molecules
    with the specific settings
    '''
    csd = MoleculeReader('CSD')
    settings = search.Search.Settings()
    settings.only_organic = True
    settings.not_polymeric = True
    settings.has_3d_coordinates = True
    settings.no_disorder = True
    settings.no_errors = True
    settings.no_ions = True
    settings.no_metals = True
    mol = []
    for i, entry in enumerate(csd):
        if settings.test(entry):
            molecule = entry.identifier
            mol.append(molecule)

    fin=[]
    csd_reader = MoleculeReader(mol)
    for i in csd_reader:
        id= i.identifier
        mol = csd_reader.molecule(id)
        smi= mol.smiles
        if smi !=  None:
            smi = smi.split('.')
            if len(Remove(smi)) == 2:
            # We make sure that the structure consist of two different molecules
                fin.append(mol.identifier)              
    final_cocrystals =[]    
    # clean the list from solvents
    for mol1 in fin:
        mol = csd_reader.molecule(mol1)
        for i in range(0, (len(mol.components))):
            if mol.components[i].smiles in solvents:
                final_cocrystals.append(mol.identifier)    
    final_cocrystals = Remove(final_cocrystals)
    final_cocrystals = [x for x in fin if x not in final_cocrystals]    
    # Clean the list from polymorphs
    cocrystals = remove_polymorphs(final_cocrystals)
    return cocrystals

if __name__ == "__main__":
    cocrystals = search_cocrystals()
