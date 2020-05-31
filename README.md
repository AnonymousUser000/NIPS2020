# NIPS2020
Implementation of methods in "Signed Network Embedding by Preserving Multi-Order Signed Proximity"

## Dependency
Matlab=R2019

Python=3.7

numpy=1.16.5

scikit-learn=0.21.3

## Usage
The methods are implemented in Matlab dynamic script. You can find detailed description in the files. Moreover, we provided a sample data based on WikiElec network. You can directly load it in Matlab.

We find that sklearn package implemented a more efficent LR model, thus we constructed a Python script for evaluation. LinkPrediction.py is for SPMF as each node has two types of representations. And LinkPredictionSP.py is for f-SPMF.
