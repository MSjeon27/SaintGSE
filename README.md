# SaintGSE: AI-Driven Signaling Pathway Analysis Tool

SaintGSE is an artificial intelligence model designed to predict gene-pathway relationships using large-scale differentially expressed gene (DEG) datasets. By leveraging an autoencoder and the SAINT transformer model, SaintGSE overcomes challenges in gene expression analysis, such as data scarcity, model compatibility, and interpretability. This project uses and modifies code from the SAINT project, licensed under the Apache License 2.0. 

Key Features

AI-Driven Pathway Prediction: Uses autoencoders and the SAINT model to analyze gene expression data and predict related signaling pathways.

Osteoarthritis Study: Applied to osteoarthritis (OA) to identify key pathways and potential therapeutic targets.

Explainability: Utilizes Shapley additive explanations (SHAP) to interpret model predictions and identify influential genes.

Installation

Clone the repository and install the dependencies:

'''bash
$ git clone https://github.com/yourusername/saintgse.git
$ cd saintgse
'''

Also, we have additional dataset for prediction


$ wget https://drive.google.com/file/d/1jLEfqp54c3VP2Mk05qCoailf_BHGnXSj/view?usp=sharing


Usage

SaintGSE can be used to analyze new gene expression datasets for pathway prediction:
