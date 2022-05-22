# MetaDTA: Meta-learning-based drug-target binding affinity prediction

Implementation for our [paper](https://openreview.net/forum?id=yzlif16IASM), accpeted to MLDD workshop in ICLR 2022.

This is a minimum working version of the code used for the paper.

## Example
We upload a small version of binding affinity dataset originally from BindingDB. The dataset size is limited for the simple test of our code, so the test performance is not same with the paper. 

## Environment setting

    conda env create --file environment.yaml
    conda activate metadta

## Quick Run
The simple model training code is 

    python train.py --use_latent_path 

The use_latent_path option is the option for the latent embedding path, which is from the [Attentive Neural Process](https://github.com/deepmind/neural-processes)

