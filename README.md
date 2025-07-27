### Installation

Run:

`conda env create -f environment.yml`

This will take care of installing all required dependencies.

The only required dependency is the latest Conda package manager, which you can download with the Anaconda Python distribution [here](https://www.anaconda.com/distribution/).

### Preparation

You should prepare the data to train QSAR models for your protein. 

The first column should be molecular SMILES, the second colomn should be the bioactivity class. you can refer to the files in `dataset` folder. 

### Training

The QSAR model train with molecular fingerprint and bioacivity value of your protein. To do this, run:

`python train_models.py`

where the `dataset_path` in train_models.py should be change to your data path.

### Optimization

After trianed the QSAR models. You can use FragOPT to optimize the target molecule for its bioacivity and druglikeness.

`python main.py -j config.json`

The configuration of optmization could refer to the `config.json`
