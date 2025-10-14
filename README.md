### Installation

Run:

`conda create -n FragOPT python=3.8.20`

`conda install numpy=1.24 pandas=2.0.3 scikit-learn=1.3 seaborn=0.13`

`conda install -c conda-forge openbabel charset-normalizer`

`pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`

`pip install tensorflow-gpu==2.10.0 xgboost-cpu==2.1.2 rdkit shap matplotlib prody`


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

The configuration of optmization could refer to the `config.json`.

The model of PD-L1(PDB ID: 5N2F) and COVID-19 (PDB ID: 7BQY) were pretrained in `models` folder, you can directly use them.
