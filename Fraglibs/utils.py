import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

RDLogger.DisableLog('rdApp.*')

def fingerprints_morgan(smiles_list: list, radius: int, hash_size: int):
    """
    Using a smiles list with radius and hash size for generating the morgan fingerprint
    :param smiles_list: a list
    :param radius: radius (1-6)
    :param hash_size: nbits (256, 512, 1024, 2048, 4096)
    :return:
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # init the bit information and the fingerprints of dataset
    bit_info = []
    fingerprints = []

    # Generating the morgan fingerprint
    for mol in mols:
        bi = {}
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, hash_size, bitInfo=bi)
        fingerprint_matrix = np.zeros((1,), dtype=np.float32)
        ConvertToNumpyArray(fingerprint, fingerprint_matrix)
        fingerprints.append(fingerprint_matrix)
        bit_info.append(bi)

    # Save as dataframe
    df_fingerprints = pd.DataFrame(fingerprints)

    return df_fingerprints, bit_info


def normalizer(value_list: list):
    """
    The gaussian normalization (Standard Z-score)
    :param value_list:
    :return:
    """
    mu = np.mean(value_list)
    sigma = np.std(value_list)
    value_list = (value_list - mu) / sigma
    df = pd.DataFrame()
    df['activity'] = value_list
    print(f"The mu value in Standard normalization for the dataset is {mu}")
    print(f"The sigma value in Standard normalization for the dataset is {sigma}")
    return df


def dataset_sklearn(smiles_list: list,
                    labels: list,
                    normalization: bool = True,
                    train_test_rate: float = 0.2,
                    random_state: int = 42,
                    val_set: bool = False,
                    test_val_size: float = 0.5,
                    radius: int = 3,
                    hash_size: int = 1024):
    """

    :param smiles_list:
    :param labels:
    :param normalization:
    :param train_test_rate:
    :param random_state:
    :param val_set:
    :param test_val_size:
    :param radius:
    :param hash_size:
    :return:
    """
    print("\n[*] Preparing to create the dataset for machine learning models")

    fingerprints, _ = fingerprints_morgan(smiles_list, radius=radius, hash_size=hash_size)

    if normalization:
        labels = normalizer(labels)
    else:
        labels = np.array(labels)

    if val_set:
        X_train, X_true, y_train, y_true = train_test_split(fingerprints,
                                                            labels,
                                                            test_size=train_test_rate,
                                                            random_state=random_state
                                                            )

        X_test, X_val, y_test, y_val = train_test_split(X_true,
                                                        y_true,
                                                        test_size=test_val_size,
                                                        random_state=random_state
                                                        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        X_train, X_test, y_train, y_test = train_test_split(fingerprints,
                                                            labels,
                                                            test_size=train_test_rate,
                                                            random_state=random_state
                                                            )

        return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred, regression_plot='regression_test_set.png', color='blue', scale=3.0):
    r2 = r2_score(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = mean_absolute_percentage_error(y_true, y_pred)

    print("Evaluating on Test set")
    print('Test set R2 score: {:.3f}'.format(r2))
    print('Test set MAE: {:.3f}'.format(mae))
    print('Test set MSE: {:.3f}'.format(mse))
    print("Test set RMSE: {:.3f}".format(rmse))
    print("Test set MAPE: {:.3f}".format(mape))
    print()
    print()
    print("Drawing the regression plot")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='none', marker='o', edgecolor=color, linewidths=1)
    plt.ylabel('Predictive Value', fontdict={'fontsize': 16})
    plt.xlabel('Experimental Value', fontdict={'fontsize': 16})
    plt.axis((-scale, scale, -scale, scale))
    plt.axline((0.0, 0.0), slope=1.0, color='gray', lw=3, linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{regression_plot}", dpi=300)
    plt.close()

