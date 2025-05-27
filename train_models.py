import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

import keras
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping
import joblib
from libs.utils import evaluate
from libs.models import lstm_model
# from components.modeling import rf_model
# from components.modeling import svr_model
# from components.modeling import mlp_model
# from components.modeling import xgb_model
from libs.utils import dataset_sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

# Parameters for LSTM model

def ensemble_model_evaluation(model_rf, model_svm, model_lstm, model_mlp, model_xgb, X_test, y_test, roc_figure='rocb1.png'):
    X_test = np.array(X_test)
    
    y_test_pred1 = model_rf.predict(X_test)
    y_test_score1 = model_rf.predict_proba(X_test)
    
    y_test_pred2 = model_svm.predict(X_test)
    y_test_score2 = model_svm.predict_proba(X_test)

    y_test_score3 = model_lstm.predict(X_test)
    y_test_pred3 = np.argmax(y_test_score3, axis=1)
    
    y_test_pred4 = model_mlp.predict(X_test)
    y_test_score4 = model_mlp.predict_proba(X_test)
    
    y_test_pred5 = model_xgb.predict(X_test)
    y_test_score5 = model_xgb.predict_proba(X_test)
    
    prediction_probability = np.array([[0., 0.]])
    for p1, p2, p3, p4, p5 in zip(y_test_score1, 
                                  y_test_score2, 
                                  y_test_score3, 
                                  y_test_score4, 
                                  y_test_score5):

        p = (0.20*p1 + 0.20*p2 + 0.20*p3 + 0.20*p4 + 0.20*p5)
        p = np.array([p])
        prediction_probability = np.concatenate((prediction_probability, p), axis=0)
    
    
    y_test_score0 = np.delete(prediction_probability, 0, 0)    
    y_test_pred0 = np.argmax(y_test_score0, axis=-1)
    
    y_test_preds = [y_test_pred1, y_test_pred2, y_test_pred3, y_test_pred4, y_test_pred5, y_test_pred0]
    
    for y_test_pred in y_test_preds:
        print('\n#Test set')
        print("##########################################")
        print("Test set Evaluation: Detail")
        print("#Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred, adjusted=False))
        print("#Accuary Score:", accuracy_score(y_test, y_test_pred))
        print("#Precision Score:", precision_score(y_test, y_test_pred, average='binary'))
        print("#Recall Score:", recall_score(y_test, y_test_pred, average='binary'))
        print("#F-score:", f1_score(y_test, y_test_pred, average='binary'))
        print("#Matthews_corrcoef:", matthews_corrcoef(y_test, y_test_pred))
        print("##########################################")
    
    # ROC-AUC
    y_test_scores = [y_test_score1, y_test_score2, y_test_score3, y_test_score4, y_test_score5, y_test_score0]
    names = ['RF', 'SVM', 'LSTM', 'MLP', 'XGB', 'Ensemble']
    colors = ['red', 'orange', 'green', 'cyan', 'blue', 'purple']
    plt.figure(figsize=(6, 6))
    
    for y_test_score, color, name in zip(y_test_scores, colors, names):
        fpr_test = dict()
        tpr_test = dict()
        roc_auc_test = dict()
    
        # Active
        fpr_test[1], tpr_test[1], _ = roc_curve(y_test, y_test_score[:, 1])
        roc_auc_test[1] = auc(fpr_test[1], tpr_test[1])
    
        plt.plot(fpr_test[1], tpr_test[1], color=color, linewidth=2, label=f"{name} (area = {roc_auc_test[1]:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis((-0.01, 1.01, -0.01, 1.01))
    plt.tick_params(labelsize=13)
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 15})
    plt.ylabel('True Positive Rate', fontdict={'fontsize': 15})
    plt.title('ROC-AUC curve on test set', fontdict={'fontsize': 16})
    plt.legend(loc="lower right")
    plt.savefig(f'{roc_figure}', dpi=300)
    plt.close()
    
    
def model_evaluation(estimator, X_test, y_test, roc_figure, cm_figure, deep=True):
    """

    :param y_test:
    :param X_test:
    :param estimator:
    :param save:
    :param model_path:
    :return:
    """
    if deep:
        y_test_score = estimator.predict(X_test)
        y_test_pred = np.argmax(y_test_score, axis=1)
    else:
        y_test_pred = estimator.predict(X_test)
        y_test_score = estimator.predict_proba(X_test)
        
    print()
    print('#Test set')
    print("##########################################")
    print("#Test set Evaluation: Overall")
    print(classification_report(y_test, y_test_pred, target_names=["Inactive", "Active"], digits=2))
    print()
    print("Test set Evaluation: Detail")
    print("#Accuary Score:", accuracy_score(y_test, y_test_pred))
    print("#Precision Score:", precision_score(y_test, y_test_pred, average='binary'))
    print("#Recall Score:", recall_score(y_test, y_test_pred, average='binary'))
    print("#F1-score:", f1_score(y_test, y_test_pred, average='binary'))
    print("##########################################")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig(f"{cm_figure}", dpi=300)
    plt.close()

    # ROC-AUC
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    # Active
    fpr_test[1], tpr_test[1], _ = roc_curve(y_test, y_test_score[:, 1])
    roc_auc_test[1] = auc(fpr_test[1], tpr_test[1])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_test[1], tpr_test[1], color='red', linewidth=2,
             label="ROC curve (area = {:.3f})".format(roc_auc_test[1]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis((-0.01, 1.01, -0.01, 1.01))
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 15})
    plt.ylabel('True Positive Rate', fontdict={'fontsize': 15})
    plt.title('ROC-AUC curve on test set', fontdict={'fontsize': 16})
    plt.legend(loc="lower right")
    plt.savefig(f'{roc_figure}', dpi=300)
    plt.close()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

dataset_path = './dataset/Pubchem_COVID-19_target_small_molecules_classification.csv'

rf_tuned_parameters = {'n_estimators': [100, 300, 500], 
                       'max_depth': [10, 20, 30, 40, 50],  
                       'min_samples_leaf': [1, 2, 3], 
                       'min_samples_split': [2, 3, 4]
                       }

svr_tuned_parameters = {'kernel': ['rbf', 'linear', 'poly'], 
                        'C': [0.1, 1, 10, 100]
                        }

LSTM_tuned_parameters = {'lstm_units': [64, 128, 256], 
                         'hidden_sizes':[256, 512, 1024], 
                         'dropout_rates':[0.0, 0.3, 0.5], 
                         'batch_sizes':[32, 64, 128], 
                         'learning_rates':[0.001, 0.005, 0.01]
                         }


mlp_tuned_parameters = {"hidden_layer_sizes": [(100,),(200,),(300,),(400,)] 
                        "learning_rate_init": [0.001, 0.005, 0.01]
                        }
                        
xgb_tuned_parameters = {"n_estimators": [100, 300, 500], 
                        "learning_rate": [0.001, 0.01, 0.1, 1], 
                        "max_depth": [5, 10, 15, 20, 30]
                        }

def main():
    print("Starting...")
    seed_everything()
    df = pd.read_csv(dataset_path)
    smi = df.iloc[:, 0].tolist()
    act = df.iloc[:, 1].tolist()

    X_train, X_test, y_train, y_test = dataset_sklearn(smiles_list=smi, labels=act, normalization=False, random_state=42)
    
    parameters_list = [rf_tuned_parameters, svr_tuned_parameters, mlp_tuned_parameters, xgb_tuned_parameters]
    
    model1 = RandomForestClassifier(random_state=42, n_jobs=-1)
    model2 = SVC(random_state=42, probability=True)
    model3 = MLPClassifier(random_state=42)
    model4 = XGBClassifier(random_state=42)
    
    models_list = [model1, model2, model3, model4]
    names_list = ['RF_model', 'SVR_model', 'MLP_model', 'XGB_model']
    
    print("\nInitialize...")
    best_parameters = []
    for parameter_space, model, name in zip(parameters_list, models_list, names_list):
        print(f"\nTraining {name}")
        estimator = GridSearchCV(model, parameter_space, cv=5, scoring='accuracy', n_jobs=-1)
        estimator.fit(X_train, y_train)
        best_parameters.append(estimator.best_params_)
        print(f"{name}:{estimator.best_params_}")
    
    
    model = RandomForestClassifier(**best_parameters[0], random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, 'RF_model.pkl')
    
    model = SVC(**best_parameters[1], random_state=42, probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, 'SVR_model.pkl')
    
    model = MLPClassifier(**best_parameters[2], random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'MLP_model.pkl')
    
    model = XGBClassifier(**best_parameters[3], random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'XGB_model.pkl')
    
    # Training Deep Learning model(LSTM)
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_sklearn(smiles_list=smi, labels=act, normalization=False, val_set=True, random_state=42)
    y_train_onehot = to_categorical(y_train)
    y_val_onehot = to_categorical(y_val)
    y_test_onehot = to_categorical(y_test)
    
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='max', restore_best_weights=True)
    
    # GridSearch on the LSTM model, define the parameter space for the parameter of LSTM.
    performance = []
    print(f"\nTraining LSTM_model")
    for params in product(LSTM_tuned_parameters['lstm_units'], 
                          LSTM_tuned_parameters['hidden_sizes'], 
                          LSTM_tuned_parameters['dropout_rates'], 
                          LSTM_tuned_parameters['batch_sizes'], 
                          LSTM_tuned_parameters['learning_rates'])
        lstm_unit = params[0]
        hidden_size = params[1]
        dropout_rate = params[2]
        batch_size = params[3]
        learning_rate = params[4]
        
        model = lstm_model(input_size=1024, lstm_unit=lstm_unit, hidden_size=hidden_size, dropout_rate=dropout_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
        history = model.fit(X_train, y_train_onehot, batch_size=batch_size, epochs=100, callbacks=[early_stopping], validation_data=(X_val, y_val_onehot), verbose=0)
        loss, accuracy = model.evaluate(X_val, y_val_onehot)
        performance.append([lstm_unit, hidden_size, dropout_rate, batch_size, learning_rate, loss, accuracy])
    
    data = pd.DataFrame(performance)
    data.columns = ['lstm_unit', 'hidden_size', 'dropout_rate', 'batch_size', 'learning_rate', 'loss', 'accuracy']
    data.sort_values(by="accuracy" , inplace=True, ascending=False) 
    data.to_csv('performance.csv')
    
    lstm_parameters = data.iloc[0,:].tolist()
    print(f"LSTM_model, lstm_unit:{lstm_parameters[0]}, hidden_size:{lstm_parameters[1]}, dropout_rate:{lstm_parameters[2]}, learning_rate:{lstm_parameters[4]}, batch_size:{lstm_parameters[3]}")
    model = lstm_model(input_size=1024, lstm_unit=lstm_parameters[0], hidden_size=lstm_parameters[1], dropout_rate=lstm_parameters[2])
    opt = optimizers.Adam(learning_rate=lstm_parameters[4])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, y_train_onehot, batch_size=lstm_parameters[3], epochs=100, callbacks=[early_stopping], verbose=1)
    model.evaluate(X_val, y_val_onehot)
    model.evaluate(X_test, y_test_onehot)
    model.save('LSTM_model.h5')
    
    # load the ML models with the model path.
    rf_path = './RF_model.pkl'
    svm_path = './SVM_model.pkl'
    lstm_path = './LSTM_model.h5'
    mlp_path = './MLP_model.pkl'
    xgb_path = './XGB_model.pkl'
    
    model_rf = joblib.load(rf_path)
    model_svm = joblib.load(svm_path)
    model_lstm = keras.models.load_model(lstm_path)
    model_mlp = joblib.load(mlp_path)
    model_xgb = joblib.load(xgb_path)
    
    # Evaluating the models
    print("\nEvaluating...")
    ensemble_model_evaluation(model_rf, model_svm, model_lstm, model_mlp, model_xgb, X_test, y_test)
    model_evaluation(model_rf, X_test, y_test, roc_figure='roc1.png', cm_figure='cm1.png', deep=False)
    model_evaluation(model_svm, X_test, y_test, roc_figure='roc2.png', cm_figure='cm2.png', deep=False)
    model_evaluation(model_lstm, X_test, y_test, roc_figure='roc3.png', cm_figure='cm3.png', deep=True)
    model_evaluation(model_mlp, X_test, y_test, roc_figure='roc4.png', cm_figure='cm4.png', deep=False)
    model_evaluation(model_xgb, X_test, y_test, roc_figure='roc5.png', cm_figure='cm5.png', deep=False)
    print("All models training is done! please move the ML models into the models folder")

    
    
if __name__ == '__main__':
    main()