import shap
import numpy as np

def voting_predict(models_list, models_type, X):
    sum_predict = np.array([[0., 0.]])
    num_models = len(models_type)

    for model, model_type in zip(models_list, models_type):
        if model_type == "sklearn":
            predict = model.predict_proba(X)
            # print(predicted)
            # print(type(predicted))

        elif model_type == "keras":
            predict = model.predict(X)
            # print(predicted)
            # print(type(predicted))

        else:
            print("Wrong model type of input models!")
            exit()

        sum_predict = sum_predict + predict

    prob_predict = sum_predict / num_models
    label_predict = np.argmax(prob_predict, axis=-1)

    return label_predict, prob_predict


def interpretation(models_list, models_type, explainers_type, X, X_original, num_sample=100):
    # Setting the background dataset
    X_sample = np.array(shap.sample(X_original, num_sample, random_state=42))

    # Predict the explainer
    sum_predict = np.array([[0., 0.]])
    indexes = []
    for model, model_type in zip(models_list, models_type):
        if model_type == 'sklearn':
            predict = model.predict_proba(X)

        elif model_type == 'keras':
            predict = model.predict(X)

        else:
            print("Wrong model type of input models!")
            exit()

        sum_predict = sum_predict + predict
        idx = np.argmax(predict)
        indexes.append(idx)

    mean_prediction = sum_predict / len(models_type)

    # Get the right index of model
    index = np.argmax(mean_prediction)

    # Create the explainer for each model
    shap_values_list = []
    for model, model_type, explainer_type in zip(models_list, models_type, explainers_type):
        if model_type == 'sklearn' and explainer_type == 'Kernel':
            explainer = shap.KernelExplainer(model.predict_proba, data=X_sample)
            shap_values = explainer.shap_values(X, silent=True)

        elif model_type == 'keras' and explainer_type == 'Kernel':
            explainer = shap.KernelExplainer(model.predict, data=X_sample)
            shap_values = explainer.shap_values(X, silent=True)

        elif model_type == 'sklearn' and explainer_type == 'Tree':
            explainer = shap.TreeExplainer(model, data=X_sample, model_output='predict_proba')
            shap_values = explainer.shap_values(X)

        elif model_type == 'keras' and explainer_type == 'Tree':
            explainer = shap.TreeExplainer(model, data=X_sample)
            shap_values = explainer.shap_values(X)

        else:
            print("Wrong model type of input models!")
            exit()

        shap_values_list.append(shap_values)

    active_groups = [shap_values[1][0] for shap_values in shap_values_list]
    inactive_groups = [shap_values[0][0] for shap_values in shap_values_list]

    if index == 1:
        sv = np.zeros(active_groups[0].shape)
        n = 0
        for idx, ag in zip(indexes, active_groups):
            if idx == index:
                sv = sv + ag
                n = n + 1
            else:
                continue

        sv = sv / n

    else:
        sv = np.zeros(inactive_groups[0].shape)
        n = 0
        for idx, ig in zip(indexes, inactive_groups):
            if idx == index:
                sv = sv + ig
                n = n + 1
            else:
                continue

        sv = sv / n

    return sv
    