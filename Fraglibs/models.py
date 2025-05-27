from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from keras import layers, Model


def rf_model(parameters, random_state: int = 42):
    model = RandomForestRegressor(**parameters, random_state=random_state, n_jobs=-1)
    return model


def svr_model(parameters):
    model = SVR(**parameters)
    return model


def mlp_model(parameters, random_state: int = 42):
    model = MLPRegressor(**parameters, random_state=random_state)
    return model


def xgb_model(parameters, random_state: int = 42):
    model = XGBRegressor(**parameters, random_state=random_state)
    return model


def deep_model(input_size: int, lstm_unit: int, hidden_size: int, dropout_rate: float):
    inputs = layers.Input(shape=input_size)
    x = layers.Reshape((1, input_size))(inputs)
    x, state_h, state_c = layers.LSTM(lstm_unit, return_state=True, return_sequences=True)(x)
    x, _, _ = layers.LSTM(lstm_unit, return_state=True)(x, [state_h, state_c])
    x = layers.Dense(hidden_size, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(lstm_unit, activation='linear')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def lstm_model(input_size: int, lstm_unit: int, hidden_size: int, dropout_rate: float):
    inputs = layers.Input(shape=input_size)
    x = layers.Reshape((1, input_size))(inputs)
    x, state_h, state_c = layers.LSTM(lstm_unit, return_state=True, return_sequences=True)(x)
    x = x[:, -1, :]
    # x = layers.Dropout(dropout_rate)(x)
    # x, _, _ = layers.LSTM(lstm_unit, return_state=True)(x, [state_h, state_c])
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_size, activation='tanh')(x)
    # x = layers.Activation('tanh')(x)
    x = layers.Dropout(dropout_rate)(x)
    # x = layers.Dense(lstm_unit, activation='linear')(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
