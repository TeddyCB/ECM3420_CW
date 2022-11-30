import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.metrics import RootMeanSquaredError
from keras.backend import clear_session
# Log return function
def lg_return(df):
    lg_return_list = []
    up_down = []
    for index, row in df.iterrows():
        # use lg return formula
        daily_return = (np.log(row["Close"]) - np.log(row["Open"]))
        lg_return_list.append(daily_return)
        if daily_return > 0:
            up_down.append(1)
        else:
            up_down.append(0)
    return lg_return_list, up_down

def even_out_frames(df1, df2):
    
    df1_index = df1.index.to_numpy()
    for index, row in df2.iterrows():
        if index not in df1_index:
            df2.drop(index=index, inplace=True)
    
    df2_index = df2.index.to_numpy()
    for index, row in df1.iterrows():
        if index not in df2_index:
                df1.drop(index=index, inplace=True)
    return 0

def LSTM_GRU(LSTM_layer, GRU_layer,  input_shape):
    model = Sequential()
    model.add(LSTM(LSTM_layer, return_sequences=True, input_shape= input_shape))
    model.add(GRU(GRU_layer, return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def LSTM_RNN(LSTM_layer, input_shape):
    model = Sequential()
    model.add(LSTM(LSTM_layer, return_sequences=False, input_shape= input_shape))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def GRU_RNN(GRU_layer, input_shape):
    model = Sequential()
    model.add(GRU(GRU_layer, return_sequences=False, input_shape=input_shape))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def train(x, y, x_val, y_val, model, epochs, loss):
    model.compile(optimizer='adam', loss=loss, metrics=[RootMeanSquaredError()])
    hist = model.fit(x, y, batch_size= 1, epochs=epochs, validation_data=(x_val, y_val))
    return hist

def format_df(arr, columns):
    data = {}
    for i in range(len(columns)):
        data[columns[i]] = arr[:, i]
    return pd.DataFrame(data=data)

def inverse(x,y, min_max_scaler):
    final = np.append(x, y.reshape(y.shape[0],1,1), axis=1)
    final = final.reshape(x.shape[0],x.shape[1]+1)
    final_trans = min_max_scaler.inverse_transform(final)
    return final_trans

def evaluate(X_train, Y_train, X_test, Y_test, rnn_type, epochs, iterations, scaler):
    predictions = [] # list of predictions
    model_history = [] # list of model history
    prediction_error = [] # list of prediction error
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    for i in range(1, iterations + 1):
        if rnn_type == "GRU":
            model = GRU_RNN(128, input_shape=(X_train.shape[1], 1))
        elif rnn_type == "LSTM":
            model = LSTM_RNN(128, input_shape=(X_train.shape[1], 1))
        else: 
            model = LSTM_GRU(128, input_shape=(X_train.shape[1], 1))
        # train the data against the RMSE and validate the data against the test data
        hist = train(X_train, Y_train, X_test, Y_test, model, epochs, loss="mean_squared_error")
        model_history.append(hist.history)

        # predict the future data
        prediction = model.predict(X_test)

        # format the predictions into a data frame and add to the list of predictions
        prediction_df = format_df(inverse(X_test, prediction, scaler), columns=["Open", "High", "Low", "Close","Volume", "Prev Tokyo Close", "Predicted Next Tokyo Open"])
        prediction_df.insert(6, "Actual Next Tokyo Open", inverse(X_test, Y_test, scaler)[:, 6])
        predictions.append(prediction_df)
        # add the RMSE of the final epoch to the prediction error list
        prediction_error.append(mean_squared_error(Y_test, prediction, squared=False))
        model = None
    return predictions, model_history, prediction_error

def group_by_epoch(model_history, epoch):
    epoch_group = {}
    for history in model_history:
        for k in history.keys():
            if k not in epoch_group:
                epoch_group[k] = [history[k][: epoch]]
                continue
            epoch_group[k].append(history[k][: epoch])
    return epoch_group


def plot_dfs_in_range(lower, upper, data):
    ax = None
    c = 0
    for prediction in data:
        if ax is None:
            ax = prediction[(prediction.index > lower) & (prediction.index < upper)].plot(y="Actual Next Tokyo Open")
            c += 1
            continue
        prediction[(prediction.index > lower) & (prediction.index < upper)].plot(ax=ax, y="Predicted Next Tokyo Open", linestyle = "dashed")
        c += 1
    
    return ax