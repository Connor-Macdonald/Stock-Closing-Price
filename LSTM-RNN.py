import fix_yahoo_finance as yf
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import sklearn.preprocessing as prep



# Loads IBM's stock data using the Yahoo Finance API and sorts the columns
# Return the sorted data

def load_data() : 
    df = yf.download("ibm", start="2013-03-17", end="2018-03-11")
    df.to_csv('ibm.csv', sep=',', encoding='utf-8', index=False)
    data = pd.read_csv('ibm.csv')

    col_list = data.columns.tolist()
    print(col_list)

    col_list.remove('Close')
    col_list.append('Close')

    print(col_list)

    data = data[col_list]
    return data

# Scales the data from 0-1
# Seperates training and test data and X and Y data for each
def preprocess(x, step_back):
   
    amount_of_features = len(x.columns)
    data = prep.MinMaxScaler().fit_transform(x)
    data = np.flipud(data)
    step_back_length = step_back + 1
    result = []
    for index in range(len(data) - step_back_length):
        result.append(data[index : index + step_back_length])
        
    result = np.array(result)
    row = round(0.8 * result.shape[0])
    train = result[: int(row), :]

    X_train = train[:, : -1]
    y_train = train[:, -1][: ,-1]
    print(train.shape)
    X_test = result[int(row) :, : -1]
    y_test = result[int(row) :, -1][ : ,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

    return [X_train, y_train, X_test, y_test]

# Builds a 2 layer Long Short Term Memeory Reccurent Neural Network
def build_model(layers):
    model = Sequential()
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("sigmoid"))
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    return model

stepBack = 20 
data = load_data()
X_train, y_train, X_test, y_test = preprocess(data[::-1], stepBack)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

model = build_model([X_train.shape[2], stepBack, 100, 1])

# Training Network
model.fit(
    X_train,
    y_train,
    batch_size=200,
    epochs=1500,
    validation_split=0.1,
    verbose=1)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


pred = model.predict(X_test)
plt.plot(pred, color='red', label='Prediction')
plt.plot(y_test, color='blue', label='Real')
plt.legend(loc='upper left')
plt.show()

predData = pd.DataFrame(pred)
predData.to_csv("output.csv")
realData = pd.DataFrame(y_test)
realData.to_csv("desired.csv")