import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError,MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
df = pd.read_excel("C:/Users/AJAY/Mini Project/rainfall&discharge chimoni.xlsx")

df.info()
df.isnull().sum()
df.describe()
plt.plot(df.Date.head(10),df['Rainfall(mm)'].head(10),linewidth=3)
plt.show()
df1 = df.copy(deep=True)
df1.loc[0:1836,'Date'] = pd.to_datetime(df1.loc[0:1836,'Date'], format = "%Y-%m-%d %H:%M:%S")
df1.loc[1837:3544,'Date'] = pd.to_datetime(df1.loc[1837:3544,'Date'], format = "%Y.%m.%d ")
df1.loc[4012:4611,'Date'] = pd.to_datetime(df1.loc[4012:4611,'Date'], format = "%d.%m.%Y")
df1.loc[4612:,'Date'] = pd.to_datetime(df1.loc[4612:,'Date'], format = "%Y-%m-%d %H:%M:%S")
df1.Date.info()
df1.Date.astype('str')
df1['Date']=pd.to_datetime(df1['Date'], format = "%Y-%m-%d %H:%M:%S")
df1.Date.min(),df1.Date.max(),print(df1.Date.max()-df1.Date.min())
df1.set_index('Date',inplace=True)
# @title Rainfall(mm)

df1['Rainfall(mm)'].plot(kind='line', figsize=(12, 5), title='Rainfall(mm)')
plt.gca().spines[['top', 'right']].set_visible(False)
# @title Total discharge (m3/sec)
df1['Total discharge (m3/sec)'].plot(kind='line', figsize=(12, 5), title='Total discharge (m3/sec)')
plt.gca().spines[['top', 'right']].set_visible(False)
df1= df1[['Rainfall(mm)','Total discharge (m3/sec)']]
pearson_corr_predictor1 = df1['Rainfall(mm)'].corr(df1['Total discharge (m3/sec)']) #pearson

# Calculate Spearman correlation coefficient
spearman_corr_predictor1 =  df1['Rainfall(mm)'].corr(df1['Total discharge (m3/sec)'], method='spearman')

print("Pearson correlation between predictor1 and target:", pearson_corr_predictor1)

print("Spearman correlation between predictor1 and target:", spearman_corr_predictor1)

def series_to_supervised(data, n_in, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

values = df1.values
data_l1=series_to_supervised(values,1)
#print(data_l1)

data_l2 = series_to_supervised(values,2)
#print(data_l2.shape)

data_l7 = series_to_supervised(values,7)
#print(data_l7.shape)

data_l12 = series_to_supervised(values,12)
#print(data_l12.shape)

# split data with lag=1 into train, validation, and test sets
values_l1 = data_l1.values
l1_train_days = int(4684 * 0.7)
l1_val_days = int(4684 * 0.15) #size of the validation dataset
train_l1 = values_l1[:l1_train_days, :]
val_l1 = values_l1[l1_train_days:l1_train_days + l1_val_days, :]
test_l1 = values_l1[l1_train_days + l1_val_days:, :]

# split into input and outputs
l1_train_X, l1_train_y = train_l1[:, :-1], train_l1[:, -1]
l1_val_X, l1_val_y = val_l1[:, :-1], val_l1[:, -1]
l1_test_X, l1_test_y = test_l1[:, :-1], test_l1[:, -1]

# reshape input to be 3D [samples, timesteps, features]
l1_train_X = l1_train_X.reshape((l1_train_X.shape[0], 1, l1_train_X.shape[1]))
l1_val_X = l1_val_X.reshape((l1_val_X.shape[0], 1, l1_val_X.shape[1]))
l1_test_X = l1_test_X.reshape((l1_test_X.shape[0], 1, l1_test_X.shape[1]))

#print(l1_train_X.shape, l1_train_y.shape, l1_val_X.shape, l1_val_y.shape, l1_test_X.shape, l1_test_y.shape)
#print(l1_train_X)

# split data with lag=2 into train, validation, and test sets
values_l2 = data_l2.values
l2_train_days = int(4684 * 0.7)
l2_val_days = int(4684 * 0.15) #size of the validation dataset
train_l2 = values_l2[:l2_train_days, :]
val_l2 = values_l2[l2_train_days:l2_train_days + l2_val_days, :]
test_l2 = values_l2[l2_train_days + l2_val_days:, :]

# split into input and outputs
l2_train_X, l2_train_y = train_l2[:, :-1], train_l2[:, -1]
l2_val_X, l2_val_y = val_l2[:, :-1], val_l2[:, -1]
l2_test_X, l2_test_y = test_l2[:, :-1], test_l2[:, -1]

# reshape input to be 3D [samples, timesteps, features]
l2_train_X = l2_train_X.reshape((l2_train_X.shape[0], 1, l2_train_X.shape[1]))
l2_val_X = l2_val_X.reshape((l2_val_X.shape[0], 1, l2_val_X.shape[1]))
l2_test_X = l2_test_X.reshape((l2_test_X.shape[0], 1, l2_test_X.shape[1]))

#print(l2_train_X.shape, l2_train_y.shape, l2_val_X.shape, l2_val_y.shape, l2_test_X.shape, l2_test_y.shape)

# split data with lag=7 into train, validation, and test sets
values_l7 = data_l7.values
l7_train_days = int(4684 * 0.7)
l7_val_days = int(4684 * 0.15) #size of the validation dataset
train_l7 = values_l7[:l7_train_days, :]
val_l7 = values_l7[l7_train_days:l7_train_days + l7_val_days, :]
test_l7 = values_l7[l7_train_days + l7_val_days:, :]

# split into input and outputs
l7_train_X, l7_train_y = train_l7[:, :-1], train_l7[:, -1]
l7_val_X, l7_val_y = val_l7[:, :-1], val_l7[:, -1]
l7_test_X, l7_test_y = test_l7[:, :-1], test_l7[:, -1]

# reshape input to be 3D [samples, timesteps, features]
l7_train_X = l7_train_X.reshape((l7_train_X.shape[0], 1, l7_train_X.shape[1]))
l7_val_X = l7_val_X.reshape((l7_val_X.shape[0], 1, l7_val_X.shape[1]))
l7_test_X = l7_test_X.reshape((l7_test_X.shape[0], 1, l7_test_X.shape[1]))

#print(l7_train_X.shape, l7_train_y.shape, l7_val_X.shape, l7_val_y.shape, l7_test_X.shape, l7_test_y.shape)

# split data with lag=12 into train, validation, and test sets
values_l12 = data_l12.values
l12_train_days = int(4684 * 0.7)
l12_val_days = int(4684 * 0.15) #size of the validation dataset
train_l12 = values_l12[:l12_train_days, :]
val_l12 = values_l12[l12_train_days:l12_train_days + l12_val_days, :]
test_l12 = values_l12[l12_train_days + l12_val_days:, :]

# split into input and outputs
l12_train_X, l12_train_y = train_l12[:, :-1], train_l12[:, -1]
l12_val_X, l12_val_y = val_l12[:, :-1], val_l12[:, -1]
l12_test_X, l12_test_y = test_l12[:, :-1], test_l12[:, -1]

# reshape input to be 3D [samples, timesteps, features]
l12_train_X = l12_train_X.reshape((l12_train_X.shape[0], 1, l12_train_X.shape[1]))
l12_val_X = l12_val_X.reshape((l12_val_X.shape[0], 1, l12_val_X.shape[1]))
l12_test_X = l12_test_X.reshape((l12_test_X.shape[0], 1, l12_test_X.shape[1]))

#print(l12_train_X.shape, l12_train_y.shape, l12_val_X.shape, l12_val_y.shape, l12_test_X.shape, l12_test_y.shape)

#lag=1
model_l1 = Sequential()
model_l1.add(LSTM(64, input_shape=(l1_train_X.shape[1], l1_train_X.shape[2])))
model_l1.add(Dense(1))
model_l1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001),metrics=[RootMeanSquaredError()])
# fit network
history_l1 = model_l1.fit(l1_train_X, l1_train_y, epochs=125, batch_size=64,validation_data=(l1_val_X, l1_val_y), verbose=2, shuffle=False)
# plot history
#plt.plot(history_l1.history['loss'], label='train')
#plt.plot(history_l1.history['val_loss'], label='test')
#plt.legend()
#plt.show()
import math
from sklearn.metrics import mean_squared_error,r2_score
# make a prediction
ypred_l1 = model_l1.predict(l1_test_X).flatten()
l1_test_y = l1_test_y.reshape((len(l1_test_y), 1)).flatten()

# calculate RMSE
mse_l1 = mean_squared_error(l1_test_y, ypred_l1)
rmse_l1 = math.sqrt(mean_squared_error(l1_test_y, ypred_l1))
r2_l1 = r2_score(l1_test_y, ypred_l1)
#print('Test RMSE: %.3f' % rmse_l1)
#print('test mse:',mse_l1)
#print('test r2 score: ',r2_l1)
def create_df(ypred, testy):
    df=pd.DataFrame()
    testy = pd.Series(testy)
    ypred = pd.Series(ypred)
    cols=[testy,ypred]
    df = pd.concat(cols,axis=1)
    df.columns = ["Actual_y","Pred_y"]
    return df

l1_test_results = create_df(ypred_l1, l1_test_y)
#l1_test_results

#lag=2
model_l2 = Sequential()
model_l2.add(LSTM(64, input_shape=(l2_train_X.shape[1], l2_train_X.shape[2])))
model_l2.add(Dense(1))
model_l2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001),metrics=[RootMeanSquaredError()])
# fit network
history_l2 = model_l2.fit(l2_train_X, l2_train_y, epochs=125, batch_size=64,validation_data=(l2_val_X, l2_val_y), verbose=2, shuffle=False)
# plot history
#plt.plot(history_l2.history['loss'], label='train')
#plt.plot(history_l2.history['val_loss'], label='test')
#plt.legend()
#plt.show()

# make a prediction
ypred_l2 = model_l2.predict(l2_test_X).flatten()
l2_test_y = l2_test_y.reshape((len(l2_test_y), 1)).flatten()

# calculate RMSE
rmse_l2 = math.sqrt(mean_squared_error(l2_test_y, ypred_l2))
mse_l2 = mean_squared_error(l2_test_y, ypred_l2)
r2_l2 = r2_score(l2_test_y, ypred_l2)
#print('Test RMSE: %.3f' % rmse_l2)
#print('test mse:',mse_l2)
#print('test r2 score: ',r2_l2)
l2_test_results = create_df(ypred_l2, l2_test_y)
#l2_test_results.head(100).plot(alpha=0.6)

#lag=7

model_l7 = Sequential()
model_l7.add(LSTM(64, input_shape=(l7_train_X.shape[1], l7_train_X.shape[2])))
model_l7.add(Dense(1))
model_l7.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001),metrics=[RootMeanSquaredError()])
# fit network
history_l7 = model_l7.fit(l7_train_X, l7_train_y, epochs=150, batch_size=64,validation_data=(l7_val_X, l7_val_y), verbose=2, shuffle=False)
# plot history
#plt.plot(history_l7.history['loss'], label='train')
#plt.plot(history_l7.history['val_loss'], label='test')
#plt.legend()
#plt.show()

# make a prediction
ypred_l7 = model_l7.predict(l7_test_X).flatten()
l7_test_y = l7_test_y.reshape((len(l7_test_y), 1)).flatten()

# calculate RMSE
rmse_l7 = math.sqrt(mean_squared_error(l7_test_y, ypred_l7))
mse_l7 = mean_squared_error(l7_test_y, ypred_l7)
r2_l7 = r2_score(l7_test_y, ypred_l7)
l7_test_results = create_df(ypred_l7, l7_test_y)
l7_test_results.head(100).plot(alpha=0.6)
#print('Test RMSE: %.3f' % rmse_l7)
#print('test mse:',mse_l7)
#print('test r2 score: ',r2_l7)

#lag=12
model_l12 = Sequential()
model_l12.add(LSTM(64, input_shape=(l12_train_X.shape[1], l12_train_X.shape[2])))
model_l12.add(Dense(1))
model_l12.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001),metrics=[RootMeanSquaredError()])
# fit network
history_l12 = model_l12.fit(l12_train_X, l12_train_y, epochs=218, batch_size=64,validation_data=(l12_val_X, l12_val_y), verbose=2, shuffle=False)
# plot history
#plt.plot(history_l12.history['loss'], label='train')
#plt.plot(history_l12.history['val_loss'], label='test')
#plt.legend()
#plt.show()

# make a prediction
ypred_l12 = model_l12.predict(l12_test_X).flatten()
l12_test_y = l12_test_y.reshape((len(l12_test_y), 1)).flatten()

# calculate RMSE
rmse_l12 = math.sqrt(mean_squared_error(l12_test_y, ypred_l12))
mse_l12 = mean_squared_error(l12_test_y, ypred_l12)
r2_l12 = r2_score(l12_test_y, ypred_l12)
l12_test_results = create_df(ypred_l12, l12_test_y)
l12_test_results.head(100).plot(alpha=0.6)
#print('Test RMSE: %.3f' % rmse_l12)
#print('test mse:',mse_l12)
#print('test r2 score: ',r2_l12)

models = ['t-1', 't-2', 't-7', 't-12']
mse_scores = [mse_l1, mse_l2, mse_l7, mse_l12]  # Mean Squared Error
rmse_scores = [rmse_l1, rmse_l2, rmse_l7, rmse_l12]
r2_scores = [r2_l1, r2_l2, r2_l7, r2_l12]  # R2 Score

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Mean Squared Error plot
axes[0].bar(models, mse_scores, color='skyblue')
axes[0].set_title('Mean Squared Error')
axes[0].set_xlabel('Models', fontsize=12)
axes[0].set_ylabel('MSE', fontsize=12)
axes[0].tick_params(axis='x', rotation=90, labelsize=10)

# Mean Absolute Error plot
axes[1].bar(models, rmse_scores, color='salmon')
axes[1].set_title('Root Mean Squared Error')
axes[1].set_xlabel('Models', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].tick_params(axis='x', rotation=90, labelsize=10)

# R2 Score plot
axes[2].bar(models, r2_scores, color='lightgreen')
axes[2].set_title('R2 Score')
axes[2].set_xlabel('Models', fontsize=12)
axes[2].set_ylabel('R2 Score', fontsize=12)
axes[2].tick_params(axis='x', rotation=90, labelsize=10)

plt.tight_layout()
plt.show() 
