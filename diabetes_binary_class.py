#There will be several print statements commented
# out in this code.  They will be helpful in your
# attempt

import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_diabetes()

#plt.hist(data.target)
#plots distribution of target values

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#X_train.shape
#take a quick look at the shape of the training datasets

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

#put training data into a dataframe for eda
df_train = pd.DataFrame(X_train, columns=[i for i in range(10)])
#dr_train.head()

input_units = X_train.shape[1]
hidden_units = input_units
model.add(Dense(hidden_units, input_dim=input_units, activation='relu'))
model.add(Dense(1))

from keras.optimizers import Adam
adam = Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=10, batch_size=None)
