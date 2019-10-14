import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from keras.utils import to_categorical

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


np.unique(y_train)
model = Sequential()
n_input = X_train.shape[1]
n_hidden = n_input
n_output = 3
model.add(Dense(n_hidden, input_dim=n_input, activation='relu'))
model.add(Dense(n_output, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=400, batch_size=None, verbose=0)

#plot the train and test loss over all 400 epochs
train_loss = history.history['loss']
test_loss = history.history['val_loss']
# plt.plot(train_loss, label='Training loss')
# plt.plot(test_loss, label='Testing loss')
# plt.legend()
