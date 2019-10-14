#There will be several print statements commented
# out in this code.  They will be helpful in your
# attempt

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
y_train = y_train*-1 + 1
y_test = y_test*-1 + 1
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#X_train.shape
#take a look at the shape of the data

# Start compiling the layers
model = Sequential()
n_input = X_train.shape[1]
n_hidden = n_input
model.add(Dense(n_hidden, input_dim=n_input, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=None, verbose=0)

#plot the train and test loss over all 100 epochs
# train_loss = history.history['loss']
# test_loss = history.history['val_loss']
# plt.plot(train_loss, label='Training loss')
# plt.plot(test_loss, label='Testing loss')
