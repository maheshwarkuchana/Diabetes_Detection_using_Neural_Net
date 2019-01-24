import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df = pd.read_csv("pima-indians-diabetes-data.csv")

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train = np_utils.to_categorical(y_train)

# Using Keras Neural Networks--Model Training
model = Sequential()
model.add(Dense(units=32, input_dim=8, activation='relu'))
model.add(Dense(units=19, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size=10)

filename = "Trained_Model.pickle"
pickle.dump(model, open(filename, 'wb'))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(accuracy_score(y_test, y_pred))

plt.plot(model.history.epoch, model.history.history['loss'])
plt.show()
