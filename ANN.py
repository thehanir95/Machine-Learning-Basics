import pandas as pd

# read csv file
dataset = pd.read_csv("Finalized_V2.csv")

x = dataset.iloc[:, 0:17]
y = dataset.iloc[:, 17]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

from keras import Sequential
from keras.layers import Dense

classifier = Sequential()

# adding the first input layer along with the hidden layer
classifier.add(Dense(input_dim=17, activation="relu", init="uniform", output_dim=9))

classifier.add(Dense(output_dim=9, init="uniform", activation="relu"))

classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentrophy", metrics=["accuracy"])

classifier.fit(x_train, y_train, batch_size=10, nb_epoch=20)

