import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

def main():

    number_of_classes = len(os.listdir('Data')) - 1

    # Load data from the CSV file
    df = pd.read_csv('hand_landmarks.csv', header=None)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, 1:].values, df.iloc[:, 0].values, test_size=0.2, random_state=42)

    # Normalize the feature values to between 0 and 1
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert the labels to one-hot encoded vectors
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=number_of_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=number_of_classes)

    # Define the model architecture
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(42,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, batch_size=32, epochs= 1550, validation_data=(X_test, y_test))

    # Evaluate the model on the testing set
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')

    # Save the trained model to a file
    model.save('hand_gesture_model.h5')

