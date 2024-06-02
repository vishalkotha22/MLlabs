from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

(trainX, trainy), (testX, testy) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainY = to_categorical(trainy)
testY = to_categorical(testy)

def prep_pixels(train, test):
 train_norm = train.astype('float32')
 test_norm = test.astype('float32')
 train_norm = train_norm / 255.0
 test_norm = test_norm / 255.0
 return train_norm, test_norm

trainX, testX = prep_pixels(trainX, testX)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history=model.fit(trainX, trainY, batch_size=64, epochs=8, validation_data=(testX, testY))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(history.history['accuracy'])
print(history.history['val_accuracy'])