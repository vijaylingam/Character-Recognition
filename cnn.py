#CNN for Handwritten digit recognition
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy
import matplotlib.image as mpimg
import itertools


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plt.imshow(X_train[11], cmap=plt.get_cmap('gray'))
# plt.show()
#print(type(X_train[11]))
# print(X_train[11].shape[0])
# print(X_train[11].shape[1])

# print(y_train[11])

# X_train = X_train.reshape(X_train.shape[0], 784)
# X_test = X_test.reshape(X_test.shape[0], 784)

print("shape", X_train[0].shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print("Shape: ", X_train[0].shape)
exit(1)

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# #print(y_train[11])

img=mpimg.imread('2.jpg') # path to image
img = img/ 255.0
#img = img.reshape(1, 784)
img = img.reshape(1, 28,28,1)

# # model = Sequential()
# # model.add(Dense(784, input_dim=784, kernel_initializer='normal', activation='relu')) 
# # model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2, batch_size=32)
# # model.summary()
# # l = model.predict(img)[0]
# # m = max(l)
# # print("Predicted Value is: ", [i for i, j in enumerate(l) if j == m][0])

model = Sequential()
model.add(Conv2D(32, (5, 5), padding='valid', input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2, batch_size=32)
l = model.predict(img)[0]
m = max(l)
print("Predicted Value is: ", [i for i, j in enumerate(l) if j == m][0])
