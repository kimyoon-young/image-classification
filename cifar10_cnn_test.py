'''
#Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

#pip install tensorflow keras matplotlib sklearn  scipy  pillow
# pillow  PIL , convert numpy to PIL images


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

# for comfusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from vis import plot_confusion_matrix  as conf_mat
from vis import show_images



# ipmort 모듈
# form 모듈 ipmort 변수나 함수

# from
# import


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = False
num_predictions = 20


checkpoint_dir = './checkpoint/'


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_test_list = np.argmax(y_test, axis=1)



model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


#model_path = os.path.join(save_dir, model_name)
#model.load_model(model_path)
#model = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(checkpoint_dir + '03-0.5411.hdf5')
print('loaded trained model at %s ' % checkpoint_dir)



# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# Score trained model.
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])



#Confution Matrix and Classification Report
Y_pred = model.predict(x_test, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')

# labels=["ant", "bird", "cat"]
# 순서

print(confusion_matrix(y_test_list, y_pred))

print('Classification Report')
print(classification_report(y_test_list, y_pred))





class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

np_class = np.asarray(class_names)

# Plot normalized confusion matrix
conf_mat.plot_confusion_matrix(y_test_list, y_pred, classes=np_class, normalize=True,
                      title='Normalized confusion matrix')


conf_mat.plot_confusion_matrix(y_test_list, y_pred, classes=np_class, normalize=False,
                      title='confusion matrix')




plt.show()