from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import time
classifier=Sequential()
classifier.add(Convolution2D(64, 3, 3, input_shape=(64,64,3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 64, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(classifier.summary())

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set= train_datagen.flow_from_directory('flowers/train_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='categorical')
test_set= test_datagen.flow_from_directory('flowers/test_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='categorical')
classifier.fit_generator(training_set,
                         samples_per_epoch = 3300,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples=1000)
classifier.save('classifier1.h5')

# from keras.models import load_model
# import numpy as np
# from keras.preprocessing import image
# classifier1=load_model('classifier.h5')
# test_image = image.load_img('2.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier1.predict(test_image)
# # result=training_set.class_indices
# print(result)
# if result[0][0] == 0:
#     prediction = 'daisy'
# elif result[0][0]==1:
#     prediction = 'dandellions'
# elif result[0][0]==2:
#     prediction='rose'
# elif result[0][0]==3:
#     prediction='sunflower'
# else:
#     prediction='tulip'
#
# print(prediction)
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# dimensions of our images
img_width, img_height = 64, 64
# load the model we saved
#model = load_model('classifier.h5')
#model.compile(loss='binary_crossentropy',
 #             optimizer='adam',
  #            metrics=['accuracy'])
# predicting images
img = image.load_img('dan4.jpg', target_size=(img_width, img_height))
img= image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
classes = classifier.predict_classes(img)
print(classes)