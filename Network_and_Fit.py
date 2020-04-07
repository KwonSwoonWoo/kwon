import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import numpy as np
from FileMake import *
from keras_preprocessing.image import ImageDataGenerator

#모든 이미지를 1/255로 스케일 조정
train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)
validation_data = ImageDataGenerator(rescale=1./255)

train_dir = './train'
test_dir = './test'
validation_dir = './validation'

train_generator = train_data.flow_from_directory(
    train_dir, # 타겟 디렉토리(학습 디렉토리)
    target_size = (150,150), 
    batch_size=3,
    class_mode='categorical')

test_generator = test_data.flow_from_directory(
    test_dir, # 타겟 디렉토리(validation 디렉토리) 
    target_size = (150,150), 
    batch_size=3,
    class_mode='categorical')

validation_generator = validation_data.flow_from_directory(
    validation_dir, # 타겟 디렉토리(validation 디렉토리) 
    target_size = (150,150), 
    batch_size=3,
    class_mode='categorical')

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='selu', padding='same', strides= 1,
            input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2), padding='same'))

model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64,(3,3),activation='selu', padding='same',strides=1))
model.add(layers.MaxPool2D((2,2), padding='same'))

model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(128,(3,3),activation='selu', padding='same',strides=1))
model.add(layers.MaxPool2D((2,2), padding='same'))

model.add(layers.Conv2D(128,(3,3),activation='selu', padding='same',strides=1))
model.add(layers.MaxPool2D((2,2), padding='same'))

model.add(layers.Flatten())


model.add(layers.Dense(512,activation='selu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3,activation='softmax'))

input(model.summary())

model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.adam(lr=0.001),
            metrics=['acc'])


history = model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50        
)


print("---------evaludate---------")
scores = model.evaluate(validation_generator)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print(np.argmax(validation_generator[1,150,150,3]))
print(np.argmax(validation_generator[1,150,150,3]))
print(np.argmax(validation_generator[1,150,150,3]))


print("---------prediction for validation data---------") # --> validation값 실제 값과 비교


output = model.predict(test_generator) 
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}) 
# np.set_printoptions: 넘파이 출력 옵션 변경, 
# 모두 3자리수로 출력 옵션이 변경됨
print(test_generator.class_indices) #class_indices: 해당 열의 class명 알려 줌. 
print("예측값:", output)

print("---------prediction for new data---------")

from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('./predict_image/ferrite.png')

#show image
fname = './predict_image/ferrite.png'
gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#transform image
img = cv2.resize(img, (150,150))
img = img.reshape(1,150,150,3)
new_prediction = model.predict(img)
print(new_prediction) #new_prediction --> numpy 배열
#print(np.argmax(new_prediction))

# new_prediction[0] 이 제일 크면 martensite 1이 제일 크면 austenite 2면 fer


if np.argmax(new_prediction) == 0:
    print("Austenite입니다.")
elif np.argmax(new_prediction) == 1:
    print("ferrite입니다.")
elif np.argmax(new_prediction) == 2:
    print("martensite입니다.")



model.save('phase_prediction.h5')
# import cv2
# print(cv2.__version__)
# X=[]
# img_h = 150
# img_w = 150
# img = Image.open('./predict_image/ferrite.png')
# img = img.convert('RGB')
# img = img.resize((img_w,img_h))

# data = np.asarray(img)
# X.append(data)

# prediction = model.predict(X)
# print(prediction)
print("cv_version:", cv2.__version__)
