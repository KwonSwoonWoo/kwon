import os
import shutil
import copy


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


#folder_make

createFolder('./train/train_for_martensite')
createFolder('./test/test_for_martensite')
createFolder('./validation/validation_for_martensite')

createFolder('./train/train_for_austenite')
createFolder('./test/test_for_austenite')
createFolder('./validation/validation_for_austenite')

createFolder('./train/train_for_ferrite')
createFolder('./test/test_for_ferrite')
createFolder('./validation/validation_for_ferrite')


for i in range (1,4):
    shutil.copy('image/martensite'+format(i)+'.png','./train/train_for_martensite/martensite'+format(i)+'.jpg') 

for i in range (4,7):
    shutil.copy('image/martensite'+format(i)+'.png','./test/test_for_martensite/martensite'+format(i)+'.jpg')

for i in range (7,10):
    shutil.copy('image/martensite'+format(i)+'.png','./validation/validation_for_martensite/martensite'+format(i)+'.jpg')  



for i in range (1,4):
    shutil.copy('image/austenite'+format(i)+'.png','./train/train_for_austenite/austenite'+format(i)+'.jpg') 

for i in range (4,7):
    shutil.copy('image/austenite'+format(i)+'.png','./test/test_for_austenite/austenite'+format(i)+'.jpg')

for i in range (7,10):
    shutil.copy('image/austenite'+format(i)+'.png','./validation/validation_for_austenite/austenite'+format(i)+'.jpg')  

    

for i in range (1,4):
    shutil.copy('image/ferrite'+format(i)+'.png','./train/train_for_ferrite/ferrite'+format(i)+'.jpg') 

for i in range (4,7):
    shutil.copy('image/ferrite'+format(i)+'.png','./test/test_for_ferrite/ferrite'+format(i)+'.jpg')

for i in range (7,10):
    shutil.copy('image/ferrite'+format(i)+'.png','./validation/validation_for_ferrite/ferrite'+format(i)+'.jpg')  