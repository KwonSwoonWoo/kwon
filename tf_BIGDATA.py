# -*- coding: utf-8 -*- 
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import plot_model
import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
import seaborn as sns
import os
import subprocess
import sys
from sklearn.preprocessing import MinMaxScaler
import random
import pickle
#t_featurename__  = ['C', 'Cr', 'Mo', 'Co', 'Ni', 'Nb', 'W', 'Mn', 'Si', 'V', 'aus', 'tem', 'time', 'AF', 'DQ']


def h():
   print("- HELP")
   print("1. python entry file.py: \t\t            Statstics program")
   print("2. python entry file <TARGET>.py: \t            Neural network program")
   print("3. python entry file <TARGET>.py:                     Genetic algorithm program")
   print("4. python entry file <TARGET>.py:\t\t            Prediction program with using neural network")
   print("5. python program structure:\t\t\t    Use Configuration file: DB - Feature and Target")   
   print("\n")

   print("6. Sequence of Activation of file(Machine learning > Genetic algorithm >save and predict.)")
   print("6.1 (IF DB changed), Modify statistics_entry.py, then Activate python statistics_entry.py")
   print("6.1.1 statistics program is independant program.")
   print("6.2 If DB changed, Modify ML_entry.py then Activate python ML_entry.py")
   print("6.2.1 Else(DB not changed) goto 6.3 - 6.5 - 6.7 - 6.9")
   print("6.3 Activate python ML_entry.py TARGET in command")
   print("6.4 Activate Jupyter notebook then If DB changed, Modify DB.csv then Activate all")
   print("6.5 In data_minmax.csv file, delete targets")
   print("6.6 If DB changed, Activate entry_ga.py then modify DB and targets")
   print("6.7 Activate python entry_ga.py TARGET in command")
   print("6.8 If DB changed Activate entry_save_predict.py then modify DB and target splitted")
   print("6.9 Activate python entry_save_predict.py TARGET in command") 
   print("\n")

   print("--------------------------------------------------------")
   print("--------------------DB 바뀔 시 수정---------------------")
   print("at 'tf_BIGDATA'")
   print("\nConfiguration file:ML_entry.py")
   print("DB's target =?Entries target")
   print("\n")
   print("at 'ga.py'")
   print("\n")
   print("Configuration file: entry_ga.py")
   print("DB's target =?Entries target")
   print("Modify Fe balanced value to optimize program")
   print("Modify Fe_bal variable")
   print("\n")
   print("at 'mutation.py'")
   print("No need of change")


   print("\n")
   print("at 'predict.py'\n")
   print("Configuration file: entry_save_predict.py")
   print("DB's target =?Entries target")
   print("IN CONFIGURATION FILE, If DB Is different, error occured")



#1. Dataload and split data
def create_model():
   try: #DB입력하지 않았을 때
      input_data = sys.argv[2]   
   except IndexError:
      print("error: DB Syntax and target syntax required or check the syntax, goto python h.py")
      exit(0)

   try: #DB가 없을 때, 잘못 입력 했을 때. 
      data = pd.read_csv(input_data) 
   except FileNotFoundError:
      print("error: Exact DB Input Required or check the syntax, goto python h.py")
      exit(0)

   try: #target을 잘못 입력했을 때
      target_data = sys.argv[4] # UTS ? YS ? ...?
   except IndexError:
      print("error: Target syntax required or check the syntax, goto python h.py")
    #  h()
      exit(0)

   try: #target을 잘못 입력했을 때
      targets_data = sys.argv[5] # UTS ? YS ? ...?
   except IndexError:
      print("error: Target syntax required or check the syntax, goto python h.py")
      exit(0)

   targets_data = targets_data.split('/')
   #data = data.dropna()
   t_featurename__=list(data.columns)
   for item in targets_data:
      t_featurename__.remove(item)  # 타겟값 잘못되면 에러 메세지 넣기
   t_featurename__.append(target_data)
   data = data.loc[:,t_featurename__]
   data = data.dropna()
   t_featurename__.remove(target_data)
      
   
   try: #target을 입력하지 않았을 때. 
      y = data.loc[:, [target_data]].values
   except KeyError:
      print("error: Exact target value or check the syntax, goto python h.py")
      exit(0)
   
   x = data.loc[:, t_featurename__].values
   command_i = ['-i','-input'] #input 실행
   if sys.argv[1] not in command_i:
      print("error: -i or input is required, goto help(h)")
      exit(0)

   command_t = ['-T','-target'] #target 실행
   if sys.argv[3] not in command_t:
      print("error: -T or target is required, goto help(h)")
      exit(0)
               
      #print (str(len(data)))
      #print (str(len(y)))
      #import time
      #time.slleep(1000)
      #sys.exit(0)
   R1 = random.randint(0,10)*20
   R2 = random.randint(0,10)*20
   #-----------------------------------split data to test, train, validation-------------------------------------------------#
   X_train_total, X_test_origin, y_train_total, y_test = train_test_split(x, y, train_size=0.8, random_state= R1)
   #X_train_origin, X_val_origin, y_train, y_val = train_test_split(X_train_total, y_train_total, train_size=1, random_state= R2)
   X_train_origin, X_val_origin, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state= R2)
   #print len(data)
   #print len(y)
   #------------------------------------------------------------------------------------#
   #1.1 Scaling - using standard scaler
   Scaler = StandardScaler()
   Scaler.fit(X_train_origin)
   X_train = Scaler.transform(X_train_origin)
   X_val = Scaler.transform(X_val_origin)
   X_test = Scaler.transform(X_test_origin)
   
   f = open('scaler', 'wb') 
   pickle.dump(Scaler,f)
   f.close()

   
   keras.initializers.he_uniform(seed=None) # He initizlier - Relu와 적합
   #2. modeling
   a = 48
   b = 48
   c = 48
   d = 48
   e = 48
   f = 48
   g = 48
   h = 48
   i = 48
   j = 48
   k = 48
   l = 48
   m = 48
   n = 48
   o = 48
   p = 48
   q = 48
   r = 48
   s = 48
   t = 48
   u = 48
   v = 48
   w = 48
   x = 48
   y = 48
   z = 48
   aa =48
   
 # bb = 48
 # cc = 48
 # dd = 48
 # ee  = 48
 # ff  = 48
 # gg  = 48
 # hh  = 48
 # ii  = 48
 # kk  = 48
 # ll  = 48
 # mm  = 48
 # nn  = 48
 # oo  = 48
 # pp  = 48
 # qq  = 48
 # rr  = 48
 
    

   model = Sequential()
   
   model.add(Dense(a,activation='selu',))
   model.add(Dense(b,activation='selu',))
   model.add(Dense(c,activation='selu',))
   model.add(Dense(d,activation='selu',))
   model.add(Dense(e,activation='selu',))
   model.add(Dense(f,activation='selu',))
   model.add(Dense(g,activation='selu',))
   model.add(Dense(h,activation='selu',))
   model.add(Dense(i,activation='selu',))
   model.add(Dense(g,activation='selu',))
   model.add(Dense(k,activation='selu',))
   model.add(Dense(l,activation='selu',))
   model.add(Dense(m,activation='selu',))
   model.add(Dense(n,activation='selu',))
   model.add(Dense(o,activation='selu',))
   model.add(Dense(p,activation='selu',))
   model.add(Dense(q,activation='selu',))
   model.add(Dense(r,activation='selu',))
   model.add(Dense(s,activation='selu',))
   model.add(Dense(t,activation='selu',))
   model.add(Dense(u,activation='selu',))
   model.add(Dense(v,activation='selu',))
   model.add(Dense(w,activation='selu',))
   model.add(Dense(x,activation='selu',))
   model.add(Dense(y,activation='selu',))
   model.add(Dense(z,activation='selu',))
   model.add(Dense(aa,activation='selu',))

 # model.add(Dense(bb,activation='selu',))
 # model.add(Dense(cc,activation='selu',))
 # model.add(Dense(dd,activation='selu',))
 # model.add(Dense(ee,activation='selu',))
 # model.add(Dense(ff,activation='selu',))
 # model.add(Dense(gg,activation='selu',))
 # model.add(Dense(hh,activation='selu',))
 # model.add(Dense(ii,activation='selu',))
 # model.add(Dense(kk,activation='selu',))
 # model.add(Dense(ll,activation='selu',))
 # model.add(Dense(mm,activation='selu',))
 # model.add(Dense(nn,activation='selu',))
 # model.add(Dense(oo,activation='selu',))
 # model.add(Dense(pp,activation='selu',))
 # model.add(Dense(qq,activation='selu',))
 # model.add(Dense(rr,activation='selu',))
   #model.add(Dropout(0.2))
   model.add(Dense(1))  

   X_train = np.array(X_train)
   y_train = np.array(y_train)
   

   #RMSE
   def root_mean_squared_error(y_true,y_pred):
        from keras import backend as K
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

   #R2
   def R_Squared(y_true, y_pred):
      from keras import backend as K
      SS_res =  K.sum(K.square( y_true-y_pred ))
      SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
      return ( 1 - SS_res/(SS_tot + K.epsilon()) )

   

   #Complie model , MAE 사용
   #model.compile(loss = root_mean_squared_error, optimizer= Adam(=0.001), metrics=[root_mean_squared_error,R_Squared]) #- rmse
   # sgd=optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
   model.compile(loss = 'mean_absolute_error', optimizer= Adam(lr=0.00002), metrics=['mae',R_Squared]) 
   history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val))
   model.save('model_'+target_data+'.h5') 
   

   #3. plotting 
   # 3.1. loss file
   while True:
            print("\n")
            namel = input("1. Loss Graph and NAN -l help -h and next -n\n")
            print("\n")
            if namel == '-n':
               break

            elif namel =='-l':

               #loss_to_csv
               np.savetxt(target_data+"_Train_Loss.csv", history.history['mean_absolute_error'], delimiter=",")
               np.savetxt(target_data+"_Test_Loss.csv", history.history['val_mean_absolute_error'], delimiter=",")
               
               plt.plot(history.history['loss'],c='r', label = 'train loss')
               plt.plot(history.history['val_loss'], 'b', label = 'test loss')
               plt.xlabel('epoch')
               plt.ylabel('loss')
               plt.legend(loc='upper left')
               plt.show()
    
            elif namel =='-h':
               print("-h\tIf you want to see the help file\n")
               print("-n\tIf you want to go to next and see Train and Test graph.\n")
               #print("-m\tto see the mae graph.\n")
               print("-l\tto see the loss graph.\n")
               

   #3.2 
   # train set predict.
   # train and test
   while True:
            print("\n")
            names = input("2. TRain graph -TR and TesT graph -TT help -h and Quit -q\n")
            print("\n")
            if names == '-q':
               break
            elif names =='-TR':
               real_value_tr_x = X_train_origin
               predict_tr = model.predict(X_train)
               real_value_tr = y_train
#csv X축 REAL Y축 NN
               result_train_csv = pd.DataFrame({'real_value_tr':list(real_value_tr.T[0]),    'predict_tr':list(predict_tr.T[0])},)
               name = pd.DataFrame(X_train_origin)
               name.columns = t_featurename__
               result_train_csv = pd.concat([name,result_train_csv],axis=1)
               result_train_csv.to_csv(target_data+"_Train_Predict"+".csv",header = True, index = False) # To .csv_Train
               
#matplotlib로 그리는 것
               result1 = pd.DataFrame({'real_value_tr':list(real_value_tr.T[0]),    'predict_tr':list(predict_tr)},)
               result1 = result1.sort_values(by='predict_tr')
               result1.index = range(len(result1))
               fig, axes = plt.subplots(figsize = (15,10))
               plt.plot(result1['predict_tr'], label = 'Prediction')
               plt.plot(result1['real_value_tr'], 'o', label = 'Real Value')
               plt.ylabel(target_data)
               plt.xlabel('index')
               plt.legend(loc = 'best')
               plt.show()            
   # 3.2. test set predict            
            elif names =='-TT':
               real_value_tt_x = X_test_origin
               predict_tt = model.predict(X_test)
               real_value_tt = y_test
#X축 REAL Y축 NN 
               result_test_csv = pd.DataFrame({'real_value_tt':list(real_value_tt.T[0]),    'predict_tt':list(predict_tt.T[0])},)
               name = pd.DataFrame(X_test_origin)
               name.columns = t_featurename__
               result_test_csv = pd.concat([name,result_test_csv],axis=1)
               result_test_csv.to_csv(target_data+"_Test_Predict"+".csv",header = True, index = False) # To .csv_Test
#matplotlib로 그리는 것 
               result1 = pd.DataFrame({'real_value_tt':list(real_value_tt.T[0]),    'predict_tt':list(predict_tt)},) #
               result1 = result1.sort_values(by='predict_tt') #Graph for ascending order
               result1.index = range(len(result1)) # index length
               fig, axes = plt.subplots(figsize = (15,10))
               plt.plot(result1['predict_tt'], label = 'Prediction')
               plt.plot(result1['real_value_tt'], 'o', label = 'Real Value')
               plt.ylabel(target_data)
               plt.xlabel('index')
               plt.legend(loc = 'best')
               plt.show()              
#HELP         
            elif names =='-h':
               print("-q\tQuit this file..\n")
               print("-TR\tto see the Train Y^ graph.\n")
               print("-TT\tto see the Test Y^ graph.\n")    
               print("-TT\tTo see the help file.\n")      
# 4.print R2 and MAE

   x = model.evaluate(X_train, y_train)
   z = model.evaluate(X_test, y_test)
   print("MAE of train: ", x[1])
   print("R2 of train: ", x[2])
 
   print("MAE of test: ", z[1])
   print("R2 of test: ", z[2])
   #pearson correlation...?   



   
   
   return

  
                             

 
create_model()
