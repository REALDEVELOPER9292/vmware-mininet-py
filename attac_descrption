# -*- coding: utf-8 -*-
"""
Created on Tue May 24 01:03:35 2022

@author: MKF
"""
import pandas as pd
import numpy as np
from tensorflow.python import tf2
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Bidirectional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score

#------------------------Training Parameters-----------------------------------
epochs = 100
No_samples = 1000
validation_spliting=0.2
Test_spliting = 0.2
#------------------------------------------------------------------------------

#-----------------------Loading Normal dataset---------------------------------
DataNormal = pd.read_csv('./dataset/D_Normal.csv', nrows = No_samples)
DataNormal.columns=[ 'frame.len', 'frame.protocols', 'ip.hdr_len',
       'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
       'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
       'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
       'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
       'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
       'tcp.time_delta','class']
DataNormal=DataNormal.drop(['ip.src', 'ip.dst','frame.protocols'],axis=1)
print(DataNormal)
#------------------------------------------------------------------------------

#----------------------Loading DDoS Attack Dataset-----------------------------
DataAttack = pd.read_csv('./dataset/D_Attack.csv', nrows = No_samples)
DataAttack.columns=[ 'frame.len', 'frame.protocols', 'ip.hdr_len',
       'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
       'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
       'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
       'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
       'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
       'tcp.time_delta','class']
DataAttack=DataAttack.drop(['ip.src', 'ip.dst','frame.protocols'],axis=1)
print(DataAttack)
#------------------------------------------------------------------------------


# ----------------------------Feature extraction-------------------------------
fech=[ 'frame.len', 'ip.hdr_len',
       'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
       'ip.ttl', 'ip.proto', 'tcp.srcport', 'tcp.dstport',
       'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
       'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
       'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
       'tcp.time_delta']
Xnormal= DataNormal[fech].values
Ynormal= DataNormal['class']
Xattack= DataAttack[fech].values
Yattack= DataAttack['class']
X=np.concatenate((Xnormal,Xattack))
Y=np.concatenate((Ynormal,Yattack))
print(X)
print(Y)
#------------------------------------------------------------------------------
#Preprocessing-----------------------------------------------------------------
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X)
X = scaler.transform(X)

for i in range(0,len(Y)):
  if Y[i] =="DDoS Attack":
    Y[i]=0
  else:
    Y[i]=1
    fech = len(X[0])
    trainLen = 25
samp = X.shape[0]
input_len = samp - trainLen
I = np.zeros((samp - trainLen, trainLen, fech))

for i in range(input_len):
    temp = np.zeros((trainLen, fech))
    for j in range(i, i + trainLen - 1):
        temp[j-i] = X[j]
    I[i] = temp
    
    X.shape
    (100000, 25)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(I, Y[25:100000], test_size = Test_spliting)
    print(X.shape)

# ---------------------------------------BP_NN---------------------------------
    def Make_baseline():
        NN_model = Sequential()
        NN_model.add(Bidirectional(LSTM(64, activation='tanh', kernel_regularizer='l2')))
        NN_model.add(Dense(64, activation='relu', input_shape=(4,)))
        NN_model.add(Dense(32, activation='relu'))
        NN_model.add(Dense(16, input_dim=1, activation='relu'))
        NN_model.add(Dense(12, activation='relu'))
        NN_model.add(Dense(1, activation='sigmoid'))
        NN_model.compile(loss = 'binary_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])
        return NN_model

NN_model = Make_baseline()
history = NN_model.fit(Xtrain, Ytrain, epochs = epochs,validation_split=validation_spliting, verbose = 1)


