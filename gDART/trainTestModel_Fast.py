import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
#from tensorflow import set_random_seed
#set_random_seed(1)
tf.random.set_seed(2)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import pickle
import pandas as pd

import keras
from keras import backend as k
from keras.models import Model
from keras.models import load_model
import os
import sys
import pickle

import rumour_Loss_Transformer as rL
import getXY_Loss_Transformer as gxy
import resCal_transformer as rt

pout = "repo_Fast/"

Epochs = XXX
trainStatsFile = pout+"dfTrainStats"+str(Epochs)+".csv"
saveModel = pout+"vsrModel.h5"
predFile = pout+"predicted"+str(Epochs)+".pkl"


def train_test():
	f = open("embeddingMatrix.pkl", "rb") #create Embedding matrix using word2vec embedding.
	embedding_Matrix = pickle.load(f)
	f.close()
	
	# sD["source"]: maximum no of words in source tweet
	# sD["branch"]: maximum length of branch.
	sD={"source":(XXX, ), "branch":(YYY, ), "lstmOP":1, "lstmOP2":100, "dense_Size":100, "regularizer":0.0001}
	sD["vocab_size"]=10739; sD["out_dim"]=300; sD["input_length"]=YYY
	sD["embedding_Matrix"]=embedding_Matrix
	sD["combN"]=4; sD["mha"]=16; sD["outdim"]=100
	
	##get data
	xtr, ytr, xts, yts=gxy.xyRet() #prepare training and testing data
	trm=int(sD["combN"]*(sD["combN"]-1)/2); trm=trm+1
	ytr=ytr[:trm]; yts=yts[:trm]
		
	##get model
	model=rL.getBaseModel(**sD)
	

	print ("\n data \n", trm, "\n")
	print ("train ", [x.shape for x in xtr], [x.shape for x in ytr], "\n")
	print ("test  ", [x.shape for x in xts], [x.shape for x in yts], "\n")
	print (" model inputs ", [x.shape for x in model.inputs], "\n")
	print (" model outputs ", [x.shape for x in model.outputs], "\n")

	##training & validation
	dfStats=pd.DataFrame(columns=["private_vsr_loss", "val_private_vsr_loss", "private_vsr_accuracy", "val_private_vsr_accuracy"])
	his=model.fit(xtr, ytr, epochs=Epochs, batch_size=32, validation_split=0.2, verbose=2)
	#print(list(his.history.keys()))
	statsL=[ his.history["private_vsr_loss"], his.history["val_private_vsr_loss"], his.history["private_vsr_accuracy"], his.history["val_private_vsr_accuracy"] ]
	#print (statsL)
	dfStats["private_vsr_loss"], dfStats["val_private_vsr_loss"], dfStats["private_vsr_accuracy"], dfStats["val_private_vsr_accuracy"]=statsL[0], statsL[1], statsL[2], statsL[3]
	dfStats.to_csv(trainStatsFile, index=False)

	#saving model
	model.save(saveModel)
	

	pred=model.predict(xts)
	f=open(predFile, "wb")
	pickle.dump(pred, f)
	f.close()
	
	rt.measure("repo_Fast/predicted"+str(Epochs)+".pkl", "repo/testFinal.csv") #testFinal.csv contains test data

	return
	

train_test()

	
