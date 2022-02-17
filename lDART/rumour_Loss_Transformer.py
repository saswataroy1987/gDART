# Multiple Inputs
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
#from tensorflow import set_random_seed
#set_random_seed(1)
tf.random.set_seed(2)
import pandas as pd

import keras
from keras import backend as k
from keras.layers import Dot, Concatenate, Flatten, TimeDistributed, BatchNormalization,LayerNormalization
from keras.layers import Input, Embedding, LSTM, Dense, Lambda, LeakyReLU
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import load_model
import math
from keras.optimizers import Adam
opt=Adam(lr=0.00001)

'''keras.regularizers.l1(0.01)
keras.regularizers.l2(0.01)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)'''
from keras.regularizers import l1, l2, l1_l2
#l1N=0.01; l2N=0.01; l1_l2N=0.01

np.random.seed(0)
#from keras.layers import Layer
from MHA import MultiHeadAttention


def vecT(x):
	l=x.get_shape().as_list()
	x=tf.reshape(x, [-1, l[1]*l[2]])
	return x

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

  return pos * angle_rates

def positional_encoding(x):
	l = x.get_shape().as_list()
	#print(l);input()
	position = l[1]
	d_model = l[2]
	#print(position,d_model);input()
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	pos_encoding = angle_rads[np.newaxis, ...]
	positional_tensor = tf.cast(pos_encoding, dtype=tf.float32)
	x = x + positional_tensor
	return x

def ortho_Loss(custom_out):
	X, Y= custom_out
	ortho = Dot(1, normalize=True)([X, Y])#; print("shape of ortho:  ",ortho.shape);input()
	ortho = Dense(1, activation="sigmoid", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l1_l2(l1=1e-5, l2=1e-4))(ortho)
	return ortho

def shape_change(src):
	src_shape = src.get_shape().as_list()
	src = tf.transpose(src,[0,2,1])
	return src

def createReshape(tensor_sample):
	shape0, shape1, shape2 = tensor_sample.get_shape().as_list()
	tensor_sample = tf.reshape(tensor_sample,[-1,shape1*shape2])
	return tensor_sample

def add_Norm(sc, custom_out):
	custom_out_add = Lambda(lambda x:tf.add_n(x))([sc, custom_out])
	custom_out_norm = LayerNormalization()(custom_out_add)
	return custom_out_norm

def getBaseModel(**kwargs):
	source=kwargs["source"]
	branch=kwargs["branch"]
	dense_Size=kwargs["dense_Size"]
	lstmOP2=kwargs["lstmOP2"]
	vocab_size=kwargs["vocab_size"]
	out_dim=kwargs["out_dim"]
	input_length=kwargs["input_length"]
	embedding_Matrix=kwargs["embedding_Matrix"]
	l2N=kwargs["regularizer"]
	combN=kwargs["combN"] ; mha=kwargs["mha"]; outdim=kwargs["outdim"]

	visibleSrc=Input(shape=source)
	src=Embedding(input_dim=vocab_size, output_dim=out_dim, weights=[embedding_Matrix], input_length=source[0], trainable=False)(visibleSrc)	
	visible=Input(shape=branch)
	s=Embedding(input_dim=vocab_size, output_dim=out_dim, weights=[embedding_Matrix], input_length=branch[0], trainable=False)(visible)
	sc=Concatenate(axis=1)([src, s])
	sc=BatchNormalization()(sc)#;print("shape befire MHA:  ",sc.shape);input("wait...........")
	
	def series_MHA(sc, flag):
		#print("before MHA:  ",sc.shape)
		custom_out,ZL = MultiHeadAttention(outdim, mha, mha*outdim)(sc)
		#print("after MHA:  ",custom_out.shape)
		#First ADD_NORM
		if flag == 0:
			sc = Dense(100, activation="sigmoid", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l2(l2N))(sc)
		custom_out_norm1 = add_Norm(sc, custom_out)#;print("after First add_Norm:  ",custom_out_norm1.shape)
		custom_out_dense= Dense(100, activation="sigmoid", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l2(l2N))(custom_out_norm1) 
		#Second ADD_NORM
		custom_out_norm2 = add_Norm(custom_out_norm1, custom_out_dense)#;print("after second add_Norm:  ",custom_out_norm2.shape)
		return custom_out_norm2,ZL

	#list_ortho_Loss = [ortho_Loss([custom_out[i]]+[custom_out[j]]) for i in range(len(custom_out)) for j in range(len(custom_out)) if i < j ] 
	
	#custom_out = Concatenate(axis=2)(custom_out)
	for i in range(4):
		#print("-------Iteration{0}--------".format(i))
		sc, ZL = series_MHA(sc,i)
		#print(i,":", sc.shape,len(ZL))
		#input("enter...")

	list_ortho_Loss = [ortho_Loss([ZL[i]]+[ZL[j]]) for i in range(len(ZL)) for j in range(len(ZL)) if i < j ] 
	sc = Dense(100, activation="sigmoid", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l2(l2N))(sc)
	custom_out = Dropout(0.5)(sc)
	#print("shape after MHA:  ",custom_out.shape);input("wait...........")
	ax0, ax1, ax2=custom_out.get_shape().as_list()
	custom_out=Lambda(lambda x:tf.reshape(x, [-1, ax1*ax2]))(custom_out)
	#print("shape after MHA:  ",custom_out.shape);input("wait...........")
	custom_out = Dense(dense_Size, activation="sigmoid", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l1_l2(l1=1e-5, l2=1e-4))(custom_out)
	custom_out = Dense(dense_Size, activation="sigmoid", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l1_l2(l1=1e-5, l2=1e-4))(custom_out)
	veracity_out = Dense(3, activation="softmax", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l1_l2(l1=1e-5, l2=1e-4), name="private_vsr")(custom_out)

	model=Model([visibleSrc, visible], [veracity_out]+list_ortho_Loss)
	#model=Model([visibleSrc, visible], veracity_out)
	loss = ['categorical_crossentropy']+['mean_squared_error' for i in range(len(list_ortho_Loss))]
	#loss = ['categorical_crossentropy']
	loss_weights = [1.4]+[0.1 for i in range(len(list_ortho_Loss))]
	#loss_weights = [1.4]
	model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights, metrics=['accuracy'])

	#print model summary
	print(model.summary())#;input("HOld on....")
	#plot graph
	#plot_model(model, to_file='repo/Transformer.png')
		
	return model
	
