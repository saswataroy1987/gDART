#from sklearn.model_selection import train_test_split
import pandas as pd
from ast import literal_eval as le
import os
import pickle
import numpy as np
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#=============================
import tensorflow as tf
#opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
import tensorflow_hub as hub
import tensorflow_text as text
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
#=============================
from MHA import MultiHeadAttention

pout="repo_Fast/"

Epochs = XXX



trainStatsFile=pout+"dfTrainStats"+str(Epochs)+".csv"
saveModel=pout+"vsrModel.h5"
predFile=pout+"predicted"+str(Epochs)+".pkl"

orthoNumber = XXX



def ortho_Loss(custom_out):
	X, Y= custom_out
	##shape retrieval
	sp=tf.shape(X)#tf.get_shape().as_list(X)
	v1=tf.reshape(X, [-1, sp[-2]*sp[-1]])
	v2=tf.reshape(Y, [-1, sp[-2]*sp[-1]])
	#https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot
	ortho = tf.keras.layers.Dot(1, normalize=True)([v1, v2])#; print("shape of ortho:  ",ortho.shape);input()
	#ortho = Dense(1, activation="sigmoid", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l1_l2(l1=1e-5, l2=1e-4))(ortho)
	ortho = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(ortho)
	return ortho



	
def getData(filename):
	
	di_encode = {"real":[0,1], "fake":[1,0]}
	path = ""
	if filename == "train":
		df = pd.read_csv(path+"Constraint_English_Train - Sheet1.csv")

	if filename == "test":
		df = pd.read_csv(path+"english_test_with_labels - Sheet1.csv")
		
	if filename == "val":
		df = pd.read_csv(path+"Constraint_English_Val - Sheet1.csv")
	
	
	# dropping the rows having NaN values
	df = df.dropna()
 
	# To reset the indices
	df = df.reset_index(drop = True)
	#df=df[:5]

	print(filename,": ",df.shape)

	df['digit_label']=df['label'].apply(lambda x: 1 if x=='fake' else 0)
	#df['encoded_label'] = df['label'].apply(lambda x: di_encode[x])
	
	X_split = np.array(df["tweet"].tolist())
	
	#Y_split = np.array(df['digit_label'].tolist())

	sp = df.shape
	# veracity = np.zeros((sp[0], len(df["encoded_label"].tolist()[0])))
	# for i, row in df.iterrows():
	# 	veracity[i, :]=row["encoded_label"]

	veracity = np.array(df['digit_label'].tolist())
	#veracity=np.expand_dims(veracity, axis=1)
	
	orthoL=[np.zeros((sp[0], 1)) for i in range(orthoNumber)]
	Y_split=[ veracity ]+orthoL
	
	
	return df, X_split, Y_split



if __name__ == "__main__":
	
	# ##get data	
	df_train, Xtr, Ytr = getData("train")
	df_test, Xts, Yts = getData("test")
	df_val, Xvl, Yvl = getData("val")	
	print("printing xtr ytr ", Xtr.shape,[x.shape for x in Ytr])
	print("printing xtr ytr ", Xts.shape,[x.shape for x in Yts])
	print("printing xtr ytr ", Xvl.shape,[x.shape for x in Yvl])
	

	combN=4; mha=16; outdim=100; l2N = 0.0001

	trm = int(combN*(combN-1)/2); trm = trm+1
	Ytr = Ytr[:trm]; Yts = Yts[:trm]; Yvl = Yvl[:trm]


	# Bert layers
	text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
	preprocessed_text = bert_preprocess(text_input)
	outputs = bert_encoder(preprocessed_text)



	# Neural network layers
	l = tf.keras.layers.Dropout(0.1, name="dropout1")(outputs['sequence_output'])

	custom_out = MultiHeadAttention(outdim, mha, combN, mha*outdim)(l);print(custom_out[0].shape)
	
	list_ortho_Loss = [ortho_Loss([custom_out[i]]+[custom_out[j]]) for i in range(len(custom_out)) for j in range(len(custom_out)) if i < j ] 



	custom_out= [ tf.keras.layers.Dense(100, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2N))(x) for x in custom_out ];print(custom_out[0].shape)
	custom_out = tf.keras.layers.Concatenate(axis=2)(custom_out);print("shape of custom_out: ", custom_out.shape)
	ax0, ax1, ax2=custom_out.get_shape().as_list()
	custom_out=tf.keras.layers.Lambda(lambda x:tf.reshape(x, [-1, ax1*ax2]))(custom_out);print(custom_out.shape)

	custom_out = tf.keras.layers.Dropout(0.1, name="dropout2")(custom_out)

	veracity_out = tf.keras.layers.Dense(1, activation='sigmoid', name="gDART")(custom_out)



	model = tf.keras.Model(inputs=[text_input], outputs = [veracity_out]+list_ortho_Loss)
	loss = ['binary_crossentropy']+['mean_squared_error' for i in range(len(list_ortho_Loss))]
	
	loss_weights = [1.4]+[0.1 for i in range(len(list_ortho_Loss))]



	METRICS = [
	      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
	      tf.keras.metrics.Precision(name='precision'),
	      tf.keras.metrics.Recall(name='recall')
	]

	# model.compile(optimizer='adam',
	#               loss='binary_crossentropy',
	#               metrics=METRICS)

	model.compile(optimizer='adam',
	              loss=loss,
	              loss_weights=loss_weights,
	              metrics=METRICS)

	print(model.summary())



	
	dfStats=pd.DataFrame(columns=["gDART_loss", "val_gDART_loss", "gDART_accuracy", "val_gDART_accuracy"])
	his = model.fit(Xtr, Ytr, epochs=Epochs,  batch_size=32, validation_data=(Xvl, Yvl), verbose=2)
	
	statsL=[ his.history["gDART_loss"], his.history["val_gDART_loss"], his.history["gDART_accuracy"], his.history["val_gDART_accuracy"] ]

	dfStats["gDART_loss"], dfStats["val_gDART_loss"], dfStats["gDART_accuracy"], dfStats["val_gDART_accuracy"]=statsL[0], statsL[1], statsL[2], statsL[3]
	dfStats.to_csv(trainStatsFile, index=False)

	y_predicted = model.predict(Xts)
	y_predicted=y_predicted[0]
	y_predicted = y_predicted.flatten()

	y_predicted = np.where(y_predicted > 0.5, 1, 0)


	print(y_predicted)
	print (Yts[0])

	f=open(predFile, "wb")
	pickle.dump(y_predicted, f)
	f.close()


	from sklearn.metrics import confusion_matrix, classification_report

	cm = confusion_matrix(Yts[0], y_predicted)

	print(cm)


	print(classification_report(Yts[0], y_predicted))


