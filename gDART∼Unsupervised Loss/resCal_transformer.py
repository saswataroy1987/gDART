import sys
import pandas as pd
import numpy as np
import pickle 
from ast import literal_eval as le

folderName="repo_Fast/"

def sM(a):
	l=[]
	for i in range(len(a)):
		v=a[i].index(max(a[i])); v=2-v
		#print (a[i], v)
		l.append(v)
	return l

def conf(p, r):	

	def cap(v):
		tp, tn, fp, fn=0, 0, 0, 0
		pre, rec, f1, mac, acc=0, 0, 0, 0, 0
		for x, y in zip(p, r):
			if x==y and y==v: tp+=1
			if x==v and y!=v: fp+=1
			if x!=v and y!=v: tn+=1
			if x!=v and y==v: fn+=1
		if tp==0: tp=1
		if fp==0: fp=1
		if tn==0: tn=1
		if fn==0: fn=1
		pre=1.0*tp/(tp+fp)
		rec=1.0*tp/(tp+fn)
		f1=2.0*pre*rec/(pre+rec)
		acc=1.0*(tp+tn)/(tp+fn+tn+fp)
		print (" tp fp tn fn ", tp, fp, tn, fn)
		return pre, rec, f1, acc

	pre, rec, f10, acc0=cap(0)
	print (" v 0 ___", pre, rec, f10, acc0)
	
	pre, rec, f11, acc1=cap(1)
	print (" v 1 ___", pre, rec, f11, acc1)

	pre, rec, f12, acc2=cap(2)
	print (" v 2 ___", pre, rec, f12, acc2)

	print ("************** macroF1 ", np.mean([f10, f11, f12]))
	print ("************** acc ", np.mean([acc0, acc1, acc2]))

	return

def measure(pred, real):
	df=pd.read_csv(real)
	df["veracity"]=df["veracity"].apply(lambda x: le(x))
	#topics=df["topics"].tolist()
	r=df["veracity"].tolist(); r=sM(r)
	#print (r[0], r[0][0], type(r[0][0]))

	f=open(pred, "rb")
	p=pickle.load(f)
	f.close()
	print ([x.shape for x in p])
	#p=[x.tolist() for x in p]	
	p=p.tolist()
	#p1, p2, p3=sM(p[0]), sM(p[1]), sM(p[2])
	#print (p1, type(p1[0]))
	p1=sM(p)
	#dfRes=pd.DataFrame({"real":r, "prv":p1, "aux":p2, "srd":p3})
	#dfRes.to_csv(folderName+"resCal.csv", index=False)
	dfRes=pd.DataFrame({"real":r, "prv":p1})
	dfRes.to_csv(folderName+"resCal.csv", index=False)

	conf(p1, r)
	#conf(p2, r)
	#conf(p3, r)

	return

