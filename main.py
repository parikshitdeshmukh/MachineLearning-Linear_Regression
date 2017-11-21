import numpy as np
import math
from scipy.cluster.vq import kmeans2
from lib import *
from sklearn.cross_validation import train_test_split


syn_input_data = np.loadtxt('/home/csgrad/pdeshmuk/ML_project2/input.csv', delimiter=',')
syn_output_data = np.loadtxt('/home/csgrad/pdeshmuk/ML_project2/output.csv', delimiter=',').reshape([-1, 1])

syn_input_train, syn_input_test, syn_output_train, syn_output_test = train_test_split(syn_input_data, syn_output_data, test_size=0.2)
syn_input_test, syn_input_val, syn_output_test, syn_output_val = train_test_split(syn_input_test, syn_output_test, test_size=0.5)

print("UBitName:"+ "anantram")
print("personNumber:" + "50249127")

print("UBitName: pdeshmuk")
print("personNumber: 50247649")

######################   Synthetic data   ####################

##########################################
###Close Form
##########################################
WcloseSyn = []
erms_val_old = 100
erms_tr_close_syn =[]
erms_val_close_syn = []
paramsCloseSyn = []
for K in range(10,20,2):
	for l in [x * 0.01 for x in range(1, 100,10)]:
		erms_val_new, erms_train, w, centers, spread = mainFuncClose(K, l, syn_input_train,syn_input_train.shape[0],syn_input_train.shape[1],syn_input_val,syn_output_train,syn_output_val)
		erms_tr_close_syn.append(erms_train)
		erms_val_close_syn.append(erms_val_new)
		if erms_val_new < erms_val_old:
			paramsCloseSyn = [K, l,  erms_val_new, erms_train, centers, spread]
			WcloseSyn = w
			erms_val_old = erms_val_new

design_mat_test = computeDesignMatrix(syn_input_test[np.newaxis, :, :], paramsCloseSyn[4], paramsCloseSyn[5])
erms_test_close_syn = eRMS(syn_output_test, design_mat_test, WcloseSyn, paramsCloseSyn[1])

print("erms_val_close_syn",paramsCloseSyn[2] )
print("erms_tr_close_syn", paramsCloseSyn[3])
print("erms_test_close_syn", erms_test_close_syn)
print("KClose_syn", paramsCloseSyn[0])
print("LambdaCloseSyn", paramsCloseSyn[1])



##########################################
###SGD
##########################################
erms_val_old = 100

patience = 10
validationSteps = 10
paramsSGDSyn = []
A = []
W=[]
learning_rate=0.1
num_epochs=100
erms_tr_sgd = []
erms_val_sgd = []

for K in range(10,20, 2):
	for l in [x * 0.01 for x in range(1, 100,10 )]:
		erms_val_new, erms_train,centers, spreads, w, stepsTaken, epoch, count= mainFuncSGD(K, l, syn_input_train,syn_input_train.shape[0],syn_input_train.shape[1], syn_input_val,syn_output_train,syn_output_val,learning_rate,validationSteps,num_epochs,patience )
		erms_tr_sgd.append(erms_train)
		erms_tr_sgd.append(erms_val_new)
		if erms_val_new < erms_val_old:
			paramsSGDSyn = [K, l,erms_val_new, erms_train,centers, spreads]
			W = w
			erms_val_old = erms_val_new




design_mat_test = computeDesignMatrix(syn_input_test[np.newaxis, :, :], paramsSGDSyn[4], paramsSGDSyn[5])
erms_test_sgd = eRMS(syn_output_test, design_mat_test, W, paramsSGDSyn[1])

print("erms_val_sgd_Syn", paramsSGDSyn[2])
print("erms_train_sgd_Syn", paramsSGDSyn[3])
print("erms_test_sgd_syn", erms_test_sgd)
print("KSGD_syn", paramsSGDSyn[0])
print("Lambda SGD_syn", paramsSGDSyn[1])


######################   LeoTR   ####################


letor_input_data = np.genfromtxt('/home/csgrad/pdeshmuk/ML_project2/Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt('/home/csgrad/pdeshmuk/ML_project2/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])

from sklearn.cross_validation import train_test_split
letor_input_train, letor_input_test, letor_output_train, letor_output_test = train_test_split(letor_input_data, letor_output_data, test_size=0.2)
letor_input_test, letor_input_val, letor_output_test, letor_output_val = train_test_split(letor_input_test, letor_output_test, test_size=0.5)

##########################################
###Close Form
##########################################
WcloseLetor = []
erms_val_old = 100
erms_tr_close_letor =[]
erms_val_close_letor = []
paramsClose_letor = []
for K in range(10,20,2):
	for l in [x * 0.01 for x in range(1, 100,10)]:
		erms_val_new, erms_train, w, centers, spread = mainFuncClose(K, l, letor_input_train,letor_input_train.shape[0],letor_input_train.shape[1],letor_input_val,letor_output_train,letor_output_val)
		erms_tr_close_letor.append(erms_train)
		erms_val_close_letor.append(erms_val_new)
		if erms_val_new < erms_val_old:
			paramsClose_letor = [K, l,  erms_val_new, erms_train, centers, spread]
			Wclose_letor = w
			erms_val_old = erms_val_new

design_mat_test_letor = computeDesignMatrix(letor_input_test[np.newaxis, :, :], paramsClose_letor[4], paramsClose_letor[5])
erms_test_close_letor = eRMS(letor_output_test, design_mat_test_letor, Wclose_letor, paramsClose_letor[1])

print("erms_val_close_letor",paramsClose_letor[2] )
print("erms_tr_close_letor", paramsClose_letor[3])
print("erms_test_close_letor", erms_test_close_letor)
print("KClose_Letor", paramsClose_letor[0])
print("LambdaClose_letor", paramsClose_letor[1])



##########################################
###SGD
##########################################
erms_val_old = 100

patience = 10
validationSteps = 10
paramsSGD_letor = []
A = []
W=[]
learning_rate=0.1
num_epochs=100
erms_tr_sgd_letor = []
erms_val_sgd_letor = []

for K in range(10,20, 2):
	for l in [x * 0.01 for x in range(1, 100,10 )]:
		erms_val_new, erms_train,centers, spreads, w, stepsTaken, epoch, count= mainFuncSGD(K, l, letor_input_train,letor_input_train.shape[0],letor_input_train.shape[1],letor_input_val,letor_output_train,letor_output_val,learning_rate,validationSteps,num_epochs,patience )
		erms_tr_sgd_letor.append(erms_train)
		erms_tr_sgd_letor.append(erms_val_new)
		if erms_val_new < erms_val_old:
			paramsSGD_letor = [K, l,erms_val_new, erms_train,centers, spreads]
			W = w
			erms_val_old = erms_val_new




design_mat_test_sgd_letor = computeDesignMatrix(letor_input_test[np.newaxis, :, :], paramsSGD_letor[4], paramsSGD_letor[5])
erms_test_sgd_letor = eRMS(letor_output_test, design_mat_test_sgd_letor, W, paramsSGD_letor[1])

print("erms_val_sgd_letor", paramsSGD_letor[2])
print("erms_train_sgd_letor", paramsSGD_letor[3])
print("erms_test_sgd_letor", erms_test_sgd_letor)
print("KSGD_letor", paramsSGD_letor[0])
print("Lambda SGD_letor", paramsSGD_letor[1])




