import numpy as np
import math
from scipy.cluster.vq import kmeans2
from lib import *
from sklearn.cross_validation import train_test_split



syn_input_data = np.loadtxt('/home/csgrad/pdeshmuk/ML_project2/input.csv', delimiter=',')
syn_output_data = np.loadtxt('/home/csgrad/pdeshmuk/ML_project2/output.csv', delimiter=',').reshape([-1, 1])

syn_input_train, syn_input_test, syn_output_train, syn_output_test = train_test_split(syn_input_data, syn_output_data, test_size=0.2)
syn_input_test, syn_input_val, syn_output_test, syn_output_val = train_test_split(syn_input_test, syn_output_test, test_size=0.5)


##########################################
###SGD
##########################################
erms_val_old = 100

patience = 10
validationSteps = 10
paramsSGDSyn = []
A = []
W= []
learning_rate=0.1
num_epochs=100
erms_tr_sgd = []
erms_val_sgd = []

for K in range(2,50,5):
	for l in [x * 0.01 for x in range(1, 100,10 )]:
		erms_val_new, erms_train,centers, spreads, W, A = mainFuncSGD(K, l, syn_input_train,syn_input_train.shape[0],syn_input_train.shape[1], syn_input_val,syn_output_train,syn_output_val,learning_rate,validationSteps,num_epochs,patience )
		erms_tr_sgd.append(erms_train)
		erms_tr_sgd.append(erms_val_new)
		if erms_val_new < erms_val_old:
			paramsSGDSyn = [K, l,erms_val_new, erms_train,centers, spreads, A]
			erms_val_old = erms_val_new

p = paramsSGDSyn[5]
print(p)

design_mat_test = computeDesignMatrix(syn_input_test[np.newaxis, :, :], paramsSGDSyn[4], paramsSGDSyn[5])
erms_test_sgd = eRMS(syn_output_test, design_mat_test, p[0], paramsSGDSyn[1])

print("erms_test_sgd", erms_test_sgd)
print("KSGD", paramsSGDSyn[0])
print("Lambda SGD", paramsSGDSyn[1])

