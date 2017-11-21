import numpy as np
import math
from scipy.cluster.vq import kmeans2

def computeDesignMatrix(X, centers, spreads):
        basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads)*(X - centers),axis=2)/(-2)).T
        return np.insert(basis_func_outputs, 0, 1, axis=1)

def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix),
        np.matmul(design_matrix.T, output_data)).flatten()

def eRMS(dataset, design_mat, wML, lambdas):
    out = np.matmul(design_mat, wML).reshape([-1,1])
    err = np.sum(np.power((dataset - out),2))/2 + lambdas*0.5*np.matmul(wML.T, wML)
    erms = math.sqrt(2*err/dataset.shape[0])
    return erms


def SGD_sol(learning_rate,validationSteps,num_epochs,L2_lambda,design_matrix,output_data, p, validation_dataset, design_mat_val ):
        weights = np.random.randn(1, design_matrix.shape[1])
        count = 0
        m=0
        err_val_old=100
        ##minibatch_size=20
        minibatch_size = int(design_matrix.shape[0]/ validationSteps)
        ##print("minibatch_size", minibatch_size)
        ##print("num_epochs", num_epochs)
        for epoch in range(num_epochs):
            for i in range(int(design_matrix.shape[0]/minibatch_size)):
                    indx = np.random.choice(design_matrix.shape[0],minibatch_size,replace=False)
                    phi = design_matrix[indx]
                    t = output_data[indx]
                    ED = np.matmul((np.matmul(phi, weights.T) - t).T, phi)
                    E = (ED + L2_lambda*weights)/minibatch_size 
                    weights = weights - learning_rate * E
            m = m + validationSteps
            err_val_new = eRMS(validation_dataset, design_mat_val, weights.T, L2_lambda)
            ##print("err_val_new", err_val_new)
            if err_val_new <= err_val_old:
               err_val_old = err_val_new
               wOPT = weights
            else:
                if count > p: 
                    break
                else: 
                    count = count + 1
        return wOPT.flatten(),m, epoch, count

def mainFuncSGD(K, lambdas, trainInData, N, M, valInData, trainOutData, valOutData,learning_rate,validationSteps,num_epochs,patience ):  
    A=[]
    W=[]
    erms_old = 100
    kmeans = kmeans2(trainInData, K, minit='points')
    centers = kmeans[0]
    centers = centers[:, np.newaxis, :]
    labels = np.array(kmeans[1])
    spreads = np.ndarray(shape=(K,M,M))

    for i in range(K):
        ##cluster_wise_data[i]=letor_input_train[np.where(labels ==i)]
        spreads[i] = np.linalg.pinv(np.cov(trainInData[np.where(labels ==i)].T))

    design_mat_train = computeDesignMatrix(trainInData[np.newaxis, :, :], centers, spreads)
    design_mat_val = computeDesignMatrix(valInData[np.newaxis, :, :], centers, spreads)

    w, stepsTaken, epoch, count = SGD_sol(learning_rate,validationSteps,num_epochs,lambdas,design_mat_train,trainOutData, patience, valOutData, design_mat_val )
    ##print("W",w)
    ###array([ 0.31814967,  2.24195525,  0.18953134, -1.96074679,  1.85905163])
    erms_val = eRMS(valOutData, design_mat_val, w, lambdas)
    erms_train = eRMS(trainOutData, design_mat_train, w, lambdas)
    ##wML = closed_form_sol(lambdas, design_mat_train, letor_output_train)

    return erms_val, erms_train, centers, spreads, w, stepsTaken, epoch, count

def mainFuncClose(K, lambdas, trainInData, N, M, valInData, trainOutData, valOutData ):  
    kmeans = kmeans2(trainInData, K, minit='points')
    centers = kmeans[0]
    centers = centers[:, np.newaxis, :]
    labels = np.array(kmeans[1])
    spreads = np.ndarray(shape=(K,M,M))

    for i in range(K):
        ##cluster_wise_data[i]=letor_input_train[np.where(labels ==i)]
        spreads[i] = np.linalg.pinv(np.cov(trainInData[np.where(labels ==i)].T))

    design_mat_train = computeDesignMatrix(trainInData[np.newaxis, :, :], centers, spreads)
    w = closed_form_sol(lambdas, design_mat_train, trainOutData)
    ###array([ 0.31814967,  2.24195525,  0.18953134, -1.96074679,  1.85905163])

    erms_train = eRMS(trainOutData, design_mat_train, w, lambdas)

    design_mat_val = computeDesignMatrix(valInData[np.newaxis, :, :], centers, spreads)
    ##wML = closed_form_sol(lambdas, design_mat_train, letor_output_train)
    erms_val = eRMS(valOutData, design_mat_val, w, lambdas)

    return erms_val, erms_train, w, centers, spreads

