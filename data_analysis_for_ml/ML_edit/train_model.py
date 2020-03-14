import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from ml_helpers import *
import pickle

output_path='././data_analysis_for_ml/svm.pkl'
data_path='./data_analysis_for_ml/output_after_sliding.txt'
if __name__ == "__main__":
    print("INFO: Loading Dataset from {}".format(data_path))
    with open(data_path) as f:
        content=f.readlines()
    element_list=[] # initialize a list
    test={} # initialize a dict
    for i in range(len(content)):
        content[i] = content[i].strip('\n')
        content[i] = content[i].strip('[')
        content[i] = content[i].strip(']')
        content[i] = content[i].split(", ")
        content[i]=[float(c) for c in content[i]]
    #     print(content)
        key = i+1
        test[key] = []
        for j in range(len(content[i])):
            element = (content[i][j])
            test[key].append(content[i][j])
    test_list = [[data, label] for data, labels in test.items() for label in labels ] 
    df = pd.DataFrame(test_list, columns=['labels', 'att_data'])
    arrays = df.to_numpy()
    med_array = np.random.randint(20, 65, size=len(arrays))
    med_array = np.array(med_array).astype(np.float)
    df_med = pd.DataFrame(med_array, columns = ['med_data'])
    df = pd.concat([df, df_med], axis=1)
    df[['att_data','med_data' ]]
    print("INFO: Done Loading")

    # create training and testing vars
    # separate into four categories, 
    # test_size=0.2 => the percentage of data that should be held over for testing, usually around 80/20 
    X_train, X_test, y_train, y_test = train_test_split(df[['att_data','med_data' ]], arrays[:,0], test_size=0.2) 
    X_train=np.reshape(X_train,(682, 2))

    print("INFO: Done Train Test Split")


    C_list = [1, 10, 100, 1000, 10000]
    gamma_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    opt_e_training = 1.0   # Optimal training error.
    opt_classifier = None  # Optimal classifier.
    opt_C          = None  # Optimal C.
    opt_gamma      = None  # Optimal gamma.

    #initialize everything
    best_C=-1
    best_gamma=-1
    best_error=10000000
    best_classifier=None 
    training_errors=np.zeros((len(C_list),len(gamma_list))) #initialize training_errors

    for i in range(len(C_list)):
        
        for j in range(len(gamma_list)): 
            c=C_list[i]
            gamma=gamma_list[j] 
            classifier=svm.SVC(C=c, gamma=gamma) 
            classifier=classifier.fit(X_train,y_train)
            e_training=calc_error(X_train, y_train, classifier) 
            training_errors[i][j]=e_training 
            
            if(e_training<best_error):
                best_error=e_training 
                #print(best_error)
                best_C=c
                best_gamma=gamma 
                best_classifier=classifier

    print("INFO: SVM BEST TRAIN ERROR: {:.3f}".format(best_error))

    with open(output_path, 'wb') as f:
        pickle.dump(best_classifier, f)

    print("INFO: Saved Model at {}".format(output_path))