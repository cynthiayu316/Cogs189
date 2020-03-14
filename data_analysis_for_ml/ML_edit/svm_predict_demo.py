import pickle
import numpy as np

with open('./data_analysis_for_ml/svm.pkl', 'rb') as fid:
    svm_classifier = pickle.load(fid)

counter,num_iter=(0,10)
while(counter<num_iter):
    attention,medidation=np.random.randint(low=0,high=100,size=2)
    data=[[attention,medidation]]
    tag=svm_classifier.predict(data)[0]
    print("Current Tag: {}".format(str(tag)))
    counter+=1