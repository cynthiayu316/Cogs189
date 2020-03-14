from sklearn.metrics import accuracy_score

def calc_error(X, Y, classifier):
    Y_pred = classifier.predict(X) 
    #print(Y_pred)
    e = accuracy_score(Y, Y_pred) 
    #print("test error:", 1-e)
    return 1-e #1-e