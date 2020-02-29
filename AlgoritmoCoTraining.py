# -*- coding: utf-8 -*-
from cotraining import CoTrainingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

class AlgoritmoCoTraining:
    
    def __init__(self, classificador):
        self.classificador = classificador
        self.cotrain = CoTrainingClassifier(self.classificador, k=100)
    
    def fit(self, L, U, y):
        metade = int((np.size(L, axis=1)/2))
        
        X = np.concatenate([L, U])
        
        X1 = X[ : , 0:metade]
        X2 = X[ : , metade:np.size(X, axis=1)]
                
        yn = np.zeros(np.size(U,axis=0)) - 1
        y = np.concatenate([y, yn])
        
        self.cotrain.fit(X1,X2,y)
    
    def predict(self, X):
        
        metade = int((np.size(X, axis=1)/2))
                
        X1 = X[ : , 0:metade]
        X2 = X[ : , metade:np.size(X, axis=1)]
        
        return self.cotrain.predict(X1, X2)
    
    """
    
    def treinar(self, L, U, y):
        total = np.size(L, axis=1)
        metade = int(total / 2)
        
        L1 = L[:, 0:metade]
        L2 = L[:, metade:total]
        U1 = U[:, 0:metade]
        U2 = U[:, metade:total]
        
        L1 = L1.tolist()
        L1.extend(U1.tolist())
        L2 = L2.tolist()
        L2.extend(U2.tolist())
        
        L1 = np.array(L1)
        L2 = np.array(L2)
        
        yn = np.zeros(np.size(U, axis=0)) - 1
        yr = y.tolist()
        yr.extend(yn)
        yr = np.array(yr)
        
        self.cotrain.fit(L1, L2, yr)
        preditas = self.cotrain.predict(U1, U2)
        
        return preditas
        """