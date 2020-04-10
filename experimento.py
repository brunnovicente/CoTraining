import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.utils import shuffle
from AlgoritmoCoTraining import AlgoritmoCoTraining
from sklearn.metrics import accuracy_score, cohen_kappa_score

sca = MinMaxScaler()
base = 'epilepsia'
bases = ['mnist', 'fashion', 'usps', 'cifar10','stl10','covtype','epilepsia','reuters']
modelos = ['KNN','MLP','RF']

for modelo in modelos:

    caminho = 'D:/Drive UFRN/bases/'
    dados = pd.read_csv(caminho + base +'.csv')
    X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
    Y = dados['classe'].values
    
    dados = pd.DataFrame(X)
    dados['classe'] = Y
    rotulados = [50 , 100, 150, 200, 250, 300]
    porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]
    
    resultado = pd.DataFrame()
    acuraciai = []
    acuraciat = []
    kappai = []
    kappat = []
    
    
    for r, p in enumerate(porcentagem):
        
        
        inicio = time.time()
        
        
        print('Teste: '+str(rotulados[r]) +' - '+base+' - '+modelo)
        
        X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)
                
        """ PROCESSO TRANSDUTIVO """
        L, U, y, yu = train_test_split(X_train, y_train, train_size = p, test_size= 1.0 - p, stratify=y_train)
    
        if modelo == 'MLP':      
            classificador = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100)
        elif modelo == 'KNN':
            classificador = KNeighborsClassifier(n_neighbors=5)
        elif modelo == 'SVM':
            classificador = SVC(probability=True)
        elif modelo == 'RF':
            classificador = RandomForestClassifier(n_estimators=20)
        elif modelo == 'NB':
            classificador = GaussianNB()
        else:
            classificador = LogisticRegression()
            
        selfT = AlgoritmoCoTraining(classificador)
        
        #X_treino = np.concatenate((L, U))
        #Y_treino = np.concatenate((y.astype(str), np.full_like(yu.astype(str), "unlabeled")))
                
        selfT.fit(L, U, y)
            
        """ FASE TRANDUTIVA """
        acuraciat.append(accuracy_score(yu, selfT.predict(U)))
        kappat.append(cohen_kappa_score(yu, selfT.predict(U)))
        
        """ FASE INDUTIVA """
        acuraciai.append(accuracy_score(y_test, selfT.predict(X_test)))
        kappai.append(cohen_kappa_score(y_test, selfT.predict(X_test)))
             
        fim = time.time()
        tempo = np.round((fim - inicio)/60,2)
        print('........ Tempo: '+str(tempo)+' minutos.')
    
    resultado['R'] = rotulados
    resultado['AT'] = acuraciat
    resultado['KT'] = kappat
    resultado['KI'] = acuraciai
    resultado['KI'] = kappai
    
    resultado.to_csv('D:/Drive UFRN/Doutorado/Resultados/Artigo KBS/Co-training/'+modelo+'/'+base+'.csv')
    
dados = pd.read_csv('D:/Drive UFRN/bases/slt10.csv')