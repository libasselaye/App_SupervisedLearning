from ArbreGeneration import ArbreGeneration
from ArbreGeneration import Make_Prediction
import pandas as pd
import numpy as np
import random
import math

def RandomForest(X, Y, theta, k) :
    #data = pd.concat([X, Y], axis=1)
    data = X.join(Y)
    #liste des arbres
    all_forest = []
    
    for j in range(k) :
        #print("Inference Arbre Numero : " + str(j))
        frac = random.uniform(0.2,1)
        data_j = data.sample(frac = frac, replace  = False) # Echantillon bootstraping       
        #Ici nous allons sample le dataframe initial et le subdiviser par la suite en features d'une part
        #et de la target d'autre part 
        X_m = data_j.iloc[:,:-1]
        Y_m = data_j.iloc[:,-1]
        # Sélection aléatoire de r attributs
        r = random.randint(2,len(X_m.columns))
        r_attrib = np.random.choice(X_m.columns, size = r, replace  = False) 
        #r_attrib = np.random.choice(X_m.columns, size = round( math.sqrt(len(X_m.columns)) ), replace  = False)
        #On cree un new Xm contenant que les nouvelles variables choisies aléatoirement
        #X_m = X_m.loc[:,r_attrib] 
        X_m = X_m[r_attrib]
        fj = ArbreGeneration(X_m, Y_m, theta)  # Génére l'arbre 
        all_forest.append(fj)
        
    return all_forest

def prediction_RandomForest(forest, X_test):
    #declaration de liste qui vont contenir les prédictions
    Y_pred = []
    Y_final = []
    for i in range(len(forest)):
        Y = Make_Prediction(forest[i], X_test)
        Y_pred.append(Y)
    #on recupere l'argmax de chaque colonne du dataframe
    mat_forest = pd.DataFrame(Y_pred)
    for i in range(len(mat_forest.columns)):
        Y_final.append( mat_forest.iloc[:,i].mode()[0] )
    return Y_final



 