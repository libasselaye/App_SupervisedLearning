from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
from RandomForest import prediction_RandomForest
from ArbreGeneration import ArbreGeneration
from ArbreGeneration import Make_Prediction
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

#Chargement du Jeu de données
data = pd.read_csv('StudentsPerformance.csv', sep=";")
#data.dtypes
#categorical columns
categorical_cols = ['race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
#transfomations des variables catégorielles en matrices indicatrices
data = pd.get_dummies(data, columns=categorical_cols)

data.gender.value_counts()
data.head()
data.describe()

X = data.drop(['gender'], axis=1)
Y = data['gender']
#Subdivision en Apprentissage - Test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#Application sur Arbre Generation
model = ArbreGeneration(X_train, y_train, 0.75)
# print(model)
Y_pred = Make_Prediction(model,X_test)
score_tree_model = accuracy_score(y_test, Y_pred)
print("SCORE TREE DECISION MODEL:" ,score_tree_model)

#Application Sklearn
clf = tree.DecisionTreeClassifier(criterion="entropy",ccp_alpha=0.01)
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)
#prediction sur l'echantillon test
y_pred = clf.predict(X_test)
#Taux d'accuracy
score_tree_sklearn = accuracy_score(y_test, y_pred)
print("SCORE TREE DECISION SKLEARN:" ,score_tree_sklearn)
    
###############################################################################
######################        RANDOM FOREST         ###########################
###############################################################################

#Random Forest of our model
forest = RandomForest(X_train, y_train, 0.35, 100)
# print(forest)
Y_pred = prediction_RandomForest(forest,X_test)
score_random_model = accuracy_score(y_test, Y_pred)
print("SCORE RANDOM FOREST MODEL :" , score_random_model)

#Random Forest of Sklearn 
#n_estimators : The number of trees in the forest.
clf = RandomForestClassifier(n_estimators = 500) 
# Training the model on the training dataset
clf.fit(X_train, y_train)
#prediction sur l'echantillon test
y_pred_sklearn = clf.predict(X_test)
#Taux d'accuracy
score_random_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("SCORE RANDOM FOREST SKLEARN:", score_random_sklearn)

###############################################################################
#################   Deep Learning avec Keras        ###########################
###############################################################################
import keras
import numpy
#architecture
from keras.models import Sequential
#couches
from keras.layers import Dense

#transformation de la target variable pour l'apprentissage sur keras
my_train = pd.get_dummies(y_train)
my_test = pd.get_dummies(y_train)
# y_train.value_counts()
# my_train.value_counts()

############################# Perceptron simple ###############################
#instanciation du modèle
PS = Sequential()

#ajout de la couche "entrée -> sortie"
#my_train.shape[1] : nb de modalités de la var. cible en sortie
#X_train.shape[1] : nb de var. explicatives en entrée
PS.add(Dense(units=my_train.shape[1],input_dim=X_train.shape[1],activation='softmax'))

#vérification
PS.get_config()

#compilation - fonction à optimiser et algo. d'optimisation
PS.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

#entraînement sur l'échantillon d'apprentissage
PS.fit(X_train,my_train,epochs=150,batch_size=20)

#prédiction sur l'échantillon test
predPS = PS.predict(X_test)
print(predPS.shape)

#on convertit les probabilités en numéro classes d'appartenance
idPredPS = numpy.argmax(predPS,axis=1)
print(idPredPS[:5])
#transformation des numéros en classes prédites
clPredPS = my_train.columns[idPredPS]
print(clPredPS[:5])
score_perceptron = accuracy_score(y_test, clPredPS)
print("SCORE perceptron simple:", score_perceptron)

############################ Perceptron Multicouches ##########################
#instanciation du modèle
PMC = Sequential()

#ajout de la première couche "entrée -> cachée"
#50 neurones dans la couche cachée
PMC.add(Dense(units=50,input_dim=X_train.shape[1],activation='relu'))

#ajout de la seconde couche "cachée -> sortie"
#mYTrain.shape[1] neurones dans la couche cachée, nb de modalités de la var. cible
PMC.add(Dense(units=my_train.shape[1],activation='softmax'))

#vérification
PMC.get_config()

#compilation - fonction à optimiser et algo. d'optimisation
PMC.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

#entraînement sur l'échantillon d'apprentissage
PMC.fit(X_train,my_train,epochs=150,batch_size=20)

#Nous pouvons également utiliser l'option évaluation de l'outil keras
#en meme temps nous allons transformer y_test en matrice d'indicatrices
print(PMC.evaluate(X_test,pd.get_dummies(y_test)))
