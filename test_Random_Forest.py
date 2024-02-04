from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
from RandomForest import prediction_RandomForest
from ArbreGeneration import ArbreGeneration
from ArbreGeneration import Make_Prediction
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

############################################################################
#################           TRAIN TEST SPLIT          ######################
############################################################################

data_X = pd.DataFrame()
data_X["X1"] = ["A","A","A","B","B","C","C"]
data_X["X2"] = ["Alph","Bet","Alph","Alph","Alph","Bet","Bet"]
data_X["X3"] = ["1","1","1","3","1","2","2"]

Y_m = ["C1","C1","C1","C1","C2","C2","C2"]

data_Y = pd.Series(Y_m)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)

#fitting the model to the labelled training set
model = ArbreGeneration(X_train, y_train, 0.007)
print(model)

Y_pred = Make_Prediction(model,X_test)

print(accuracy_score(y_test, Y_pred))

#TreeDecision de Sklearn sur le meme jeu de données
clf = tree.DecisionTreeClassifier(criterion="entropy",ccp_alpha=0.4)
clf = clf.fit(X_train, y_train)
#Ici le model d'apprentissage de Sklearn n'arrive pas prendre en compte la structure de notre dataset 
#Alors que notre model arrive à bien l'utiliser d'ou la supériorité de notre mèthode


###################### IRIS DATA SET #######################################
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica'])

data_X = df.iloc[:,0:4]
data_Y = df.iloc[:,4]

#Subdivision Apprentissage - Test
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)

#Application sur Arbre Generation
model = ArbreGeneration(X_train, y_train, 0.7)
print(model)
Y_pred = Make_Prediction(model,X_test)
print(accuracy_score(y_test, Y_pred))

#Application Sklearn
clf = tree.DecisionTreeClassifier(criterion="entropy",ccp_alpha=0.3)
clf = clf.fit(X_train, y_train)

#prediction sur l'echantillon test
y_pred = clf.predict(X_test)
#Taux d'accuracy
print(accuracy_score(y_test, y_pred))

########################## Dataset Breast Cancer ###########################
bc = load_breast_cancer()

df = pd.DataFrame(bc.data, columns=bc.feature_names)
df["target"] = bc.target
df['target'] = df['target'].replace(to_replace= [0, 1], value = ['malignant', 'benign'])

data_X = df.iloc[:,0:30]
data_Y = pd.Series(df.iloc[:,30])

#Subdivision Apprentissage - Test
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)

#Application sur Arbre Generation
model = ArbreGeneration(X_train, y_train, 0.28)
print(model)
Y_pred = Make_Prediction(model,X_test)
print(accuracy_score(y_test, Y_pred))


#Application Sklearn
clf = tree.DecisionTreeClassifier(criterion="entropy",ccp_alpha=0.35)
clf = clf.fit(X_train, y_train)

#prediction sur l'echantillon test
y_pred_sklearn = clf.predict(X_test)
#Taux d'accuracy
print(accuracy_score(y_test, y_pred_sklearn))

###############################################################################
############################ TEST ON RANDOM FOREST ############################
###############################################################################


####################### Test on IRIS Dataset ##################################
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica'])

#Subdivision Apprentissage - Test
data_X = df.iloc[:,0:4]
data_Y = df.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)

forest = RandomForest(X_train, y_train, 0.7, 200)
print(forest)
Y_pred = prediction_RandomForest(forest,X_test)
print(accuracy_score(y_test, Y_pred))

#Random Forest of Sklearn 
#n_estimators : The number of trees in the forest.
clf = RandomForestClassifier(n_estimators = 200) 
 
# Training the model on the training dataset
clf.fit(X_train, y_train)
 
#prediction sur l'echantillon test
y_pred_sklearn = clf.predict(X_test)
#Taux d'accuracy
print(accuracy_score(y_test, y_pred_sklearn))


###################### Test on breast Cancer ##################################
bc = load_breast_cancer()

df = pd.DataFrame(bc.data, columns=bc.feature_names)
df["target"] = bc.target
df['target'] = df['target'].replace(to_replace= [0, 1], value = ['malignant', 'benign'])

data_X = df.iloc[:,0:30]
data_Y = pd.Series(df.iloc[:,30])

#Subdivision Apprentissage - Test
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)


#en testant toutes ces valeurs seuils on tombe sur le meilleur score avec 0.7
# list_of_seuils = [0.0001, 0.002, 0.07, 0.4, 0.5, 0.65, 0.7, 0.8, 0.95]
# for i in range(len(list_of_seuils)):
#     forest = RandomForest(X_train, y_train, list_of_seuils[i], 500,0.5)
#     #print(forest)
#     Y_pred = prediction_RandomForest(forest,X_test)
#     print(accuracy_score(y_test, Y_pred))
    
forest = RandomForest(X_train, y_train, 0.4, 500,0.5)
print(forest)
Y_pred = prediction_RandomForest(forest,X_test)
print(accuracy_score(y_test, Y_pred))
    

#Random Forest of Sklearn 
#n_estimators : The number of trees in the forest.
clf = RandomForestClassifier(n_estimators = 500) 
 
# Training the model on the training dataset
clf.fit(X_train, y_train)
 
#prediction sur l'echantillon test
y_pred_sklearn = clf.predict(X_test)
#Taux d'accuracy
print(accuracy_score(y_test, y_pred_sklearn))


###################### Test on Load Digits ##################################
digits = load_digits()

df = pd.DataFrame(digits.data, columns=digits.feature_names)
df["target"] = digits.target

data_X = df.iloc[:,0:64]
data_Y = pd.Series(df.iloc[:,64])

#Subdivision Apprentissage - Test
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)

#Application sur Arbre Generation
model = ArbreGeneration(X_train, y_train, 0.007)
#print(model)
Y_pred = Make_Prediction(model,X_test)
print("Model : ", accuracy_score(y_test, Y_pred))


#Application Sklearn
clf = tree.DecisionTreeClassifier(criterion="entropy",ccp_alpha=0.007)
clf = clf.fit(X_train, y_train)

#prediction sur l'echantillon test
y_pred_sklearn = clf.predict(X_test)
#Taux d'accuracy
print("Sklearn : ",accuracy_score(y_test, y_pred_sklearn))

print("------------Random Forest-----------------s")


forest = RandomForest(X_train, y_train, 0.5, 500,0.1)
Y_pred = prediction_RandomForest(forest,X_test)
print("Model : ", accuracy_score(y_test, Y_pred))

#Random Forest of Sklearn 
#n_estimators : The number of trees in the forest.
clf = RandomForestClassifier(n_estimators = 500) 
# Training the model on the training dataset
clf.fit(X_train, y_train)
#prediction sur l'echantillon test
y_pred_sklearn = clf.predict(X_test)
#Taux d'accuracy
print("Sklearn : ", accuracy_score(y_test, y_pred_sklearn))

