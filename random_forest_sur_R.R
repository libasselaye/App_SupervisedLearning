## Fichier: random_forest_sur_R.r
## Etudiants : Mouhamadou Mansour LO & Mame Libasse MBOUP
## Description : Projet Advanced Supervised Learning

#chargement des données
data <- read.csv("StudentsPerformance.csv", stringsAsFactors=TRUE, sep=";")
summary(data)

#affichage des premières lignes du jeu de données
head(data)
str(data)

data$gender = as.factor(data$gender)

# Subdivision en echantillon app et test
set.seed(123)
sample <- floor(0.70 * nrow(data))
spliter <- sample(seq_len(nrow(data)), size = sample)
app = data[spliter,]
test = data[-spliter,]

#Regardons la répartition de la classe sur les données initiales ainsi que sur 
print(table(data$gender))

#De meme sur l'app et le test 
table(app$gender) 
table(test$gender)
#Nous constatons que les proportions de la variable cible sont assez équilibrées

################################################################################
################        TREE DECISION           ################################
################################################################################
library(rpart)
arbre <- rpart(gender ~ ., data = app)
print(arbre)

#fonction d'évaluationde la qualité de la prédiction prenant en entrée la variable 
#cible observée et la prédiction du modèle

error_rate <- function(yobs,ypred){
  #matrice de confusion
  mc <- table(yobs,ypred)
  #taux d'erreur
  err <- 1.0 - sum(diag(mc))/sum(mc)
  return(err)
}

#prédiction sur échantillon test
pred <- predict(arbre ,newdata=test,type="class")
#taux d'erreur
print(error_rate(test$gender,pred))


################################################################################
################        RANDOM FOREST           ################################
################################################################################
library(randomForest)
#ntree étant le nombre d'arbres
rf <- randomForest(gender ~ ., data = app, ntree = 200)
#prédiction
predrf <- predict(rf,newdata=test,type="class")
#erreur en test
print(error_rate(test$gender,predrf))


#How to install Random Forest
urlPackage <- "https://cran.r-project.org/src/contrib/Archive/randomForest/randomForest_4.6-12.tar.gz"
install.packages(urlPackage, repos=NULL, type="source") 
