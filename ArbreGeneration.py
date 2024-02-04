import math
import pandas as pd
import numpy as np

from Arbre import Node
from DivisionAttribut import Division_Attribut

#---------------------------------------------------------------------------------------
# Presentation : Cette fonction genere l'arbre de decision
# Entrée       : la X_m matrice des donnees relatives au noeud m (n*p), Y_m(n*1), teta : condition d'arret
# Sortie       : arbre de decision

def ArbreGeneration(X_m,Y_m,teta,E_anc=50,label= "-> ") :
    
    #On definit le nombre d'element du Noeud
    N_m = len(X_m) 
    
    Y_m = Y_m
    
    #On compte le nombre de classe de Y et ses frequences
    modal_value_Y = Y_m.value_counts()
    #Nous recuperons le nom des classes pour permettre la prediction
    modal_value_Y = modal_value_Y.reset_index()
    #On instancie l'entropie
    E = 0
    for l in range(len(modal_value_Y)):
        
        
        #On definit le nombre d'element du noeaud m pour la classe l  
        N_m_l = modal_value_Y.iloc[l,1]
                 
        #Nous calculons sum(p_m_l*log(p_m_l))
        E = E + (-N_m_l/N_m)*math.log2(N_m_l/N_m)
         
     #Nous comparons l'entropie trouvée avec la condition d'arret  
    if(E <= teta or E == E_anc):
        #Nous recuperons la classe majoritaire et nous etiquetons la feuille
        index_max = np.argmax(modal_value_Y.iloc[:,1])
        leaf = Node(modal_value_Y.iloc[index_max,0])
        leaf.set_label([label + str(modal_value_Y.iloc[index_max,0]),"Entropie = {}".format(E)])
        leaf.set_status()
        
        return leaf
    #Nous recuperons la colonne qui minimise l'entropie ainsi que le deltaj
    j_etoile,deltaj = Division_Attribut(X_m,Y_m)
    
    #Si nous ne trouvons pas de colonnes minimisant l'entropie en renvoie la classe majoritaire
    if(j_etoile == -21):
        #Nous recuperons la classe majoritaire et nous etiquetons la feuille
        index_max = np.argmax(modal_value_Y.iloc[:,1])
        leaf = Node(modal_value_Y.iloc[index_max,0])
        leaf.set_label([label + str(modal_value_Y.iloc[index_max,0]),"Entropie = {}".format(E)])
        leaf.set_status()
        
        return leaf
    
    label_node = [X_m.columns[j_etoile],"Entropie = {}".format(E)]
    
    #S1 est le noeud central et S est le noeud genere par la colonne qui minimise l'entropie
    S1 = Node(0)
    S1.set_label([label])
    
    S = Node(0)
    S.set_label(label_node)
  
    # S est un fils de S1
    S1.Ajouter_Branche(S)
    
    #Nous recuperons la colonne qui minimise l'entropie
    columns = pd.DataFrame(X_m.iloc[:,j_etoile])
    
    #Nous verifions si la colonne est qualitative
    if(deltaj == -2021.12345):
       
        #Nous recuperons les modalités de la colonne
       modal_value_X = columns.value_counts()
       modal_value_X = modal_value_X.reset_index()
       
       #Pour chaque modalité , Nous instancions une nouvelle sous-branche
       for i in range(len(modal_value_X)):
           
           X_m_prim = X_m[X_m.iloc[:,j_etoile] == modal_value_X.iloc[i,0]]
           
           Y_m_prim = Y_m.loc[X_m_prim.index]
                   
           #Nous generons un nouvel arbre(fils)
           S_prim = ArbreGeneration(X_m_prim,Y_m_prim,teta,E,"{} == {} -> ".format(X_m.columns[j_etoile],modal_value_X.iloc[i,0]))
           
           S_prim.set_column(X_m.columns[j_etoile])
           S_prim.set_deltaj(deltaj)
           S_prim.set_modalite(modal_value_X.iloc[i,0])
           
           S.Ajouter_Branche(S_prim)
           
     #Si la variable est quantitative alors :
    else:
        
        #Nous recuperons la partie gauche de l'arbre 
        X_m_priml = X_m[X_m[X_m.columns[j_etoile]] <= deltaj]
           
        Y_m_priml = Y_m.loc[X_m_priml.index]
             
        #Nous generons un nouvel arbre(fils gauche)   
        S_priml = ArbreGeneration(X_m_priml,Y_m_priml,teta,E,"{} <= {} -> ".format(X_m.columns[j_etoile],deltaj))
        
        S_priml.set_column(X_m.columns[j_etoile])
        S_priml.set_deltaj(deltaj)
        S_priml.set_sens("left")
        
        S.Ajouter_Branche(S_priml)
        
        
        #Nous recuperons la partie droite de l'arbre
        X_m_primr = X_m[X_m[X_m.columns[j_etoile]] > deltaj]
           
        Y_m_primr = Y_m.loc[X_m_primr.index]
           
         #Nous generons un nouvel arbre(fils droit)   
        S_primr = ArbreGeneration(X_m_primr,Y_m_primr,teta,E,"{} > {} -> ".format(X_m.columns[j_etoile],deltaj))
        
        S_primr.set_column(X_m.columns[j_etoile])
        S_primr.set_deltaj(deltaj)
        S_primr.set_sens("right")
                 
        S.Ajouter_Branche(S_primr)
        
    #Nous retournons le noeud obtenu
    return S1
#----------------------------------------------------------------------

# Presentation : Cette fonction Permet de faire une prediction
# Entrée       : L'arbre et X_m_Test
# Sortie       : La liste des predictions
def Make_A_Prediction(Branche_Arbre,vector_Test): 

    #La condition d'arret de la recurssion
    
    #Si la branche est une feuille et en plus une prediction
    if(Branche_Arbre.is_leaf() and Branche_Arbre.status() == 1):
        column = Branche_Arbre.get_column()
        deltaj = Branche_Arbre.get_deltaj()
        pred = ""
        
        #Si elle esr quantitative et possede un noeud pere
        if(Branche_Arbre.get_deltaj() == -2021.12345 and column != "vide"):
            modalite =   Branche_Arbre.get_modalite()
            if(vector_Test[column].values[0] == modalite):
             return Branche_Arbre.Valeur_noeud()
         
        #Si elle est qualitative
        else:
            sens = Branche_Arbre.get_sens() 
            if(sens == 0):
                if(vector_Test[column].values[0] <= deltaj):
                    return Branche_Arbre.Valeur_noeud()
                    
            if(sens == 1):
                if(vector_Test[column].values[0] > deltaj):
                    return Branche_Arbre.Valeur_noeud()
     
    #Nous cherchons plus profondemment la solution parmis les fils du noeud
    else:
        
        #Nous ne prenons que les branches avec des fils
        pred = ""
        if(Branche_Arbre.is_leaf() == False ):
            
            #Si la branche est un aggregat alors on passe
            if(Branche_Arbre.get_column() == "vide" ):
                
               fils = Branche_Arbre.get_fils()
            
               for i in range(len(fils)):
                  
                  pred = Make_A_Prediction(fils[i], vector_Test)
                  if(pred != ""):
                      return pred
            
                #Si la branche n'est pas un noeud aggregat on cherche notre prediction
            else:
                
               fils = Branche_Arbre.get_fils()
               column = Branche_Arbre.get_column()
                
               #Si la branche est qualitative
               if(Branche_Arbre.get_deltaj() == -2021.12345 ):
                  
                   modalite =   Branche_Arbre.get_modalite()    
                   
                   if(vector_Test[column].values[0] == modalite ):
                       
                        for i in range(len(fils)):
                            
                          pred = Make_A_Prediction(fils[i], vector_Test)
                          if(pred != ""):
                              return pred
                 
               #Si elle est quantitative   
               else:
                     
                    deltaj = Branche_Arbre.get_deltaj()
                    sens = Branche_Arbre.get_sens()
                    
                    if(sens == 0 ):
                        if(vector_Test[column].values[0] <= deltaj):
                            for i in range(len(fils)):
                               pred = Make_A_Prediction(fils[i], vector_Test)
                               if(pred != ""):
                                 return pred
                              
                    if(sens == 1 ):
                        if(vector_Test[column].values[0] > deltaj):
                            for i in range(len(fils)):
                               pred = Make_A_Prediction(fils[i], vector_Test)
                               if(pred != ""):
                                  return pred
    return pred
            
 #------------------------------------------------------------------------------------


# Presentation : Cette fonction Permet de faire des predictions
# Entrée       : L'arbre et X_m_Test
# Sortie       : La liste des predictions

def Make_Prediction(Branche_Arbre, Xm_Test) :
    
    prediction = []
    
    for i in range(len(Xm_Test)):
        prediction.append(Make_A_Prediction(Branche_Arbre,Xm_Test.iloc[i:i+1,:]))
        
    return prediction
#-------------------------------------------------------------------------------------
