#Chargement des packages
import math
import numpy as np
from collections import Counter
import collections


# Presentation : Cette fonction calcule le j* qui minimise l'entropie de l'arbre au noeud m
# Entrée       : une liste
# Sortie       : index des valeurs jamais vus dans une liste avec des valeurs dupliquées

def index_deltaj(liste):
    seen = set()
    seen_add = seen.add
    return [idx for idx, element in enumerate(liste) if not (element in seen or seen_add(element))]

#-------------------------------------------------------------------------------------------------------------

# Presentation : Cette fonction calcule le j* qui minimise l'entropie de l'arbre au noeud m
# Entrée       : la X_m matrice des donnees relatives au noeud m (n*p) et Y_m(n*1)
# Sortie       : indice j* qui minimise l'entropie du noeud m, et deltaj si il existe

def Division_Attribut(X_m,Y_m):
    
    #A l'initialisation , l'entropie minimale est mise a l'infini
    Min_E = math.inf
    
    #On initialise j_etoile
    j_etoile = -21
    
    #On initialise deltaj
    deltajr = -2021.12345
    
    #On initialise la variable N_m
    N_m = len(X_m)
    
    for i in range(len(X_m.columns)):
        
        if (X_m.iloc[:,i].dtypes == "float64" or X_m.iloc[:,i].dtypes == "int64" ) :
            
            #Si la variable n'est pas Quantitative donc Qualitative
            #Nous chargeons les colonnes correspondantes en les castant en liste
            columns = X_m.iloc[:,i].tolist()
            cible = Y_m.tolist()
            
            #Nous creons tupes permettant de compter les occurences (un peu comme groupby)
            data = zip(columns, cible)

            counter = Counter(data)
            
            #Nous ordonnons les comptes
            ordered_counter = counter.most_common()
            
            #Nous recuperons les valeurs de la colonnes ainsi que la prediction
            list1 = [row[0] for row in ordered_counter]
            
            list1,list2 = zip(*list1)
            
            list1 = list(list1)
            list2 = list(list2)
            
            #Nous recuperons l'index des valeurs dupliquées mais en ne prenant que le plus
            #grand des comptes pour qu'il soit le representant de cette classe de valeurs
            index = index_deltaj(list(list1))
             
            #Nous eliminons les valeurs non representatives 
            values = [list(list1)[i] for i in index]
            prediction = [list(list2)[i] for i in index]
            
            #Nous mettons dans un dictionnaire ordonne les donnees
            data = dict(zip(values,prediction))
            data = collections.OrderedDict(sorted(data.items()))
            keyList=sorted(data.keys())
            
            #Le Tableau contenant l'ensemble des deltaj que Nous allons tester
            Tabdeltaj = []
            
            #Le Tableau contenant L'entropie des deltaj que Nous allons tester
            TabEntdeltaj = []
            
            E = 0
            deltaj = 0
             
            #Nous ne prenons en compte que les changements de classes comme deltaj
            for z,v in enumerate(keyList):
              if z < len(keyList)-1:
               nextk = keyList[z+1]
               if(nextk != None):
                   
                #Si on a un changement de classe de 2 objet adjacent
                if(data[v] != data[nextk]):
                       E = 0
           
                       #Le delta j est pris entre les deux valeurs des deux objets
                       deltaj = (v + nextk)/2
                       
                       #La valeur est rajouté dans notre tab de delta
                       Tabdeltaj.append(deltaj)
                       
                       #Nous calculons les branches gauches et droites
                       index_left = [idx for idx, element in enumerate(columns) if element <= deltaj]
                       index_right = [idx for idx, element in enumerate(columns) if element > deltaj]
                       data_left = [columns[i] for i in index_left]
                       cible_left = [cible[i] for i in index_left]
                       data_right = [columns[i] for i in index_right]
                       cible_right = [cible[i] for i in index_right]

           
                       #Nous calculons les valeurs de chaque branche N_m_k
                       N_m_k_r =len(data_right)
                       
                       N_m_k_l = len(data_left)
                       
                       #Nous calculons les nombre d'elements de chaque branche
                       #Suivant chaque classe
                       modal_value_Y_r = Counter(cible_right)
                       
                       modal_value_Y_l = Counter(cible_left)
           
           
                       somme_r = 0
                       
                       somme_l = 0
                                                    
                       #Nous calculons les valeurs d'entropie
                       for l in modal_value_Y_r.values():
                           N_m_kl_r = l
                       
                           #Nous calculons sum(p_m_kl_r*log(p_m_kl_r))
                           somme_r = somme_r + (N_m_kl_r/N_m_k_r)*math.log2(N_m_kl_r/N_m_k_r)
                       
                       for v in modal_value_Y_l.values():
                           
                           N_m_kl_l = v
                       
                           #Nous calculons sum(p_m_kl_l*log(p_m_kl_l))
                           somme_l = somme_l + (N_m_kl_l/N_m_k_l)*math.log2(N_m_kl_l/N_m_k_l)
                       E = (-N_m_k_r/N_m)*somme_r + (-N_m_k_l/N_m)*somme_l
                       
                       TabEntdeltaj.append(E)
                       
            if(len(TabEntdeltaj)!= 0):
              index_min = np.argmin(TabEntdeltaj)
              E_min = TabEntdeltaj[index_min]
              deltaj_min = Tabdeltaj[index_min]
                   
            
              #Nous comparons la valeur de l'entropie trouvée a celle du minimum actuel
              if(E_min < Min_E):
                
                Min_E = E_min
                j_etoile = i
                deltajr = deltaj_min
                          
        else:
            
            #Si la variable n'est pas Quantitative donc Qualitative
            #Nous chargeons les colonnes correspondantes en les castant en liste
            columns = X_m.iloc[:,i].tolist()
            cible = Y_m.tolist() 
            
            #Nous recuperons ses modalités et les frequences relatives
            modal_value_X = Counter(columns)
            
            E = 0
            
            for k,v in modal_value_X.items():
                
               #Nous creons le subset correspondant
               index = [idx for idx, element in enumerate(columns) if element == k]
               cible1 = [cible[i] for i in index]
              
               #Calcul des N_m_k 
               N_m_k = v
               
               #Nous recuperons les frequences de chaque classes par rapport a une modalité
               modal_value_Y = Counter(cible1)
              
               somme = 0
               
               for l,m in modal_value_Y.items():
                  
                 N_m_kl = m
                 
                 #Nous calculons sum(p_m_kl*log(p_m_kl))
                 somme = somme + (N_m_kl/N_m_k)*math.log2(N_m_kl/N_m_k)
               
                #Nous calculons -sum(p_m_l*...)
               E = E + (-N_m_k/N_m)*somme

            
            #Nous comparons la valeur de l'entropie trouvé a celle du minimum actuel
            if(E < Min_E):
                
                Min_E = E
                j_etoile = i
                deltajr = -2021.12345
                
    return j_etoile,deltajr

#----------------------------------------------------------------------

