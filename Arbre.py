#La classe Node permet de generer l'Arbre(Graphe acyclique)
class Node:
    
    #Initialisation du Noeud
    def __init__(self, data):
        self.fils = []
        self.data = data
        self.mod = ""
        self.label = []
        self.sens = -1
        self.column = "vide"
        self.deltaj = -2021.12345
        self.statut_pred = 0
     
    #Cette fonction permet d'ajouter des Branches(fils)
    def Ajouter_Branche(self,Node):
        self.fils.append(Node)
        
    #Cette fonction permet d'ajouter des Branches(fils)
    def get_fils(self):
        return self.fils
     
    #Cette fonction permet de definir la valeur du noeud
    def set_data(self,data):
        self.data = data
        
    #Cette fonction permet de recuperer la valeur d'un noeud
    def Valeur_noeud(self):
        return self.data
    
    #Cette fonction permet de verifier si un noeud est une feuille
    def is_leaf(self):
        return (len(self.fils) == 0)
    
    #Cette fonction permet de dire qu'une feuille est une prediction
    def set_status(self):
        self.statut_pred = 1
    
    #Cette fonction permet de savoir si une feuille est une prediction
    def status(self):
        return self.statut_pred
    
     #Cette fonction permet de definir le label d'une feuille
    def set_label(self,label):
        self.label = label
        
    #Cette fonction permet de definir le sens de l'inegalité du deltaj
    def set_sens(self,sens):
        if(sens == "left"):
         self.sens = 0
         
        else:
         self.sens = 1
         
    #Cette fonction permet de d'obtenir le sens de l'inegalité du deltaj
    def get_sens(self):
         return self.sens
    
    #Cette fonction permet de definir la colonne relié au noeud
    def set_column(self,column):
        self.column = column
        
    #Cette fonction permet d'obtenir la colonne relié au noeud
    def get_column(self):
        return self.column
    
    #Cette fonction permet de definir le deltaj
    def set_deltaj(self,deltaj):
        self.deltaj = deltaj
    
    #Cette fonction permet d'obtenir le deltaj
    def get_deltaj(self):
        return self.deltaj
    
    #Cette fonction permet de definir une modalite
    def set_modalite(self,mod):
        self.mod = mod
    
    #Cette fonction permet d'obtenir la modalite
    def get_modalite(self):
        return self.mod
    #Cette fonction permet de print un Node(L'arbre est un Noeud contenant des branches fils)     
    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.label)+"\n"
        for child in self.fils:
            ret += child.__repr__(level+1)
        return ret
          