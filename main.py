# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:20:38 2022

@author: Lucie
"""

import ACP as acp
import numpy as np

#%% ---------------------------------------importation des données----------------------------------------

Ebrut = np.genfromtxt("iris.csv", dtype = str, delimiter=',')
Elabelscolonne = Ebrut[0,:-1]                                                   #sepal_length,sepal_width,petal_length,petal_width
Elabelsligne = Ebrut[1:,-1]                                                     #espece
E = Ebrut[1:,:-1].astype('float')

Hbrut = np.genfromtxt("Howellmod.csv", dtype = str, delimiter=',')
Hlabelscolonne = Hbrut[0,1:]                                                    #dimensions
label = Hbrut[1:,0]  
for color, x in enumerate(np.unique(label)):
    label[label == x] = color
Hlabelsligne = np.array(list(map(lambda x: int(x), label)))                                                    #population
H = Hbrut[1:,1:].astype('float')

Pbrut = np.genfromtxt("Pizzamod.csv", dtype = str, delimiter=',')
Plabelscolonne = Pbrut[0,:-1]                                                   #mois,prot,fat,ash,sodium,carb,cal
Plabel = Pbrut[1:,-1]  
for color, x in enumerate(np.unique(Plabel)):
    Plabel[Plabel == x] = color
Plabelsligne = np.array((list(map(lambda x: int(x), Plabel))))                                            #brand
P = Pbrut[1:,1:].astype('float')

#%% test approx

print(acp.approx(E,2))

#%%% test correlationdirprinc

print(acp.correlationdirprinc(E,2))

#%% --------------------------------------------test ACP2D------------------------------------------------

print(acp.ACP2D(E,Elabelsligne,Elabelscolonne))
#%%
print(acp.ACP2D(P,Plabelsligne,Plabelscolonne))
#%%
print(acp.ACP2D(H,Hlabelsligne,Hlabelscolonne))

#%% --------------------------------------------test ACP3D------------------------------------------------

print(acp.ACP3D(E,Elabelsligne,Elabelscolonne))
#%%
print(acp.ACP3D(P,Plabelsligne,Plabelscolonne))
#%%
print(acp.ACP3D(H,Hlabelsligne,Hlabelscolonne))

#%% ---------------------------------------------test ACP------------------------------------------------- 

print(acp.ACP(E,Elabelsligne,Elabelscolonne))
#%%
print(acp.ACP(P,Plabelsligne,Plabelscolonne))
#%%
print(acp.ACP(H,Hlabelsligne,Hlabelscolonne))
