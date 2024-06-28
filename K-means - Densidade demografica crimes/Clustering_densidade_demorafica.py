# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:46:22 2024

@author: Matheus Miyamoto
"""

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%%Importando o banco de dados e tratando as variaveis
densidade_parana = pd.read_csv("consulta (3).csv", sep=";", decimal=",",encoding='latin1')
banco_parana = densidade_parana

banco_parana['Densidade Demográfica (hab/km²)'] = banco_parana['Densidade Demográfica (hab/km²)'].str.replace('.', '')
banco_parana['Densidade Demográfica (hab/km²)'] = banco_parana['Densidade Demográfica (hab/km²)'].str.replace(',', '.')
banco_parana['Densidade Demográfica (hab/km²)'] = pd.to_numeric(banco_parana['Densidade Demográfica (hab/km²)'], errors='coerce')

banco_parana['Crimes de Estupro '] = banco_parana['Crimes de Estupro '].str.replace('.', '')
banco_parana['Crimes de Estupro '] = banco_parana['Crimes de Estupro '].str.replace(',', '.')
banco_parana['Crimes de Estupro '] = pd.to_numeric(banco_parana['Crimes de Estupro '], errors='coerce')

banco_parana['Crimes de Roubo '] = banco_parana['Crimes de Roubo '].str.replace('.', '')
banco_parana['Crimes de Roubo '] = banco_parana['Crimes de Roubo '].str.replace(',', '.')
banco_parana['Crimes de Roubo '] = pd.to_numeric(banco_parana['Crimes de Roubo '], errors='coerce')

banco_parana = banco_parana.drop(columns=['Localidade', 'Ano'])
banco_parana = banco_parana.dropna()
densidade_parana = densidade_parana.dropna()
#%%Fazendo padronização Z-Score

par_pad = banco_parana.apply(zscore,ddof=1)
#%%Metodos Elbow e Silhueta

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(par_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) 
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(par_pad)
    silhueta.append(silhouette_score(par_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()
#%%Estou separando em 3 clusters pois quero ter uma visão de baixo, medio e alto
kmeans_final = KMeans(n_clusters=3, init = 'random', random_state=100).fit(par_pad)

kmeans_clusters = kmeans_final.labels_
densidade_parana['cluster_kmeans'] = kmeans_clusters
par_pad['cluster_kmeans'] = kmeans_clusters
densidade_parana['cluster_kmeans'] = densidade_parana['cluster_kmeans'].astype('category')
par_pad['cluster_kmeans'] = par_pad['cluster_kmeans'].astype('category')

#%% ANOVA

# Densidade Demográfica (hab/km²)
pg.anova(dv='Densidade Demográfica (hab/km²)', 
         between='cluster_kmeans', 
         data=par_pad,
         detailed=True).T

# Crimes de Estupro  
pg.anova(dv='Crimes de Estupro ', 
         between='cluster_kmeans', 
         data=par_pad,
         detailed=True).T

# Crimes de Roubo 
pg.anova(dv='Crimes de Roubo ', 
         between='cluster_kmeans', 
         data=par_pad,
         detailed=True).T
#%%

fig = px.scatter_3d(densidade_parana, 
                    x='Densidade Demográfica (hab/km²)', 
                    y='Crimes de Roubo ', 
                    z='Crimes de Roubo ',
                    color='cluster_kmeans')
fig.show()