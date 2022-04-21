# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:32:18 2022

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ks_2samp
import math
import seaborn as sns


df_r = pd.read_csv("./Data/‏‏131_subjects.csv")
df_s = pd.read_csv("./Data/400_synthetic.csv")

x, y = df_r.iloc[:,1:-3], df_s.iloc[:,1:-1]

print('exploring data...')
corr1 = x.corr(method='spearman')
plt.figure(figsize=(20, 10))
matrix = np.triu(corr1)
heatmap = sns.heatmap(corr1, vmin=-1, vmax=1, cmap='RdYlGn',mask=matrix)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12)
heatmap.set(xlabel='Features')
heatmap.collections[0].colorbar.set_label("Corr. [r]")
plt.savefig('./Figs/real_heatmap.png', dpi=300, bbox_inches='tight') 

corr2 = y.corr(method='spearman')
plt.figure(figsize=(20, 10))
matrix = np.triu(corr2)
heatmap = sns.heatmap(corr2, vmin=-1, vmax=1, cmap='RdYlGn',mask=matrix)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12)
heatmap.set(xlabel='Features')
heatmap.collections[0].colorbar.set_label("Corr. [r]")
plt.savefig('./Figs/syn_heatmap.png', dpi=300, bbox_inches='tight') 

corr_diff = abs(corr1-corr2)
plt.figure(figsize=(20, 10))
matrix = np.triu(corr_diff)
#median = np.median(corr_diff)
threshold = 0.3
corr_diff = corr_diff.clip(threshold)
corr_diff = corr_diff.replace(threshold,0)
heatmap = sns.heatmap(corr_diff, vmin=-1, vmax=1, cmap='RdYlGn',mask=matrix)
heatmap.set_title(f'Correlation diff. between real and synthetic \n(above: {threshold})', fontdict={'fontsize':16}, pad=12)
heatmap.set(xlabel='Features')
heatmap.collections[0].colorbar.set_label("Corr. [r]")
plt.savefig(f'./Figs/compare_heatmap_by_{threshold}.png', dpi=300, bbox_inches='tight') 
print('-correlation figure created')

