# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:23:52 2022

@author: User
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ks_2samp
import math
import seaborn as sns

### load data 
df = pd.read_csv("./Data/‏‏131_subjects.csv")
x = df.iloc[:,1:-3] # data without tags
y = df.iloc[:,-3] # tags
data = df.iloc[:,1:-2] # data

##preprocessing for CTGAN: declare discrete variables 
discrete_columns = [
    'gender',
    'E_score',
    'E_score2',
    'A_score',
    'A_score2',
    'C_score',
    'C_score2',
    'N_score',
    'N_score2',
    'O_score',
    'O_score2',
    'Gpain',
    'Gsound',
    'physical',
    'sterm_prior',
    'lterm_prior',
    'L1',
    'L2',
    'L3',
    'N1',
    'N2',
    'N3',
    'SP1_d',
    'SP1_m',
    'SP2_d',
    'SP2_m',
    'tag'
]

###part 1: CTGAN
from ctgan import CTGANSynthesizer
from table_evaluator import TableEvaluator
import time
import warnings
warnings.filterwarnings("ignore")

t_n_bad = []
n_epochs = [4800] # optimal
for e in n_epochs:
    t0 = time.time()
    ctgan = CTGANSynthesizer(epochs=e)
    ctgan.fit(data, discrete_columns)
    t1 = time.time()
    total = t1-t0
    print(f'epochs:{e} took: {total:.3f} sec to train')
    
    num_bad = []
    samples = [400] # Optimal
    for s in samples:
        print(f'-epoch:{e}')
        path = str(f'./Figs/{e}/{s}/')
        synthetic_data = ctgan.sample(s)
        #print(synthetic_data.head())
        table_evaluator = TableEvaluator(data, synthetic_data)
        table_evaluator.visual_evaluation(save_dir=path)
        res = table_evaluator.evaluate(target_col='tag',return_outputs = True)
        k_s=res.get('Kolmogorov-Smirnov statistic') # exctract KS test pval
        features = []
        count = 0
        for k,v in k_s.items():
            if v.get('equality') != 'identical':
                features.append(k)
                count+=1
        num_bad.append(count)
        print (f'--sample: {s}: ---features {features} are bad')
    t_n_bad.append(num_bad)
    plt.close('all')
    
resample = [400]
num_bad = []
t_n_bad = []
for s in resample:
    synthetic_data = ctgan.sample(s)
    table_evaluator = TableEvaluator(data, synthetic_data)
    table_evaluator.visual_evaluation(save_dir=path)
    res = table_evaluator.evaluate(target_col='tag',return_outputs = True)
    k_s=res.get('Kolmogorov-Smirnov statistic') # exctract KS test pval
    features = []
    count = 0
    for k,v in k_s.items():
        if v.get('equality') != 'identical':
            features.append(k)
            count+=1
    num_bad.append(count)
    print (f'--sample: {s}: ---features {features} are bad')
    t_n_bad.append(num_bad)
    plt.close('all')
    
# if one bad feature, save file: 
synthetic_data.to_csv('data/400_synthetic.csv')


##### part 2: Capula
from sdv.tabular import CopulaGAN
from sdv.constraints import Between
from sdv.tabular import GaussianCopula

#add constains to important faetures relative to health tag (above corr > 0.1)
reasonable_confL1 = Between(column='L1_conf', low=0, high=1, handling_strategy='reject_sampling')
reasonable_confL3 = Between(column='L3_conf', low=0, high=1, handling_strategy='reject_sampling')
reasonable_confN3 = Between(column='N3_conf', low=0, high=1, handling_strategy='reject_sampling')
reasonable_confsterm = Between(column='sterm_prior_conf', low=0, high=1, handling_strategy='reject_sampling')

constraints = [reasonable_confsterm, 
               reasonable_confL1,
               reasonable_confL3,
               reasonable_confN3]

t_n_bad = []
es = [300, 500, 600, 1200]
samples = [131, 262, 400, 700, 1000]
for e in es:
    t0 = time.time()
    model = CopulaGAN(constraints=constraints, epochs=e)
    model.fit(data)
    t1 = time.time()
    total = t1-t0
    print(f'epochs:{e} took: {total:.3f} sec to train')
    
    num_bad = []
    for s in samples:
        print(f'-epoch:{e}')
        path = str(f'./Figs/CopulaGAN/{e}/{s}/')
        new_data = model.sample(s)
        table_evaluator = TableEvaluator(data, new_data)
        table_evaluator.visual_evaluation(save_dir=path)
        res = table_evaluator.evaluate(target_col='tag',return_outputs = True)
        k_s=res.get('Kolmogorov-Smirnov statistic') # exctract KS test pval
        features = []
        count = 0
        for k,v in k_s.items():
            if v.get('equality') != 'identical':
                features.append(k)
                count+=1
        num_bad.append(count)
        print (f'--sample: {s}: ---features {features} are bad')
    t_n_bad.append(num_bad)
    plt.close('all')
        

'''
count = 0
for f in data.columns:
    _, pval = ks_2samp(data[f], new_data[f])
    if pval<0.05:
        print(f'features: {f} was differernt')
        count+=1
print(count)

table_evaluator = TableEvaluator(data, new_data)
path = str('./Figs/CopulaGAN/')
table_evaluator.visual_evaluation(save_dir=path)

#res2 = CopulaGAN 
res2 = table_evaluator.evaluate(target_col='tag',return_outputs = True)
k_s=res.get('Kolmogorov-Smirnov statistic') # extract KS test pval
features = []
for k,v in k_s.items():
    if v.get('equality') != 'identical':
        features.append(k)
print (f'SDV algo:\nfeatures {features} are bad')
plt.close('all')

#note that tag and age are fine! maybe take them from here?
#If p-value is lower than a, then it is very probable that the two distributions are different.


#post-processing using min-max features scaling
e=1200
s=500
ctgan = CTGANSynthesizer(epochs=e)
ctgan.fit(data, discrete_columns)
synthetic_data = ctgan.sample(s)

table_evaluator = TableEvaluator(data, synthetic_data)
path = str(f'./Figs/ctgan')
table_evaluator.visual_evaluation(save_dir=path)
#res1 = CTGAN
res1 = table_evaluator.evaluate(target_col='tag',return_outputs = True)


synthetic_data_norm = synthetic_data.copy()
bad_features=['age', 'duration' ,'sterm_prior_conf' ,'L1_conf' ,'L3_conf' ,'N3_conf']
for col in bad_features:
    for i, val in enumerate(synthetic_data_norm[col]):
        a = data[col].min()
        b = data[col].max()
        xmin = synthetic_data_norm[col].min()
        xmax = synthetic_data_norm[col].max()
        synthetic_data_norm[col][i] = a + (((val - xmin)*(b-a))/(xmax - xmin))
            
#print(synthetic_data_norm['L1_conf'].min(),synthetic_data_norm['L1_conf'].max())

#print(synthetic_data.head())
table_evaluator = TableEvaluator(data, synthetic_data_norm)
path = str(f'./Figs/minmaxnorm')
table_evaluator.visual_evaluation(save_dir=path)

#res3 = max-min norm on CTGAN
res3 = table_evaluator.evaluate(target_col='tag',return_outputs = True)
k_s=res.get('Kolmogorov-Smirnov statistic') # extracts KS test pval
features = []
for k,v in k_s.items():
    if v.get('equality') != 'identical':
        features.append(k)
print (f'epoch:{e},sample:{s}\nfeatures {features} are bad')
plt.close('all')

#save the synthetic data with changes
synthetic_data['L1_conf'] = synthetic_data_norm['L1_conf']
synthetic_data['L3_conf'] = synthetic_data_norm['L3_conf']
synthetic_data['age'] = new_data['age']
synthetic_data.to_csv('./data/131_synthetic.csv')

table_evaluator = TableEvaluator(data, synthetic_data)
path = str(f'./Figs/selected_norm')
table_evaluator.visual_evaluation(save_dir=path)

#res4 = CTGAN with selected features
res4 = table_evaluator.evaluate(target_col='tag',return_outputs = True)

synthetic_data['duration'].delete()
synthetic_data['sterm_prior_conf'].delete()
synthetic_data['N3_conf'].delete()
data['duration'].delete()
data['sterm_prior_conf'].delete()
data['N3_conf'].delete()

table_evaluator = TableEvaluator(data, synthetic_data)
path = str(f'./Figs/selected_norm_withoutbad')
table_evaluator.visual_evaluation(save_dir=path)
#without bad features
res5 = table_evaluator.evaluate(target_col='tag',return_outputs = True)


#correlation of features with tags
corr = synthetic_data.corr(method='spearman')
plot_corr = corr[:-1]  # get the important corr values
plot_corr_sorted = plot_corr.sort_values('tag') # sort them
plt.bar(plot_corr_sorted.index, height = plot_corr_sorted['tag'], data = plot_corr_sorted)
plt.xticks(rotation=45, ha='center')
plt.xlabel('Features')
plt.ylabel('Corr. [rho]')
plt.rcParams.update({'font.size': 12})

plt.savefig('./Figs/synth__corr_features_tags.png', dpi=300)



#Some basic visualization - age histogram
#g = sns.FacetGrid(df, row='tag', hue='tag')
#g.map(sns.histplot, "age")
#g.set(xlabel='Age [years]', ylabel='Count [#]')
#plt.savefig('./Figs/ages.png', dpi=300, bbox_inches='tight')
#h, nh = df['tag'].value_counts()
#print(f'Pilot data with N = 131:\nN = {nh} non-healthy,\nN = {h} healthy')
# gender: 1 - female
# physical: Tiles with design - 1
# lterm prior: 1 - towards you, 0 - flat
# sterm prior: 3 - I see a part of a person, 2 - I see a face, 1 - I see a whole person, 0 - I don't see anyone
# sp1_d: 1 - left, sp2_d: 1 - right
'''