# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 16:14:32 2021
@author: Yuval S-Katz
"""
#from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ks_2samp
import math
import seaborn as sns
from ctgan import CTGANSynthesizer # pip install first
from table_evaluator import TableEvaluator
import time
import warnings
warnings.filterwarnings("ignore")

#background: https://towardsdatascience.com/how-to-generate-tabular-data-using-ctgans-9386e45836a6
#df = pd.read_csv("./Data/‏‏131_subjects.csv") # real data
df = pd.read_csv("./Data/train/train_20.csv")
data = df.iloc[:,1:]
discrete_columns = ['gender', 'E_score', 'E_score2', 'A_score', 'A_score2', 
                    'C_score', 'C_score2', 'N_score', 'N_score2', 'O_score',
                    'O_score2', 'Gpain', 'Gsound', 'physical', 'sterm_prior',
                    'lterm_prior', 'SP1_d', 'SP2_d', 'tag']

im_fs = ['N31_conf','N3L1_conf', 'N3L3_conf', 'N3N1_conf', 'L31_conf', 'L1N1_conf']
x = df.iloc[:,1:-3] # tabular data
y = df.iloc[:,-3] # tags

def explore_data(data):
    '''
    unsupervised exploration of features with:
        1 - Spearman's correlation
        2 - PCA analysis
        3 - Normality statistical test
    param: data = tabular data without tags
    '''
    print('exploring data...')
    corr = data.corr(method='spearman')
    plt.figure(figsize=(20, 10))
    matrix = np.triu(corr)
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, cmap='RdYlGn',mask=matrix)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12)
    heatmap.set(xlabel='Features')
    heatmap.collections[0].colorbar.set_label("Corr. [r]")
    plt.savefig('./Figs/heatmap.png', dpi=300, bbox_inches='tight') 
    print('-correlation figure created')
    
    norm_data = preprocessing.scale(data)
    pca = PCA()
    pca.fit(norm_data)
    pca_data = pca.transform(norm_data)
    exp_var = np.round(pca.explained_variance_ratio_ *100, decimals=1)
    plt.clf()
    plt.title('PC space')
    plt.scatter(range(1,len(exp_var)+1),exp_var,label='PC variance')
    plt.plot(range(1,len(exp_var)+1),exp_var.cumsum(), marker = 'x', c='r', linestyle='--',label='Cumulative sum')
    plt.xlabel('Number of PCs [#]')
    plt.ylabel('Variance [%]')
    plt.legend()
    plt.savefig('./Figs/PCA_screeplot.png', dpi=300, bbox_inches='tight')
    print('-PCA figure created')
    
    print('-Normality tests over features:')
    feature = []
    boll = False
    for f in data:
        feature = data[f]
        _, p = shapiro(feature)
        alpha = 0.05
        if p > alpha:
            print(f'--feature: {f} looks Gaussian (fail to reject H0)')
            boll == True
        else:
            pass
    if boll == False:
        print('--no normally distributed features detected')
    #explore_data(df.iloc[:,1:-3])
    
    #Some basic visualization - correlation of features with tags
    plt.rcParams['figure.figsize'] = [10, 5]
    corr = df.iloc[:,1:-3].corr(method='spearman')
    plot_corr = abs(corr['tag'])  # get the important corr values
    corr_mat = np.matmul(plot_corr.T, corr)
    plot_corr_sorted = corr_mat.sort_values('tag') # sort 
    tags = plot_corr_sorted['tag']
    
    colors = []
    for i in range(0,len(plot_corr_sorted)):
        colors.append('gray')
        
    threshold = 0.15
    for i,c in enumerate(colors):
        if tags[i] > threshold or tags[i] < -threshold:
            colors[i] = 'k'
        
            
    plt.bar(plot_corr_sorted.index, height = plot_corr_sorted['tag'], data = plot_corr_sorted, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Corr. [rho]')
    plt.axhline(threshold, color='k', linestyle='--')
    plt.axhline(-threshold, color= 'k', linestyle='--')
    plt.savefig(f'./Figs/corr_features_tags_mul{threshold}.png', dpi=600, bbox_inches='tight')
        
    plt.rcParams['figure.figsize'] = [10, 5]
    df['AE_L'] = (df['L1'] - df['L3']) / 15
    new_df = pd.concat([df['AE_L'], df['health']],axis=1)
    new_df['health'] = new_df["health"].replace("None", "Healthy", regex=True)
    df_plot = new_df.groupby('health').agg(['mean','sem'])
    df_plot.columns = df_plot.columns.map('_'.join)
    df_plot = df_plot.reset_index()
    plot_sorted = df_plot.sort_values('AE_L_mean') # sort them
    labels = plot_sorted.health.unique()
    x_pos = np.arange(len(plot_sorted['health']))
    plt.bar(x_pos, plot_sorted['AE_L_mean'],yerr=plot_sorted['AE_L_sem'], align='center',color=['peru','peru','cornflowerblue', 'peru'])
    plt.xticks(x_pos, labels)
    plt.ylabel('AE to length [A.U.]')
    plt.savefig('./Figs/AE_L.png', dpi=600, bbox_inches='tight')
      

def CTGAN_run(data, discrete_columns, n_epochs, n_samples, glr, dlr, imp_fs):
    '''
    finetune of CTGAN hyperparameters
    '''
    
    t_n_bad = [] # how many features failed in each epoch
    t_n_imp_bad = []
    log_t=[]
    for e in n_epochs:
        t0 = time.time()
        ctgan = CTGANSynthesizer(epochs=e, generator_lr = glr, discriminator_lr = dlr)
        ctgan.fit(data, discrete_columns)
        t1 = time.time()
        total = t1-t0
        print(f'epochs:{e} took: {total/60:.1f} min')
        num_bad = []
        num_im_bad = []
        for s in samples:
            #print(f'-samples: {s}')
            path = str(f'./Figs/train_20/{e}/{s}/')
            synthetic_data = ctgan.sample(s)
            #print(synthetic_data.head())
            table_evaluator = TableEvaluator(data, synthetic_data)
            table_evaluator.visual_evaluation(save_dir=path)
            res = table_evaluator.evaluate(target_col='tag',return_outputs = True)
            k_s=res.get('Kolmogorov-Smirnov statistic') # exctract KS test pval
            features = []
            count = 0
            count_im_fs = 0
            for k,v in k_s.items():
                if v.get('equality') != 'identical':
                    features.append(k)
                    count+=1
                    if k in im_fs:
                        count_im_fs+=1
            num_im_bad.append(count_im_fs)
            num_bad.append(count)
            log = str(f'lr={dlr}, amples: {s}, important / total: {count_im_fs} / {count}')
            log_t.append(log)
        t_n_imp_bad.append(num_im_bad)
        t_n_bad.append(num_bad)
        plt.close('all')
    return log_t

import csv
epochs = [9600]
samples = [104, 208, 400, 700, 1000] # sensativity on synthetic sample sizes
glr = [0.005, 0.001, 0.0005, 0.0001]
dlr = 0.0002
t_log = []
for lr in glr:
    print(f'Generator learning rate:{lr}')
    log = CTGAN_run(data, discrete_columns, epochs, samples, lr, dlr, im_fs)
    t_log.append(log)
    
with open("res_g.csv", 'w', newline="\n") as f:
    writer = csv.writer(f)
    writer.writerows(t_log)
    
dlr = [0.005, 0.001, 0.0005, 0.0001]
glr = 0.0002
t_log = []

for lr in dlr:
    print(f'Discriminator learning rate:{lr}')
    #change func!
    log = CTGAN_run(data, discrete_columns, epochs, samples, glr, lr, im_fs)
    t_log.append(log)

with open("res_d.csv", 'w', newline="\n") as f:
    writer = csv.writer(f)
    writer.writerows(t_log)
    

#https://arxiv.org/abs/2101.00598
from sdv.tabular import CopulaGAN
from sdv.constraints import Between
from sdv.tabular import GaussianCopula

def Copula_run(data, discrete_columns, n_epochs, n_samples, glrs, dlrs, constraints = None):
        '''
    finetune of copulaGAN hyperparameters
    '''
    t_n_bad = [] # how many features failed in each epoch
    for e in n_epochs:
        t0 = time.time()
        cg = CopulaGAN(epochs=e, generator_lr = glrs, discriminator_lr = dlrs, constraints=constraints )
        cg.fit(data, discrete_columns)
        t1 = time.time()
        total = t1-t0
        print(f'epochs:{e} took: {total:.3f} sec to train')
        num_bad = []
        for s in samples:
            print(f'--sampling:{s}')
            path = str('./Figs/CopulaGAN/train_20/{e}/{s}/')
            synthetic_data = cg.sample(s)
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
            print (f'---failed: {features}')
        t_n_bad.append(num_bad)
        plt.close('all')
    return t_n_bad

epochs = [4800, 7200, 9600, 10800]
samples = [104, 208, 400, 700, 1000] # sensativity on synthetic sample sizes
glrs = 0.0002
dlrs = 0.0002
#reasonable_age_constraint = Between(column='age', low=data['age'].min(), high=data['age'].max(), handling_strategy='reject_sampling')
#reasonable_confL1 = Between(column='L1_conf', low=0, high=1, handling_strategy='reject_sampling')
#constraints = [reasonable_confsterm,
               #reasonable_age_constraint, 
               #reasonable_confL1]
n_bad_features_CG = Copula_run(data, discrete_columns, epochs, samples, glrs, dlrs, constraints)

#post-processing of CTGAN using min-max features scaling
e=9600
s=400
ctgan = CTGANSynthesizer(epochs=e)
ctgan.fit(data, discrete_columns)
synthetic_data = ctgan.sample(s)
table_evaluator = TableEvaluator(data, synthetic_data)
path = str(f'./Figs/pre_norm_ctgan')
table_evaluator.visual_evaluation(save_dir=path)
res1 = table_evaluator.evaluate(target_col='tag',return_outputs = True)
plt.close('all')
k_s=res1.get('Kolmogorov-Smirnov statistic') # extracts KS test pval
features = []
for k,v in k_s.items():
    if v.get('equality') != 'identical':
        features.append(k)
print (f'epoch:{e},sample:{s}\nfeatures {features} are bad')


#normalize by original values
synthetic_data_norm = synthetic_data.copy()
bad_features=features
for col in bad_features:
    for i, val in enumerate(synthetic_data_norm[col]):
        a = data[col].min()
        b = data[col].max()
        xmin = synthetic_data_norm[col].min()
        xmax = synthetic_data_norm[col].max()
        synthetic_data_norm[col][i] = a + (((val - xmin)*(b-a))/(xmax - xmin))
            
table_evaluator = TableEvaluator(data, synthetic_data_norm)
path = str(f'./Figs/post_norm_CTGAN')
table_evaluator.visual_evaluation(save_dir=path)
res3 = table_evaluator.evaluate(target_col='tag',return_outputs = True)
plt.close('all')
k_s=res3.get('Kolmogorov-Smirnov statistic') # extracts KS test pval
features = []
for k,v in k_s.items():
    if v.get('equality') != 'identical':
        features.append(k)
print (f'sample:{s}\nfeatures {features} are bad')

#save the synthetic data with changes
#synthetic_data['L1_conf'] = synthetic_data_norm['L1_conf']
#synthetic_data['L3_conf'] = synthetic_data_norm['L3_conf']
synthetic_data.to_csv('./data/pre_synthetic.csv')
synthetic_data_norm.to_csv('./data/post_synthetic.csv')


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






