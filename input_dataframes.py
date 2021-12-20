import pandas as pd
import numpy as np
import os 
import operator
import albumentations as A
import imageio
#import ipyplot
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from PIL import Image

""" GEDAAN
df_f_train = pd.read_csv("../../heather/CSV/train_full.csv")
df_f_aug = pd.read_csv("../total_train_full_aug.csv")

df_full_train = df_f_train.merge(df_f_aug[['ID','n_aug']], on ='ID', how='left')

df_full_train.to_csv("/media/beta/mitko-beps/mandy/train_full_aug.csv")
"""

#full
df_f_train = pd.read_csv("../train_full_aug.csv")
df_f_val = pd.read_csv("../../heather/CSV/valid_full.csv")

#unique
df_u_train = pd.read_csv("../../heather/CSV/train_unique_aug.csv")
df_u_val = pd.read_csv("../../heather/CSV/valid_unique.csv")

#make dataframe with individual names and only column ID & Label
def make_dataframe_train(df):
    df_new = pd.DataFrame(columns=['ID', 'Label'])
    for i, row in df.iterrows(): 
        img = row.ID
        lbl= row.Label
        n_aug = row.n_aug
        n_cell = row.n_cells
      
        if n_aug < 1:
            for j in range(1, n_cell+1):
                df_new = df_new.append({'ID': img+ "_"+str(j), 'Label': lbl}, ignore_index=True)
    
        if n_aug >= 1:
            for k in range(1, n_cell+1):
                df_new = df_new.append({'ID': img+ "_"+str(k), 'Label': lbl}, ignore_index=True)
                for l in range(1, n_aug+1):
                    df_new = df_new.append({'ID': "aug_"+str(l)+"_"+img+ "_"+str(k), 'Label': lbl}, ignore_index=True)

    return df_new   

def make_dataframe_val(df):
    df_new = pd.DataFrame(columns=['ID', 'Label'])
    for i, row in df.iterrows(): 
        img = row.ID
        lbl= row.Label
        n_cell = row.n_cells
      
        for j in range(1, n_cell+1):
            df_new = df_new.append({'ID': img+ "_"+str(j), 'Label': lbl}, ignore_index=True)


    return df_new 
""" 
df_full_train = make_dataframe(df_f_train)
df_full_train.to_csv("/media/beta/mitko-beps/mandy/CSVinput/df_f_train.csv")
"""

df_full_val = make_dataframe_val(df_f_val)
df_full_val.to_csv("/media/beta/mitko-beps/mandy/CSVinput/df_f_val.csv")

df_unique_train = make_dataframe_train(df_u_train)
df_unique_train.to_csv("/media/beta/mitko-beps/mandy/CSVinput/df_u_train.csv")

df_unique_val = make_dataframe_val(df_u_val)
df_unique_val.to_csv("/media/beta/mitko-beps/mandy/CSVinput/df_u_val.csv")