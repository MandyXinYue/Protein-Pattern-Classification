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

def get_img_id(df): 
    path_mt = []
    path_nu = []
    path_er = [] 
    path_tp = []

    for i, row in df[0:len(df)].iterrows(): 
        img = row.ID
        ids = os.path.basename(img)
        count = row.n_cells
        for i in range(1, count+1):
            path_mt.append(ids+"_"+str(i)+"_mt.png")
            path_nu.append(ids+"_"+str(i)+"_nu.png")
            path_er.append(ids+"_"+str(i)+"_er.png")
            path_tp.append(ids+"_"+str(i)+"_tp.png")
    return path_mt, path_nu, path_er, path_tp

def augment(img_path, img_id_mt, img_id_nu, img_id_er, img_id_tp, times): 
    transform = A.Compose([

        A.Rotate(limit = 90,p=0.5),
        
        A.Transpose(p=0.5),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(p=0.5),

        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5)],

        
        additional_targets = {"image1": "image", "image2": "image", "image3": "image"}, p=1)

    img_mt = imageio.imread(img_path+img_id_mt)
    img_nu = imageio.imread(img_path+img_id_nu)
    img_er = imageio.imread(img_path+img_id_er)
    img_tp = imageio.imread(img_path+img_id_tp)

    transformed = transform(image = img_mt, image1 = img_nu, image2 = img_er, image3 = img_tp)

    transformed_img_mt = transformed["image"]
    transformed_img_nu = transformed["image1"]
    transformed_img_er = transformed["image2"]
    transformed_img_tp = transformed["image3"]

    im_mt = Image.fromarray(transformed_img_mt)
    im_nu = Image.fromarray(transformed_img_nu)
    im_er = Image.fromarray(transformed_img_er)
    im_tp = Image.fromarray(transformed_img_tp)
    
    #save the augmented images
    im_mt.save("/media/beta/mitko-beps/mandy/train_full_aug/"+"aug_"+ str(times)+"_"+img_id_mt)
    im_nu.save("/media/beta/mitko-beps/mandy/train_full_aug/"+"aug_"+ str(times)+"_"+img_id_nu)
    im_er.save("/media/beta/mitko-beps/mandy/train_full_aug/"+"aug_"+ str(times)+"_"+img_id_er)
    im_tp.save("/media/beta/mitko-beps/mandy/train_full_aug/"+"aug_"+ str(times)+"_"+img_id_tp)
   
    return print("saved aug_"+str(times)+img_id_mt)

#load images and csv
img_path = "../../data/segmented_train/"
df_train_full_segmented = pd.read_csv("../../heather/CSV/train_full.csv")


#Augumentation label 11
wanted_labels11 = ['11'] 
df_wanted_labels11 = df_train_full_segmented[df_train_full_segmented['Label'].apply(lambda x: pd.Series(x.split('|')).isin(wanted_labels11).any())]
path_mt11, path_nu11, path_er11, path_tp11 = get_img_id(df_wanted_labels11)

n_aug_11 = 6
for i in range(1,n_aug_11+1):
        for y in range(len(path_mt11)): 
            augment(img_path, path_mt11[y], path_nu11[y], path_er11[y], path_tp11[y], i)


#make new dataframe without label 11
df_f_drop = df_train_full_segmented.copy()
cond = df_f_drop['ID'].isin(df_wanted_labels11['ID'])
df_f_drop.drop(df_f_drop[cond].index, inplace = True)
df_f_drop


#augmentation label 15
wanted_labels15 = ['15'] 
df_wanted_labels15 = df_f_drop[df_f_drop['Label'].apply(lambda x: pd.Series(x.split('|')).isin(wanted_labels15).any())]
path_mt15, path_nu15, path_er15, path_tp15 = get_img_id(df_wanted_labels15)

n_aug_15 = 4
for i in range(1,n_aug_15+1):
        for y in range(len(path_mt15)): 
            augment(img_path, path_mt15[y], path_nu15[y], path_er15[y], path_tp15[y], i)
            
            
#make dataframe
df_wanted_labels11["n_aug"] = int(6)
df_wanted_labels15["n_aug"] = int(4)


df_aug =df_wanted_labels11.append(df_wanted_labels15, ignore_index=True)
df= df_train_full_segmented.merge(df_aug[['ID','n_aug']], on ='ID', how='left')
df['n_aug'] = df['n_aug'].fillna(0).astype(int)

df.to_csv("/media/beta/mitko-beps/mandy/train_full_aug.csv")


