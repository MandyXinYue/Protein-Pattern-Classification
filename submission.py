# import libs 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

# Reload the model 
reload_model = tf.keras.models.load_model('/media/beta/mitko-beps/mandy/notebooks/Pretrain_FineTune/Finetune_logs/tf_finetuning_on_single_labels')

# Load the csv with image ids, cell ids and encoded masks
cell_df = pd.read_csv('/media/beta/mitko-beps/heather/cell_df.csv')
cell_df['cls'] = ''

# make with the reloaded model predictions for the earlier segmented and preprocessed test images 
preds_j = []
for n, row in cell_df.iterrows():    
    a = plt.imread("/media/beta/mitko-beps/data/segmented_test/{}_{}.png".format(row['image_id'], row['cell_id']))
    a = a.reshape(1, *a.shape)

    pred = reload_model.predict(a)[0]
    preds_j.append(pred)

# convert predictions to torch tensor 
predi = torch.FloatTensor(preds_j)

# set a threshold for what minimal probability the image can be taken into account. Then, return in the cell_df the probability 
# and the cls that belongs to the probability 
threshold = 0.0

for i in range(predi.shape[0]): 
    p = torch.nonzero(predi[i] > threshold).squeeze().numpy().tolist()
    if type(p) != list: 
        p = [p]
        
    if len(p) == 0: 
        cls = [(predi[i].argmax().item(), preds[i].max().item())]
        #print(cls)
    else: 
        cls = [(x, predi[i][x].item()) for x in p]
        
    cell_df['cls'].loc[i] = cls
    
# function to combine the enclosed mask and the prediction (so prob + class)
def combine(r):
    cls = r[0]
    #print(cls)
    enc = r[1]
    classes = [str(c[0]) + ' ' + str(c[1]) + ' ' + enc for c in cls]
    return ' '.join(classes)

# test if it works 
combine(cell_df[['cls', 'enc']].loc[24])

# create column pred where the encoded mask and cls is combined for 1 cell 
cell_df['pred'] = cell_df[['cls', 'enc']].apply(combine, axis=1)

# combine the pred column for all the images with the same image_id 
subm = cell_df.groupby(['image_id'])['pred'].apply(lambda x: ' '.join(x)).reset_index()

# load sample_submission 
sample_submission = pd.read_csv("/media/beta/mitko-beps/data/sample_submission.csv")

# merge the 2 dataframes 
sub = pd.merge(
    sample_submission,
    subm,
    how="left",
    left_on='ID',
    right_on='image_id',
)

# where NaN, fill in the example PredicitionString
def isNaN(num):
    return num != num

for i, row in sub.iterrows():
    if isNaN(row['pred']): continue
    sub.PredictionString.loc[i] = row['pred']
    
# use the correct culumn names 
sub = sub[sample_submission.columns]

# save as csv to load in Kaggle 
sub.to_csv('/media/beta/mitko-beps/mandy/submissions/submission_try.csv', index=False)