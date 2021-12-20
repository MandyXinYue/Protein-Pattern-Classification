# import libs 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import imageio
from PIL import Image
import albumentations as A

# Reload the model 
reload_model = tf.keras.models.load_model('/media/beta/mitko-beps/mandy/notebooks/Pretrain_FineTune/Finetune_logs/tf_finetuning_on_single_labels')

# Load the csv with image ids, cell ids and encoded masks
cell_df = pd.read_csv('/media/beta/mitko-beps/heather/cell_df.csv')
cell_df['cls'] = ''

# ---------------------------------------------------------------------

def tta_aug(img):
    
    Geo_transform = A.Compose([

            A.Rotate(limit = 90, p=0.5),
            #A.RandomRotate90(always_apply=False, p=0.5),

            A.Transpose(p=0.5),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.ShiftScaleRotate(p=0.5),
        
            A.FromFloat(dtype='uint8', p=1),
            A.RandomToneCurve(scale=0.1, p=0.5),
            A.ToFloat(p=1)
    ])
    
    b = Geo_transform(image = img)
    transformed = b['image']             #array
    
    #e=Image.fromarray(transformed)
    
    aug_img = transformed.reshape(1, *transformed.shape)
    
    return aug_img
# -------------------------------------------------------------------

# make with the reloaded model predictions for the earlier segmented and preprocessed test images 

tta_n = 39
preds_j = []
for n, row in cell_df.iterrows():
    a = plt.imread("/media/beta/mitko-beps/data/segmented_test/{}_{}.png".format(row['image_id'], row['cell_id']))   
    
    aug_pred =[]
    for i in range(tta_n):
        d = tta_aug(a)
        pred_img = reload_model.predict(d)[0]
        aug_pred.append(pred_img)
    
    a_og = a.reshape(1, *a.shape)
    pred_og = reload_model.predict(a_og)[0]
    aug_pred.append(pred_og)
    
    pred = np.mean(aug_pred, axis=0)
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
        cls = [(predi[i].argmax().item(), predi[i].max().item())]
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
sub.to_csv('/media/beta/mitko-beps/mandy/submissions/submission_Geo_Intensity_40.csv', index=False)