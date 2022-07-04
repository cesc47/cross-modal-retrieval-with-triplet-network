"""
File: manage_Flickr_dataset.py
Authors: Juan A. Rodriguez , Igor Ugarte, Francesc Net, David Serrano
Description:
    This script is used to prepare the Flickr dataset, generating splits for the available VGG and FastText Features.
"""
import json

import numpy as np
import os
import pickle
import scipy.io

PATH_TO_DATASET = os.path.join('..', '..', 'data', 'Flickr30k')

# Read VGG Features in .mat format
mat = scipy.io.loadmat(PATH_TO_DATASET + '/vgg_feats.mat')
feats = mat['feats']

# Read FastText Features in .pkl format
with open(PATH_TO_DATASET + '/fasttext_feats.npy', 'rb') as f:
    fasttext_feats = np.load(f, allow_pickle=True)

# Read the JSON files of image captions
with open(PATH_TO_DATASET + '/train.json', 'rb') as f:
    train_ann = json.load(f)
with open(PATH_TO_DATASET + '/test.json', 'rb') as f:
    test_ann = json.load(f)
with open(PATH_TO_DATASET + '/val.json', 'rb') as f:
    val_ann = json.load(f)

train_ids = [ann['imgid'] for ann in train_ann]
test_ids = [ann['imgid'] for ann in test_ann]
val_ids = [ann['imgid'] for ann in val_ann]

# Train features form list if ids
train_feats = feats[:,train_ids]
test_feats = feats[:,test_ids]
val_feats = feats[:,val_ids]

train_texts = fasttext_feats[train_ids]
test_texts = fasttext_feats[test_ids]
val_texts = fasttext_feats[val_ids]

# Store the features in .pkl format
with open(PATH_TO_DATASET + '/train_vgg_features.pkl', 'wb') as f:
    pickle.dump(train_feats, f)
with open(PATH_TO_DATASET + '/test_vgg_features.pkl', 'wb') as f:
    pickle.dump(test_feats, f)
with open(PATH_TO_DATASET + '/val_vgg_features.pkl', 'wb') as f:
    pickle.dump(val_feats, f)

with open(PATH_TO_DATASET + '/train_fasttext_features.pkl', 'wb') as f:
    pickle.dump(train_texts, f)
with open(PATH_TO_DATASET + '/test_fasttext_features.pkl', 'wb') as f:
    pickle.dump(test_texts, f)
with open(PATH_TO_DATASET + '/val_fasttext_features.pkl', 'wb') as f:
    pickle.dump(val_texts, f)


print(
    "VGG Features: {}\nFastText Features: {}\nTrain Annotations: {}\nTest Annotations: {}\nVal Annotations: {}".format(
        feats.shape, fasttext_feats.shape, len(train_ann), len(test_ann), len(val_ann)))
