

import pickle
ROOT_PATH = '../../data/'
TRAIN_TEXT_EMB = ROOT_PATH + "Flickr30k/train_bert_features.pkl"
# Load the text embeddings
with open(ROOT_PATH + "Flickr30k/big_bert_features/train_bert_features.pkl", 'rb') as f:
    text_embeddings = pickle.load(f)

foo = text_embeddings[0]

#
# with open(ROOT_PATH + "Flickr30k/train_bert_features.pkl", 'wb') as f:
#     pickle.dump(text_embeddings, f)