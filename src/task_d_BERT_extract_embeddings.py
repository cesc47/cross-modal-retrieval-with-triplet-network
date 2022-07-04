from abc import ABC

from transformers import BertModel, BertTokenizer
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import numpy as np

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

ROOT_PATH = "../../data/"

TRAIN_IMG_EMB = ROOT_PATH + "Flickr30k/train_vgg_features.pkl"
TEST_IMG_EMB = ROOT_PATH + "Flickr30k/val_vgg_features.pkl"
TRAIN_TEXT_EMB = ROOT_PATH + "Flickr30k/train_fasttext_features.pkl"
TEST_TEXT_EMB = ROOT_PATH + "Flickr30k/val_fasttext_features.pkl"


class FlickrTextDataset(Dataset):
    def __init__(self, split):
        with open(ROOT_PATH + 'Flickr30k/' + split + '.json') as f:
            data = json.load(f)
        sentences = []
        for item in data:
            image_sentences = [t['raw'] for t in item['sentences']]
            for s in image_sentences:
                sentences.append(s)
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return [self.sentences[idx]]


def extractBertFeatures(dataloader):
    sentences = []
    for batch_idx, (data) in enumerate(dataloader):
        # if batch_idx > 0:
        #     break
        print("batch_idx: ", batch_idx, "of ", len(dataloader))
        bert_model.eval()
        for sentence in data[0]:
            inputs = bert_tokenizer(sentence, return_tensors="pt").to('cuda')
            out = bert_model(**inputs)
            last_hidden_state = out.pooler_output
            sentences.append(last_hidden_state.cpu().squeeze().detach().numpy())
    return sentences


for split in ['train', 'val', 'test']:
    print("Processing split: ", split)
    dataset = FlickrTextDataset(split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    bert_embeddings = np.asarray(extractBertFeatures(dataloader))

    with open(ROOT_PATH + 'Flickr30k/'+split+'_bert_features.pkl', 'wb') as f:
        pickle.dump(bert_embeddings, f)
