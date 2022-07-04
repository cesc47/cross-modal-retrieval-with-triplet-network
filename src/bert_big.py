

import pickle
import numpy as np
import tqdm
ROOT_PATH = '../../data/'
# Load the text embeddings
for split in ['train', 'val', 'test']:
    print('Loading', split, 'text embeddings...')
    with open(ROOT_PATH + "Flickr30k/big_bert_features/"+split+"_bert_features.pkl", 'rb') as f:
        text_embeddings = pickle.load(f)
    print('Done!')
    all_cls_tokens = []
    i = 0
    for item in text_embeddings:
        print("sample: ", i)
        # if i >10:
        #     break
        cls_token = item[0, :]
        all_cls_tokens.append(cls_token)
        i += 1
    text_embeddings_cls = np.asarray(all_cls_tokens)
    # foo = text_embeddings_cls.reshape(len(all_cls_tokens)//5, 768)

    with open(ROOT_PATH + "Flickr30k/"+split+"_bert_features.pkl", 'wb') as f:
        pickle.dump(text_embeddings_cls, f)