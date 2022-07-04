"""
File: task_a_evaluate_image_to_text_retrieval.py
Authors: Juan A. Rodriguez , Igor Ugarte, Francesc Net, David Serrano
Description:
    - This script is used to evaluate the image to text retrieval system for task a.
    - It uses the test set for retrieval using KNN
    - Quantitative and qualitative results are presented
"""

import json
import os.path
import pickle
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from datasets import Flickr30k
from models import ResnetFlickr, EmbeddingTextNet, TripletTextImage, TripletImageText
from evaluation_metrics import mapk

cuda = torch.cuda.is_available()


def extract_embeddings(dataloader, model, out_size=256, model_id=''):
    model.to('cuda')
    with torch.no_grad():
        model.eval()
        image_embeddings = np.zeros((len(dataloader.dataset), out_size))
        text_embeddings = np.zeros((len(dataloader.dataset) * 5, out_size))
        k = 0
        for images, texts in dataloader:
            if cuda:
                images = images.cuda()
                texts = texts.cuda()

            im_emb, text_emb = model.get_embedding_pair(images, texts)
            image_embeddings[k:k + len(images)] = im_emb.data.cpu().numpy()
            text_embeddings[k:k + len(texts) * 5] = text_emb.data.cpu().numpy().reshape(len(texts) * 5, out_size)
            k += len(images)

    return image_embeddings, text_embeddings


def main():
    # Load the datasets
    ROOT_PATH = "../../data/"
    TEST_IMG_EMB = ROOT_PATH + "Flickr30k/test_vgg_features.pkl"
    TEST_TEXT_EMB = ROOT_PATH + "Flickr30k/test_bert_features.pkl"

    # Method selection
    base = 'ImageToText'
    text_aggregation = 'BERT'
    image_features = 'VGG'
    emb_size = 768
    out_size = 4096
    input_size = 4096
    info = 'out_size_' + str(out_size)
    model_id = base + '_' + image_features + '_' + text_aggregation + '_textagg_' + info

    PATH_MODEL = 'models/'
    PATH_RESULTS = 'results/'
    # Create folder if it does not exist
    if not path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)

    # Load the test dataset
    test_dataset = Flickr30k(TEST_IMG_EMB, TEST_TEXT_EMB, train=False,
                             text_aggregation=text_aggregation)  # Create the test dataset

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=1)

    margin = 1.
    embedding_text_net = EmbeddingTextNet(embedding_size=emb_size, output_size=out_size, sequence_modeling=None)
    embedding_image_net = ResnetFlickr(input_size=input_size, output_size=out_size)
    model = TripletImageText(embedding_text_net, embedding_image_net, margin=margin)

    # Check if file exists
    if path.exists(PATH_MODEL + model_id + '.pth'):
        print('Loading the model from the disk, {}'.format(model_id + '.pth'))
        checkpoint = torch.load(PATH_MODEL + model_id + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Obtain ground truth from the json file (test.json)
    with open(ROOT_PATH + 'Flickr30k/test.json') as f:
        data = json.load(f)

    gt = {}  # Ground truth as a dictionary with the image filename as key and the list of text id as value
    dict_sentences = {}  # Dictionary with the text id as key and the sentence as value
    count = 0
    for item in data:
        gt[item['filename']] = [x['raw'] for x in item['sentences']]
        for sentence in item['sentences']:
            dict_sentences[count] = sentence['raw']
            count += 1

    # Extract embeddings
    image_embeddings, text_embeddings = extract_embeddings(test_loader, model, out_size, model_id)
    # Compute the labels for each embedding
    image_labels = [i for i in range(1, 1000 + 1)]
    text_labels = [j for j in range(1, 1000 + 1) for i in range(5)]  # Trick to obtain the same
    # number of labels, copying the same labels 5 (5 text embeddings)

    # Compute the nearest neighbors
    print('Computing the nearest neighbors...')
    k = 5  # Number of nearest neighbors

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree').fit(text_embeddings, text_labels)

    # Make predictions
    distances, indices = knn.kneighbors(image_embeddings)
    # pickle.dump((distances, indices), open(PATH_MODEL + model_id + '_knn.pkl', 'wb'))

    # Compute mAPk
    image_labels_pred = []
    # We create a dict to map the index of a single text with its corresponding label (image)
    for k_predictions in indices.tolist():
        # map indices with the corresponding labels
        k_labels_pred = [text_labels[i] for i in k_predictions]
        image_labels_pred.append(k_labels_pred)

    im_labels = [[i] for i in image_labels]
    map_k = mapk(im_labels, image_labels_pred, k=k)
    print(f'mAP@{k}: {map_k}')

    # Compute the accuracy
    knn_accuracy = knn.score(image_embeddings, image_labels)
    print('KNN accuracy: {}%'.format(100 * knn_accuracy))

    # Qualitative results
    num_samples = 10
    # Create random samples
    random_samples = np.random.choice(image_labels, num_samples, replace=False)
    # im_labels, image_labels_pred
    for sample in random_samples:
        print("Example:" + str(sample))
        print("--------------------------------")

        # Get image embedding from batch
        filename = list(gt)[sample]
        print("Ground truth: ")
        for t in gt[filename]:
            print(t)

        predictions = indices[sample]

        print("Predictions:")
        for pred in predictions:
            print(dict_sentences[pred])

        im = plt.imread(ROOT_PATH + 'Flickr30k/flickr30k-images/' + filename)
        plt.imshow(im)
        plt.show()
        print("--------------------------------------------------------------------------------")


# Main
if __name__ == '__main__':
    main()
