"""
File: datasets.py
Authors: Juan A. Rodriguez , Igor Ugarte, Francesc Net, David Serrano
Description:
    File that defines the datasets used in the project.
"""
import json
import pickle
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class Flickr30k(Dataset):
    def __init__(self, image_embedding_path, text_embedding_path, train, text_aggregation='mean'):
        self.train = train
        self.text_aggregation = text_aggregation
        # Load the image embeddings
        with open(image_embedding_path, 'rb') as f:
            self.image_embeddings = pickle.load(f)

        # Ugly trick to manage transposed embeddings
        if self.image_embeddings.shape[1] == 1024:
            self.image_embeddings = np.transpose(self.image_embeddings)

        # Load the text embeddings
        with open(text_embedding_path, 'rb') as f:
            self.text_embeddings = pickle.load(f)

        # Number of captions per image
        self.num_captions = len(self.text_embeddings[0])

        if self.text_aggregation is not 'BERT':
            self.text_embeddings = self.aggregate_text_embedding(self.text_embeddings)
        else:
            self.text_embeddings = self.text_embeddings.reshape(self.text_embeddings.shape[0] // 5, 5, -1)

        # Length of the dataset
        self.length_dataset = len(self.image_embeddings[1])

    def __getitem__(self, index):
        img_embedding = self.image_embeddings[:, index]
        text_embedding = self.text_embeddings[index]

        return img_embedding, text_embedding

    def __len__(self):
        return self.length_dataset

    def aggregate_text_embedding(self, text_embeddings):
        aggregated_text_embeddings = []
        for i in range(len(text_embeddings)):
            aggregated_sentences = []
            for j in range(len(text_embeddings[i])):
                if self.text_aggregation == 'mean':
                    aggregated = np.mean(text_embeddings[i][j], axis=0)
                elif self.text_aggregation == 'sum':
                    aggregated = np.sum(text_embeddings[i][j], axis=0)
                aggregated_sentences.append(aggregated)
            aggregated_text_embeddings.append(aggregated_sentences)
        agg = np.asarray(aggregated_text_embeddings)
        return agg


class TripletFlickr30kImgToText(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.dataset = dataset
        self.n_samples = dataset.length_dataset
        self.train = split == 'train'
        # Transform the output of the Dataset object into Tensor
        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        if not self.train:  # Test triplets
            # Loop to create fixed triplets for testing
            triplets = []
            for anchor_index in range(self.n_samples):
                # Get anchor image and positive text embedding (random from the set of 5 texts)
                anchor_img, positive_text = self.dataset[anchor_index]

                # Chose only a random text from the 5
                positive_text = positive_text[random.randint(0, 4)]

                # Get index of the negative text embedding
                negative_text_index = anchor_index
                while negative_text_index == anchor_index:
                    negative_text_index = np.random.choice(range(self.n_samples))

                # Get negative text embedding (random from the set of 5 texts)
                negative_text = self.dataset.text_embeddings[negative_text_index, random.randint(0, 4)]

                # Create triplet and add it to the list
                triplets.append([anchor_img, positive_text, negative_text])

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            # Get anchor image and positive text embedding (random from the set of 5 texts)
            anchor_img, positive_text = self.dataset[index]

            # Chose only a random text from the 5
            positive_text = positive_text[random.randint(0, 4)]
            # Get index of the negative text embedding
            negative_text_index = index
            while negative_text_index == index:
                negative_text_index = np.random.choice(range(self.n_samples))

            # Get negative text embedding (random from the set of 5 texts)
            negative_text = self.dataset.text_embeddings[negative_text_index, random.randint(0, 4)]

        else:
            anchor_img = self.test_triplets[index][0]
            positive_text = self.test_triplets[index][1]
            negative_text = self.test_triplets[index][2]

        return (anchor_img, positive_text, negative_text), []

    def __len__(self):
        return self.n_samples


# To Do
class TripletFlickr30kTextToImg(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.dataset = dataset
        self.n_samples = dataset.length_dataset
        self.train = split == 'train'

        if not self.train:  # Test triplets
            # Loop to create fixed triplets for testing
            triplets = []
            for anchor_index in range(self.n_samples):
                # Get anchor image and positive text embedding (random from the set of 5 texts)
                positive_image, anchor_text = self.dataset[anchor_index]
                # Chose only a random text from the 5
                anchor_text = anchor_text[random.randint(0, 4)]
                # Get index of the negative text embedding
                negative_image_index = anchor_index
                while negative_image_index == anchor_index:
                    negative_image_index = np.random.choice(range(self.n_samples))

                # Get negative text embedding (random from the set of 5 texts)
                negative_image = self.dataset.image_embeddings[:, negative_image_index]

                # Create triplet and add it to the list
                triplets.append([anchor_text, positive_image, negative_image])

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            # Get anchor image and positive text embedding (random from the set of 5 texts)
            positive_image, anchor_text = self.dataset[index]
            # Chose only a random text from the 5
            anchor_text = anchor_text[random.randint(0, 4)]
            # Get index of the negative text embedding
            negative_image_index = index
            while negative_image_index == index:
                negative_image_index = np.random.choice(range(self.n_samples))

            # Get negative text embedding (random from the set of 5 texts)
            negative_image = self.dataset.image_embeddings[:, negative_image_index]

        else:
            anchor_text = self.test_triplets[index][0]
            positive_image = self.test_triplets[index][1]
            negative_image = self.test_triplets[index][2]

        return (anchor_text, positive_image, negative_image), []

    def __len__(self):
        return self.n_samples


class Flickr30kEndToEnd(Dataset):
    def __init__(self, root_path, text_embedding_path, split, text_aggregation='mean'):
        self.text_aggregation = text_aggregation
        self.root_path = root_path
        with open(root_path + 'Flickr30k/' + split + '.json') as f:
            data = json.load(f)
        filenames = []
        for item in data:
            image_file = item['filename']
            filenames.append(image_file)
        self.image_names = filenames

        self.transform = transforms.Compose(
            [
                # RandomHorizontalFlip(),
                # RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                transforms.CenterCrop(256),
            ])

        # Load the text embeddings
        with open(text_embedding_path, 'rb') as f:
            self.text_embeddings = pickle.load(f)

        # Number of captions per image
        self.num_captions = len(self.text_embeddings[0])

        if self.text_aggregation is not 'BERT':
            self.text_embeddings = self.aggregate_text_embedding(self.text_embeddings)
        else:
            self.text_embeddings = self.text_embeddings.reshape(self.text_embeddings.shape[0] // 5, 5, -1)

        # Length of the dataset
        self.length_dataset = len(self.image_names)

    def __getitem__(self, index):
        img = Image.open(self.root_path + '/Flickr30k/flickr30k-images/' + self.image_names[index])
        img = self.transform(img)
        text_embedding = self.text_embeddings[index]

        return img, text_embedding

    def __len__(self):
        return self.length_dataset

    def aggregate_text_embedding(self, text_embeddings):
        aggregated_text_embeddings = []
        for i in range(len(text_embeddings)):
            aggregated_sentences = []
            for j in range(len(text_embeddings[i])):
                if self.text_aggregation == 'mean':
                    aggregated = np.mean(text_embeddings[i][j], axis=0)
                elif self.text_aggregation == 'sum':
                    aggregated = np.sum(text_embeddings[i][j], axis=0)
                aggregated_sentences.append(aggregated)
            aggregated_text_embeddings.append(aggregated_sentences)
        agg = np.asarray(aggregated_text_embeddings)
        return agg


class TripletFlickr30kTextToImgEndToEnd(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.dataset = dataset
        self.n_samples = dataset.length_dataset
        self.train = split == 'train'

        if not self.train:  # Test triplets
            # Loop to create fixed triplets for testing
            triplets = []
            for anchor_index in range(self.n_samples):
                # Get anchor image and positive text embedding (random from the set of 5 texts)
                positive_image, anchor_text = self.dataset[anchor_index]
                # Chose only a random text from the 5
                anchor_text = anchor_text[random.randint(0, 4)]
                # Get index of the negative text embedding
                negative_image_index = anchor_index
                while negative_image_index == anchor_index:
                    negative_image_index = np.random.choice(range(self.n_samples))

                # Get negative text embedding (random from the set of 5 texts)
                negative_image = Image.open(
                    self.dataset.root_path + '/Flickr30k/flickr30k-images/' + self.dataset.image_names[
                        negative_image_index])
                negative_image = self.dataset.transform(negative_image)

                # Create triplet and add it to the list
                triplets.append([anchor_text, positive_image, negative_image])

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            # Get anchor image and positive text embedding (random from the set of 5 texts)
            positive_image, anchor_text = self.dataset[index]
            # Chose only a random text from the 5
            anchor_text = anchor_text[random.randint(0, 4)]
            # Get index of the negative text embedding
            negative_image_index = index
            while negative_image_index == index:
                negative_image_index = np.random.choice(range(self.n_samples))

            # Get negative text embedding (random from the set of 5 texts)
            negative_image = Image.open(
                self.dataset.root_path + '/Flickr30k/flickr30k-images/' + self.dataset.image_names[
                    negative_image_index])
            negative_image = self.dataset.transform(negative_image)

        else:
            anchor_text = self.test_triplets[index][0]
            positive_image = self.test_triplets[index][1]
            negative_image = self.test_triplets[index][2]

        return (anchor_text, positive_image, negative_image), []

    def __len__(self):
        return self.n_samples
