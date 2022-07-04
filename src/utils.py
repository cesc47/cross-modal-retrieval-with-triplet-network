import numpy as np
import os
import cv2
from tqdm import tqdm
import torch
import time

def aggregate_text_embedding(sentence, embedding_model):
    """
    Aggregate the text embedding of a sentence using the embedding model.
    """
    return np.mean(embedding_model.wv[sentence], axis=0)


def load_images_from_folder(folder, toTensor=False):
    """
    Load all images from a folder.
    If toTensor is True, the images are converted to tensors.
    :param folder:      Path to the folder.
    :param toTensor:    If True, the images are converted to tensors.
    :param device:      Device to load the images to.
    :return:            List of images.
    """
    images = []
    print(f"Loading images from {folder}")
    for idx, filename in tqdm(enumerate(os.listdir(folder))):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            if toTensor:
                img = torch.as_tensor(img.transpose(2, 0, 1)).float()
            images.append(img)

    return images
