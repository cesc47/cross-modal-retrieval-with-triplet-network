import os
import cv2
import torch
import json

from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle

"""
    Get Faster-RCNN features of flickr30 dataset.
    CAREFUL! Delete first of all the readme.md file in flickr30k-images folder if you have just downloaded it!!
"""


def initialise():
    """
    Initialise Faster-RCNN model with detectron2.
    :return: cfg: configuration object
    """

    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg = get_cfg()

    # Run a model in detectron2's core library: get file and weights
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Hyper-params
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


class Flickr30kImages(Dataset):
    """
    Dataset for Flickr30k images.
    """
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split

        with open(root_dir + 'Flickr30k/' + split + '.json') as f:
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            # transforms.CenterCrop(256),
        ]
    )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir + '/Flickr30k/flickr30k-images', self.image_names[idx])
        image = Image.open(image_name)
        image = self.transform(image)
        return image


# --- PREDICTION ---
def get_feature_map_rcnn(cfg, processFeatures='mean_pool2d'):
    """
    Get Faster-RCNN features of flickr30 dataset.
    :param cfg: configuration object
    :param processFeatures: function to process features. implemented functions: 'mean_pool2d', 'max_pool1d'.
    Default: 'mean_pool2d' (see below). 'max_pool1d' returns a bigger feature vector
    (of around 16k features instead of around 1k from 'mean_pool2d')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and set it to eval mode
    model = build_model(cfg)
    model.eval()

    for split in ['train', 'val', 'test']:
        # Load images and convert them in tensor
        images = Flickr30kImages(root_dir="../../data/", split=split)
        loader = DataLoader(images, batch_size=64, shuffle=False)

        print(f"Loading images and generating features... "
              f"{np.ceil(images.__len__() / loader.batch_size)} iterations are needed")

        for batch_idx, images in tqdm(enumerate(loader)):
            # Get feature maps from that batch
            with torch.no_grad():
                features = model.backbone(images.to(device))
            # add to all_features the features tensor that is converted to numpy array and cpu
            features = features['res4'].cpu().detach().numpy()

            if processFeatures == 'mean_pool2d':
                # Reduce the size of the features tensor by doing a mean pooling
                features = np.mean(features, axis=(2, 3))
            elif processFeatures == 'mean_pool1d':
                # Reduce the size of the features tensor by doing a mean pooling only in one dimension (16x16 => 16)
                features = np.mean(features, axis=-1)
                # Reshape the features numpy vector into a 2D matrix
                features = features.reshape(features.shape[0], -1)
            else:
                raise NotImplementedError('only mean pooling in 2d and 1d is implemented until now')

            # store features in all_features variable, where for each image
            if batch_idx != 0:
                all_features = np.concatenate((all_features, features))
            else:
                all_features = features
        # Transpose all_features vector
        all_features = all_features.T
        # Save features in a pickle file
        with open("../../data/Flickr30k/" + 'v2_' + split + '_FasterRCNN_features.pkl', 'wb') as f:
            pickle.dump(all_features, f)


if __name__== "__main__":
    cfg = initialise()
    get_feature_map_rcnn(cfg, processFeatures='mean_pool1d')
    print('finished')