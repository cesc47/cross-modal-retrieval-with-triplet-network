"""
File: datasets.py
Authors: Juan A. Rodriguez , Igor Ugarte, Francesc Net, David Serrano
Description:
    File corresponding to task a: Image to text retrieval through triplet metric learning
"""
import os
import os.path
from os import path

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from datasets import Flickr30k, TripletFlickr30kTextToImg

from train import fit
from losses import TripletLoss
from models import ResnetFlickr, EmbeddingTextNet, TripletTextImage
import wandb

wandb.init(project="M5-src", entity="celulaeucariota")


def main():
    # cuda management
    DEVICE = 'cuda'
    cuda = torch.cuda.is_available()

    # Find which device is used
    if cuda and DEVICE == "cuda":
        print(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CAREFUL!! Training the model with CPU')

    # Output directory
    OUTPUT_MODEL_DIR = './models/'

    # Create the output directory if it does not exist
    if not path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    # Load the datasets
    ROOT_PATH = "../../data/"
    TRAIN_IMG_EMB = ROOT_PATH + "Flickr30k/v2_train_FasterRCNN_features.pkl"
    TEST_IMG_EMB = ROOT_PATH + "Flickr30k/v2_val_FasterRCNN_features.pkl"
    TRAIN_TEXT_EMB = ROOT_PATH + "Flickr30k/train_fasttext_features.pkl"
    TEST_TEXT_EMB = ROOT_PATH + "Flickr30k/val_fasttext_features.pkl"

    # Method selection
    base = 'TextToImage'
    text_aggregation = 'mean'
    image_features = 'FasterRCNN'
    emb_size = 300
    out_size = 4096
    inp_size = 14336
    info = 'out_size_' + str(out_size)
    model_id = base + '_' + image_features + '_' + text_aggregation + '_textagg_' + info

    # Load the datasets
    train_dataset = Flickr30k(TRAIN_IMG_EMB, TRAIN_TEXT_EMB, train=True,
                              text_aggregation=text_aggregation)  # Create the train dataset
    test_dataset = Flickr30k(TEST_IMG_EMB, TEST_TEXT_EMB, train=False,
                             text_aggregation=text_aggregation)  # Create the test dataset

    train_dataset_triplet = TripletFlickr30kTextToImg(train_dataset, split='train')
    test_dataset_triplet = TripletFlickr30kTextToImg(test_dataset, split='test')

    batch_size = 1024

    # Create the dataloaders
    triplet_train_loader = torch.utils.data.DataLoader(train_dataset_triplet, batch_size=batch_size, shuffle=True)
    triplet_test_loader = torch.utils.data.DataLoader(test_dataset_triplet, batch_size=batch_size, shuffle=False)

    margin = 1.
    embedding_text_net = EmbeddingTextNet(embedding_size=emb_size, output_size=out_size, sequence_modeling=None)
    embedding_image_net = ResnetFlickr(input_size=inp_size, output_size=out_size)
    model = TripletTextImage(embedding_text_net, embedding_image_net, margin=margin)

    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)

    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    start_epoch = 0
    # Check if file exists
    if path.exists(OUTPUT_MODEL_DIR + model_id + '.pth'):
        print('Loading the model from the disk')
        checkpoint = torch.load(OUTPUT_MODEL_DIR + model_id + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    print('Starting training, EPOCH: ', start_epoch)
    n_epochs = 40
    log_interval = 10

    # Wandb configuration
    wandb.config = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "text_aggregation": text_aggregation,
        "image_features": image_features,
        "out_size": out_size,
        "model_id": model_id,
    }

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        model_id, start_epoch=start_epoch)


# main function
if __name__ == "__main__":
    main()
