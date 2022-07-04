import torch

if __name__ == "__main__":

    # Number of retrievals
    num_retrievals = 1

    # Initialize the model. delete pickle file of train and test if you want to recompute the features!
    baseline = 'resnet50'
    method = 'base'
    model_id = baseline + '_' + method

    model = torch.hub.load('pytorch/vision:v0.10.0', baseline, pretrained=True)
    # Remove the last layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)