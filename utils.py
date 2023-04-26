import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset
##TODO: Remove!
import matplotlib.pyplot as plt

def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id<0 or cnn_id>2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model

class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """
    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        top_class_pred = torch.max(pred, dim=1)
        total += labels.shape[0]
        correct += (top_class_pred.indices == labels).float().sum()

    return (correct / total)

##TODO: Assuming we should operate only on fist batch?
def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    adv_by_batch = []
    labels_by_batch = []
    for images, org_labels  in data_loader:
        images = images.to(device)
        org_labels = org_labels.to(device)
    
        if (targeted == True):
            #TODO: Check if labels are indeed in range (1, num_classes) or start at 0
            label_pert = torch.randint_like(org_labels, 0, high=n_classes).to(device)
            labels = torch.remainder(org_labels + label_pert, n_classes).to(device)
        
        else:
            labels = org_labels

        adv_images = attack.execute(images, labels, targeted)
        adv_by_batch.append(adv_images)
        labels_by_batch.append(labels)

    total_adv_imgs = torch.cat(adv_by_batch)
    total_labels= torch.cat(labels_by_batch)
    return total_adv_imgs, total_labels
    

def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    adv_by_batch = []
    labels_by_batch = []
    queries_by_batch = []
    i = 0

    for images, org_labels  in data_loader:
        images = images.to(device)
        org_labels = org_labels.to(device)
    
        if (targeted == True):
            label_pert = torch.randint_like(org_labels, 0, high=n_classes).to(device)
            labels = torch.remainder(org_labels + label_pert, n_classes).to(device)
        
        else:
            labels = org_labels

        adv_images, queries_by_sample = attack.execute(images, labels, targeted)
        print(f'Batch {i}: {queries_by_batch}')
        adv_by_batch.append(adv_images)
        labels_by_batch.append(labels)
        queries_by_batch.append(queries_by_sample)
        i += 1
        
    total_adv_imgs = torch.cat(adv_by_batch)
    total_labels= torch.cat(labels_by_batch)
    total_queries_by_sample = torch.cat(queries_by_batch)

    return total_adv_imgs, total_labels, total_queries_by_sample

def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    x_adv = x_adv.to(device)
    y = y.to(device)

    model.eval()
    pred = model(x_adv)
    top_class_pred = torch.max(pred, dim=1)
    is_successful = top_class_pred.indices == y if targeted == True else top_class_pred.indices != y
    correct = is_successful.float().sum()
    return (correct / x_adv.shape[0])

def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass # FILL ME

def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass # FILL ME

def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass # FILL ME

#TODO: Remove!
def imshow(img):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 5))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
