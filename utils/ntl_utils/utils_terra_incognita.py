import numpy as np
from collections.abc import Iterable
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import math
from scipy.special import softmax
import scipy.io as sio
import torchvision.datasets as datasets
import cv2

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_ti_l38():
    return get_terra_incognita_data('location_38')

def get_ti_l43():
    return get_terra_incognita_data('location_43')

def get_ti_l46():
    return get_terra_incognita_data('location_46')

def get_ti_l100():
    return get_terra_incognita_data('location_100')
    
def get_terra_incognita_data(domain):
    # Init
    list_img = []
    list_label = []
    data_size = 0
    
    # Load img
    domain_dir = "./data/terra_incognita/{}".format(domain) 
    classes = os.listdir(domain_dir)
    classes = sorted(classes)
    for i in range(len(classes)):
        class_path = os.path.join(domain_dir, classes[i])
        img_path = os.listdir(class_path)
        for j in range(len(img_path)):
            img_path_temp = os.path.join(class_path, img_path[j])
            img = cv2.imread(img_path_temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
                
            list_img.append(img)
            list_label.append(np.eye(len(classes))[i])
            data_size += 1
            
    # Shuffle
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    print(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]
    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

def get_augment_data(data_name):
    list_img = []
    list_label = []
    data_size = 0

    augment_trainset = os.listdir("./data_ntl_aug/augment_{}/".format(data_name))
    augment_labels = np.loadtxt("./data_ntl_aug/augment_{}/labels".format(data_name))

    for i in range(len(augment_trainset) - 1):
        img = cv2.imread("./data_ntl_aug/augment_{}/img_{}.png".format(data_name, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        list_img.append(img)
        list_label.append(np.eye(10)[int(augment_labels[i])])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

    
def add_watermark(dataset_list, img_size, value=80):
    list_img, list_label, data_size = dataset_list
    
    mask = np.zeros(list_img[0].shape)
    for i in range(img_size):
        for j in range(img_size):
            if i % 2 == 0 or j % 2 == 0:
                mask[i,j,0] = value
    img_list_len = list_img.shape[0]
    #print(np.max(list_img[i]))
    for i in range(img_list_len):
        list_img[i] = np.minimum(list_img[i].astype(int) + mask.astype(int), 255).astype(np.uint8)
    return [list_img, list_label, data_size]


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)
