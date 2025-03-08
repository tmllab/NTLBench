
import os
import cv2
import pickle
import numpy as np
import torchvision.datasets as datasets

def get_home_art():
    return get_home_data('Art')

def get_home_cli():
    return get_home_data('Clipart')

def get_home_pd():
    return get_home_data('Product')

def get_home_rw():
    return get_home_data('Real World')

def get_home_data(source):
    list_img = []
    list_label = []
    data_size = 0
    root_temp = f'./data/office_home/{source}'
    class_path = os.listdir(root_temp)
    class_path = sorted(class_path)
    for i in range(len(class_path)):
        class_temp = os.path.join(root_temp, class_path[i])
        img_path = os.listdir(class_temp)
        for j in range(len(img_path)):
            img_path_temp = os.path.join(class_temp, img_path[j])
            img = cv2.imread(img_path_temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            list_img.append(img)
            list_label.append(np.eye(len(class_path))[i])
            data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
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
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        list_img.append(img)
        list_label.append(np.eye(65)[int(augment_labels[i])])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

if __name__ == '__main__':
    a = get_augment_data('OP')
    exit(0)