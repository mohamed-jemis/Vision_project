import os
import pathlib
import cv2 as cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def read_images():
    letters = {"A", "B", "C", "D", "E"}
    Images = []
    Labels = []
    for i in letters:
        __dir = 'C:/Users/avata/Desktop/Vision project/data/person{}/Train/'.format(i)
        images_path = os.listdir(__dir)
        for image_path in images_path:
            if pathlib.Path(image_path).suffix[1] == 'p':
                full_path = os.path.join(__dir, image_path)
                image = cv2.imread(full_path, 0)
                # print(image_path[6])
                Images.append(image)
                Labels.append(image_path[6])
                # print(image_path[6])
    return Images, Labels


# Create sift and extract kp and desc
def extract_features(Images):
    Kp = []
    Desc = []
    sift = cv2.SIFT_create()
    for i in range(len(Images)):
        kp, desc = sift.detectAndCompute(Images[i], None)
        Kp.append(kp)
        Desc.append(desc)
    return Kp, Desc


def format_ND(l):
    #  vertical stacking of l
    vStack = np.array(l[0])
    for remaining in l[1:]:
        vStack = np.vstack((vStack, remaining))
    descriptor_vstack = vStack.copy()
    return descriptor_vstack


def cluster(n_clusters, descriptor_vstack):
    kmeans_obj = KMeans(n_clusters=n_clusters)
    kmeans_ret = kmeans_obj.fit_predict(descriptor_vstack)


def vocab():
    return 1


def svm_model(labeled_features):
    return 1


def plot_hist():
    return 1


def standardize():
    return 1


def train():
    return 1


def train():
    return 1

# main function
images, Labels = read_images()
Key_points, Descriptors = extract_features(images)
labeled_features = np.array([Key_points, Descriptors, Labels])
labeled_features = labeled_features.reshape(3, -1)
# 0 is the kp and 1 is the desc and labels is the 3
print((labeled_features[1][1]))
print((labeled_features[1][2]))
print((labeled_features[1][3]))

# print(labeled_features[3])
# print(labeled_features[2])


# print(Key_points)
# labeled_features = np.concatenate((Key_points,Descriptors), axis=1)
# print(labeled_features)

