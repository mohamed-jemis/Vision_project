import os
import pathlib
import cv2 as cv2


# Read the 200 images
def read_images():
    letters = {"A", "B", "C", "D", "E"}
    Images = []
    for i in letters:
        __dir = 'C:/Users/avata/Desktop/Vision project/data/person{}/Train/'.format(i)
        images_path = os.listdir(__dir)
        for image_path in images_path:
            if pathlib.Path(image_path).suffix[1] == 'p':
                full_path = os.path.join(__dir, image_path)
                image = cv2.imread(full_path, 0)
                Images.append(image)
    return Images


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


# main function
images = read_images()
Key_points, Descriptors = extract_features(images)
print(len(Key_points), len(Descriptors))
