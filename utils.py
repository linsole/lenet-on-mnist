import pandas as pd
import numpy as np
import os
import cv2
import torch
import yaml
from model import LeNet
import matplotlib.pyplot as plt
import random

def parse_data(csv_file, train_test):
    # This function aims to convert MNIST dataset from csv file redistribution to images.
    # Dataset comprises of 60000 train images and 10000 test images.
    # The csv files are downloaded from the following website: 
    # https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    data_ori = pd.read_csv(csv_file)
    data = np.array(data_ori.copy(), ndmin=2)

    counts = np.zeros(10, dtype=int) # counts of every number category
    root_path = os.path.join("MNIST", train_test)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    for vector in data:
        label = vector[0]

        # put every image to its corresponding sub directories according to its number
        sub_path = os.path.join(root_path, str(label))
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # write to jpg file
        img = np.reshape(vector[1:], (28, -1))
        img_path = os.path.join(sub_path, str(counts[label]).zfill(4)) + ".jpg"
        cv2.imwrite(img_path, img)
        counts[label] += 1 # increment corresponding count


def inference(img):
    # inference one handwritten digit image

    # initialize model
    with open("config.yaml", "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    lenet = LeNet(data)
    lenet.load_state_dict(torch.load("saves.pth"))
    lenet.eval()

    # pass input data to the model, involving a little work on image dimensions
    img = torch.unsqueeze(torch.Tensor(img), 0) # convert to tensor and add a batch dimension
    img = torch.unsqueeze(img, 0)
    pred = torch.squeeze(lenet(img), 0) # squeeze out the batch dimension
    
    return pred.argmax(0).item()


if __name__ == "__main__":
    # testing code for parse_data
    # parse_data(os.path.join("MNIST", "csv", "mnist_train.csv"), "train")
    # parse_data(os.path.join("MNIST", "csv", "mnist_test.csv"), "test")
    # print("Done.")

    # testing code for inference

    # initialize a dictionary for randomly selecting images
    category_count_dict = {}
    for i in range(10):
        category_count_dict[i] = len(os.listdir(os.path.join("MNIST", "test", str(i))))

    number_of_imgs = 9
    rows = 3
    cols = 3
    fig = plt.figure()
    for i in range(number_of_imgs):
        # randomly select a test image and read it
        category = random.randint(0, 9)
        filename = str(random.randint(0, category_count_dict[category] - 1)).zfill(4) + ".jpg"
        img = cv2.imread(os.path.join("MNIST", "test", str(category), filename), cv2.IMREAD_GRAYSCALE)

        # get the inference result and plot a subplot
        pred_res = inference(img)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"predict result: {pred_res}")

    plt.show()