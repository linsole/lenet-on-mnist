# lenet_on_mnist
This is a pytorch implementation of LeNet handwritten digit recognition model.
## About Model
The model and the parameters used to create the layers (convolutional layers and fully connected layers, etc.) are seperated into two files for the convinience of modifying model layers. Model implementation resides in model.py, and parameters are in config.yaml.
## About Dataset
This repository uses the CSV format redistributions of MNIST dataset, which can be downloaded in https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. For training purpose you can just use the csv file since the data has already been organized, but for inferencing purpose you may find it useful to hava images at hand. Functions involving parsing csv file to images and inferencing reside in utils.py.
## Get Started
First download dataset from the website described above into a subfolder (be aware of the hard-coded path in the code, you may need to modify them). Then just execute train.py, if everything works fine, you will get a "saves.pth" in the folder.
That's the training part. For inference, use the "parse_data" function in utils.py to convert dataset into folders of images, then use "inference" function to plot a single image on a pop up window with a predicted result as the title.
Happy hacking.
