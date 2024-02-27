#seperate into 3 channels
#convolutional of k x k
#max pooling
#flatten amoung channels
#2 layer linear with relu on first, softmax into c classes on second

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy 
import os

class TrafficSignClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = 

    def forward(self, X):
        return self.net(X)

#Data
class TrafficSignDataset(Dataset):
    def __init__(self, test=False):
        sub_dir = "test" if test else "train"
        self.image_directory = "../data/traffic_sign_classification/" + sub_dir + "/images/"
        self.labels_directory = "../data/traffic_sign_classification/" + sub_dir + "/labels/"
        self.class_labels = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']

        #get image labels
        column_names = ["id", "class_label_index", "x1", "y1"]
        image_ids = set()

        #read all files in images directory
        with os.listdir(self.image_directory) as directory_list: 
            #only handling jpg (which should be the only data type)
            image_ids = set(map(lambda image_name: image_name[-3], filter(lambda file_name: True if file_name[-3] == '.jpg' else False, directory_list)))

        #filter out all image ids for which labels don't exist
        for image_id in image_ids.iter():
            if not os.path.isfile(self.labels_directory + image_id + ".txt"):
                image_ids.remove(image_id)

        #create dataframe with ids, and empty columns for label data

    
    def __len__(self):
        return len(self.image_locations) 

    def __getitem__(self, ind):
        #find row in image labels set by row index

        #put important information (all columns except image id and row index)

        #with image id from row, retrive image

        #decompose into channels, into tensor
        #maybe tensor.read_image function in pytorch???

        return (image_tensor, label_data_tensor)