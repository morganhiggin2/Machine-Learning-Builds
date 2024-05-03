import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms as non_max_surpression 
from torchvision.ops import box_iou, clip_boxes_to_image, roi_pool
import numpy 
import pandas
import os
import glob
import xml.etree.ElementTree as ET 
import math
import matplotlib.pyplot as pyplot 
import matplotlib
from progress.bar import Bar as ProgressBar

#methods we need
#generate anchor boxes
#get overlap matrix
#get best region predictions by method of maximum ...


#Data
class CowDataset(Dataset):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu 

        self.sub_dir = "../data/cow_object_detection/"
        bounding_box_file_path = self.sub_dir + "annotations.xml"

        #get bouding boxs for each images, and their classification
        bounding_box_dataframe = self.__read_xml(bounding_box_file_path)

        # list of images names 
        self.image_names = bounding_box_dataframe['filename'].unique().tolist()

        # list of list, containing bouding boxes tuples for each image
        self.target_data = bounding_box_dataframe 

        self.image_dimensions = (4200, 3200)

    def __read_xml(self, path):
        # Read xml file, credit to STPETE_ISHII of Kaggle
        dataset = []

        for anno in glob.glob(path):
            tree = ET.parse(anno)
            root = tree.getroot()
        
        for image_elem in root.iter("image"):
            image_attrs = image_elem.attrib
            image_data = {
                "filename": image_attrs['name'],
                "width": float(image_attrs['width']),
                "height": float(image_attrs['height']),
                "boxes": []
            }
            
            for box_elem in image_elem.iter("box"):
                box_attrs = box_elem.attrib
                box_data = {
                    "label": box_attrs['label'],
                    "occluded": int(box_attrs['occluded']),
                    "xtl": float(box_attrs['xtl']),
                    "ytl": float(box_attrs['ytl']),
                    "xbr": float(box_attrs['xbr']),
                    "ybr": float(box_attrs['ybr'])
                }
                image_data["boxes"].append(box_data)
            
            dataset.append(image_data)

        dataset = pandas.DataFrame(dataset)

        flattened_dataset = pandas.DataFrame() 

        for i in range(len(dataset)):
            boxes=dataset.loc[i,'boxes']
            for box in boxes:
                a=dataset.loc[i,'filename']
                b=dataset.loc[i,'width']
                c=dataset.loc[i,'height']
                d=box['xtl']
                e=box['ytl']
                f=box['xbr']
                g=box['ybr']
                add_df=pandas.DataFrame([[a,b,c,d,e,f,g]])
                flattened_dataset=pandas.concat([flattened_dataset,add_df],axis=0)
        flattened_dataset.columns=['filename','width','height','xtl','ytl','xbr','ybr']

        return flattened_dataset

    def __len__(self):
        return len(self.image_names) 

    def __getitem__(self, ind):
        # Get image name
        image_name = self.image_names[ind] 
        
        # Get image as tensor
        image_tensor = read_image(self.sub_dir + image_name)
        image_size = image_tensor.size()

        # Pad image
        padded_image_tensor = torch.zeros((3, self.image_dimensions[0], self.image_dimensions[1]))
        padded_image_tensor[:, 0:image_size[1], 0:image_size[2]] = image_tensor[:,:,:]

        if self.use_gpu:
            if torch.cuda.device_count() >= 1:
                    image_tensor.to(torch.device(f'cuda:{0}'))
        # Get class tensor of (N, 2) with first being if cow, second being not. Binary
        class_tensor = torch.tensor([self.target_data[self.target_data['filename'] == image_name], self.target_data[self.target_data['filename'] == image_name]])

        # Return image tensor, and bounding box information for this image
        return padded_image_tensor, class_tensor

    # Helper methods to get tensor data from target data
    # TODO also include use_gpu move in this as well

# Apply bounding box offsets to anchor boxes
def apply_bounding_box_offsets_to_anchors(anchors, offsets):
    #TODO is it width and height, or x2 and y2
    #anchors: (n, 4), x1, y1, 
    #offsets: (n, 4)
    return anchors + offsets

class CowDetectionPredictor(torch.nn.Module):
    def __init__(self, image_dimensions, use_gpu=False):
        super().__init__()

        #stacked CNN layers, maybe VGG style
        def block(in_channels, out_channels):
            conv_1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
            conv_2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
            max_pool = torch.nn.MaxPool2d(kenel_size=(2, 2), stride=2)

            return torch.nn.Sequential(conv_1, conv_2, max_pool)

        # Create VGG blocks
        block_1 = block(3, 2) 
        block_2 = block(2, 1)
         
        # Create sequence of blocks 
        self.conv_sequence = torch.nn.Sequential(block_1, block_2)

        rpn = RegionPurposalNetwork(image_dimensions, use_gpu)

        # Fully connected layer
        linear_1_1 = torch.nn.Linear(2 * 28 * 28, 12)
        self.fc = torch.nn.Sequential(linear_1_1)

        # Bouding box predictor
        linear_2_1_1 = torch.nn.Linear(12, 4)

        # Classifier
        lienar_2_2_1 = torch.nn.Linear(12, 2)
        softmax_1 = torch.nn.Softmax()

        if use_gpu:
            if torch.cuda.device_count() >= 1:
                self.net.to(torch.device(f'cuda:{0}'))

    def forward(self, X):
        # convo layers
        reduced_image = self.conv_sequence(X)
        
        # region purposal network
        region_purposals = self.rpn(reduced_image)  

        # roi pooling (num_region_purposals, channels, width, hegith)
        pooled_rois = roi_pool(reduced_image, [region_purposals])

        # flatten region purposals (num_region_purposals, channels * width * height)
        flattened_rois = torch.flatten(4, 2 * pooled_rois.shape[0] * pooled_rois.shape[1])

        # Apply fully connected layer
        fc_out = self.fc(flattened_rois)

        # Predict bounding box offsets
        offset_prediction = self.offset_predictor(fc_out)
        # Get bounding box predictions
        bounding_box_predictions = apply_bounding_box_offsets_to_anchors(region_purposals, offset_prediction)

        # Clip purposals
        clipped_bouding_box_predictions = clip_boxes_to_image(bounding_box_predictions, (X.shape[0], X.shape[1]))

        # Predict bouding box classification (cow or not cow)
        classification_prediction = self.classifier(fc_out)
       
        # Return as bouding box predictions, bouding box classification
        return clipped_bouding_box_predictions, classification_prediction 

class RegionPurposalNetwork(torch.nn.Module):
    def __init__(self, image_dimensions, use_gpu=False):
        super().__init__()
        self.num_anchors=9
        self.bounding_box_limit = 5

        self.image_dimensions = image_dimensions

        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3), padding=1)

        self.anchor_box_generator = AnchorGenerator(sizes=((16, 32, 64), ), aspect_ratios=((0.5, 1.0, 2.0), )) 
        self.feature_maps = [torch.empty((16, 16))]

        # Predict the offsets
        self.bounding_box_predictor = torch.nn.Conv2d(in_channels=2, out_channels=self.num_anchors*4, kernel_size=(3, 3), padding=(1, 1))
        
        # Predict class labels
        self.class_predictor = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3, 3), padding=(1, 1))
        self.softmax = torch.nn.Softmax2d()
        self.class_confidence = 0.95


    def forward(self, X):
        # Get conv pass output
        Y = self.conv(X)

        # Generate anchor boxes ()
        anchor_boxes = self.anchor_box_generator(ImageList(Y.unsqueeze(0), ), self.feature_maps)

        # Bounding box prediction (num_anchors * 4, x.size.0, x.size.1)
        bounding_box_offsets = self.bounding_box_predictor(Y) 

        # Reshape into (num_anchors, 4, x.size.0, y.size.0) TODO
        bounding_box_offsets = torch.reshape(bounding_box_offsets, (self.num_anchors, 4, X.shape[0], X.shape[1]))

        # Binary Class Prediction (object = 1, background = 0) (1, x.size.0, x.size.1)
        object_predictions = self.class_predictor(Y) 
        object_predictions = self.softmax = self.softmax(object_predictions)

        # Based on a confidence interval, will determine if it can label class probability as object or background
        object_prediction_mask = torch.gt(object_predictions[0,:,:], self.class_confidence)

        # Filter out bouding box predictions which are not object labeled (from class_prediction)
        filtered_anchor_boxes = torch.masked_select(anchor_boxes, object_prediction_mask)
        filtered_bounding_box_offsets = torch.masked_select(bounding_box_predictions, object_prediction_mask)

        bounding_box_predictions = apply_bounding_box_offsets_to_anchors(filtered_bounding_box_offsets, filtered_anchor_boxes)

        # Clip out of image anchor boxes
        filtered_bounding_box_prections = clip_boxes_to_image(filtered_bounding_box_prections, (X.shape[0], X.shape[1]))

        #TODO filter anchor boxes, for now, lets get say at most the top n anchor boxes
        # Calculate IOU of every box with respect to every other
        #iou_scores = box_iou(filtered_bounding_boxes, anchor_boxes) 
        # Get iou scores (num filtered bouding boxes, num anchor boxes)
        # Non maximum surpression using the filtered predicted bounding boxes
        # Gets indicies of kept region purposals
        #region_purposals_indicies = [non_max_surpression(anchor_boxes, anchor_box_iou_scores, iou_threshold=0.7) for anchor_box_iou_score in iou_scores]
        if filtered_bounding_box_prections.shape[0] > self.bounding_box_limit:
            filtered_bounding_box_prections = filtered_bounding_box_prections[0:5, :, :, :]

        # Return filtered bounding boxes as region purposals
        return filtered_bounding_box_prections 


#Model Variables
num_epochs = 4 
learning_rate = 1.0e-3 
training_data_batch_size = 10 
use_gpu = True 

training_dataset = CowDataset(use_gpu=True)
test_dataset = CowDataset(test=True, use_gpu=True)

#Generate data loaders for data
training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=training_data_batch_size, num_workers=0)
test_data_loader = DataLoader(test_dataset, shuffle=True, num_workers=0)

#Components
model = CowDetectionPredictor()
optimizer = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=learning_rate
)
bounding_box_loss_function = torch.nn.L1Loss(reduction='none')
classification_loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

def loss_function(bounding_box_predictions, classification_predictions, bounding_boxes, classes):
    bounding_box_loss = bounding_box_loss_function(bounding_box_predictions, bounding_boxes)
    classification_loss = classification_loss_function(classification_predictions, classes)

    return bounding_box_loss + classification_loss

#Dynamic Graph
store_losses = []

#Progress Bar
progress_bar = ProgressBar('processing', max=num_epochs * (len(training_dataset) / training_data_batch_size))

for epoch in range(num_epochs):
    avg_loss = 0

    #Move forward
    for i, (inputs, targets) in enumerate(training_data_loader):
        # Get targets
        bounding_boxes = targets[0]
        classes = targets[1]

        #Get predictions
        bounding_box_predictions, class_predictions = model.forward(inputs)

        #Compute Loss
        loss = loss_function(bounding_box_predictions, class_predictions, bounding_boxes, classes)

        optimizer.zero_grad()

        #Compute Gradients, increment weight
        loss.backward()
        optimizer.step()

        #Compute batch loss for graph
        store_losses += [loss.item()]

        progress_bar.next()

progress_bar.finish()

exit()
#MAJOR TODO DOWN HERE 

# Compute Loss over training data
training_sample_inputs, training_sample_targets = next(iter(training_data_loader))
sample_loss = loss_function(model(training_sample_inputs), training_sample_targets).detach().item()

# Compute Loss over test data
avg_loss = 1

for i, (inputs, targets) in enumerate(test_data_loader):
    print('predition was {pred} and target was {target}'.format(pred=model(inputs), target=targets))
    avg_loss = avg_loss + loss_function(model(inputs), targets).detach().item()

    #TODO DELETE
    if i == 1:
        break

avg_loss = avg_loss / len(test_dataset)

print("model has " + str(sum([param.nelement() for param in model.parameters()])) + " parameters")
print(f"sample loss was {sample_loss}")
print(f"test data loss was {avg_loss}")

#Plot with matplotlib
pyplot.style.use('ggplot')
matplotlib.use('TkAgg')           

step_size = 1.0 / math.ceil(len(training_dataset) / training_data_batch_size)
graph_epochs = numpy.arange(0, len(store_losses)) * step_size
graph_losses = numpy.array(store_losses)


figure = pyplot.figure()
axis = figure.add_subplot(111)
line, = axis.plot(graph_epochs, graph_losses, 'o')
axis.set_xlabel('epoch')
axis.set_ylabel('loss')

pyplot.show()
#watch -n 0.5 nvidia-smi