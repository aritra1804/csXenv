#!/usr/bin/env python3
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Final project_Trash_Detector!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import sys
sys.path.append('..')
import cv2
import argparse
import numpy as np
import hashlib
from lib.camera_v2 import Camera
from lib.ros_environment import ROSEnvironment
from lib.robot import Robot


from imutils.object_detection import non_max_suppression
from PIL import Image
import pyocr
import pyocr.builders
import time

# import os
# import torch
# import torchvision
# from torch.utils.data import random_split
# import torchvision.models as models
# import torch.nn as nn
# import torch.nn.functional as F
# import zipfile as zf
#
# files = zf.ZipFile("dataset-resized.zip",'r')
# files.extractall()
# files.close()
# os.listdir(os.path.join(os.getcwd(),"dataset-resized"))
#
# data_dir  = './dataset-resized'
#
# classes = os.listdir(data_dir)
# print(classes)
#
# from torchvision.datasets import ImageFolder
# import torchvision.transforms as transforms
# transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#
# dataset = ImageFolder(data_dir, transform = transformations)
# random_seed = 42
# torch.manual_seed(random_seed)
# train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
#
# from torch.utils.data.dataloader import DataLoader
# batch_size = 32
#
# train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
# val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)
#
# from torchvision.utils import make_grid
#
# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#
#
#

#
# class ImageClassificationBase(nn.Module):
#
#     def training_step(self, batch):
#         images, labels = batch
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss
#         return loss
#
#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
#
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
#
#     def epoch_end(self, epoch, result):
#         print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
#
# class ResNet(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#
#         # Use a pretrained model
#         self.network = models.resnet50(pretrained=True)
#
#         # Replace last layer
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
#
#     def forward(self, xb):
#         return torch.sigmoid(self.network(xb))
#
# model = ResNet()
#
# def get_default_device():
#
#     """Pick GPU if available, else CPU"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')
#
# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list,tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)
#
# class DeviceDataLoader():
#     """Wrap a dataloader to move data to a device"""
#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device
#
#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)
#
#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)
#
# device = get_default_device()
#
# train_dl = DeviceDataLoader(train_dl, device)
# val_dl = DeviceDataLoader(val_dl, device)
# to_device(model, device)
#
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)
#
# def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
#     history = []
#     optimizer = opt_func(model.parameters(), lr)
#     for epoch in range(epochs):
#
#         # Training Phase
#         model.train()
#         train_losses = []
#         for batch in train_loader:
#             loss = model.training_step(batch)
#             train_losses.append(loss)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#         # Validation phase
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         model.epoch_end(epoch, result)
#         history.append(result)
#     return history
#
# model = to_device(ResNet(), device)
#
# evaluate(model, val_dl)
#
# num_epochs = 1
# opt_func = torch.optim.Adam
# lr = 5.5e-5
#
# history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
#
# def predict_image(img, model):
#     # Convert to a batch of 1
#     xb = to_device(img.unsqueeze(0), device)
#     # Get predictions from model
#     yb = model(xb)
#     # Pick index with highest probability
#     prob, preds  = torch.max(yb, dim=1)
#     # Retrieve the class label
#     return dataset.classes[preds[0].item()]
#
# from PIL import Image
# from pathlib import Path
#
# loaded_model = model
# def predict_external_image(image_name):
#     image = Image.open(Path('./' + image_name))
#
#     example_image = transformations(image)
#

avail_tools = pyocr.get_available_tools()
avail_tools = avail_tools[0]
print("Available Tools: ", avail_tools.get_name())
print("Available Language: ", avail_tools.get_available_languages())
#
def postProcess(scores, geometry, minConfidence):
    # Go through the row and column to find the bounding box and confidence score
    (rows, cols) = scores.shape[2:4]
    bboxes = []
    confidence_scores = []

    for rowIndex in range(0, rows):
        score = scores[0, 0, rowIndex]
        # Need to use geometric data to determine the bounding box
        geoData0 = geometry[0, 0, rowIndex]
        geoData1 = geometry[0, 1, rowIndex]
        geoData2 = geometry[0, 2, rowIndex]
        geoData3 = geometry[0, 3, rowIndex]
        angleData = geometry[0, 4, rowIndex]

        for colIndex in range(0, cols):
            # If the confidence score is too low, we can skip
            if score[colIndex] < minConfidence:
                continue
            # Need to find the offset because the feature map is 4 times smaller than the original image size
            (xOffset, yOffset) = (colIndex * 4.0, rowIndex * 4.0)

            # Get the rotation angle
            angle = angleData[colIndex]
            # Compute cos and sin of the rotation angle
            cosAngle = np.cos(angle)
            sinAngle = np.sin(angle)

            # Compute the width and height
            width = geoData1[colIndex] + geoData3[colIndex]
            height = geoData0[colIndex] + geoData2[colIndex]

            endPtX = int(xOffset + (cosAngle * geoData1[colIndex]) + (sinAngle * geoData2[colIndex]))
            endPtY = int(yOffset - (sinAngle * geoData1[colIndex]) + (cosAngle * geoData2[colIndex]))
            startPtX = int(endPtX - width)
            startPtY = int(endPtY - height)
            bboxes.append((startPtX, startPtY, endPtX, endPtY))
            confidence_scores.append(score[colIndex])
    return bboxes, confidence_scores
#
# # The bounding box from the text detector might be too small so that it cuts some of the text
# # We want to create a bigger bounding box but without going over the border
def createBuffer(startX, startY, endX, endY):
    startX = startX -5
    endX = endX+5
    startY = startY -5
    endY = endY +5
    if(startX < 0):
        startX = 1
    if(startX > 640):
        startX = 640
    if(endX < 0):
        endX = 1
    if(endX > 640):
        endX = 640
    if(startY < 0):
        startY = 1
    if(startY > 480):
        startY = 480
    if(endY < 0):
        endY = 1
    if(endY > 480):
        endY = 480
    return startX, startY, endX, endY




#If you have to change, change path
cfg_path = "./yolov3.cfg"
weight_path= "./yolov3.weights"
class_name_path = "./yolov3.txt"


camera = None
classes = None
COLORS = None


def loadClasses(filename):
    """ Load classes into 'classes' list and assign a random but stable color to each class. """
    global classes, COLORS

    # Load classes into an array
    try:
        with open(filename, 'r') as file:
            classes = [line.strip() for line in file.readlines()]
    except EnvironmentError:
        print("Error: cannot load classes from '{}'.".format(filename))
        quit()

    # Assign a random (but constant) color to each class
    # Method: convert first 6 hex characters of md5 hash into RGB color values
    COLORS = []
    for idx,c in zip(range(0, len(classes)), classes):
        cstr = hashlib.md5(c.encode()).hexdigest()[0:6]
        c = tuple( int(cstr[i:i+2], 16) for i in (0, 2, 4))
        COLORS.append(c)


def drawAnchorbox(frame, class_id, confidence, box):
    """ Draw an anchorbox identified by `box' onto frame and label it with the class name and confidence. """
    global classes, COLORS

    conf_str = "{:.2f}".format(confidence).lstrip('0')
    label = "{:s} ({:s})".format(classes[class_id], conf_str)
    color = COLORS[class_id]

    # Make sure we do not print outside the top/left corner of the window
    lx = max(box[0] + 5, 0)
    ly = max(box[1] + 15, 0)

    # 3D "shadow" effect: print label with black color shifted one pixel right/down,
    #                     then print the colored label at the indented position.
    cv2.rectangle(frame, box, color, 2)
    cv2.putText(frame, label, (lx+1, ly+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    


def main():
    global cfg_path, weight_path, class_name_path, classes, COLORS

    #
    # Set default parameters
    #
    # blobFromImage
    scale = 1.0/255         # scale factor: normalize pixel value to 0...1
    meansub = (0, 0, 0)     # we do not use mean subtraction
    outsize = (416, 416)    # output size (=expected input size for YOLOv3)
    # result detection
    conf_threshold = 0.50   # confidence threshold
    nms_threshold = 0.4     # threshold for non-maxima suppression (NMS)

    
    # Parse command line arguments
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False, help = 'path to input image')
    ap.add_argument('-c', '--confidence', required=False, help = 'confidence threshold', type=float)
    ap.add_argument('-m', '--nms', required=False, help = 'NMS threshold', type=float)
    ap.add_argument('-n', '--network', required=False, help='network path')
    ap.add_argument('-w', '--weight', required=False, help='weights path')
    ap.add_argument('-l', '--label', required=False, help='labels path')
    ap.add_argument('-p', '--preview', required=False, help = 'enable preview window', type=bool, choices=[True, False])
    ap.add_argument('--isCam', required=False, help= 'enable Camera', type=bool, choices=[True, False])

    args = ap.parse_args()

    if args.confidence is not None:
        conf_threshold = args.confidence
    if args.nms is not None:
        nms_threshold = args.nms
    if args.network is not None:
        cfg_path = args.network
    if args.weight is not None:
        weight_path = args.weight
    if args.label is not None:
        class_name_path = args.label

    isCam = True           
    hasPreview = False      

    if args.isCam is False:
        isCam = False
    if args.preview is True:
        hasPreview = True

    #
    # Print configuration
    #
    print("Configuration:\n"
          "  Network:\n"
          "    config:      {}\n"
          "    weights:     {}\n"
          "    classes:     {}\n"
          "  Preprocessing:\n"
          "    scale        {:.3f}\n"
          "    mean subtr.  {}\n"
          "    output size  {}\n"
          "  Detection:\n"
          "    conf. thld   {:.3f}\n"
          "    nms. thld    {:.3f}"
          "\n"
          .format(cfg_path, weight_path, class_name_path, scale, meansub, outsize, conf_threshold, nms_threshold))

    #
    # Initialize network
    #
    # load DNN
    net = cv2.dnn.readNet(weight_path, cfg_path)

    # load classes
    loadClasses(class_name_path)

    # it has version issue. so if it doesn't work, then change to i[0]
    
    layer_names = net.getLayerNames()
    output_layers = [ layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers() ]
   

    # print the names of all layers and, separately, all output layers
    print("Network information:\n"
          "  layer names:\n    {}"
          "  output layers:\n    {}"
          "\n"
          .format(layer_names, output_layers))

    #
    # Setup windows
    #
    cv2.namedWindow("ObjectDetection")
    #if hasPreview:
       # cv2.namedWindow("Preview")
    if isCam:
        ROSEnvironment()
        camera = Camera()
        camera.start()
        robot = Robot()
        print("camera starting")
    robot.start()
    robot.center()

    textlist=[]
    text_x_list=[]
    text_y_list=[]


    while(True):
        
        if isCam:
            key = cv2.waitKey(5)
            input_image = camera.getImage()
            
        else:
            key = cv2.waitKey()
            print("Where is cam?")

        #
        # Load image from disk
        #
        if args.image is not None:
            input_image = cv2.imread(args.image)
        if input_image is None:
            print("Error: cannot load image '{}'.".format(args.image))
            quit()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        width = input_image.shape[1]
        height = input_image.shape[0]

        #
        # Preprocess image
        #
        # create blob from input image after mean subtraction and normalization
        blob = cv2.dnn.blobFromImage(input_image, scale, outsize, meansub, swapRB=False, crop=False)

        #
        # Inference: run forward pass through network
        #
        # feed the image blob to the neural net
        net.setInput(blob)

        # run inference and return results from identified output layers (2 for YOLOv3-tiny, 3 for YOLOv3)
        preds = net.forward(output_layers)

        #
        # Iterate through all detected anchor boxes in each output layer
        #
        # the format of an anchor box is
        #   [0:1] center (x/y), range 0..1 (multiply by image width/height)
        #   [2:3] dim    (x/y), range 0..1 (multiply by image width/height)
        #   [4]   p_obj  probability that there is an object in this box
        #   [5:]  probabilities for each class (80 with YOLOv3 / COCO)
        #
        # an anchor's total confidence is obj * argmax(class probabilities)
        #
        # initialize empty result lists
        class_ids = []
        confidence_values = []
        bounding_boxes = []

        net2 = cv2.dnn.readNet("./frozen_east_text_detection.pb")

        img_copy = input_image.copy()
        img_output = input_image.copy()

        # Change the size of the image
        # Smaller images are faster, but might be less accurate
        newWidth = 320
        newHeight = 320
        originWidth = 640
        originHeight = 480
        ratioWidth = originWidth / float(newWidth)
        ratioHeight = originHeight / float(newHeight)
        img_copy = cv2.resize(img_copy, (newWidth, newHeight))

        # Select the layer that we will be using. Conv_7/Sigmoid is used to get the probability.
        # concat_3 is used to determine the bounding box,. 
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        """
        # identify all output layers (depend on network: YOLOv3 has 3, YOLOv3-tiny has 2)
        layer_names = net.getLayerNames()
        output_layers = [ layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers() ]

        # print the names of all layers and, separately, all output layers
        print("Network information:\n"
              "  layer names:\n    {}"
              "  output layers:\n    {}"
              "\n"
              .format(layer_names, output_layers))
        """


        # Create blobs that will be forward passed in the network
        blob2 = cv2.dnn.blobFromImage(img_copy, 1.0, (newWidth,newHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Input the blobs and forward pass the blobs
        net2.setInput(blob2)
        (scores, geometry) = net2.forward(layerNames)

        # The postProcess method takes in scores, geometry, and minimum score. If the score is less than the minimum score, it is removed.
        # The postProcess method outputs the bounding boxes (x, y, width, height) and the confidence scores
        (bboxes, confidences) = postProcess(scores, geometry, 0.9)

        # Removes overlapping boxes
        bboxes = non_max_suppression(np.array(bboxes), probs=confidences)

        # Draw rectangles using the bounding boxes
        for (startPtX, startPtY, endPtX, endPtY) in bboxes:
            # Need to consider the resizing and the cv2.rectangle takes in start points and end points. So, need to compute the start points and the end points
            startPtX = int(startPtX * ratioWidth)
            startPtY = int(startPtY * ratioHeight)
            endPtX = int(endPtX * ratioWidth)
            endPtY = int(endPtY * ratioHeight)

            # For the detected text and its bounding box, we want to recognize the text. We will create some buffer around the text and make sure it doesn't go over the image size.
            (startPtX, startPtY, endPtX, endPtY) = createBuffer(startPtX, startPtY, endPtX, endPtY)

            # We cut the region in the image where we want to recognize text
            roi = input_image[startPtY:endPtY, startPtX:endPtX, :] # ROI is short for region of interest

            # Convert the format of the image
            img_pil = Image.fromarray(roi)

            # Use pyocr to recognize the text in the roi image
            text = avail_tools.image_to_string(img_pil, lang='eng', builder=pyocr.builders.TextBuilder())

            # Remove non-ascii characters with a space
            text = "".join([char if ord(char) < 128 else "" for char in text]).strip()

            # Print recognized text
            print("{}\n".format(text))

            # Print the text on the image
            cv2.putText(img_output, text, (startPtX, startPtY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.rectangle(img_output, (startPtX, startPtY), (endPtX, endPtY), (0, 255, 255), 2)

            #for robot
            textlist.append(text)
            text_x_list.append((startPtX+endPtX)//2)
            text_y_list.append((startPtY+endPtY)//2)
        

       



        for pred in preds:
            for anchorbox in pred:
                class_id = np.argmax(anchorbox[5:])
                confidence = anchorbox[4] * anchorbox[class_id + 5]


                # for analysis/debugging, we allow a separate (manual) threshold
                # replace 0 with 1 if you are not interested in predictions below the confidence threshold
                if (confidence >= conf_threshold or confidence >= 1) and (classes[class_id]=="Food" or  classes[class_id]=="Garbage" or classes[class_id]=="recycle"):
                    cx = int(anchorbox[0] * width)  # center of anchorbox in image
                    cy = int(anchorbox[1] * height)  # in x & y direction
                    dx = int(anchorbox[2] * width)  # dimensions of ancherbox in image
                    dy = int(anchorbox[3] * height)  # in x & y direction

                    x = int(cx - dx / 2)  # x/y coordinate of
                    y = int(cy - dy / 2)  # anchorbox

                    # print result
                    print(
                        "[ ({:3d}/{:3d}) - ({:3d}/{:3d}): Pobj: {:.3f}, Pclass: {:.3f} -> Ptotal: {:.3f}, class: {:s} ]"
                        .format(x, y, x + dx, y + dy, anchorbox[4], anchorbox[class_id + 5], confidence,
                                classes[class_id]))
                                       # only consider prediction if total confidence is above threshold
                    if confidence >= conf_threshold:
                        class_ids.append(class_id)
                        confidence_values.append(
                            float(confidence))  # find out what happens in NMSBoxes if you remove the typecast...
                        bounding_boxes.append([x, y, dx, dy])

                        #if classes[class_id]=="bottle":
                           # print("bottle")
                        try:
                            if (classes[class_id] in textlist):
                                textlist.clear()
                                (ox, oy, oz) = camera.convert2d_3d(cx, cy)
                                (ox, oy, oz) = camera.convert3d_3d(ox, oy, oz)
                                (x,y,z)=camera.convert2d_3d(text_x_list[0], text_y_list[0])
                                (x, y, z)=camera.convert3d_3d(x, y, z)

                                for i in range(1,6):

                                    robot.lookatpoint(ox,oy,oz,100)
                                    
                                    robot.down(0.3)
                                    robot.up(0.3)
                                    # time.sleep(0.5)
                                    robot.lookatpoint(x, y, z,100)
                                    robot.up(0.3)
                                    robot.down(0.3)

                                    

                                text_x_list.clear()
                                text_y_list.clear()
                                time.sleep(1)

                                robot.center()

                        except:
                            textlist.clear()
                            text_x_list.clear()
                            text_y_list.clear()


                        if nms_threshold >= 0:
                            indices = cv2.dnn.NMSBoxes(bounding_boxes, confidence_values, conf_threshold, nms_threshold)
                        else:
                        # include all boxes
                            indices = range(0, len(bounding_boxes))        
                        #Overlay final results on image        
                        indices = np.reshape(indices, -1)
                        for idx in indices:
                            drawAnchorbox(img_output, class_ids[idx], confidence_values[idx], bounding_boxes[idx])
            

        #
        # Display results
        #l
        img_output2 = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
        cv2.imshow("ObjectDetection", img_output2[..., ::-1])  # idem

        # key = cv2.waitKey()

        if key > 0:
            break

    cv2.destroyAllWindows()


#
# Program entry point when started directly
#
if __name__ == '__main__':
    main()
