#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Object detection with YOLOv3
# In this exercise, you learn how to perform object detection
# with YOLOv3 on Gretchen's video stream
#

import sys
sys.path.append('..')

# Import required modules
import cv2
import argparse
import numpy as np
import hashlib

from lib.camera_v2 import Camera
from lib.ros_environment import ROSEnvironment

#
# Default parameters for network
# (YOLOv3)
#
cfg_path = "./yolov3.cfg"
weight_path= "./yolov3.weights"
class_name_path = "./yolov3.txt"

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

    #
    # Parse command line arguments
    #
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False, help = 'path to input image')
    ap.add_argument('-c', '--confidence', required=False, help = 'confidence threshold', type=float)
    ap.add_argument('-m', '--nms', required=False, help = 'NMS threshold', type=float)
    # TODO
    # (1) add more command line arguments that allow
    # - the specification of the network, the weights, and the labels
    # - to enable/disable the preview window
    # (2) define a boolean flag, such as 'isCam' that is True when the feed should be taken from the
    #     camera and is False otherwise
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
    # TODO
    # evaluate newly added parameters
    # ...

    if args.network is not None:
        cfg_path = args.network
    if args.weight is not None:
        weight_path = args.weight
    if args.label is not None:
        class_name_path = args.label

    isCam = False           # TODO: fix
    hasPreview = False      # TODO: fix

    if args.isCam is True:
        isCam = True
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

    # identify all output layers (depend on network: YOLOv3 has 3, YOLOv3-tiny has 2) !!! by James
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
    if hasPreview:
        cv2.namedWindow("Preview")

    #
    # We only need to initialize the camera if we are actually going to use it
    #
    if isCam:
        ROSEnvironment()
        camera = Camera()
        camera.start()
        print("camera starting")

    #
    # Here we go...
    #
    while(True):
        #
        # TODO
        #
        # - if the feed comes from the camera, set 'input_image' to the camera image,
        #   otherwise load image from disk
        # - determine height, width of image to analyze
        # - perform preprocessing, inference, and result extraction the same way as
        #   in example_image_detector.py (i.e., copy-paste the code and adjust a bit)
        print("While loop start")
        if isCam:
            key = cv2.waitKey(10)
            input_image = camera.getImage()
            print("Is cam testing")
        else:
            key = cv2.waitKey()

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

        for pred in preds:
            for anchorbox in pred:
                class_id = np.argmax(anchorbox[5:])
                confidence = anchorbox[4] * anchorbox[class_id + 5]

                # for analysis/debugging, we allow a separate (manual) threshold
                # replace 0 with 1 if you are not interested in predictions below the confidence threshold
                if confidence >= conf_threshold or confidence >= 1:
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

        #
        # Preview: show all anchorboxes with a total confidence > conf_threshold
        #
        preview = input_image.copy()
        for idx, classid in enumerate(class_ids):
            drawAnchorbox(preview, classid, confidence_values[idx], bounding_boxes[idx])


        #
        # Run the NMS (Non Maximum Suppression) algorithm to eliminate boxes with a low confidence
        # and those that overlap too much
        #
        if nms_threshold >= 0:
            indices = cv2.dnn.NMSBoxes(bounding_boxes, confidence_values, conf_threshold, nms_threshold)
        else:
            # include all boxes
            indices = range(0, len(bounding_boxes))

        #
        # Overlay final results on image
        #
        indices = np.reshape(indices, -1)
        for idx in indices:
            drawAnchorbox(input_image, class_ids[idx], confidence_values[idx], bounding_boxes[idx])

        #
        # Display results
        #
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Preview", preview[..., ::-1])  # swap RGB -> BGR
        cv2.imshow("ObjectDetection", input_image[..., ::-1])  # idem
        # key = cv2.waitKey()

        if key > 0:
            break

    cv2.destroyAllWindows()


#
# Program entry point when started directly
#
if __name__ == '__main__':
    main()