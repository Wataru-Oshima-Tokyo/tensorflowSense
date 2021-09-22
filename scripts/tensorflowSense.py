#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import sys
import time
from threading import Thread
import importlib.util
import cv2 as cv
from sensor_msgs.msg import Image
import rospy
import cv_bridge
import numpy as np
from geometry_msgs.msg import Twist

class Depth_Retriever():
    def __init__(self):
        
        self.bridge = cv_bridge.CvBridge()

        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.image_pub = rospy.Publisher('/camera/tensorflow/image_raw', Image, queue_size=1)
    def callback(self,ros_pics):
        global width, height, frame_rate_calc
        try: 
            t1 = cv.getTickCount()
            my_image = self.bridge.imgmsg_to_cv2(ros_pics, desired_encoding = "bgr8")
            # cv.imshow('Object detector', my_image)
            # frame1 = videostream.read()
            frame1 = my_image
            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # frame_resized = cv.resize(frame_rgb, (width, height))
            frame_resized = cv.resize(frame_rgb, (300, 300))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv.FILLED) # Draw white box to put label text in
                    cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            # cv.imshow('Object detector', frame)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame))
            # Calculate framerate
            t2 = cv.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # Press 'q' to quit
            cv.waitKey(1) 
        except cv_bridge.CvBridgeError as e:
            print("CvBridge could not convert images from realsense to opencv")
        height,width, channels = my_image.shape
        my_height = my_image.shape[0]
        # print(my_height)

        self.vert_len = ros_pics.height # retrieve height information from Image msg
        # print(self.vert_len)

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv.getTickFrequency()

# # Initialize video stream
# videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
# time.sleep(1)

# #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
# while True:

#     # Start timer (for calculating frame rate)
#     t1 = cv.getTickCount()

#     # Grab frame from video stream
    



def main():
    print("start")
    rospy.init_node("Depth")
    Depth=Depth_Retriever()
    rospy.spin()


if __name__ == '__main__':

    main()