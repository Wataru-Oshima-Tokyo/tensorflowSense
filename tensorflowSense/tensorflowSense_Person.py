#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import os
import argparse
import sys
import time
from threading import Thread
import importlib.util
from traceback import print_last
from types import LambdaType
import cv2 as cv
from sensor_msgs.msg import Image
import cv_bridge
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class Depth_Retriever(Node):
    def __init__(self):
        super().__init__('detectPerson')
        self.declare_parameter('modeldir')
        try:
            self.MODEL_NAME = str(self.get_parameter('modeldir').value)
        except:
            print("not found")
            self.MODEL_NAME = ""
        print(self.MODEL_NAME)
        self.GRAPH_NAME = 'detect.tflite'
        self.LABELMAP_NAME = 'labelmap.txt'
        self.min_conf_threshold = float(0.5)
        self.resW, self.resH = 640, 480
        self.imW, self.imH = int(self.resW), int(self.resH)
        self.use_TPU = False
        # print(MODEL_NAME)
        # print(GRAPH_NAME)
        # print(LABELMAP_NAME)
        # print(use_TPU)

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        
        #it works if it is run on python3 but python2 dodes not have importlib.util
        #so just make it false always

        pkg = importlib.util.find_spec('tflite_runtime')
        #pkg = False
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (self.GRAPH_NAME == 'detect.tflite'):
                self.GRAPH_NAME = 'edgetpu.tflite'       

        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # # Path to .tflite file, which contains the model that is used for object detection
        # PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
        # 
        # # Path to label map file
        # PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
        self.PATH_TO_CKPT = os.path.join(self.MODEL_NAME,self.GRAPH_NAME)
        self.PATH_TO_LABELS = os.path.join(self.MODEL_NAME, self.LABELMAP_NAME)

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del(self.labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if self.use_TPU:
            self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT,
                                    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(self.PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5
        # self.input_mean = 45.5
        # self.input_std = 45.5

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv.getTickFrequency()

        # # Initialize video stream
        # videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
        # time.sleep(1)

        # #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        # while True:

        #     # Start timer (for calculating frame rate)
        #     t1 = cv.getTickCount()

        #     # Grab frame from video stream
        self.ymin = 0
        self.xmin = 0
        self.ymax = 0
        self.xmax = 0
        self.bridge = cv_bridge.CvBridge()

        self.sub = self.create_subscription(Image,'/camera/color/image_raw', self.callback, 1)
        self.sub = self.create_subscription(Image, '/camera/depth/image_rect_raw',  self.callback, 1)
        self.sub = self.create_subscription(Image,'/camera/depth/image_rect_raw', self.depthCallback, 1)
        self.image_pub = self.create_publisher(Image,'/camera/tensorflow/image_raw',  1)
        self.person = self.create_publisher( String,'/camera/tensorflow/object', 1)
        self.depth = self.create_publisher( String,'/camera/tensorflow/distance', 1)

    def depthCallback(self, depth_pic):
     # Use cv_bridge() to convert the ROS image to OpenCV format
        try:  
            msg = String()
            #Convert the depth image using the default passthrough encoding
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_pic, desired_encoding="passthrough")
            depth_array = np.array(self.depth_image, dtype=np.float32)
            center_idx = np.array(depth_array.shape) / 2
            x_mean, y_mean = (self.xmax+self.xmin) /2, (self.ymax+self.ymin)/ 2
            if x_mean >480:
                x_mean = 479
            elif y_mean >480:
                y_mean = 479
            # print ('center depth:', depth_array[center_idx[0], center_idx[1]])
            # print(depth_array[center_idx[0], center_idx[1]])
            msg.data = str(depth_array[int(x_mean), int(y_mean)])
            #self.depth.publish(msg)
            # self.depth.publish(str(self.xmin))
        except cv_bridge.CvBridgeError:
            pass
        #Convert the depth image to a Numpy array

    def callback(self,ros_pics):
        try: 
            msg = String()
            t1 = cv.getTickCount()
            my_image = self.bridge.imgmsg_to_cv2(ros_pics, desired_encoding = "bgr8")
            #my_image = ros_numpy.numpify(ros_pics)
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
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
            self.interpreter.invoke()

            # Retrieve detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    # self.ymin = int(max(1,(boxes[i][0] * self.imH)))
                    # self.xmin = int(max(1,(boxes[i][1] * self.imW)))
                    # self.ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                    # self.xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                    self.ymin = 0
                    self.xmin = 0
                    self.ymax = 0
                    self.xmax = 0
                    # cv.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    category = 'no'
                    if(object_name == 'person' and int(scores[i]*100) >= 60):
                        self.ymin = int(max(1,(boxes[i][0] * self.imH)))
                        self.xmin = int(max(1,(boxes[i][1] * self.imW)))
                        self.ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                        self.xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                        x_mean, y_mean = int((self.xmax+self.xmin) /2), int((self.ymax+self.ymin)/ 2)
                        category = 'yes'
                        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(self.ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv.rectangle(frame, (self.xmin,self.ymin), (self.xmax,self.ymax), (10, 255, 0), 2)
                        cv.rectangle(frame, (x_mean,y_mean), (x_mean+10,y_mean+10), (255, 255, 0), 2)
                        cv.rectangle(frame, (self.xmin, label_ymin-labelSize[1]-10), (self.xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv.FILLED) # Draw white box to put label text in
                        cv.putText(frame, label, (self.xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv.putText(frame,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv.imshow('Object detector', frame)
            try:
                size = str(np.array(frame, dtype=np.float32).shape)
                msg.data = size
                self.person.publish(msg)
            except:
                pass
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame))
            # Calculate framerate
            t2 = cv.getTickCount()
            time1 = (t2-t1)/self.freq
            frame_rate_calc= 1/time1

            # Press 'q' to quit
            cv.waitKey(1) 
        except cv_bridge.CvBridgeError as e:
            print("CvBridge could not convert images from realsense to opencv")
        #height,width, channels = my_image.shape
        #my_height = my_image.shape[0]
        # print(my_height)

        #self.vert_len = ros_pics.height # retrieve height information from Image msg
        # print(self.vert_len)

# def startup(tfFolder):
#     # Define and parse input arguments
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
#     # #                    required=True)
#     # 		    default='/home/ubuntux86/catkin_ws/src/tensorflowSense/Sample_TFLite_model')
#     #             # default=tfFolder)
#     # parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
#     #                     default='detect.tflite')
#     # parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
#     #                     default='labelmap.txt')
#     # parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
#     #                     default=0.5)
#     # parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
#     #                     default='1280x720')
#     # parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
#     #                     action='store_true')

#     # args = parser.parse_args()
#     # #args = parser.parse_known_args()
#     # MODEL_NAME = args.modeldir
#     # GRAPH_NAME = args.graph
#     # LABELMAP_NAME = args.labels
#     # min_conf_threshold = float(args.threshold)
#     # resW, resH = args.resolution.split('x')
#     # imW, imH = int(resW), int(resH)
#     # use_TPU = args.edgetpu
    
        



def main(args=None):
    print("start")
    rclpy.init(args=args)
    # startup()
    Depth=Depth_Retriever()
    rclpy.spin(Depth)


if __name__ == '__main__':

    main()
