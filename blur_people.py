# Script to blur people from a video stream (Real time)
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import time
import yolo_v3
import yolo_v3_tiny
from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, convert_to_original_size
from PIL import ImageDraw, Image, ImageFilter

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')

tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

# Function to load the graph from the .pb file
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

# Function to open the pipeline of the camera
def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=20,
                       flip_method=0):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (
            capture_width, capture_height, framerate, flip_method, display_width, display_height))


# Class defining the tracking algorithm and the necessary variables
class TrackingAlgorithm:

    def __init__(self):
        # Iteration, necessary to asign an iteration to the data
        self.count = 0

        # Is the first time an object is recognized?
        self.first_time = True

        # Variable to add new objects
        self.consecutive = 0
        self.discarded=[]

        # Dictionary with recognized objects
        self.objects = dict()

        # Initialize object to save video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.writeVideo = cv2.VideoWriter('output.mp4', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))
        self.writeVideo2 = cv2.VideoWriter('output2.mp4', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))
        self.personBox = []
        self.personClass = 0
    # Take the boxes, supress the non_max, draw the boxes and show the images
    def draw_and_show(self,detected_boxes,pil_im):
        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)
        self.draw_boxes_and_objects(filtered_boxes, pil_im, classes, (FLAGS.size, FLAGS.size), True)
        img = np.array(pil_im)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #self.writeVideo2.write(img)
        if len(self.personBox)>0:
            y1 = int(self.personBox[0])
            y2 = int(self.personBox[1])
            x1 = int(self.personBox[2])
            x2 = int(self.personBox[3])
            if y1>0 and y2>0 and x1>0 and x2>0:
                print(x1,x2,y1,y2)
                img[y2:x2, y1:x1] = cv2.blur(img[y2:x2, y1:x1], (23, 23))
                y = 460
                x = 1020
                h = 50
                w = 50
                img[y:y + h, x:x + w] = cv2.blur(img[y:y + h, x:x + w], (23, 23))
        cv2.imshow('CSI Camera', img)
        if  self.count%5 ==0:
            self.writeVideo.write(img)

    # Process every detected box, that means: Draw the boxes in the images, and create and update tracked objects
    def draw_boxes_and_objects(self,boxes, img, cls_names, detection_size, is_letter_box_image):
        draw = ImageDraw.Draw(img)
        if not self.objects:
            self.first_time = True
        # For every detected box from all clases
        for cls, bboxs in boxes.items():
            color = tuple(np.random.randint(0, 256, 3))

            for box, score in bboxs:
                # Box processing, changing from Yolo format
                box = convert_to_original_size(box, np.array(detection_size),
                                               np.array(img.size),
                                               is_letter_box_image)
                # Draw boxes and names
                if cls ==0:
                    draw.rectangle(box, outline=colors_array[cls],fill=(0,0,0))



    # Change coordinates from x0, y0, x1, y1 to x, y, width, height
    def change_coordinates(self,box):
        width = box[2]-box[0]
        height = box[3]-box[1]
        x = box[0]+width/2
        y = box[1]+height/2
        return x,y

    # Function to change from cv2 to pil and resizing
    def prepare_image(self,img):
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        img_resized = letter_box_image(pil_im, FLAGS.size, FLAGS.size, 128)
        img_resized = img_resized.astype(np.float32)
        return img_resized,pil_im

def colors(classes):
    farbe =dict()
    for i,classe in enumerate(classes):
        farbe[i] = tuple(np.random.randint(0, 256, 3))
    return farbe

# Load the classes file and the graph
classes = load_coco_names(FLAGS.class_names)
frozenGraph = load_graph(FLAGS.frozen_model)

colors_array = colors(classes)

# Initialize the pipeline for the camera
# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=2))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

# Prepare the cv2 window
if cap.isOpened():
    window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

# Create session and load graph
# Configure the tensorflow session, especially with allow_growth, so it doesnt fails to get memory
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False)

# Initialize the session: ACHTUNG!! This is the more efficient way, in comparison to with tf.Session as sess:
#I am not sure why, but that way freezes the Nanoboard and make the loading process really slow
tf_sess = tf.Session(graph=frozenGraph,config=config)

# Get the names of the inputs and outputs of the networks
boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)


tracker = TrackingAlgorithm()

# While you get something from the camera
while cv2.getWindowProperty('CSI Camera', 0) >= 0:
    ret_val, img = cap.read()


    # Prepare the image
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 180, 1)
    img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    img_resized, pil_im = tracker.prepare_image(img)

    # Run the network
    detected_boxes = tf_sess.run(boxes, feed_dict={inputs: [img_resized]})

    # Show the detected image and process tracking
    tracker.draw_and_show(detected_boxes,pil_im)
    tracker.count = tracker.count + 1

    # Check if the window should be closed
    keyCode = cv2.waitKey(30) & 0xff

    # Stop the program on the ESC key
    if keyCode == 27:
        break
        tracker.writeVideo.release()
        cap.release()
        cv2.destroyAllWindows()

# Close the tf Session
tf_sess.close()
