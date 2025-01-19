import os
import tensorflow as tf
import numpy as np
import cv2
from tkinter import Tk, Button, Label, filedialog, Frame, StringVar, OptionMenu
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import time
from yolo_v3 import Yolo_v3
from utils_v3 import load_images, load_class_names, draw_boxes, draw_frame

# Configuration
_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './coco.names'
_MAX_OUTPUT_SIZE = 50
iou_threshold = 0.3
confidence_threshold = 0.3

# Load class names
class_names = load_class_names(_CLASS_NAMES_FILE)
n_classes = len(class_names)

# Initialize YOLOv3 model
model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE, max_output_size=_MAX_OUTPUT_SIZE,
                iou_threshold=iou_threshold, confidence_threshold=confidence_threshold)

# Initialize TensorFlow placeholders and saver
inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
detections = model(inputs, training=False)
saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

# Global variables for GUI
image_path = ""
video_path = ""
detection_result = {}
csv_data = {}

# Load weights
def load_weights():
    with tf.Session() as sess:
        saver.restore(sess, 'weights/model.ckpt')

# GUI Functions
def load_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        img = Image.open(image_path)
        img = img.resize((500, 500))  # Adjust image size
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

def apply_yolov3():
    global image_path
    if not image_path:
        return
    
    batch = load_images(image_path, model_size=_MODEL_SIZE)
    
    start_time = time.time()  # Start timer
    with tf.Session() as sess:
        saver.restore(sess, 'weights/model.ckpt')
        detection_result = sess.run(detections, feed_dict={inputs: batch})
    end_time = time.time()  # End timer

    elapsed_time = end_time - start_time
    time_label.config(text=f"Detection Time: {elapsed_time:.2f} seconds")

    # Update image with detection result
    draw_boxes([image_path], detection_result, class_names, _MODEL_SIZE)
    detected_image_path = './detections/' + os.path.basename(image_path)[:-4] + '_yolo.jpg'
    img_with_boxes = Image.open(detected_image_path)
    img_with_boxes = img_with_boxes.resize((500, 500))  # Adjust image size
    img_with_boxes = ImageTk.PhotoImage(img_with_boxes)
    image_label.configure(image=img_with_boxes)
    image_label.image = img_with_boxes

def apply_video_yolov3():
    global video_path
    if not video_path:
        return

    win_name = 'Video detection'
    cv2.namedWindow(win_name)
    cap = cv2.VideoCapture(video_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not os.path.exists('detections'):
        os.mkdir('detections')
    head, tail = os.path.split(video_path)
    name = './detections/'+tail[:-4]+'_yolo.mp4'
    out = cv2.VideoWriter(name, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    with tf.Session() as sess:
        saver.restore(sess, './weights/model.ckpt')

        try:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
                detection_result = sess.run(detections, feed_dict={inputs: [resized_frame]})
                draw_frame(frame, frame_size, detection_result, class_names, _MODEL_SIZE)
                if ret:
                    cv2.imshow(win_name, frame)
                    out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            cap.release()
            out.release()

def load_video():
    global video_path
    video_path = filedialog.askopenfilename()
    if video_path:
        apply_video_yolov3()

# Function to load CSV data
def load_csv_data():
    global csv_data
    csv_path = filedialog.askopenfilename()
    if csv_path:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            csv_data.clear()  # Clear existing data
            for row in reader:
                try:
                    _class = int(row['Class'])
                    csv_data[_class] = {
                        'Accuracy': float(row['Accuracy']),
                        'Precision': float(row['Precision']),
                        'Recall': float(row['Recall']),
                        'F1 Score': float(row['F1 Score'])
                    }
                except ValueError as e:
                    print(f"Error parsing row: {row}. Error: {e}")
        update_plot(stat_type.get())

def generate_plots(stat_type):
    if not csv_data:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjusted size to be smaller and fit better

    available_classes = sorted(csv_data.keys())  # Only use available classes
    values = [csv_data[_class][stat_type] for _class in available_classes]

    ax.bar(available_classes, values, color='b')
    ax.set_title(f'{stat_type} per Class', fontsize=10)
    ax.set_xlabel('Class', fontsize=8)
    ax.set_ylabel(stat_type, fontsize=8)
    ax.set_xticks(available_classes)
    ax.set_xticklabels(available_classes, rotation=90, fontsize=6)
    plt.tight_layout()

    return fig

# Function to update plot based on slider selection
def update_plot(stat_type):
    fig = generate_plots(stat_type)
    
    # Embed the plot in the tkinter window
    for widget in plot_frame.winfo_children():
        widget.destroy()  # Clear any previous plot
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# Initialize GUI
root = Tk()
root.title("YOLOv3 Object Detection")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to fill the screen but not in full screen mode
root.geometry(f"{screen_width}x{screen_height}")

# Create a frame for the buttons and center it
button_frame = Frame(root)
button_frame.pack(pady=10, side='top')

# Create a frame for the plots on the left side
plot_frame = Frame(root)
plot_frame.pack(side='left', padx=10, pady=10)

# Create a frame for displaying the image on the right side
image_frame = Frame(root)
image_frame.pack(side='right', padx=10, pady=10)

image_label = Label(image_frame)
image_label.pack()

# Create a label for displaying detection time above the image
time_label = Label(image_frame, text="")
time_label.pack()

# Create buttons and place them in the button frame in a row
btn_load_image = Button(button_frame, text="Load Image", command=load_image)
btn_load_image.pack(side='left', padx=5)

btn_apply_yolo = Button(button_frame, text="Apply YOLOv3", command=apply_yolov3)
btn_apply_yolo.pack(side='left', padx=5)

btn_load_video = Button(button_frame, text="Apply Yolov3 on Video", command=load_video)
btn_load_video.pack(side='left', padx=5)

btn_load_csv = Button(button_frame, text="Load CSV Data", command=load_csv_data)
btn_load_csv.pack(side='left', padx=5)

# Create a dropdown menu for selecting statistics type
stat_type = StringVar(root)
stat_type.set("Select Statistic")  # default value
stat_menu = OptionMenu(button_frame, stat_type, "Accuracy", "Precision", "Recall", "F1 Score", command=lambda _: update_plot(stat_type.get()))
stat_menu.pack(side='left', padx=5)

# Load weights initially
load_weights()

# Start the GUI loop
root.mainloop()
