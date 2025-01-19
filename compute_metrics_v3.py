import os
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from yolo_v3 import Yolo_v3
from utils_v3 import load_images, load_class_names, draw_boxes
import numpy as np
import cv2
import csv

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = 'coco.names'
_MAX_OUTPUT_SIZE = 50
iou_threshold = 0.5
confidence_threshold = 0.5

def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

class_names = load_class_names(_CLASS_NAMES_FILE)
n_classes = len(class_names)

model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE, max_output_size=_MAX_OUTPUT_SIZE,
                iou_threshold=iou_threshold, confidence_threshold=confidence_threshold)

inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
detections = model(inputs, training=False)
saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
with open('instances_train2017.json', 'r') as f:
    info = json.load(f)

def load_images(img_name, model_size):
    imgs = []
    
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, model_size)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img[:, :, :3], axis=0)
    imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs

def apply_yolov3(image_path: str):
    if not image_path:
        print("Please load an image first.")
        return
    
    batch = load_images(image_path, model_size=_MODEL_SIZE)
    
    with tf.Session() as sess:
        saver.restore(sess, 'weights/model.ckpt')
        detection_result = sess.run(detections, feed_dict={inputs: batch})
        aux = detection_result[0]

        detected_categories = set()
        for category_index, arr in aux.items():
            if arr.size != 0:
                detected_categories.add(category_index+1)

        return detected_categories


output_labels = []
true_output_labels = []
folder_path = 'train2017'

files = os.listdir(folder_path)
for i in range(int(1 * len(files))):
    print(i)
    filename = files[i]
    detected_categories = apply_yolov3(os.path.join(folder_path, filename))
    output_labels.append(detected_categories)


    aux = os.path.splitext(filename)[0]
    file_id =int(aux)

    true_categories = set()
    for annotaion in info['annotations']:

        if annotaion['image_id'] == file_id:
            true_categories.add(annotaion['category_id'])
    true_output_labels.append(true_categories)
    print(i)

tp = {_class: 0 for _class in range(n_classes)}
tn = {_class: 0 for _class in range(n_classes)}
fp = {_class: 0 for _class in range(n_classes)}
fn = {_class: 0 for _class in range(n_classes)}

for lable, true_lable in zip(output_labels, true_output_labels):
    print(lable)
    print(true_lable)
    print()
    for _class in range(n_classes):
        if _class in lable and _class in true_lable:
            tp[_class] += 1
        if _class not in lable and _class in true_lable:
            fn[_class] += 1
        if _class in lable and _class not in true_lable:
            fp[_class] += 1
        if _class not in lable and _class not in true_lable:
            tn[_class] += 1

accuracy = {}
precision = {}
recall = {}
f1 = {}

for _class in range(n_classes):
    accuracy[_class] = tp[_class] / (tp[_class] + tn[_class] + fp[_class] + fn[_class]) if tp[_class] + tn[_class] + fp[_class] + fn[_class] != 0 else 0
    precision[_class] = tp[_class] / (tp[_class] + fp[_class]) if tp[_class] + fp[_class] != 0 else 0
    recall[_class] = tp[_class] / (tp[_class] + fn[_class]) if tp[_class] + fn[_class] != 0 else 0
    f1[_class] = 2 * precision[_class] * recall[_class] / (precision[_class] + recall[_class]) if precision[_class] + recall[_class] != 0 else 0

# Save the metrics into a CSV file
with open('yolo_metrics.csv', 'w', newline='') as csvfile:
    fieldnames = ['Class', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for _class in range(n_classes):
        writer.writerow({
            'Class': _class,
            'Accuracy': accuracy[_class],
            'Precision': precision[_class],
            'Recall': recall[_class],
            'F1 Score': f1[_class]
        })

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Accuracy
classes = list(range(n_classes))
axs[0, 0].bar(classes, list(map(lambda x: x[1], accuracy.items())), color='b')
axs[0, 0].set_title('Accuracy per Class')
axs[0, 0].set_xlabel('Class')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_xticks(classes)
axs[0, 0].set_xticklabels(classes, rotation=90, fontsize=6)

# Precision
axs[0, 1].bar(classes, list(map(lambda x: x[1], precision.items())), color='g')
axs[0, 1].set_title('Precision per Class')
axs[0, 1].set_xlabel('Class')
axs[0, 1].set_ylabel('Precision')
axs[0, 1].set_xticks(classes)
axs[0, 1].set_xticklabels(classes, rotation=90, fontsize=6)

# Recall
axs[1, 0].bar(classes, list(map(lambda x: x[1], recall.items())), color='r')
axs[1, 0].set_title('Recall per Class')
axs[1, 0].set_xlabel('Class')
axs[1, 0].set_ylabel('Recall')
axs[1, 0].set_xticks(classes)
axs[1, 0].set_xticklabels(classes, rotation=90, fontsize=6)

# F1 Score
axs[1, 1].bar(classes, list(map(lambda x: x[1], f1.items())), color='purple')
axs[1, 1].set_title('F1 Score per Class')
axs[1, 1].set_xlabel('Class')
axs[1, 1].set_ylabel('F1 Score')
axs[1, 1].set_xticks(classes)
axs[1, 1].set_xticklabels(classes, rotation=90, fontsize=6)

plt.tight_layout()
plt.savefig('yolo_metrics_plots.png')
plt.show()
