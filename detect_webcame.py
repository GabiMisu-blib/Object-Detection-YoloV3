import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import cv2
from yolo_v3 import Yolo_v3
from utils_v3 import load_class_names, draw_boxes, draw_frame

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = 'coco.names'
_MAX_OUTPUT_SIZE = 50

detection_result = {}

def main(iou_threshold, confidence_threshold):
    global detection_result
    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

    inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
    detections = model(inputs, training=False)
    saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

    with tf.Session() as sess:
        saver.restore(sess, './weights/model.ckpt')

        win_name = 'Webcam Detection'
        cv2.namedWindow(win_name)
        cap = cv2.VideoCapture(0)  # Accesăm camera web
        if not cap.isOpened():
            print("Error: Camera not accessible.")
            return

        try:
            print("Starting webcam...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to capture video.")
                    break

                resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
                detection_result = sess.run(detections, feed_dict={inputs: [resized_frame]})
                draw_frame(frame, (frame.shape[1], frame.shape[0]), detection_result, class_names, _MODEL_SIZE)

                cv2.imshow(win_name, frame)

                # Apăsați 'q' pentru a ieși
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            cap.release()
            print('Webcam detection stopped.')

if __name__ == '__main__':
    main(0.5, 0.5)
