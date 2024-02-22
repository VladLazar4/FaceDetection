from collections import OrderedDict
import numpy as np
import cv2
import dlib

facial_features_cordinates = {}

FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Mouth", (48, 68))
])

predictor_name = "shape_predictor_68_face_landmarks.dat"
def shape_to_numpy_array(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates

def visualize_facial_landmarks(image, shape, facial_details, alpha=0.75):
    overlay = image.copy()
    output = image.copy()

    colors = [(211, 137, 34), (211, 137, 34), (0, 200, 200), (70, 100, 255)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        if facial_details[i]==True:
            (j, k) = FACIAL_LANDMARKS_INDEXES[name]
            pts = shape[j:k]
            facial_features_cordinates[name] = pts

            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def detect_image(image, eyes, nose, mouth):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    output=[]
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)

        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output.append(visualize_facial_landmarks(image, shape, [eyes, eyes, nose, mouth]))

    return output