import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "./models/face_landmarker.task"

cam = cv2.VideoCapture(3)

left_eye_ids = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    133,
]

right_eye_ids = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    362,
]

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
)

MATRIX_WIDTH = 32
MATRIX_HEIGHT = 32


class Detector:
    def __init__(self):
        self.cam = cv2.VideoCapture(3)
        self.landmarker = FaceLandmarker.create_from_options(options)

    def stretch_eye_landmarks(self, eye_landmarks):
        # TODO: implement this function
        return eye_matrix

    def detect(self, eye_type):
        _, img = self.cam.read()

        timestamp = cv2.getTickCount()
        # resize the image
        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        # flip the image
        img = cv2.flip(img, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = self.landmarker.detect(mp_image)

        # get eye connections
        eye_connection = []
        if eye_type == "left":
            eye_connection = left_eye_ids
        else:
            eye_connection = right_eye_ids

        eye_matrix = np.zeros((MATRIX_WIDTH, MATRIX_HEIGHT))

        eye_landmarks = []
        for id in eye_connection:
            if len(detection_result.face_landmarks) == 0:
                continue
            landmark = detection_result.face_landmarks[0][id]
            x = int(landmark.x * MATRIX_WIDTH)
            y = int(landmark.y * MATRIX_HEIGHT)
            eye_landmarks.append((x, y))

        # stretch out the eye landmarks to fill the matrix
        eye_matrix = self.stretch_eye_landmarks(eye_landmarks)

        if len(eye_landmarks) == 0:
            return eye_matrix

        # center the eye landmarks

        # calculate the centroid of the eye landmarks
        centroid_x = sum(x for x, y in eye_landmarks) / len(eye_landmarks)
        centroid_y = sum(y for x, y in eye_landmarks) / len(eye_landmarks)

        # calculate the translation needed to move the centroid to the center of the frame
        translation_x = MATRIX_WIDTH // 2 - centroid_x
        translation_y = MATRIX_HEIGHT // 2 - centroid_y

        # apply the translation to each landmark to center them
        centered_eye_landmarks = [
            (x + translation_x, y + translation_y) for x, y in eye_landmarks
        ]

        # create a matrix to draw the centered eye landmarks
        eye_matrix = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH), dtype=np.uint8)

        # convert the centered landmarks to a contour array
        centered_eye_contour = np.array(centered_eye_landmarks, dtype=np.int32).reshape(
            (-1, 1, 2)
        )

        # draw the centered landmarks on the matrix
        cv2.fillPoly(eye_matrix, [centered_eye_contour], 1)

        return eye_matrix
