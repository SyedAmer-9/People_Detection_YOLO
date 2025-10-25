import cv2
import cvzone
import math
from ultralytics import YOLO
from sort import *
import numpy as np

video_path = 'people.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error : Could not open the video file {video_path}")
    exit()