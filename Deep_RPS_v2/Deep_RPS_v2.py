import cv2
import tensorflow as tf
import time

"""
constants
"""
NUM_CAM = 1


cap = cv2.VideoCapture(NUM_CAM)

# Build network model
