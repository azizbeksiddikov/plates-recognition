# import cv2
# import numpy as np

valid_vehicles = [2, 3, 5, 7]


def get_car(licence_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        licence_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
        
    Return:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, _, _ = licence_plate
    
    for xcar1, ycar1, xcar2, ycar2, track_id in vehicle_track_ids:
        if x1 >= xcar1 and y1 >= ycar1 and x2 <= xcar2 and y2 <= ycar2:
            return xcar1, ycar1, xcar2, ycar2, track_id
    return -1, -1, -1, -1, -1

def detect_vehicles(frame, model):
    detections = []
    coco_detections = model(frame, verbose=False)[0]
    for detection in coco_detections.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = detection
        x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
        if class_id in valid_vehicles:
            detections.append([x1, y1, x2, y2, confidence])
    return detections



coco_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
         5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
         10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
         15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
         20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
         25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
         30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
         35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
         40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
         45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
         50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
         55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
         60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
         65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
         70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
         75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}