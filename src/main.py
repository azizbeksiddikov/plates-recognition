from ultralytics import YOLO
import cv2
import numpy as np
from utils import *
from sort.sort import *
from utils.file_utils import validate_paths, write_csv
from utils.detection_utils import detect_vehicles, get_car
from utils.license_plate_utils import process_plate_image
from utils.add_missing_data import add_missing_data
from utils.visualize import vizualize
import os

def process_license_plates(frame, frame_nmr, track_ids, detector, results):

    license_plates = detector(frame, verbose=False)[0]
    for plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = plate
        x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)
        if car_id == -1:
            continue
        
        plate_text, text_score = process_plate_image(frame[y1:y2, x1:x2])
        # plate_text, text_score = process_plate_image(frame[y1:y2, x1:x2].copy())
        
        if plate_text:
            results[frame_nmr][car_id] = {
                "car": {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                "license_plate": {
                    "bbox": [x1, y1, x2, y2],
                    "text": plate_text,
                    "bbox_score": score,
                    "text_score": text_score
                }
            }
    return results
                
def process_video(input_video_path, coco_model, mot_tracker, license_plate_detector):
    results = {}

    # Load video
    cap = cv2.VideoCapture(input_video_path)
    
    # Read frames
    frame_nmr = 0
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        results[frame_nmr] = {}
        detections = detect_vehicles(frame, coco_model)
        track_ids = mot_tracker.update(np.array(detections))
        results = process_license_plates(frame, frame_nmr, track_ids, license_plate_detector, results)
            
            
    cap.release()
    return results

if __name__ == "__main__":
    print("*" * 50)
    print("Starting...")
    # Set paths 
    input_video_path = os.path.abspath("data/videos/video_1.mp4")
    output_video_path = os.path.abspath("data/videos/video_1_output.mp4")
    initial_csv = os.path.abspath("data/output.csv")
    optimized_csv = os.path.abspath("data/output_optimized.csv")
    lpd_model_path = os.path.abspath("src/model/license_plate_detector.pt")

    validate_paths([input_video_path, lpd_model_path])
    
    # Load models
    mot_tracker = Sort()
    coco_model = YOLO("yolo11n.pt", verbose=False)
    license_plate_detector = YOLO(lpd_model_path, verbose=False)

    print("*" * 50)
    print("Processing video...")
    results = process_video(input_video_path, coco_model, mot_tracker, license_plate_detector)
    
    # Write results
    print("*" * 50)
    print("Writing results...")
    os.makedirs(os.path.dirname(initial_csv), exist_ok=True)
    os.makedirs(os.path.dirname(optimized_csv), exist_ok=True)

    write_csv(results, initial_csv)
    add_missing_data(initial_csv, optimized_csv)
    
    # Save and display results
    print("*" * 50)
    print("Visualizing results...")
    vizualize(input_video_path, output_video_path, optimized_csv)
    
    print("*" * 50)
    print("Done!")