import ast
import cv2
import numpy as np
import pandas as pd

# Configuration
VIDEO_CONFIG = {
    'output_size': (1280, 720),
    'font_scale': 4.3,
    'font_thickness': 17,
    'border_thickness': 25,
    'line_length': 200
}

def video_init(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return cap, out

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    lines = [
        ((x1, y1), (x1, y1 + line_length_y)),
        ((x1, y1), (x1 + line_length_x, y1)),
        ((x1, y2), (x1, y2 - line_length_y)),
        ((x1, y2), (x1 + line_length_x, y2)),
        ((x2, y1), (x2 - line_length_x, y1)),
        ((x2, y1), (x2, y1 + line_length_y)),
        ((x2, y2), (x2, y2 - line_length_y)),
        ((x2, y2), (x2 - line_length_x, y2))
    ]
    
    for start, end in lines:
        cv2.line(img, start, end, color, thickness)
    return img

def parse_bbox(bbox_str):
    return ast.literal_eval(bbox_str.replace('[ ', '[')
                          .replace('   ', ' ')
                          .replace('  ', ' ')
                          .replace(' ', ','))
    
def process_license_plates(results, cap):
    license_plates = {}
    for car_id in np.unique(results['car_id']):
        car_data = results[results['car_id'] == car_id]
        max_score = np.amax(car_data['license_number_score'])
        max_score_data = car_data[car_data['license_number_score'] == max_score]
        
        frame_number = max_score_data['frame_nmr'].iloc[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            bbox = parse_bbox(max_score_data['license_plate_bbox'].iloc[0])
            x1, y1, x2, y2 = map(int, bbox)
            license_crop = frame[y1:y2, x1:x2, :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
            
            license_plates[car_id] = {
                'license_crop': license_crop,
                'license_number': max_score_data['license_number'].iloc[0]
            }
    
    return license_plates

def draw_frame_info(frame, row_data, license_plate_info):
    car_bbox = parse_bbox(row_data['car_bbox'])
    license_bbox = parse_bbox(row_data['license_plate_bbox'])
    
    # Draw car border
    draw_border(frame, 
                (int(car_bbox[0]), int(car_bbox[1])),
                (int(car_bbox[2]), int(car_bbox[3])),
                (0, 255, 0),
                VIDEO_CONFIG['border_thickness'],
                VIDEO_CONFIG['line_length'],
                VIDEO_CONFIG['line_length'])
    
    # Draw license plate
    cv2.rectangle(frame,
                 (int(license_bbox[0]), int(license_bbox[1])),
                 (int(license_bbox[2]), int(license_bbox[3])),
                 (0, 0, 255), 12)
    
    try:
        overlay_license_info(frame, car_bbox, license_plate_info)
    except:
        pass
    
    return frame

def overlay_license_info(frame, car_bbox, license_info):
    car_x1, car_y1, car_x2, _ = car_bbox
    license_crop = license_info['license_crop']
    H, W, _ = license_crop.shape
    
    # Overlay license plate image
    frame[int(car_y1) - H - 100:int(car_y1) - 100,
          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
    
    # Add white background
    frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)
    
    # Add text
    text = license_info['license_number']
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        VIDEO_CONFIG['font_scale'],
        VIDEO_CONFIG['font_thickness']
    )
    
    cv2.putText(frame, text,
                (int((car_x2 + car_x1 - text_width) / 2),
                 int(car_y1 - H - 250 + (text_height / 2))),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIDEO_CONFIG['font_scale'],
                (0, 0, 0),
                VIDEO_CONFIG['font_thickness'])


def process_video(input_path, output_path, results_path):
    # Initialize video
    cap, out  = video_init(input_path, output_path)
        
    # Load and process data
    results = pd.read_csv(results_path)
    license_plates = process_license_plates(results, cap)
    
    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_nmr = -1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_nmr += 1
        df_ = results[results['frame_nmr'] == frame_nmr]
        
        for _, row in df_.iterrows():
            frame = draw_frame_info(frame, row, license_plates[row['car_id']])
        
        out.write(frame)
        frame = cv2.resize(frame, VIDEO_CONFIG['output_size'])
    
    out.release()
    cap.release()
    

def vizualize(input_video_path, output_video_path, output_csv_path):
    process_video(input_video_path, output_video_path, output_csv_path)
