import csv
import numpy as np
from scipy.interpolate import interp1d

def load_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        return list(reader)
    
def write_output_csv(filepath, data, header):
    with open(filepath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    
def extract_columns(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])
    return frame_numbers, car_ids, car_bboxes, license_plate_bboxes


def interpolate_for_car(car_id, frame_numbers, car_ids, car_bboxes, license_plate_bboxes, data):
    car_mask = car_ids == car_id
    car_frame_numbers = frame_numbers[car_mask]
    car_bboxes_interpolated = []
    license_plate_bboxes_interpolated = []

    first_frame_number = car_frame_numbers[0]
    last_frame_number = car_frame_numbers[-1]

    for i in range(len(car_bboxes[car_mask])):
        frame_number = car_frame_numbers[i]
        car_bbox = car_bboxes[car_mask][i]
        license_plate_bbox = license_plate_bboxes[car_mask][i]

        if i > 0:
            prev_frame_number = car_frame_numbers[i-1]
            prev_car_bbox = car_bboxes_interpolated[-1]
            prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

            if frame_number - prev_frame_number > 1:
                frames_gap = frame_number - prev_frame_number
                x = np.array([prev_frame_number, frame_number])
                x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                interpolated_car_bboxes = interp_func(x_new)
                interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                interpolated_license_plate_bboxes = interp_func(x_new)

                car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

        car_bboxes_interpolated.append(car_bbox)
        license_plate_bboxes_interpolated.append(license_plate_bbox)

    interpolated_data = []
    frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
    for i in range(len(car_bboxes_interpolated)):
        frame_number = first_frame_number + i
        row = {
            'frame_nmr': str(frame_number),
            'car_id': str(car_id),
            'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
            'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i]))
        }

        if str(frame_number) not in frame_numbers_:
            # Imputed row, set the following fields to '0'
            row.update({
                'license_plate_bbox_score': '0',
                'license_number': '0',
                'license_number_score': '0'
            })
        else:
            # Original row, retrieve values from the input data if available
            original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
            row.update({
                'license_plate_bbox_score': original_row.get('license_plate_bbox_score', '0'),
                'license_number': original_row.get('license_number', '0'),
                'license_number_score': original_row.get('license_number_score', '0')
            })

        interpolated_data.append(row)

    return interpolated_data

def interpolate_bounding_boxes(data):
    frame_numbers, car_ids, car_bboxes, license_plate_bboxes = extract_columns(data)
    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        interpolated_data.extend(interpolate_for_car(car_id, frame_numbers, car_ids, car_bboxes, license_plate_bboxes, data))
    return interpolated_data

# Main execution
def add_missing_data(input_csv, output_csv):
    data = load_csv(input_csv)
    interpolated_data = interpolate_bounding_boxes(data)
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    write_output_csv(output_csv, interpolated_data, header)