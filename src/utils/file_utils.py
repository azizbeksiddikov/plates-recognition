import os

# Check whether folders/files exist
def validate_paths(filepaths):
    for filepath in filepaths:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input video not found: {filepath}")

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')

        for frame_nmr, frame_data in results.items():
            for car_id, car_data in frame_data.items():
                if 'car' in car_data and 'license_plate' in car_data and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write(f"{frame_nmr},{car_id},"
                            f"[{car_data['car']['bbox'][0]} {car_data['car']['bbox'][1]} {car_data['car']['bbox'][2]} {car_data['car']['bbox'][3]}],"
                            f"[{car_data['license_plate']['bbox'][0]} {car_data['license_plate']['bbox'][1]} {car_data['license_plate']['bbox'][2]} {car_data['license_plate']['bbox'][3]}],"
                            f"{car_data['license_plate']['bbox_score']},"
                            f"{car_data['license_plate']['text']},"
                            f"{car_data['license_plate']['text_score']}\n")
