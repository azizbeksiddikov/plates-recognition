import string
import easyocr
import cv2

# Initialize easyocr reader
reader = easyocr.Reader(['en'])

# Mapping dictionaries for character conversion
dict_char_to_int = {
    "O":"0",
    "I":"1",
    "J":"3",
    "A":"4",
    "G":"6",
    "S":"5",
}

dict_int_to_char = {
    "0":"O",
    "1":"I",
    "3":"J",
    "4":"A",
    "6":"G",
    "5":"S",
}


def license_complies_format(text):
    """
    Check if the license plate text complies with the format.

    Args:
        text (str): License plate text.
    
    Returns:
        bool: True if the license plate text complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    return all([
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char),
        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char),
        (text[2] in string.digits or text[2] in dict_char_to_int),
        (text[3] in string.digits or text[3] in dict_char_to_int),
        (text[4] in string.ascii_uppercase or text[4] in dict_char_to_int),
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char),
        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char)
    ])

def format_license(text):
    """
    Format the license plate text.

    Args:
        text (String): License plate text.
    
    Returns:
        String: Formatted license plate text.
    """ 
    formatted = []
    
    for idx in range(len(text)):
        char = text[idx]
        if idx in [2, 3]:
            if char in dict_char_to_int:
                char = dict_char_to_int[char]
        else:
            if char in dict_int_to_char:
                char = dict_int_to_char[char]
        formatted.append(char)

    return ''.join(formatted)

      
      


def read_license_plate(licence_plate_crop):
    """
    Read the license plate text from the cropped image.
    
    Args:
        licence_plate_crop (PIL.Image.Image): Cropped image of the license plate.
    Returns:
        Tuple: tuple containing the formatted license plate and the confidence score.
    """
    detections = reader.readtext(licence_plate_crop)
    for detection in detections:
        _, text, text_score = detection
        text = text.upper().replace(" ", "")
        
        if license_complies_format(text):
            return (format_license(text), text_score)
    
    return None, None


def process_plate_image(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)[1]
    return read_license_plate(thresh)
