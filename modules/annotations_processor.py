import cv2
import numpy as np
import easyocr

# Initialize reader globally to save VRAM/time
reader = easyocr.Reader(['en'], gpu=True)

def process_annotated_sketch(img):
    """
    Main entry point for the pipeline.
    Takes a numpy image (from Gradio) and returns a dictionary of found text.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. DEFINE HSV COLOR RANGES
    # --- RED ---
    lower_red1, upper_red1 = np.array([0, 120, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 100]), np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # --- BLUE ---
    lower_blue, upper_blue = np.array([100, 120, 100]), np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --- BLACK ---
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 90])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 2. INTERACTION CLEANUP
    mask_black = cv2.bitwise_and(mask_black, cv2.bitwise_not(mask_blue))
    mask_black = cv2.bitwise_and(mask_black, cv2.bitwise_not(mask_red))

    kernel = np.ones((3,3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

    # 3. PREPARE OCR-FRIENDLY IMAGES
    def prepare_canvas(mask, original_img):
        canvas = np.full(original_img.shape, 255, dtype=np.uint8)
        canvas[mask > 0] = (0, 0, 0)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    red_for_ocr = prepare_canvas(mask_red, img)
    blue_for_ocr = prepare_canvas(mask_blue, img)
    black_for_ocr = prepare_canvas(mask_black, img)

    # 4. EXECUTE OCR
    red_results = reader.readtext(red_for_ocr)
    blue_results = reader.readtext(blue_for_ocr)
    black_results = reader.readtext(black_for_ocr)

    # 5. CATEGORIZE
    CONFIDENCE_THRESHOLD = 0.3
    
    def get_combined_text(results_raw):
        combined_words = []
        for (bbox, text, prob) in results_raw:
            if prob > CONFIDENCE_THRESHOLD:
                combined_words.append(text)
        return " ".join(combined_words)

    # Mapping to the keys expected by your main.py logic
    extracted_data = {
        "Red": get_combined_text(red_results),
        "Blue": get_combined_text(blue_results),
        "Black": get_combined_text(black_results)
    }

    return extracted_data