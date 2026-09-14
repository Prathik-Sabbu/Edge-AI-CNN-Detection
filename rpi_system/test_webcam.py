"""
Interactive live webcam test for Edge AI Animal Detection.
Displays real-time video feed with object-detection style bounding box and prediction badge.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

# Add rpi_system to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config

def run_webcam_test():
    print(f"Loading TFLite model from: {config.OUTPUT_TFLITE}")
    interpreter = tf.lite.Interpreter(model_path=str(config.OUTPUT_TFLITE))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(config.OUTPUT_LABELS, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("========================================================")
    print(" Live Webcam Started!")
    print(" Hold up animal photos inside the center bounding box.")
    print(" Press 'q' on the camera window to stop.")
    print("========================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 1. Crop Center Region of Interest (ROI) to avoid 16:9 distortion
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        crop_size = int(min_dim * getattr(config, "ROI_CROP_SCALE", 1.0))
        x1 = (w - crop_size) // 2
        y1 = (h - crop_size) // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        roi = frame[y1:y2, x1:x2]

        # 2. Preprocess ROI for MobileNetV3
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32)

        # 3. Forward pass inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        # 4. Extract top prediction
        top_idx = int(np.argmax(output))
        pred_label = labels[top_idx]
        confidence = float(output[top_idx])

        # 5. Draw Object-Detection Style Bounding Box around ROI
        box_color = (0, 230, 0) if confidence >= config.CONFIDENCE_THRESHOLD else (0, 140, 255)
        line_thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, line_thickness)

        # 6. Draw Attached Label Badge on top of bounding box
        label_text = f"{pred_label} {confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        badge_y1 = max(0, y1 - text_h - 12)
        badge_y2 = badge_y1 + text_h + 12
        badge_x2 = min(w, x1 + text_w + 16)

        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
        cv2.putText(
            frame,
            label_text,
            (x1 + 8, badge_y2 - 8),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

        # 7. Helper banner & CNN Receptive Field Inset (PiP)
        # Shows user the exact patch the CNN sees so they can align phone/photo properly
        pip_size = 140
        pip_img = cv2.resize(roi, (pip_size, pip_size))
        frame[15 : 15 + pip_size, w - pip_size - 15 : w - 15] = pip_img
        cv2.rectangle(frame, (w - pip_size - 15, 15), (w - 15, 15 + pip_size), (0, 255, 255), 2)
        cv2.putText(frame, "CNN Input Feed", (w - pip_size - 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Live probabilities panel on left
        for idx, (lbl, prob) in enumerate(zip(labels, output)):
            txt = f"{lbl}: {prob * 100:.1f}%"
            c = (0, 255, 0) if idx == top_idx else (200, 200, 200)
            cv2.putText(frame, txt, (15, h - 130 + (idx * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, txt, (15, h - 130 + (idx * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

        cv2.putText(
            frame,
            "Fill the box with the animal photo",
            (x1 + 10, y2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Edge AI Animal Detection (Press 'q' to Quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    run_webcam_test()
