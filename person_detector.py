import os
import sys
import time
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' is not installed.")
    print("Please install it using: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' is not installed.")
    print("Please install it using: pip install ultralytics")
    sys.exit(1)

# Import speed prediction function from predict.py
try:
    from predict import predict_speed
    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False


def detect_people_and_predict(
    source=0,
    model_path='yolov8n.pt',
    conf_threshold=0.4,
    enable_prediction=True,
    temp=32.0,
    humidity=50.0,
    gas=120.0,
    pressure=1013.25,
    voltage=12.0,
    current=2.5,
    output_path=None
):
    """
    Runs YOLO person detection on a camera stream or video file,
    counts the number of people (occupancy), and optionally predicts
    BLDC motor RPM in real-time using KNN.
    """
    print(f"Loading YOLO model: {model_path}...")
    model = YOLO(model_path)

    # Convert integer source string (like '0') to int for webcam
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {source}")
        return

    # Video writer setup if output_path is provided
    writer = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("\nStarting video stream...")
    print("Press 'q' or 'ESC' on the video window to stop.")

    prev_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Stream ended or frame could not be read.")
            break

        # Run YOLO inference specifically filtering for class 0 ('person' in COCO dataset)
        results = model(frame, conf=conf_threshold, classes=[0], verbose=False)
        
        person_count = 0
        annotated_frame = frame.copy()

        if len(results) > 0:
            boxes = results[0].boxes
            person_count = len(boxes)

            # Draw bounding boxes and labels
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {confidence:.2f}"
                cv2.putText(
                    annotated_frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Predict BLDC Motor RPM based on occupancy
        predicted_rpm = None
        if enable_prediction and PREDICTION_AVAILABLE:
            try:
                features = {
                    'Temperature_C': temp,
                    'Humidity_percent': humidity,
                    'Gas_ppm': gas,
                    'Pressure_hPa': pressure,
                    'Occupancy': float(person_count),
                    'Voltage_V': voltage,
                    'Current_A': current
                }
                predicted_rpm = predict_speed(features)
            except Exception as e:
                predicted_rpm = None

        # Draw HUD / Dashboard Overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (430, 160 if predicted_rpm is not None else 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        # Dashboard text
        cv2.putText(
            annotated_frame, f"Occupancy (Persons): {person_count}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.putText(
            annotated_frame, f"FPS: {fps:.1f}",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        if predicted_rpm is not None:
            cv2.putText(
                annotated_frame, f"Predicted Fan Speed:",
                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1
            )
            cv2.putText(
                annotated_frame, f"{predicted_rpm:.1f} RPM",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2
            )

        if writer:
            writer.write(annotated_frame)

        # Show frame
        cv2.imshow("Occupancy Detection & BLDC Speed Drive", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Detection session ended.")


def main():
    parser = argparse.ArgumentParser(description="YOLO Person Occupancy Detection & BLDC Speed Predictor")
    parser.add_argument('--source', default='0', help="Camera index (0) or path to video file / image")
    parser.add_argument('--model', default='yolov8n.pt', help="YOLO model checkpoint (default: yolov8n.pt)")
    parser.add_argument('--conf', type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument('--no-predict', action='store_true', help="Disable motor RPM prediction overlay")
    parser.add_argument('--temp', type=float, default=32.0, help="Ambient temperature in C")
    parser.add_argument('--humidity', type=float, default=50.0, help="Ambient humidity in %")
    parser.add_argument('--gas', type=float, default=120.0, help="Gas level in ppm")
    parser.add_argument('--pressure', type=float, default=1013.25, help="Pressure in hPa")
    parser.add_argument('--voltage', type=float, default=12.0, help="Feed-forward voltage in V")
    parser.add_argument('--current', type=float, default=2.5, help="Feed-forward current in A")
    parser.add_argument('--output', type=str, default=None, help="Optional output video file path (.mp4)")

    args = parser.parse_args()

    detect_people_and_predict(
        source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
        enable_prediction=not args.no_predict,
        temp=args.temp,
        humidity=args.humidity,
        gas=args.gas,
        pressure=args.pressure,
        voltage=args.voltage,
        current=args.current,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
