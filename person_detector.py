import os
import sys
import time
import json
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

# Import simulate_realtime for live simulation streaming
try:
    import simulate_realtime
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False

LIVE_BRIDGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.live_occupancy.json')


def publish_live_bridge(occupancy, source, frame=0, total_frames=0, fps=0.0):
    """
    Publishes live occupancy to .live_occupancy.json for inter-process communication
    with simulate_realtime.py.
    """
    try:
        data = {
            'occupancy': float(occupancy),
            'source': os.path.basename(str(source)),
            'frame': int(frame),
            'total_frames': int(total_frames),
            'fps': round(float(fps), 1),
            'timestamp': time.time()
        }
        temp_file = LIVE_BRIDGE_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        os.replace(temp_file, LIVE_BRIDGE_FILE)
    except Exception:
        pass


def cleanup_live_bridge():
    """Removes the live bridge file upon exit."""
    try:
        if os.path.exists(LIVE_BRIDGE_FILE):
            os.remove(LIVE_BRIDGE_FILE)
    except Exception:
        pass


def format_duration(seconds):
    """Formats seconds into MM:SS format."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def detect_people_and_predict(
    source=0,
    video=None,
    model_path='yolov8n.pt',
    conf_threshold=0.4,
    enable_prediction=True,
    pass_to_sim=False,
    sim_interval=1.0,
    sim_scenario=None,
    sim_log=None,
    sync_fps=True,
    speed=1.0,
    loop=False,
    no_display=False,
    temp=32.0,
    humidity=50.0,
    gas=120.0,
    pressure=1013.25,
    voltage=12.0,
    current=2.5,
    output_path=None
):
    """
    Runs YOLO person detection on a camera stream or custom video file.
    Validates detection performance with live metrics and optionally streams
    the detected occupancy into simulate_realtime.py.
    """
    # Custom video path overrides source if provided
    if video is not None:
        source = video

    # Identify whether input is a custom video file or webcam
    is_video_file = False
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    elif isinstance(source, str):
        is_video_file = True
        if not os.path.isfile(source):
            print(f"\n[Error] Custom video file not found: '{source}'")
            print("Please verify the file path and ensure the video exists.\n")
            return None

    print(f"\nLoading YOLO model: {model_path}...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {source}")
        return None

    # Inspect video file parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video_file else 0
    duration_s = (total_frames / video_fps) if (is_video_file and video_fps > 0) else 0

    print("=" * 80)
    if is_video_file:
        print(" CUSTOM VIDEO DETECTION & VALIDATION MODE")
        print("=" * 80)
        print(f" Video File:      {source}")
        print(f" Resolution:      {width} x {height}")
        print(f" Native Video FPS:{video_fps:.1f} fps")
        print(f" Total Frames:    {total_frames} (~{format_duration(duration_s)})")
        print(f" FPS Sync:        {'Enabled (real-time playback at ' + str(speed) + 'x)' if sync_fps else 'Disabled (maximum benchmark speed)'}")
        print(f" Loop Video:      {'Yes' if loop else 'No'}")
    else:
        print(" LIVE WEBCAM OCCUPANCY DETECTION MODE")
        print("=" * 80)
        print(f" Camera Source:   Device {source}")
        print(f" Resolution:      {width} x {height}")

    print(f" YOLO Model:      {model_path} (Confidence: {conf_threshold})")
    print(f" Motor Predictor: {'Enabled (KNN Regression)' if enable_prediction else 'Disabled'}")
    print(f" Simulation Pass: {'Enabled (streaming to simulate_realtime)' if pass_to_sim else 'Live Bridge (.live_occupancy.json active)'}")
    print("=" * 80)

    # Video writer setup if output_path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        print(f" Recording output to: {output_path}")

    # Simulation bridge in-stream setup
    sim_step_count = 0
    last_sim_time = 0.0
    sim_current_state = None
    sim_history = []
    if pass_to_sim:
        if not SIMULATION_AVAILABLE:
            print("[Warning] simulate_realtime module could not be imported. Simulation pass disabled.")
            pass_to_sim = False
        else:
            print("\n[Direct Real-Time Simulation Stream Active]")
            print(simulate_realtime.TABLE_HEADER)

    print("\nControls:")
    print("  'q' or ESC : Stop detection & print validation report")
    print("  SPACE      : Pause / Resume playback")
    print("  'r'        : Rewind video to beginning")
    print("  'd'        : Toggle HUD display")
    print("------------------------------------------------------------\n")

    # Metrics tracking
    frames_processed = 0
    start_session_time = time.time()
    prev_frame_time = time.time()
    occupancy_history = []
    rpm_history = []
    fps_history = []
    is_paused = False
    show_hud = True

    try:
        while cap.isOpened():
            if is_paused:
                key = cv2.waitKey(30) & 0xFF
                if key == 32:  # Space to resume
                    is_paused = False
                elif key == ord('q') or key == 27:
                    break
                elif key == ord('r') and is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    is_paused = False
                continue

            frame_start = time.time()
            success, frame = cap.read()
            if not success:
                if is_video_file and loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("\nEnd of video stream reached.")
                    break

            frames_processed += 1
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Run YOLO person detection (class 0 = person)
            results = model(frame, conf=conf_threshold, classes=[0], verbose=False)

            person_count = 0
            annotated_frame = frame.copy()

            if len(results) > 0:
                boxes = results[0].boxes
                person_count = len(boxes)

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # Draw bounding box and label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {confidence:.2f}"
                    cv2.putText(
                        annotated_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

            occupancy_history.append(person_count)

            # Instantaneous FPS
            now = time.time()
            instant_fps = 1.0 / (now - prev_frame_time) if (now - prev_frame_time) > 0 else video_fps
            prev_frame_time = now
            fps_history.append(instant_fps)

            # Predict BLDC Motor RPM
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
                    rpm_history.append(predicted_rpm)
                except Exception:
                    predicted_rpm = None

            # Publish live data to IPC bridge file for simulate_realtime.py
            publish_live_bridge(
                occupancy=person_count,
                source=source,
                frame=current_frame_idx,
                total_frames=total_frames,
                fps=instant_fps
            )

            # In-process simulation step update
            if pass_to_sim and (now - last_sim_time >= sim_interval):
                last_sim_time = now
                sim_step_count += 1
                record, sim_current_state, sim_rpm, row = simulate_realtime.step_simulation(
                    step_count=sim_step_count,
                    previous_state=sim_current_state,
                    occupancy=person_count,
                    scenario=sim_scenario
                )
                print(row)
                sim_history.append(record)

            # Draw HUD Overlay
            if show_hud:
                overlay = annotated_frame.copy()
                hud_height = 195 if (predicted_rpm is not None and is_video_file) else 145
                cv2.rectangle(overlay, (10, 10), (450, hud_height), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.72, annotated_frame, 0.28, 0, annotated_frame)

                # Occupancy text
                cv2.putText(
                    annotated_frame, f"Occupancy (Persons): {person_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )
                # Processing FPS
                cv2.putText(
                    annotated_frame, f"Inference FPS: {instant_fps:.1f} (Video: {video_fps:.1f})",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
                )

                # Video progress
                line_y = 100
                if is_video_file and total_frames > 0:
                    pct = (current_frame_idx / total_frames) * 100
                    elapsed_str = format_duration(current_frame_idx / video_fps) if video_fps > 0 else "00:00"
                    total_str = format_duration(duration_s)
                    cv2.putText(
                        annotated_frame, f"Progress: {current_frame_idx}/{total_frames} ({pct:.1f}%) [{elapsed_str}/{total_str}]",
                        (20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 230, 255), 1
                    )
                    line_y += 30

                # Predicted Motor Speed
                if predicted_rpm is not None:
                    cv2.putText(
                        annotated_frame, f"Predicted Fan Speed: {predicted_rpm:.1f} RPM",
                        (20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2
                    )
                    line_y += 25

                # Stream status
                stream_text = "PASSING -> simulate_realtime" if pass_to_sim else "LIVE IPC BRIDGE ACTIVE"
                cv2.putText(
                    annotated_frame, f"Status: {stream_text}",
                    (20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1
                )

            if writer:
                writer.write(annotated_frame)

            # Display window and manage frame pacing
            if not no_display:
                cv2.imshow("Occupancy Detection & BLDC Speed Drive", annotated_frame)

                # Pacing calculation to sync with video FPS
                if is_video_file and sync_fps:
                    time_spent = time.time() - frame_start
                    target_delay = (1.0 / (video_fps * max(speed, 0.1))) - time_spent
                    wait_ms = max(1, int(target_delay * 1000))
                else:
                    wait_ms = 1

                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nStopping detection (user requested).")
                    break
                elif key == 32:  # SPACE bar: pause
                    is_paused = True
                    print("\n[Playback Paused - Press SPACE to resume, 'r' to rewind]")
                elif key == ord('r') and is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("\n[Rewinding video to frame 0]")
                elif key == ord('d'):
                    show_hud = not show_hud

    except KeyboardInterrupt:
        print("\nDetection interrupted by user (Ctrl+C).")

    # Release resources
    cap.release()
    if writer:
        writer.release()
    if not no_display:
        cv2.destroyAllWindows()

    cleanup_live_bridge()

    # Save simulation history if requested
    if sim_log and sim_history:
        import pandas as pd
        df = pd.DataFrame(sim_history)
        df.to_csv(sim_log, index=False)
        print(f"\nSimulation logs saved to: {sim_log}")

    # Performance Validation Summary
    total_elapsed = time.time() - start_session_time
    avg_fps = (frames_processed / total_elapsed) if total_elapsed > 0 else 0
    min_occ = min(occupancy_history) if occupancy_history else 0
    max_occ = max(occupancy_history) if occupancy_history else 0
    mean_occ = np.mean(occupancy_history) if occupancy_history else 0.0
    min_rpm = min(rpm_history) if rpm_history else 0.0
    max_rpm = max(rpm_history) if rpm_history else 0.0
    mean_rpm = np.mean(rpm_history) if rpm_history else 0.0

    print("\n" + "=" * 80)
    print("        VIDEO DETECTION & MOTOR DRIVE VALIDATION REPORT")
    print("=" * 80)
    print(f" Source:                  {source}")
    print(f" Total Frames Processed:  {frames_processed}" + (f" / {total_frames} ({(frames_processed/total_frames)*100:.1f}%)" if is_video_file and total_frames > 0 else ""))
    print(f" Total Elapsed Time:      {total_elapsed:.2f} s")
    print(f" Average Processing FPS:  {avg_fps:.1f} fps")
    if is_video_file:
        realtime_capable = avg_fps >= (video_fps * 0.95)
        print(f" Target Video FPS:        {video_fps:.1f} fps")
        print(f" Real-Time Capability:    {'[PASSED] Real-Time Edge Processing Capable' if realtime_capable else '[NOTICE] Inference slower than video framerate'}")
    print("-" * 80)
    print(f" Detected Occupancy:      Min: {min_occ} | Max: {max_occ} | Mean: {mean_occ:.2f} persons")
    if rpm_history:
        print(f" Predicted BLDC Fan RPM:  Min: {min_rpm:.1f} | Max: {max_rpm:.1f} | Mean: {mean_rpm:.1f} RPM")
    if pass_to_sim:
        print(f" Simulation Pass Steps:   {sim_step_count} cycles sent to simulate_realtime")
    print("=" * 80 + "\n")

    return {
        'frames_processed': frames_processed,
        'elapsed_seconds': total_elapsed,
        'avg_fps': avg_fps,
        'min_occupancy': min_occ,
        'max_occupancy': max_occ,
        'mean_occupancy': mean_occ,
        'min_rpm': min_rpm,
        'max_rpm': max_rpm,
        'mean_rpm': mean_rpm
    }


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Person Occupancy Detection & FOC BLDC Drive Predictor with Custom Video Validation & Simulation Stream"
    )
    # Video input options
    parser.add_argument('--video', '-v', type=str, default=None,
                        help="Path to custom video file (.mp4, .avi, .mkv, .mov) for validation")
    parser.add_argument('--source', default='0',
                        help="Camera index (0) or path to video file / image (default: 0)")
    parser.add_argument('--model', default='yolov8n.pt',
                        help="YOLO model checkpoint (default: yolov8n.pt)")
    parser.add_argument('--conf', type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")

    # Performance & playback controls
    parser.add_argument('--no-sync-fps', '--fast', action='store_true',
                        help="Disable playback pacing and run inference at maximum speed for benchmarking")
    parser.add_argument('--speed', type=float, default=1.0,
                        help="Playback speed multiplier for video (e.g. 1.0, 1.5, 2.0)")
    parser.add_argument('--loop', action='store_true',
                        help="Loop video playback continuously")
    parser.add_argument('--no-display', '--headless', action='store_true',
                        help="Run without displaying GUI window (headless/batch evaluation)")
    parser.add_argument('--output', type=str, default=None,
                        help="Optional output video file path to save annotated detection (.mp4)")

    # Integration with simulate_realtime.py
    parser.add_argument('--simulate', '--pass-to-sim', action='store_true', dest='pass_to_sim',
                        help="Pass detected occupancy data directly into simulate_realtime.py dashboard in console")
    parser.add_argument('--sim-interval', type=float, default=1.0,
                        help="Simulation step interval in seconds when streaming to simulate_realtime (default: 1.0)")
    parser.add_argument('--sim-scenario', type=str, default=None,
                        choices=['morning_empty', 'afternoon_crowd', 'normal_evening', 'food_court_smoke'],
                        help="Mall environment scenario preset for simulation")
    parser.add_argument('--sim-log', type=str, default=None,
                        help="Save simulation stream data driven by video occupancy to CSV (e.g. video_sim.csv)")

    # Motor prediction parameters
    parser.add_argument('--no-predict', action='store_true',
                        help="Disable motor RPM prediction overlay")
    parser.add_argument('--temp', type=float, default=32.0, help="Ambient temperature in C")
    parser.add_argument('--humidity', type=float, default=50.0, help="Ambient humidity in %%")
    parser.add_argument('--gas', type=float, default=120.0, help="Gas level in ppm")
    parser.add_argument('--pressure', type=float, default=1013.25, help="Pressure in hPa")
    parser.add_argument('--voltage', type=float, default=12.0, help="Feed-forward voltage in V")
    parser.add_argument('--current', type=float, default=2.5, help="Feed-forward current in A")

    args = parser.parse_args()

    # Prioritize --video if provided, otherwise fallback to --source
    video_source = args.video if args.video is not None else args.source

    detect_people_and_predict(
        source=video_source,
        model_path=args.model,
        conf_threshold=args.conf,
        enable_prediction=not args.no_predict,
        pass_to_sim=args.pass_to_sim,
        sim_interval=args.sim_interval,
        sim_scenario=args.sim_scenario,
        sim_log=args.sim_log,
        sync_fps=not args.no_sync_fps,
        speed=args.speed,
        loop=args.loop,
        no_display=args.no_display,
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
