import argparse
import cv2
import os
from ultralytics import YOLO

def main(weights, source, conf=0.5, save_output=False, output_path="outputs/output.mp4"):
    # Load model
    model = YOLO(weights)

    # Open source (video/webcam)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open source {source}")
        return

    # Get input video properties
    fps_in = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer if saving
    writer = None
    if save_output:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        print(f"💾 Saving output video to: {output_path}")

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=conf)
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("TrashNet Detect", annotated_frame)

        # Save frame if enabled
        if writer is not None:
            writer.write(annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or path to video file)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--output", type=str, default="outputs/output.mp4", help="Path to save output video")
    args = parser.parse_args()

    # If source is numeric string, treat as webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    main(args.weights, source, conf=args.conf, save_output=args.save, output_path=args.output)


import imageio
import glob

files = sorted(glob.glob("frames/*.png"))
with imageio.get_writer("demo.gif", mode="I", duration=0.2) as writer:
    for f in files:
        image = imageio.imread(f)
        writer.append_data(image)
