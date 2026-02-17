import argparse

import cv2
from object_pipelines.object_detctor import ObjectDetector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--display", action="store_true", help="Show raw and bbox image windows"
)
parser.add_argument(
    "--headless", action="store_true", help="Runs without displaying images"
)
parser.add_argument(
    "--webcam", action="store_true", help="Use live webcam instead of image"
)
parser.add_argument(
    "--camera-index", type=int, default=0, help="Webcam index (default: 0)"
)
args = parser.parse_args()

if not args.display and not args.headless:
    parser.error("Must specify either --display or --headless")

detector: ObjectDetector = ObjectDetector("coreML")

if args.webcam:
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    print("Streaming webcam — press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        detector.run_inference(frame)
        print(f"Results: {detector.get_results()}")

        if args.display:
            cv2.imshow("raw_frame", frame)
            cv2.imshow("bbox_frame", detector.plot(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

else:
    frame = cv2.imread("./src/test/gw5pfdc9urnf1.jpeg", cv2.IMREAD_COLOR)
    if frame is None:
        print("Cannot open image")
        exit()

    detector.run_inference(frame)
    print(f"Results: {detector.get_results()}")
    print(f"Frame shape: {frame.shape}")
    print(detector.get_observations(0))

    if args.display:
        cv2.imshow("raw_frame", frame)
        cv2.imshow("bbox_frame", detector.plot(frame))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
