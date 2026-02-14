import cv2

from coprocessor.src.object_pipelines.object_detctor import object_detector

detector: object_detector = object_detector("coreML")
cap = cv2.videoCapture(0)

if not cap.isOpened():
    print("cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("cannot capture frame")
        break

    detector.run_inference(frame)

    cv2.imshow("raw_frame", frame)
    cv2.imshow("bbox_frame", detector.plot(frame))
    print(detector.get_yaw(0))
