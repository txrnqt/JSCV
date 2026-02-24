import cv2
from fiducial_pipelines import detect_fiducial
import config.source_config 

detector = detect_fiducial.fiducial_detector("cpu")
frame = cv2.imread("./src/test/gw5pfdc9urnf1.jpeg", cv2.IMREAD_COLOR)
if frame is None:
    print("Cannot open image")
    exit()

source_config.update_camera_constants_test()
detector.find_fiducial(frame)
multi_res = detector.get_multi_tag_result()
single_res = detector.get_single_tag_resutl()

print(multi_res)
print(single_res)
cv2.imshow("raw", frame)
cv2.imshow("bbox_frame", detector.plot_result(frame))
cv2.waitKey(0)
cv2.destroyAllWindows()
