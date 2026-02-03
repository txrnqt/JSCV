import camera.Camera as camera
import object_pipelines.object_detctor

if __name__ == "__main__":
    detector = object_pipelines.object_detctor.object_detector()
    cam = camera.camera()

    detector.run_infrence(cam.get_frame)
