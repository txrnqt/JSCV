import camera.Camera as camera
import object_pipelines.object_detctor
from config.source_config import LocalSourceConfig, NTConfigSource

if __name__ == "__main__":
    LocalSourceConfig.update_camera_constants
    LocalSourceConfig.update_server_config
    NTConfigSource.update
    detector = object_pipelines.object_detctor.object_detector()
    cam = camera.camera()

    detector.run_infrence(cam.get_frame)
