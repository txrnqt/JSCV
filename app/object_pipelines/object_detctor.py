import backends_yolo.detector_cpu as cpu
import backends_yolo.detector_cuda as cuda


class object_detector:
    def __init__(
        self,
        backend: str = "cpu",
        device: int = 0,
        model: str = "app/models/yolo26n.pt",
    ) -> None:
        match backend:
            case "cuda":
                self.detector = cuda.DetectorCUDA(model, device)
                pass
            case "coreML":
                pass
            case "hailo":
                pass
            case "rockchip":
                pass
            case "cpu":
                self.detector = cpu.DetectorCPU(model)
                pass
            case _:
                self.detector = cpu.DetectorCPU()
                pass

    def run_infrence(self, frame):
        self.results = self.detector.run_inference(frame)

    def get_results(self):
        if self.results is None:
            return []
        detections = []
        for results in self.results:
            for box in results.boxes:
                detections.append(
                    {
                        "class": int(box.cls[0]),
                        "confidance": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )
        return detections
