#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0); // 0 = default camera

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera!" << std::endl;
        return -1;
    }

    std::cout << "Camera opened. Press 'q' to quit." << std::endl;

    cv::Mat frame;
    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Empty frame!" << std::endl;
            break;
        }

        cv::imshow("Camera", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
