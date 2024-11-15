#include <iostream>
#include <opencv2/opencv.hpp>

// 计算两帧之间的差异
double frameDifference(const cv::Mat &frame1, const cv::Mat &frame2) {
  cv::Mat gray1, gray2;
  cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

  cv::Mat diff;
  cv::absdiff(gray1, gray2, diff);
  cv::Scalar sumDiff = cv::sum(diff);

  return sumDiff[0] / (diff.rows * diff.cols);
}

// 检测视频中的画面冻结
bool detectFreezeFrame(const std::string &videoPath, double threshold,
                       int freezeDuration) {
  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video." << std::endl;
    return false;
  }

  cv::Mat prevFrame, frame;
  int freezeCount = 0;

  if (cap.read(prevFrame)) { // 读取第一帧
    while (cap.read(frame)) {
      double diff = frameDifference(prevFrame, frame);

      if (diff < threshold) { // 小于阈值，认为画面无变化
        freezeCount++;
      } else {
        freezeCount = 0;
      }

      if (freezeCount >= freezeDuration) { // 连续几帧无变化，认为发生画面冻结
        return true;
      }

      prevFrame = frame.clone();
    }
  }

  return false;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <VideoPath>" << std::endl;
    return -1;
  }

  bool isFrozen =
      detectFreezeFrame(argv[1], 10.0, 30); // 阈值和连续帧数根据实际情况调整
  if (isFrozen) {
    std::cout << "Freeze frame detected." << std::endl;
  } else {
    std::cout << "No freeze frame detected." << std::endl;
  }

  return 0;
}