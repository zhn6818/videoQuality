#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

class AdvancedObstructionDetector {
private:
  // 关键参数配置
  struct DetectorConfig {
    float changeThreshold = 0.7;       // 变化阈值
    int persistentFrames = 15;         // 持续帧数
    int initialHistory = 500;          // 初始历史帧数
    double initialVarThreshold = 100.0; // 初始阈值
  };

  // 背景建模
  cv::Ptr<cv::BackgroundSubtractorMOG2> backgroundModel;

  // 配置参数
  DetectorConfig config;

  // 动态调整参数
  int dynamicHistory;
  double dynamicVarThreshold;

  // 预分配矩阵
  cv::Mat foregroundMask, kernel;

public:
  AdvancedObstructionDetector() {
    try {
      // 初始化背景建模
      dynamicHistory = config.initialHistory;
      dynamicVarThreshold = config.initialVarThreshold;

      backgroundModel = cv::createBackgroundSubtractorMOG2(
          dynamicHistory, dynamicVarThreshold, false);
    } catch (const cv::Exception &e) {
      std::cerr << "初始化错误: " << e.what() << std::endl;
      throw;
    }
  }

  // 核心遮挡检测方法
  bool detectObstruction(const cv::Mat &frame) {
    if (frame.empty()) {
      std::cerr << "无效帧" << std::endl;
      return false;
    }

    try {
      // 1. 图像预处理：高斯模糊去噪
      cv::Mat blurredFrame;
      cv::GaussianBlur(frame, blurredFrame, cv::Size(5, 5), 1.5);

      // 2. 背景差分
      backgroundModel->apply(blurredFrame, foregroundMask);


      return assessObstruction(foregroundMask);
    } catch (const cv::Exception &e) {
      std::cerr << "检测过程发生错误: " << e.what() << std::endl;
      return false;
    }
  }

public:


  // 遮挡评估
  bool assessObstruction(const cv::Mat &foregroundMask) {
    // 计算遮挡覆盖率
    cv::imshow("foregroud", foregroundMask);
    double maskedArea = cv::countNonZero(foregroundMask);
    double totalArea = foregroundMask.rows * foregroundMask.cols;
    double coverageRatio = maskedArea / totalArea;

    // 判断遮挡是否显著
    return coverageRatio > config.changeThreshold;
  }

  // 告警日志记录
  void logObstruction(const cv::Mat &frame, const std::string &path) {
    try {
      // 确保路径存在
      if (!fs::exists(path)) {
        fs::create_directories(path);
      }

      // 保存遮挡帧到指定目录
      std::string filename =
          path + "/obstruction_" + std::to_string(std::time(nullptr)) + ".jpg";
      cv::imwrite(filename, frame);
      std::cout << "遮挡帧已保存至: " << filename << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "保存遮挡帧失败: " << e.what() << std::endl;
    }
  }
};

int main() {
  try {
    // RTSP流地址
    std::string rtspUrl = "rtsp://admin:admin123@10.0.0.179:554/live";

    // 打开 RTSP 流
    cv::VideoCapture cap(rtspUrl);

    // 检查视频流是否成功打开
    if (!cap.isOpened()) {
      std::cerr << "无法打开RTSP流!" << std::endl;
      return -1;
    }

    // 创建遮挡检测器
    AdvancedObstructionDetector detector;

    // 设置保存路径
    std::string savePath = "./result";

    while (true) {
      cv::Mat frame;
      cap >> frame;

      if (frame.empty()) {
        std::cerr << "获取帧失败" << std::endl;
        break;
      }

      // 检测遮挡
      bool isObstructed = detector.detectObstruction(frame);

      if (isObstructed) {
        std::cout << "摄像头可能被遮挡!" << std::endl;
        detector.logObstruction(frame, savePath); // 保存遮挡帧到指定路径
      }

      // 显示帧
      cv::imshow("Camera", frame);

      // 按 'q' 退出
      if (cv::waitKey(1) == 'q')
        break;
    }
  } catch (const std::exception &e) {
    std::cerr << "发生异常: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}