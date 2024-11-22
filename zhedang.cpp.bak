#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace std;
using namespace cv;

class AdvancedObstructionDetector {
private:
  // 关键参数配置
  struct DetectorConfig {
    float changeThreshold = 0.3;    // 变化阈值
    int persistentFrames = 15;      // 持续帧数
    float iouThreshold = 0.5;       // 区域重叠阈值
    bool enableDeepLearning = true; // 启用深度学习
  };

  // 背景建模
  cv::Ptr<cv::BackgroundSubtractorMOG2> backgroundModel;

  // 深度学习目标检测模型
  cv::dnn::Net objectDetectionModel;

  // 配置参数
  DetectorConfig config;

  // 遮挡区域追踪
  std::vector<cv::Rect> trackedObstructions;

  // 预分配矩阵
  cv::Mat blob, foregroundMask, kernel;

public:
  AdvancedObstructionDetector() {
    try {
      // 初始化背景建模
      backgroundModel = cv::createBackgroundSubtractorMOG2(500, 16, true);

      // 加载预训练目标检测模型
      objectDetectionModel =
          cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");

      // 使用默认后端和目标
      objectDetectionModel.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
      objectDetectionModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
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
      // 1. 背景差分
      backgroundModel->apply(frame, foregroundMask);

      // 2. 形态学处理
      kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
      cv::morphologyEx(foregroundMask, foregroundMask, cv::MORPH_CLOSE, kernel);

      // 3. 深度学习目标检测
      blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(416, 416),
                                    cv::Scalar(0, 0, 0), true, false);
      objectDetectionModel.setInput(blob);

      // 检测潜在遮挡目标
      std::vector<cv::Mat> detections;
      std::vector<cv::String> outNames =
          objectDetectionModel.getUnconnectedOutLayersNames();

      objectDetectionModel.forward(detections, outNames);

      // 检查detections是否为空
      if (detections.empty()) {
        std::cerr << "未检测到任何目标" << std::endl;
        return false;
      }

      // 4. 区域分析
      std::vector<cv::Rect> obstructionAreas =
          analyzeDetections(detections, frame);

      // 5. 遮挡判断
      return assessObstruction(obstructionAreas, foregroundMask);
    } catch (const cv::Exception &e) {
      std::cerr << "检测过程发生错误: " << e.what() << std::endl;
      return false;
    }
  }

private:
  // 检测结果分析
  std::vector<cv::Rect>
  analyzeDetections(const std::vector<cv::Mat> &detections,
                    const cv::Mat &frame) {
    std::vector<cv::Rect> potentialObstructions;

    for (const auto &detection : detections) {
      // 检查检测结果是否为空或格式不符合
      if (detection.empty() || detection.dims != 4) {
        std::cerr << "无效的检测结果" << std::endl;
        continue;
      }

      for (int i = 0; i < detection.size[2]; ++i) {
        try {
          const float *data = detection.ptr<float>(0, 0, i);
          float confidence = data[2];
          int classId = static_cast<int>(data[1]);

          if (confidence > 0.5 && isObstructionClass(classId)) {
            float x = data[3] * frame.cols;
            float y = data[4] * frame.rows;
            float width = data[5] * frame.cols;
            float height = data[6] * frame.rows;

            potentialObstructions.push_back(cv::Rect(x, y, width, height));
          }
        } catch (const cv::Exception &e) {
          std::cerr << "处理检测结果时发生错误: " << e.what() << std::endl;
        }
      }
    }

    return potentialObstructions;
  }

  // 判断是否为遮挡类别
  bool isObstructionClass(int classId) {
    // 定义可能导致遮挡的对象类别
    std::vector<int> obstructionClasses = {
        0,  // 人
        24, // 手
        73  // 其他遮挡物
    };

    return std::find(obstructionClasses.begin(), obstructionClasses.end(),
                     classId) != obstructionClasses.end();
  }

  // 遮挡评估
  bool assessObstruction(const std::vector<cv::Rect> &obstructionAreas,
                         const cv::Mat &foregroundMask) {
    // 计算遮挡覆盖率
    double maskedArea = cv::countNonZero(foregroundMask);
    double totalArea = foregroundMask.rows * foregroundMask.cols;
    double coverageRatio = maskedArea / totalArea;

    // 区域重叠判断
    bool hasSignificantObstruction = coverageRatio > config.changeThreshold;

    // 额外的遮挡区域判断
    if (!obstructionAreas.empty()) {
      hasSignificantObstruction = true;
    }

    return hasSignificantObstruction;
  }

  // 告警日志记录
  void logObstruction(const cv::Mat &frame) {
    // 记录遮挡时的帧
    std::string filename =
        "obstruction_" + std::to_string(std::time(nullptr)) + ".jpg";
    cv::imwrite(filename, frame);
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
        // 可以在这里触发告警逻辑
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