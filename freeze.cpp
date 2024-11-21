#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 冻结检测类
class FreezeDetector {
private:
  deque<Mat> frameDeque;        // 缓存帧队列
  const int cacheSize;          // 缓存帧数
  const double freezeThreshold; // 冻结判断阈值（像素变化阈值）
  const int latence = 10;       // 差值延迟调整

public:
  // 构造函数
  FreezeDetector(int cacheSize = 5, double freezeThreshold = 100.0,
                 int latence = 10)
      : cacheSize(cacheSize), freezeThreshold(freezeThreshold),
        latence(latence) {}

  // 检测冻结的方法
  bool detect(const Mat &currentFrame) {
    Mat grayCurrent;
    cvtColor(currentFrame, grayCurrent, COLOR_BGR2GRAY);

    // 如果缓存数量不足，直接返回 false
    if (frameDeque.size() < cacheSize) {
      frameDeque.push_back(currentFrame.clone());
      return false;
    }

    int similarCount = 0; // 统计差异小于阈值的帧数

    // 遍历缓存帧队列
    for (const auto &frame : frameDeque) {
      Mat diff, grayFrame;
      cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

      // 计算帧间差分
      absdiff(grayCurrent, grayFrame, diff);

      // 对差分图像减去延迟
      diff = diff - latence;

      // 计算差值的总和
      double avgDiff = sum(diff)[0];

      // 判断是否小于冻结阈值
      if (avgDiff < freezeThreshold) {
        similarCount++;
      }
    }

    // 更新缓存队列
    frameDeque.push_back(currentFrame.clone());
    if (frameDeque.size() > cacheSize) {
      frameDeque.pop_front(); // 保持队列大小为 cacheSize
    }

    // 判断冻结条件：当前帧与缓存中大部分帧差异均小于阈值
    return similarCount > frameDeque.size() / 2;
  }
};

// 主程序
int main() {
  string rtspUrl = "rtsp://admin:admin123@10.0.0.179:554/live";

  // 打开 RTSP 流
  VideoCapture cap(rtspUrl);
  if (!cap.isOpened()) {
    cout << "Error: Cannot open RTSP stream" << endl;
    return -1;
  }

  Mat frame;
  FreezeDetector freezeDetector; // 初始化冻结检测类

  while (true) {
    cap >> frame;
    if (frame.empty())
      break;

    // 检测冻结
    bool isFrozen = freezeDetector.detect(frame);

    // 显示检测结果
    if (isFrozen) {
      putText(frame, "Freeze Detected!", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1,
              Scalar(0, 0, 255), 2);
    } else {
      putText(frame, "Normal", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1,
              Scalar(0, 255, 0), 2);
    }

    // 显示视频
    imshow("RTSP Stream", frame);

    // 按下 'q' 键退出
    if (waitKey(30) == 'q')
      break;
  }

  return 0;
}