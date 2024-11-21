#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 阈值和持续帧数配置
const int minPersistentFrames = 10;       // 持续遮挡帧数阈值
const double blockChangeThreshold = 15.0; // 像素变化阈值
const int blockSize = 16;                 // 块大小

// 自定义比较函数
struct RectCompare {
  bool operator()(const Rect &a, const Rect &b) const {
    if (a.x != b.x)
      return a.x < b.x;
    if (a.y != b.y)
      return a.y < b.y;
    if (a.width != b.width)
      return a.width < b.width;
    return a.height < b.height;
  }
};

// 遮挡区域数据结构
struct ObstructionArea {
  int frameCount; // 遮挡持续的帧数
  Rect area;      // 遮挡区域

  // 重载 == 运算符
  bool operator==(const ObstructionArea &other) const {
    return frameCount == other.frameCount && area == other.area;
  }
};

// 遮挡检测算法
void detectObstruction(VideoCapture &cap) {
  Mat frame, gray, prevGray, diff;
  map<Rect, ObstructionArea, RectCompare>
      obstructionMap; // 记录每个遮挡区域的状态
  int frameNumber = 0;

  while (true) {
    cap >> frame;
    if (frame.empty())
      break;

    // 转换为灰度图
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // 初始化第一帧
    if (frameNumber == 0) {
      gray.copyTo(prevGray);
      frameNumber++;
      continue;
    }

    // 计算帧间差分
    absdiff(gray, prevGray, diff);

    // 阈值化差分图像
    threshold(diff, diff, blockChangeThreshold, 255, THRESH_BINARY);

    // 分块处理
    for (int y = 0; y < diff.rows; y += blockSize) {
      for (int x = 0; x < diff.cols; x += blockSize) {
        Rect blockRect(x, y, blockSize, blockSize);
        Mat block = diff(blockRect & Rect(0, 0, diff.cols, diff.rows));

        // 计算块内像素变化比例
        double blockChange = sum(block)[0] / (blockSize * blockSize);

        // 如果块变化显著，标记为遮挡
        if (blockChange > blockChangeThreshold) {
          if (obstructionMap.find(blockRect) == obstructionMap.end()) {
            // 新的遮挡区域
            obstructionMap[blockRect] = {1, blockRect};
          } else {
            // 增加持续帧数
            obstructionMap[blockRect].frameCount++;
          }
        } else {
          // 如果没有显著变化，移除该区域
          if (obstructionMap.find(blockRect) != obstructionMap.end()) {
            obstructionMap.erase(blockRect);
          }
        }
      }
    }

    // 可视化遮挡区域
    int totalObstructedArea = 0; // 记录遮挡的总面积
    for (auto &[rect, obstruction] : obstructionMap) {
      if (obstruction.frameCount >= minPersistentFrames) {
        // 持续遮挡超过阈值，标记为告警区域
        rectangle(frame, obstruction.area, Scalar(0, 0, 255), 2); // 红色框
        totalObstructedArea += obstruction.area.area();
      }
    }

    // 在画面上显示告警信息
    putText(frame,
            "Obstructed Area: " + to_string(totalObstructedArea) + " pixels",
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

    // 显示结果
    imshow("Video Feed", frame);
    imshow("Pixel Change", diff);

    // 更新上一帧
    gray.copyTo(prevGray);

    // 按下 'q' 键退出
    if (waitKey(30) == 'q')
      break;

    frameNumber++;
  }
}

int main(int argc, char **argv) {
  // RTSP 地址
  string rtspUrl = "rtsp://admin:admin123@10.0.0.179:554/live";

  // 打开 RTSP 流
  VideoCapture cap(rtspUrl);
  if (!cap.isOpened()) {
    cout << "Error: Cannot open RTSP stream" << endl;
    return -1;
  }

  // 调用遮挡检测函数
  detectObstruction(cap);

  return 0;
}