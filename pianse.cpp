#include <iostream>
#include <opencv2/opencv.hpp>

bool checkBrightness(const cv::Mat &image, double lowThreshold,
                     double highThreshold) {
  if (image.empty()) {
    std::cerr << "Error: Image is empty." << std::endl;
    return false;
  }

  // 转换为灰度图像
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

  // 计算直方图
  int histSize = 256; // 亮度级别
  float range[] = {0, 256};
  const float *histRange = {range};
  cv::Mat hist;
  cv::calcHist(&grayImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange,
               true, false);

  // 统计亮度异常的像素数量
  int lowBrightCount = 0;
  int highBrightCount = 0;
  for (int i = 0; i < lowThreshold; i++) {
    lowBrightCount += cvRound(hist.at<float>(i));
  }
  for (int i = highThreshold; i < histSize; i++) {
    highBrightCount += cvRound(hist.at<float>(i));
  }

  // 判断是否亮度异常
  double totalPixels = grayImage.rows * grayImage.cols;
  double lowRatio = (double)lowBrightCount / totalPixels;
  double highRatio = (double)highBrightCount / totalPixels;

  // 输出信息，这部分可以根据需要调整
  std::cout << "Low brightness ratio: " << lowRatio << std::endl;
  std::cout << "High brightness ratio: " << highRatio << std::endl;

  // 根据经验阈值判断
  return lowRatio > 0.1 || highRatio > 0.1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <ImagePath>" << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(argv[1]);
  if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
  }

  bool isBrightnessAbnormal = checkBrightness(image, 50, 200);
  if (isBrightnessAbnormal) {
    std::cout << "Brightness is abnormal." << std::endl;
  } else {
    std::cout << "Brightness is normal." << std::endl;
  }

  return 0;
}