#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 遮挡检测算法实现
void detectObstruction(VideoCapture& cap, int blockSize, double thresholdValue) {
    Mat frame, gray, prevGray, diff;
    int frameCount = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 转换为灰度图
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 初始化上一帧灰度图
        if (frameCount == 0) {
            gray.copyTo(prevGray);
            frameCount++;
            continue;
        }

        // 计算帧间差分
        absdiff(gray, prevGray, diff);

        // 分块处理
        for (int y = 0; y < diff.rows; y += blockSize) {
            for (int x = 0; x < diff.cols; x += blockSize) {
                Rect blockRect(x, y, blockSize, blockSize);
                Mat block = diff(blockRect & Rect(0, 0, diff.cols, diff.rows));

                // 计算块内像素变化的累积值
                double blockChange = sum(block)[0] / (blockSize * blockSize);

                // 如果块的变化值超过阈值，标记为遮挡
                if (blockChange > thresholdValue) {
                    rectangle(frame, blockRect, Scalar(0, 0, 255), 2); // 用红色框标记遮挡区域
                }
            }
        }

        // 显示结果
        imshow("Video Feed", frame);
        imshow("Pixel Change", diff);

        // 更新上一帧
        gray.copyTo(prevGray);

        // 按下 'q' 键退出
        if (waitKey(30) == 'q') break;

        frameCount++;
    }
}

int main(int argc, char** argv) {
    // 打开摄像头或视频文件
    VideoCapture cap;
    if (argc == 2) {
        cap.open(argv[1]); // 从文件打开
    } else {
        cap.open(0); // 从摄像头打开
    }

    if (!cap.isOpened()) {
        cout << "Error: Cannot open video source." << endl;
        return -1;
    }

    // 参数配置
    int blockSize = 16;       // 块大小
    double thresholdValue = 15.0; // 阈值

    // 调用遮挡检测函数
    detectObstruction(cap, blockSize, thresholdValue);

    return 0;
}