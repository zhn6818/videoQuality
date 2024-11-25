#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

class ZheDang {
  private:
    deque<Mat> frameBuffer;
    const size_t bufferSize;
    const int blockSize;
    int frameCount;

    // 计算MSE函数
    double calculateMSE(const Mat &img1, const Mat &img2) {
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            cerr << "Error: Images must have the same dimensions and type." << endl;
            return -1;
        }

        Mat diff;
        absdiff(img1, img2, diff);
        diff.convertTo(diff, CV_32F);

        Mat squaredDiff;
        pow(diff, 2, squaredDiff);

        Scalar mse = mean(squaredDiff);

        double maxError = 50.0 * 50.0;
        double normalizedMSE = (mse[0] + mse[1] + mse[2]) / (maxError * img1.channels());
        return normalizedMSE;
    }

    void processFrame(Mat &frame, const Mat &grayFrame) {
        // Add frame count text
        string text = "Frame Count: " + to_string(frameCount++);
        putText(frame, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        // Process frame buffer
        if (frameBuffer.size() == bufferSize) {
            const Mat &firstFrame = frameBuffer.front();

            // Calculate similarity for each block
            for (int y = 0; y < firstFrame.rows; y += blockSize) {
                for (int x = 0; x < firstFrame.cols; x += blockSize) {
                    Rect blockRegion(x, y, blockSize, blockSize);
                    if (x + blockSize > firstFrame.cols || y + blockSize > firstFrame.rows) {
                        continue;
                    }
                    Mat block1 = firstFrame(blockRegion);
                    Mat block2 = grayFrame(blockRegion);

                    double blockSimilarity = calculateMSE(block1, block2);

                    string similarityText = format("%.2f", blockSimilarity);
                    Point textOrigin(x, y + 15);
                    putText(frame, similarityText, textOrigin, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
                }
            }
            imshow("contrast", firstFrame);
            imshow("Frame with Similarity", frame);
        }
    }

  public:
    ZheDang(size_t bufSize = 300, int blkSize = 64) : bufferSize(bufSize), blockSize(blkSize), frameCount(0) {}

    // 处理单帧图像的公开接口
    void processImage(Mat &frame) {
        if (frame.empty()) {
            cerr << "Empty frame received." << endl;
            return;
        }

        // 转为灰度图
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // 管理帧缓冲
        if (frameBuffer.size() == bufferSize) {
            frameBuffer.pop_front();
        }
        frameBuffer.push_back(grayFrame);

        // 处理帧
        processFrame(frame, grayFrame);
    }

    // 清理资源
    void cleanup() {
        frameBuffer.clear();
        destroyAllWindows();
    }

    ~ZheDang() { cleanup(); }
};

int main() {
    // 视频捕获的设置移回main函数
    string rtspUrl = "rtsp://admin:admin123@10.0.0.179:554/live";
    VideoCapture cap(rtspUrl);

    if (!cap.isOpened()) {
        cerr << "Failed to open RTSP stream." << endl;
        return -1;
    }

    // 创建遮挡检测实例
    ZheDang processor;

    while (true) {
        Mat frame;
        if (!cap.read(frame)) {
            cerr << "Failed to read frame from RTSP stream." << endl;
            break;
        }

        // 处理当前帧
        processor.processImage(frame);

        // 按ESC退出
        if (waitKey(10) == 27) {
            break;
        }
    }

    // 清理资源
    cap.release();
    processor.cleanup();
    return 0;
}