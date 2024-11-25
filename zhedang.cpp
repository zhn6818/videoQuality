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
    bool is_debug;
    const int diff_value = 50;
    const double block_threshold = 0.5;

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

        double maxError = diff_value * diff_value;
        double normalizedMSE = (mse[0] + mse[1] + mse[2]) / (maxError * img1.channels());
        return normalizedMSE;
    }

    bool processFrame(Mat &frame, const Mat &grayFrame, double &maxArea) {
        // Add frame count text
        if (is_debug) {
            string text = "Frame Count: " + to_string(frameCount++);
            putText(frame, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }

        // Process frame buffer
        if (frameBuffer.size() == bufferSize) {
            const Mat &firstFrame = frameBuffer.front();

            int totalBlocks = 0;
            int highSimilarityBlocks = 0;

            // 创建二值化掩码,用于标记不相似区域
            Mat mask;
            if (is_debug) {
                mask = Mat::zeros(firstFrame.size(), CV_8UC1);
            }

            // Calculate similarity for each block
            for (int y = 0; y < firstFrame.rows; y += blockSize) {
                for (int x = 0; x < firstFrame.cols; x += blockSize) {
                    // 确保不超出图像边界
                    int currentBlockWidth = min(blockSize, firstFrame.cols - x);
                    int currentBlockHeight = min(blockSize, firstFrame.rows - y);

                    if (currentBlockWidth < blockSize || currentBlockHeight < blockSize) {
                        continue; // 跳过不完整的块
                    }

                    totalBlocks++;

                    Rect blockRegion(x, y, blockSize, blockSize);
                    Mat block1 = firstFrame(blockRegion);
                    Mat block2 = grayFrame(blockRegion);

                    double blockSimilarity = calculateMSE(block1, block2);

                    // 根据相似度选择颜色
                    Scalar blockColor;
                    if (blockSimilarity > 1.0) {
                        blockColor = Scalar(0, 0, 255); // 红色
                        highSimilarityBlocks++;
                        if (is_debug) {
                            // 在掩码上标记不相似区域
                            mask(blockRegion).setTo(255);
                        }
                    } else {
                        blockColor = Scalar(0, 255, 0); // 绿色
                    }

                    if (is_debug) {
                        // 绘制块的边界
                        rectangle(frame, blockRegion, blockColor, 2);

                        // 显示相似度数值
                        string similarityText = format("%.2f", blockSimilarity);
                        Point textOrigin(x + 5, y + blockSize / 2);
                        putText(frame, similarityText, textOrigin, FONT_HERSHEY_SIMPLEX, 0.5, blockColor, 1);
                    }
                }
            }

            if (is_debug) {
                // 寻找连通区域
                vector<vector<Point>> contours;
                vector<Vec4i> hierarchy;
                findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                maxArea = 0.0;
                for (const auto& contour : contours) {
                    // 计算最小外接矩形
                    Rect boundingRect = cv::boundingRect(contour);
                    rectangle(frame, boundingRect, Scalar(255, 0, 0), 3); // 蓝色粗线显示连通区域

                    // 计算并显示连通区域面积
                    double area = contourArea(contour);
                    if (area > maxArea) {
                        maxArea = area;
                    }
                    string areaText = format("Area: %.0f", area);
                    Point textPos(boundingRect.x, boundingRect.y - 5);
                    putText(frame, areaText, textPos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
                }

                // 计算并显示高相似度区域占比
                double highSimilarityRatio = (double)highSimilarityBlocks / totalBlocks * 100.0;
                string ratioText = format("High Similarity Blocks: %d/%d (%.2f%%)", highSimilarityBlocks, totalBlocks,
                                          highSimilarityRatio);
                                          
                putText(frame, ratioText, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

                imshow("contrast", firstFrame);
                imshow("Frame with Similarity", frame);
                imshow("Connected Regions", mask);
            }

            // 当相似度占比超过50%时返回true
            return highSimilarityBlocks / (double)totalBlocks > block_threshold;
        }
        return false;
    }

  public:
    ZheDang(size_t bufSize = 300, int blkSize = 128, bool debug = false) : bufferSize(bufSize), blockSize(blkSize), frameCount(0), is_debug(debug) {}

    // 处理单帧图像的公开接口
    bool processImage(Mat &frame, double &maxArea) {
        if (frame.empty()) {
            cerr << "Empty frame received." << endl;
            return false;
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
        return processFrame(frame, grayFrame, maxArea);
    }

    // 清理资源
    void cleanup() {
        frameBuffer.clear();
        destroyAllWindows();
    }

    ~ZheDang() { cleanup(); }
};

int main() {
    // 视频捕获的设置
    string rtspUrl = "rtsp://admin:admin123@10.0.0.179:554/live";
    VideoCapture cap(rtspUrl);

    if (!cap.isOpened()) {
        cerr << "Failed to open RTSP stream." << endl;
        return -1;
    }

    // 创建遮挡检测实例
    ZheDang processor(300, 128, false); // 设置debug模式

    while (true) {
        Mat frame;
        if (!cap.read(frame)) {
            cerr << "Failed to read frame from RTSP stream." << endl;
            break;
        }

        double maxArea = 0.0;
        // 处理当前帧
        bool isBlocked = processor.processImage(frame, maxArea);

        // 在图片正中间写入true或false
        string text = isBlocked ? "true" : "false";
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 3.0;
        int thickness = 2;
        int baseline = 0;
        Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
        Point textOrg((frame.cols - textSize.width) / 2, (frame.rows + textSize.height) / 2);
        putText(frame, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        if (isBlocked) {
            cout << "Max blocked area: " << maxArea << endl;
        }

        // 显示帧
        imshow("Frame", frame);

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