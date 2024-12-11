#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

class CameraBlockage {
  private:
    const int blockSize;
    int frameCount;
    bool is_debug;
    int diff_value;
    double block_threshold;

    // 计算MSE函数
    /**
     * 计算两个图像之间的均方误差(MSE)
     * @param img1 第一个输入图像
     * @param img2 第二个输入图像
     * @return 归一化后的MSE值。如果输入图像尺寸或类型不匹配则返回-1
     */
    double calculateMSE(const Mat &img1, const Mat &img2) {
        // 检查两个图像的尺寸和类型是否匹配
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            cerr << "Error: Images must have the same dimensions and type." << endl;
            return -1;
        }

        // 计算两个图像的绝对差值
        Mat diff;
        absdiff(img1, img2, diff);
        // 将差值转换为32位浮点型以进行后续计算
        diff.convertTo(diff, CV_32F);

        // 计算差值的平方
        Mat squaredDiff;
        pow(diff, 2, squaredDiff);

        // 计算平方差的均值
        Scalar mse = mean(squaredDiff);

        // 计算最大可能误差值
        double maxError = diff_value * diff_value;
        // 归一化MSE值，考虑图像通道数
        double normalizedMSE = (mse[0] + mse[1] + mse[2]) / (maxError * img1.channels());
        return normalizedMSE;
    }

    /**
     * 计算两个图像块之间的相似度
     * @param img1 第一个输入图像块
     * @param img2 第二个输入图像块
     * @return 相似度值。值越大表示差异越大
     */
    double calculateSimilarity(const Mat &img1, const Mat &img2) {
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            cerr << "Error: Images must have the same dimensions and type." << endl;
            return -1;
        }

        // 1. 应用高斯模糊减少噪声影响
        Mat blur1, blur2;
        GaussianBlur(img1, blur1, Size(3, 3), 0);
        GaussianBlur(img2, blur2, Size(3, 3), 0);

        // 2. 计算差异
        Mat diff;
        absdiff(blur1, blur2, diff);

        // 3. 应用阈值，忽略微小变化
        Mat thresholded;
        threshold(diff, thresholded, 30, 255, THRESH_BINARY);

        // 4. 计算有效差异区域的比例
        int nonZeroPixels = countNonZero(thresholded);
        double totalPixels = thresholded.rows * thresholded.cols;
        double diffRatio = nonZeroPixels / totalPixels;

        // 5. 如果差异比例小于某个阈值，认为是噪声
        if (diffRatio < 0.1) { // 10%以下的差异视为噪声
            return 0.0;
        }

        // 6. 计算归一化的差异值
        Scalar meanDiff = mean(diff);
        double normalizedDiff = meanDiff[0] / diff_value;

        return normalizedDiff;
    }

  public:
    CameraBlockage(int blkSize = 128, double blockThreshold = 0.9, int diffValue = 40, bool debug = false)
        : blockSize(blkSize), frameCount(0), is_debug(debug), block_threshold(blockThreshold), diff_value(diffValue) {}

    // 处理单帧图像的公开接口
    bool processImage(Mat &grayFrame, double &maxArea, const Mat &firstFrame) {
        // Add frame count text
        Mat frame = grayFrame;
        if (is_debug) {
            string text = "Frame Count: " + to_string(frameCount++);
            putText(frame, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }

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

                double blockSimilarity = calculateSimilarity(block1, block2);

                // 根据相似度选择颜色
                Scalar blockColor;
                if (blockSimilarity > 0.3) {        // 面积阈值
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
            for (const auto &contour : contours) {
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

    // 清理资源
    void cleanup() { destroyAllWindows(); }

    ~CameraBlockage() { cleanup(); }
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
    CameraBlockage processor(128, 0.85, 80, true); // 设置debug模式

    // 创建帧缓冲队列
    deque<Mat> frameBuffer;
    const size_t bufferSize = 300;

    // 读取第一帧并初始化帧缓冲
    Mat firstFrame;
    if (!cap.read(firstFrame)) {
        cerr << "Failed to read first frame from RTSP stream." << endl;
        return -1;
    }

    // 将第一帧转为灰度图作为参考帧
    Mat grayFirstFrame;
    cv::cvtColor(firstFrame, grayFirstFrame, COLOR_BGR2GRAY);
    frameBuffer.push_back(grayFirstFrame.clone());

    while (true) {
        Mat frame;
        if (!cap.read(frame)) {
            cerr << "Failed to read frame from RTSP stream." << endl;
            break;
        }
        // 转为灰度图
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        // 更新帧缓冲
        frameBuffer.push_back(grayFrame.clone());
        if (frameBuffer.size() > bufferSize) {
            frameBuffer.pop_front();
        }

        double maxArea = 0.0;
        // 处理当前帧
        bool isBlocked = processor.processImage(frameBuffer.back(), maxArea, frameBuffer.front());

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