#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

/**
 * 计算两个图像块之间的相似度
 */
double calculateSimilarity(const Mat &img1, const Mat &img2, int diff_value) {
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

/**
 * 处理图像并检测遮挡
 * @param grayFrame 当前灰度图像
 * @param firstFrame 参考灰度图像
 * @param blockSize 分块大小
 * @param block_threshold 遮挡判定阈值
 * @param diff_value 差异值阈值
 * @param is_debug 是否显示调试信息
 * @return 是否检测到遮挡
 */
bool processImage(Mat &grayFrame, const Mat &firstFrame, 
                 int blockSize = 64, 
                 double block_threshold = 0.9, 
                 int diff_value = 50,
                 bool is_debug = false) {
    // Add frame count text
    Mat frame = grayFrame.clone();
    
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

            double blockSimilarity = calculateSimilarity(block1, block2, diff_value);

            // 根据相似度选择颜色
            Scalar blockColor;
            if (blockSimilarity > 0.5) {        // 面积阈值
                blockColor = Scalar(0, 0, 255); // 红色
                highSimilarityBlocks++;
                if (is_debug) {
                    mask(blockRegion).setTo(255);
                }
            } else {
                blockColor = Scalar(0, 255, 0); // 绿色
            }

            if (is_debug) {
                rectangle(frame, blockRegion, blockColor, 2);
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

        for (const auto &contour : contours) {
            Rect boundingRect = cv::boundingRect(contour);
            rectangle(frame, boundingRect, Scalar(255, 0, 0), 3);
            
            double area = contourArea(contour);
            string areaText = format("Area: %.0f", area);
            Point textPos(boundingRect.x, boundingRect.y - 5);
            putText(frame, areaText, textPos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
        }

        // 显示高相似度区域占比
        double highSimilarityRatio = (double)highSimilarityBlocks / totalBlocks * 100.0;
        string ratioText = format("High Similarity Blocks: %d/%d (%.2f%%)", 
                                highSimilarityBlocks, totalBlocks, highSimilarityRatio);
        putText(frame, ratioText, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        imshow("contrast", firstFrame);
        imshow("Frame with Similarity", frame);
        imshow("Connected Regions", mask);
    }

    return highSimilarityBlocks / (double)totalBlocks > block_threshold;
}

int main() {
    string rtspUrl = "rtsp://admin:admin123@10.0.0.179:554/live";
    VideoCapture cap(rtspUrl);

    if (!cap.isOpened()) {
        cerr << "Failed to open RTSP stream." << endl;
        return -1;
    }

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
    cvtColor(firstFrame, grayFirstFrame, COLOR_BGR2GRAY);
    frameBuffer.push_back(grayFirstFrame.clone());

    while (true) {
        Mat frame;
        if (!cap.read(frame)) {
            cerr << "Failed to read frame from RTSP stream." << endl;
            break;
        }
        
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        frameBuffer.push_back(grayFrame.clone());
        if (frameBuffer.size() > bufferSize) {
            frameBuffer.pop_front();
        }

        // 处理当前帧
        bool isBlocked = processImage(frameBuffer.back(), frameBuffer.front(), 64, 0.9, 50, true);

        // 在图片正中间写入true或false
        string text = isBlocked ? "true" : "false";
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 3.0;
        int thickness = 2;
        int baseline = 0;
        Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
        Point textOrg((frame.cols - textSize.width) / 2, (frame.rows + textSize.height) / 2);
        putText(frame, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        imshow("Frame", frame);

        if (waitKey(10) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}