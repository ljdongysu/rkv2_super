#pragma once

#include "SpRun.h"
#include <iostream>
#include <algorithm>
#include "timer.h"
#include <fstream>
//#include <dirent.h>
#include <unistd.h>

template<class T>
void PrintMatrix2(const T *data, const int W)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
            std::cout << std::setw(10) << std::setprecision(3) << data[i * W + j][0];
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

SpRun::SpRun()
        : conf_thresh(0.005f)
        , nms_dist(4)
        , border(4)
{
}


SpRun::SpRun(int desc_c, int pixn, int rh, int rw)
        : SpRun()
{
    desc_channel = desc_c;
    pixnum = pixn;
    rsize_h = rh;
    rsize_w = rw;
    HFeature = int(floor(rh / this->cell));
    WFeature = int(floor(rw / this->cell));
}

SpRun::~SpRun()
{

}


void SpRun::Norm(TensorType& tensor)
{
    if (tensor.empty()) return;
    if (tensor[0].empty()) return;
    std::vector<float> norm(tensor[0].size(), 0.00001);

    for (int i = 0; i < tensor[0].size(); ++i)
    {
        for (int j = 0; j < tensor.size(); ++j)
        {
            norm[i] += (tensor[j][i] * tensor[j][i]);
        }

        norm[i] = sqrt(norm[i]);
    }

    for (int i = 0; i < tensor[0].size(); ++i)
    {
        for (int j = 0; j < tensor.size(); ++j)
        {
            tensor[j][i] /= norm[i];
        }
    }
}

void SpRun::grid_sample(const TensorType &coarse_desc, const Points &points, Describes &describes)
{
    // grid_sample �˰��� ���� ��ũ
    // https://stackoverflow.com/questions/73300183/understanding-the-torch-nn-functional-grid-sample-op-by-concrete-example

    const size_t count = points.size();
    const int hFeature = this->HFeature, wFeature = this->WFeature;
    Describes result(coarse_desc.size(), std::vector<float>(count, 0));
    Describes result2(coarse_desc.size(), std::vector<float>(count, 0));
    std::vector<float> norm(count, 0);

    for (size_t channel = 0; channel < coarse_desc.size(); channel++)
    {
        // TODO : 2023-03-20 00:42:39 [hao]  speed up, exchange i and j
        for (size_t p = 0; p < count; p++)
        {
            // TODO : 2023-03-20 00:42:04 [hao]  why so complex, only a scale.
            double cvt_x = MIN(MAX(float(points[p].x + 1) / this->rsize_w * wFeature - 1, 0), wFeature - 1);
            double cvt_y = MIN(MAX(float(points[p].y + 1) / this->rsize_h * hFeature - 1, 0), hFeature - 1);
            double a = (cvt_x) - floor(cvt_x);
            double b = (cvt_y) - floor(cvt_y);
            double x = coarse_desc[channel][int(floor(cvt_y)) * wFeature + int(floor(cvt_x))];
            double y = coarse_desc[channel][int(floor(cvt_y)) * wFeature + int(ceil(cvt_x))];
            double w = coarse_desc[channel][int(ceil(cvt_y)) * wFeature + int(floor(cvt_x))];
            double z = coarse_desc[channel][int(ceil(cvt_y)) * wFeature + int(ceil(cvt_x))];

            double out = (a * b * z) + ((1 - a) * (1 - b) * x) +
                         ((1 - a) * b * w) + ((1 - b) * a * y);
            result[channel][p] = out;
        }
    }

    Norm(result);
    Describes resultT(count, std::vector<float>(coarse_desc.size(), 0));

    for (size_t channel = 0; channel < coarse_desc.size(); channel++)
    {
        for (size_t p = 0; p < count; p++)
        {
            resultT[p][channel] = result[channel][p];
        }
    }
    describes.swap(resultT);
}


void SpRun::Softmax(const TensorType &semi, TensorType &dense, bool dropLastAixs)
{
    if (semi.empty()) return;
    if (semi.at(0).empty()) return;

    dense = semi;
    std::vector<double> sum(semi.at(0).size(), 0.0001);

    for (size_t channel = 0; channel < dense.size(); channel++)
    {
        auto &denseChannel = dense[channel];
        for (size_t i = 0; i < denseChannel.size(); i++)
        {
            // TODO : 2023-03-17 15:03:21 [hao]  why use double
            double value = exp(denseChannel[i]);
            denseChannel[i] = value;
            sum[i] += value;
        }
    }

//    PrintMatrix(dense[0].data(), 80, 8, 8, "exp");
//    PrintMatrix(sum.data(), 80, 8, 8, "exp sum");

    for (size_t channel = 0; channel < dense.size(); channel++)
    {
        auto &denseChannel = dense[channel];
        for (size_t i = 0; i < pixnum; i++)
        {
            denseChannel[i] = denseChannel[i] / (sum[i]);
        }
    }

//    PrintMatrix(dense[0].data(), 80, 8, 8, "softmax");
    if (dropLastAixs)
        dense.resize(dense.size() - 1);
}

// convert (channel[cell * cel], h, w) -> (h_image[h * cell], w_image[w * cell])
void SpRun::ReshapeLocal(const TensorType &input, TensorType &output)
{
/* like:
    array = np.arange(1, Hc * Wc * cell * cell + 1)
    nodust = array.reshape((cell * cell, Hc, Wc))
    nodust_trans = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust_trans, [Hc, Wc, cell, cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])
    */

    if (input.empty()) return;
    if (input.at(0).empty()) return;

    int channelInput = input.size();

    output.resize(this->rsize_h, std::vector<float>(this->rsize_w, 0));

    for (int c = 0; c < channelInput; ++c)
    {
        for (int h = 0; h < this->HFeature; ++h)
        {
            int row = h * this->cell + (c / this->cell);
            for (int w = 0; w < this->WFeature; ++w)
            {
                float value = input[c][h * this->WFeature + w];
                int col = w * this->cell + (c % this->cell);
                output[row][col] = value;
            }
        }
    }
}

void SpRun::GetValidPoint(const TensorType &heatmap, Points &features)
{
    features.resize(0);
    int count = 0;

    for (size_t i = 0; i < this->rsize_h; i++)
    {
        for (size_t j = 0; j < this->rsize_w; j++)
        {
            FeaturePoint point;
            if (heatmap[i][j] >= conf_thresh)
            {
                point.x = j;
                point.y = i;
                point.confidence = heatmap[i][j];
                point.id = count;
                count++;
                features.push_back(point);
            }
        }
    }

}

void SpRun::nms_fast(const Points &input, Points &output)
{
    Points corners = input;
    std::vector<int> indexSorted;

    std::sort(corners.begin(), corners.end(), [](FeaturePoint &p1, FeaturePoint &p2)
    { return p1.confidence > p2.confidence; });

    float invalidConfidence = -1;
    size_t count = 0;

    for (int i = 0; i < corners.size(); ++i)
    {
        const auto &pre = corners[i];
        bool valid = pre.confidence > 0;
        if (not valid) continue;
        count++;

        for (int j = i + 1; j < corners.size(); ++j)
        {
            auto &next = corners[j];
            bool near = (abs(pre.x - next.x) <= this->nms_dist and
                         abs(pre.y - next.y) <= this->nms_dist);
            if (near)
            {
                next.confidence = invalidConfidence;
            }
        }
    }

    corners.erase(std::remove_if(corners.begin(), corners.end(), [this](FeaturePoint &p1)
    { return p1.confidence < this->conf_thresh; }), corners.end());

    std::swap(output, corners);
};


void SpRun::ParsePoints(const TensorType &semi, Points &ptsNMS)
{
    Timer timer;
    TensorType nodust;  // (channel , h*WFeature )
    TensorType heatmap;
    Points pts;

    Softmax(semi, nodust, true);
    timer.Timing("softmax", true);
    ReshapeLocal(nodust, heatmap);
    timer.Timing("reshape and transpose", true);
    GetValidPoint(heatmap, pts);
    timer.Timing("get valide feature", true);
    PRINTF(pts.size());
    nms_fast(pts, ptsNMS);
    // TODO : 2023-03-19 23:32:02 [hao]  remove near imgae border 4
    PRINTF(ptsNMS.size());
    timer.Timing("nms", true);
}


void SpRun::calc(TensorType &semi, TensorType &desc, cv::Mat img
        , Points &pointsResult, Describes &describesResult)
{
    // semi : (1,loc_channel, outpixNum)
    // coarse_desc : (1, desc_channel, pixnum)

    Points points;
    Describes describes;

    ParsePoints(semi, points);
    ParseDescribe(desc, points, describes);

    pointsResult.swap(points);
    describesResult.swap(describes);
//    WriteDescCSV(points, describes);

}

void SpRun::ShowImage(const std::string file, const Points &points, cv::Mat &image)
{
    const int W = image.cols;
    const int H = image.rows;


    cv::Mat imgcp = image;
    for (int i = 0; i < points.size(); i++)
    {
        const auto &p = points[i];
        auto color = cv::Scalar(0, 255, 0);
        float confidence = MIN(1, p.confidence + 0.2);
        color[2] = 255 - confidence * 255;
        color[1] = confidence * 255;
        cv::circle(imgcp, cv::Point(p.x, p.y), 2, color, -1, 8, 0);
    }
    cv::putText(imgcp, std::to_string(points.size()), cv::Point(W - 100, H - 30), cv::FONT_HERSHEY_COMPLEX
            , 1, cv::Scalar(0, 0, 255), 2);

    if (not file.empty())
    {
        std::string outputFile;
        char *p = getcwd(NULL, 0);
        int index = file.find_last_of('/');
        std::string path = file.substr(index + 1, -1);
        int index2 = path.find_last_of(".");
        std::string extendName = path.substr(0, index2);
        outputFile = std::string(p) + "/" + "result" + "/result_" + extendName + ".jpg";
        cv::imwrite(outputFile, imgcp);

        std::cout << outputFile << std::endl;
    }
}

void SpRun::ParseDescribe(const TensorType &desc, const Points &points, Describes &describes)
{
    Timer timer;
    // TODO : 2023-03-19 23:48:03 [hao]  why not use
    // normalize the coordinate to (-1, 1) / (rsize_w / 2.)) - 1.
    grid_sample(desc, points, describes);
    timer.Timing("grid sample", true);
}

void PrintMatrixUchar(const uint8_t *data, const int W)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
            std::cout << std::setw(10) << std::setprecision(3) << (int) (data[i * W + j]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void SpRun::WriteDescCSV(const Points &points, const std::vector<std::vector<float>> describes)
{
    std::ofstream outFile;

    if (points.size() > 0)
    {
        std::cout << "points.size: " << points.size() << std::endl;
    }
    if (describes.size() > 0)
    {
        std::cout << "shape[" << describes.size();
        if (describes[0].size() > 0)
        {
            std::cout << ", " << describes[0].size() << "]" << std::endl;
        }
    }

    outFile.open("data.csv", std::ios::out);
    outFile << "x" << "," << "y" << std::endl;
    for (int i = 0; i < points.size(); ++i)
    {
        outFile << points[i].x << ", " << points[i].y << ",";
        for (int j = 0; j < describes[0].size(); ++j)
        {
            outFile << describes[i][j] << ",";
        }
        outFile << "\n";
    }
    outFile.close();
}
