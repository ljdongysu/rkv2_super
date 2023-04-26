//
// Created by indemind on 4/26/23.
//

#ifndef RKV2_SUPER_SPRUN_H
#define RKV2_SUPER_SPRUN_H

#include <cmath>
#include "opencv2/opencv.hpp"

#define PRINT
#define PRINTF(a) (std::cout << "" << (#a) << " = " << (a) << "" << std::endl)

template<class T>
void PrintMatrix(const T *data, const int W, const int printRow = 18, const int printCol = 18
        , const std::string message = "")
{
#ifdef PRINT
    if (not message.empty())
        std::cout << message << std::endl;
    // print col id
    std::cout << std::setw(8) << " ";
    for (int j = 0; j < printCol; j++)
        std::cout << std::fixed << std::setw(8) << std::setprecision(3) << j;
    std::cout << std::endl;

    // print data
    for (int i = 0; i < printRow; i++)
    {
        std::cout << std::setw(6) << i << ": ";
        for (int j = 0; j < printCol; j++)
            std::cout << std::fixed << std::setw(8) << std::setprecision(3) << data[i * W + j];
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif
}

using TensorType = std::vector<std::vector<float>>;

void PrintMatrixUchar(const uint8_t *data, const int W);

struct FeaturePoint
{
    int id;
    int x;
    int y;
    float confidence;
};

using Points = std::vector<FeaturePoint>;
using Describes = std::vector<std::vector<float>>;

struct Features
{
    Points points;
    Describes describes;
};

class SpRun
{

protected:
    double nn_thresh;
    long long desc_channel;
    long long pixnum;
    long long rsize_h;
    long long rsize_w;
    int nms_dist;
    int border;
    long long HFeature;
    long long WFeature;
    int cell = 8;
    double conf_thresh;

    long long **pts_save;
    double *score_save;
    double **desc_save;

public:
    SpRun();

    SpRun(int desc_c, int pixn, int rsize_h, int rsize_w);

    ~SpRun();

    void grid_sample(const TensorType &coarse_desc, const Points &points, Describes& describes);

    void calc(TensorType &semi, TensorType &desc, cv::Mat img, Points &pointsResult
            , Describes &describesResult);

    void Softmax(const TensorType &semi, TensorType &dense, bool dropLastAixs = false);

    void ReshapeLocal(const TensorType &input, TensorType &output);

    void GetValidPoint(const TensorType &heatmap, Points &features);

    void nms_fast(const Points &input, Points &output);

    void ParsePoints(const TensorType &semi, Points &ptsNMS);

    void ParseDescribe(const TensorType &desc, const Points &points, Describes& describes);

    static void ShowImage(const std::string file, const Points &points, cv::Mat &image);

    static void Norm(TensorType& tensor);

    void WriteDescCSV(const Points &points ,const std::vector<std::vector<float>>);
};
#endif //RKV2_SUPER_SPRUN_H
