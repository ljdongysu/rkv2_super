#include <unistd.h>
#include <iostream>
#include <string.h>
#include <queue>
#include <opencv2/highgui.hpp>
#include "rknn_fp.h"
#include "SpRun.h"
#include "pthread.h"
//#include "rknn_api.h"
#include<sys/time.h>
#include <ctime>
#include <unistd.h>
#include <rknn_api.h>
#include <opencv2/imgproc.hpp>
#include "fstream"
#include "timer.h"

#define RAND_INT(a, b) (rand() % ((b)-(a)+1))+ (a)
// 读取rknn模型输入/输出属性
void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
    return;
}

using Points = std::vector<FeaturePoint>;
using Describes = std::vector<std::vector<float>>;
int RunSuperPoint( std::string modelFile, std::string file1, Points &points, Describes &describes);

struct Array
{
    float* data = nullptr;
    std::vector<uint> dims{};

    ~Array()
    {
        Release();
    }
    void Release()
    {
        delete[] data;
    }
};

void Reshape(Array &input, TensorType &output)
{
    int featureSize = input.dims[0] * input.dims[1];

    output.resize(input.dims[2], std::vector<float>(featureSize, 0));
    for (int i = 0; i < input.dims[2]; ++i)
    {
        int id = featureSize * i;
        memcpy(output.at(i).data(), input.data + id, featureSize * sizeof(float));
        continue;
//         whc -> chw
        for (int j = 0; j < input.dims[1]; ++j)
        {
            for (int k = 0; k < input.dims[0]; ++k)
            {
                int idSrc = k * input.dims[2] * input.dims[1] + j * input.dims[2] + i;
                int idDst = j * input.dims[1] + k;
                output.at(i).at(idDst) = input.data[idSrc];
            }
        }
    }
}


cv::Mat ConvertVectorMat(const TensorType &descResultLeft)
{
//    cv::Mat imageResult(0, descResultLeft[0].size(), cv::DataType<float>::type);
//    for (int i = 0; i < descResultLeft.size(); ++i)
//    {
//        cv::Mat Sample(1, descResultLeft[0].size(), cv::DataType<float>::type, descResultLeft[i].data());
//        imageResult.push_back(Sample);
//    }

    if (descResultLeft.size() > 0)
    {
        cv::Mat imageResult(descResultLeft.size(), descResultLeft[0].size(), CV_32F);
        for (int i = 0; i < descResultLeft.size(); ++i)
            imageResult.row(i) = cv::Mat(descResultLeft[i]).t();

        return imageResult;
    }
    else
    {
        std::cout << "describes.size() == 0" << std::endl;
        cv::Mat image;
        return image;
    }
}


void ShowMatch(cv::Mat imgLeft, cv::Mat imgRight
        , const std::string extendName
        , const Points &pointsLeft, const Points &pointsRight
        , const std::vector<cv::DMatch> matches)
{
    cv::Mat imageLeftRight;
    int drawPoints = 0;
    const int W = imgLeft.cols, H = imgLeft.rows;

    SpRun::ShowImage("", pointsLeft, imgLeft);
    SpRun::ShowImage("", pointsRight, imgRight);
    cv::hconcat(imgLeft, imgRight, imageLeftRight);

    for (int i = 0; i < matches.size(); ++i)
    {
        cv::DMatch dMatch;
        dMatch = matches[i];
        const auto &left = pointsLeft[dMatch.queryIdx];
        const auto &right = pointsRight[dMatch.trainIdx];
        cv::Point leftCV = cv::Point(left.x, left.y);
        cv::Point rightCV = cv::Point(right.x + imgLeft.cols, right.y);

        if (abs(left.y - right.y) > 20) continue;
        drawPoints += 1;

        cv::line(imageLeftRight, leftCV, rightCV, cv::Scalar(RAND_INT(0, 255), RAND_INT(0, 255)
                , RAND_INT(0, 255)), 1, cv::LINE_AA);
//        cv::circle(imageLeftRight, leftCV, 2, GetColor(left.confidence), -1, 8, 0);
//        cv::circle(imageLeftRight, rightCV, 2, GetColor(right.confidence), -1, 8, 0);
    }

    cv::putText(imageLeftRight, "match " + std::to_string(drawPoints) + "", cv::Point(50, H - 30)
            , cv::FONT_HERSHEY_COMPLEX
            , 1, cv::Scalar(0, 255, 0), 2);

    std::string outputFile;
    char *p = getcwd(NULL, 0);
    outputFile = std::string(p) + "/" + "result" + "/result_" + extendName + ".jpg";
    cv::imwrite(outputFile, imageLeftRight);

    std::cout << outputFile << std::endl;
}


std::vector<cv::DMatch>
Match(const Points &semiResultLeft, const Points &semiResultRight, const TensorType &descResultLeft
        , const TensorType &descResultRight, std::vector<cv::DMatch> &matches)
{
    cv::BFMatcher matcher;
    cv::Mat descriptorLeft = ConvertVectorMat(descResultLeft);
    cv::Mat descriptorRight = ConvertVectorMat(descResultRight);

    matcher.match(descriptorLeft, descriptorRight, matches);
    std::cout << "point-L: " << semiResultLeft.size() << " point-R: "
              << semiResultRight.size() << " , matches: " << matches.size() << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "argc数量" << argc << std::endl;    // argc 是参数的个数， 第一个是工程的名字，第二第三是要输入的参数
    if (argc < 3) {                    // 判断语句  return 0 表示完成，1 表示真，-1表示 失败
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << std::endl;
        return -1;
    }
    if (argc == 3)
    {
        Points points;
        Describes describes;
        RunSuperPoint( argv[1], argv[2], points, describes);
        cv::Mat image_in = cv::imread(argv[2]);
        SpRun::ShowImage(argv[2], points, image_in);
    }
    else if (argc == 4)
    {
        Points pointsL, pointsR;
        Describes describesL, describesR;
        std::vector<cv::DMatch> matches;

        cv::Mat imageL, imageR;
        imageL = cv::imread(argv[2]);
        imageR = cv::imread(argv[3]);

        RunSuperPoint( argv[1], argv[2], pointsL, describesL);
        RunSuperPoint( argv[1], argv[3], pointsR, describesR);
//        SpRun::ShowImage(argv[2], pointsL, image_in);

        Match(pointsL, pointsR, describesL, describesR, matches);
        ShowMatch(imageL, imageR, "tttt", pointsL, pointsR, matches);

    }
}

int RunSuperPoint( std::string modelFile, std::string imageFile, Points &points, Describes &describes)
{
//    printf("开始了没有");
    int _cpu_id;
    int _n_input;
    int _n_output;
    //Inputs and Output sets
    rknn_context ctx;
    rknn_tensor_attr _input_attrs[1];
    rknn_tensor_attr _output_attrs[2];
    rknn_tensor_mem* _input_mems[1];
    rknn_tensor_mem* _output_mems[2];
    float* _output_buff[2];

    int img_w = 640;
    int img_h = 400;
    const char *model_path = modelFile.c_str();  // 获取模型地址

    // rknn_fp 的参数，
    /*
     char model_path
     int cpuid
     int n_input
     int n_output
    */
    int cpuid=2;

    int ret = 0;
    cpu_set_t mask;

    CPU_ZERO(&mask);
    CPU_SET(1, &mask);

    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)    //
        std::cerr << "set thread affinity failed" << std::endl;


    printf("Bind NPU process on CPU %d\n", cpuid);

    _cpu_id   = 1;
//    _n_input  = n_input;     // 这个输入是什么，看不明白
    _n_input  = 1;     // 这个输入是什么，看不明白
//    _n_output = n_output;    // and this
    _n_output = 2;    // and this

    // Load model

    FILE *fp = fopen(model_path, "rb");   // 读取模型定义指针  FILE类型(文件结构)
    if(fp == NULL) {
        printf("fopen %s fail!\n", model_path);
        exit(-1);
    }
    // 文件的长度(单位字节)
    fseek(fp, 0, SEEK_END);    // 调节文件指针位置
    int model_len = ftell(fp);   // 指向文件的当前读写位置
    // 创建一个存储空间model且读入
    void *model = malloc(model_len);   // memory allocation 动态内存分配
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {   // fread 用来读取文件，只能读取指针往后的内容
        printf("fread %s fail!\n", model_path);
        free(model);
        exit(-1);
    }

    // ret = rknn_init(&ctx, model, m odel_len, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
//    init_runtime()
    ret = rknn_init(&ctx, model, model_len, 0, NULL);   // 模型初始化
    if(ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(-1);
    }

    // rknn_sdk_version
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    printf("api version: %s\n", version.api_version);
    printf("driver version: %s\n", version.drv_version);

    // rknn inputs
    printf("input tensors:\n");
    memset(_input_attrs, 0, _n_input * sizeof(rknn_tensor_attr));    // 初始化函数 memory set 作用是将某一块内存中的全部设置为指定的值
    for (uint32_t i = 0; i < _n_input; i++) {
        _input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error! ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&_input_attrs[i]);
    }

    // Create input tensor memory
    rknn_tensor_type   input_type   = RKNN_TENSOR_UINT8; // default input type is int8 (normalize and quantize need compute in outside)
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC; // default fmt is NHWC, npu only support NHWC in zero copy mode
//    rknn_tensor_format input_layout = RKNN_TENSOR_NCHW; // default fmt is NHWC, npu only support NHWC in zero copy mode
    _input_attrs[0].type = input_type;
    _input_attrs[0].fmt = input_layout;
    _input_mems[0] = rknn_create_mem(ctx, _input_attrs[0].size_with_stride);

    // rknn outputs
    printf("output tensors:\n");
    std::cout <<"sizeof(rknn_tensor_attr: " << sizeof(rknn_tensor_attr) << std::endl;
    memset(_output_attrs, 0, _n_output * sizeof(rknn_tensor_attr));

    for (uint32_t i = 0; i < _n_output; i++)
    {
        std::cout <<"_output_attrs[i]->n_dims: " << _output_attrs[i].n_dims << ", n_elems: "
        << _output_attrs[i].n_elems << "size: " << _output_attrs[i].size << std::endl;
    }

    for (uint32_t i = 0; i < _n_output; i++) {
        _output_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(_output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&_output_attrs[i]);
    }

    // Create output tensor memory
    for (uint32_t i = 0; i < _n_output; ++i) {
        // default output type is depend on model, this require float32 to compute top5
        // allocate float32 output tensor
        int output_size = _output_attrs[i].n_elems * sizeof(float);
        std::cout << "output_size: " << output_size << std::endl;
        _output_mems[i]  = rknn_create_mem(ctx, output_size);
        std::cout << "_output_mems[i].size: " << _output_mems[i]->size << std::endl;
    }

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx, _input_mems[0], &_input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        exit(-1);
    }

    // Set output tensor memory
    for (uint32_t i = 0; i < _n_output; ++i) {
        // default output type is depend on model, this require float32 to compute top5
        _output_attrs[i].type = RKNN_TENSOR_FLOAT32;
        // set output memory and attribute
        ret = rknn_set_io_mem(ctx, _output_mems[i], &_output_attrs[i]);
        std::cout <<"dims: " << _output_attrs[i].n_elems << std::endl;
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            exit(-1);
        }
    }

    //  前面是跑成功的  **********************************************************************************************
    cv::Mat dst;    // cv::Mat是OpenCV定义的用于表示任意维度的稠密数组，OpenCV使用它来存储和传递图像
    cv::Mat image_in = cv::imread(imageFile);

    cv::resize(image_in, dst, cv::Size(img_w, img_h));

    using TYPE = uint8_t;
    cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    int width  = _input_attrs[0].dims[2];  // 2
    Timer timerRun;

    //    memcpy(_input_mems[0]->virt_addr, dst.data, width*_input_attrs[0].dims[1]*_input_attrs[0].dims[3]);
    memcpy(_input_mems[0]->virt_addr, dst.data, width*_input_attrs[0].dims[1]*_input_attrs[0].dims[3]);
//    memcpy(_input_mems[0]->virt_addr, dst.data, 640*400*1);
    //
//    TYPE *ptr = (TYPE *) malloc(dst.rows * dst.cols * dst.channels() * sizeof(TYPE));
//    memcpy(ptr, dst.data , dst.rows * dst.cols * dst.channels() * sizeof(TYPE));
    //

    // if(img.data) free(img.data);
//    unsigned char * buff = (unsigned char *)_input_mems[0]->virt_addr;
    Timer timerInterface;

    // rknn inference
    ret = rknn_run(ctx, nullptr);
    timerInterface.Timing("interface", true);
    std::cout<<ret<<std::endl;
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    // query1: inference time
    rknn_perf_run perf_run;
    ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run,sizeof(perf_run));

    for(int i=0;i<_n_output;i++){
        _output_buff[i] = (float*)_output_mems[i]->virt_addr;
//        std::cout << "output.size[" << i << "]: " << _output_mems[i]->size << std::endl;
//        PrintMatrix(_output_buff[i], 80);
    }

//    std::cout << "运行时间：= " << perf_run.run_duration << std::endl;

    Timer timer, timerAll;

    Array semi, coarse_desc;
    semi.data = (float *) malloc(80 * 50 * 65 * sizeof (float));
    semi.dims = {80, 50, 65, 1};

    coarse_desc.data = (float *) malloc(80 * 50 * 256 * sizeof (float));
    coarse_desc.dims = {80, 50, 256, 1};

    memcpy(semi.data, _output_buff[0],80 * 50 * 65 * sizeof (float));
    memcpy(coarse_desc.data, _output_buff[1],80 * 50 * 256 * sizeof (float));

    ret = rknn_destroy_mem(ctx, _input_mems[0]);
    ret &= rknn_destroy_mem(ctx, _output_mems[0]);
    ret &= rknn_destroy_mem(ctx, _output_mems[1]);
    if(ret < 0) {
        printf("rknn_destroy_mem fail! ret=%d\n", ret);
        return -1;
    }
    ret = rknn_destroy(ctx);
    if(ret < 0) {
        printf("rknn_destroy fail! ret=%d\n", ret);
        return -1;
    }
    long long height = 400;
    TensorType semiResult, descResult;  // (65, 50*80) (256, 50*80)


    Reshape(semi, semiResult);
    Reshape(coarse_desc, descResult);

    timer.Timing("reshape output", true);
    SpRun::Norm(descResult);
//    PrintMatrix(descResult[0].data(), 80);
    timer.Timing("normal.", true);

//    PrintMatrix(semiResult[0].data(), 80);
    int outpixNum = 80 * 50;

    SpRun *sp = new SpRun(coarse_desc.dims[2], outpixNum, height, width);
    sp->calc(semiResult, descResult, image_in, points, describes);
    timerAll.Timing("post process", true);
    timerRun.Timing("SuperPoint", true);

}

