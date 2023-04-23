#include <unistd.h>
#include <iostream>
#include <string.h>
#include <queue>
#include <opencv2/highgui.hpp>
#include "rknn_fp.h"

#include "pthread.h"
//#include "rknn_api.h"
#include<sys/time.h>
#include <ctime>
#include <unistd.h>
#include <rknn_api.h>
#include <opencv2/imgproc.hpp>
#include "fstream"

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


int main(int argc, char **argv) {       // void main没有返回值，int main有返回值。
//    printf("开始了没有");
    int _cpu_id;
    int _n_input;
    int _n_output;
    //Inputs and Output sets
    rknn_context ctx;
    rknn_tensor_attr _input_attrs[1];
    rknn_tensor_attr _output_attrs[3];
    rknn_tensor_mem* _input_mems[1];
    rknn_tensor_mem* _output_mems[3];
    float* _output_buff[3];

    std::cout << "argc数量" << argc << std::endl;    // argc 是参数的个数， 第一个是工程的名字，第二第三是要输入的参数
    if (argc < 3) {                    // 判断语句  return 0 表示完成，1 表示真，-1表示 失败
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << std::endl;
        return -1;
    }

    int img_w = 640;
    int img_h = 400;
    const char *model_path = argv[1];  // 获取模型地址
    std::string imageFile = argv[2];   // 获取图片地址

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
    _n_output = 1;    // and this


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
//    if (core_mask == RKNN_NPU_CORE_2)
//    {
//        ret = rknn_set_core_mask(ctx, core_mask);
//        if(ret < 0)
//        {
//            printf("set NPU core_mask fail! ret=%d\n", ret);
//            exit(-1);
//        }
//    }
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
    memset(_output_attrs, 0, _n_output * sizeof(rknn_tensor_attr));
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
        _output_mems[i]  = rknn_create_mem(ctx, output_size);
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
    std::cout<<"111111111"<< width<<std::endl;
//    memcpy(_input_mems[0]->virt_addr, dst.data, width*_input_attrs[0].dims[1]*_input_attrs[0].dims[3]);
    memcpy(_input_mems[0]->virt_addr, dst.data, width*_input_attrs[0].dims[1]*_input_attrs[0].dims[3]);
//    memcpy(_input_mems[0]->virt_addr, dst.data, 640*400*1);

    //
//    TYPE *ptr = (TYPE *) malloc(dst.rows * dst.cols * dst.channels() * sizeof(TYPE));
//    memcpy(ptr, dst.data , dst.rows * dst.cols * dst.channels() * sizeof(TYPE));
    //

    // if(img.data) free(img.data);
//    unsigned char * buff = (unsigned char *)_input_mems[0]->virt_addr;

    // rknn inference
    ret = rknn_run(ctx, nullptr);
    std::cout<<"1111111112222"<<std::endl;

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
    }

    std::cout << "运行时间：= " << perf_run.run_duration << std::endl;

}

/*
 /root/super_rknn/rkv2_super.rknn
/root/super_rknn/rknn.jpg
 */
