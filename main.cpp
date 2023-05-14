#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
// 不想写这个的话 那么就在前面打上cv的作用域
//using namespace cv;
//同理 不写using namespace的std也会出错
using namespace std;
#include <onnxruntime_cxx_api.h>

// vector库的导入
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
// MODNet的头文件导入头文件一般是定义一些其他的变量
#include "MODNet.h"
#include <string>
#include <chrono>


int main()
{
    // 添加记时操作
    auto start = std::chrono::high_resolution_clock::now();

    // 指定路径这个wstring是宽字符串 为什么不用string呢
    std::wstring model_path(L"C:\\Users\\Admin\\Desktop\\onnx\\mad_best.onnx");
    std::cout << "infer...." << std::endl;
    // 实例化MODNet对象  modnet并且指定model_path和需要的参数
    // 这里传入的是path 线程被设置为1 这里的vector是input_node_dims是NCHW 那么这个vector可以成为input_node_dims
    // 定义在modnet.h的line37
    MODNet modnet(model_path, 1, { 1, 3, 512, 512 });

    // 实例化对象之后的public方法predict 
    // 传入的参数为src_image 和 destination_image
    // 定义在ModNet.cpp的line 105
    modnet.predict_image("C:\\Users\\Admin\\Desktop\\onnx\\zidane.jpg", "C:\\Users\\Admin\\Desktop\\onnx\\zidane_matting51211111.jpg");


    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差并且输出
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "程序运行时间：" << duration.count() << " ms" << std::endl;

    return 0;
}