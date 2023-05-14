#include "MODNet.h"
#include<string>
#include<iostream>
using namespace std;

// 实例化时的参数
// 第一个为onnx的路径 第二个为线程数 第三个为input_node_dims输入维度 其实没什么用 因为会被你main.cpp里面的给替代
MODNet::MODNet(std::wstring model_path, int num_threads = 1, std::vector<int64_t> input_node_dims = { 1, 3, 192, 192 }) {
	// 赋值操作 将input_node_dims赋值给input_node_dims_
	input_node_dims_ = input_node_dims;

	// 这个写法是新写法 
	// 计算输入张量（input tensor）和输出张量（output tensor）的大小
	// 等价于传统for循环中的 "for (size_t j = 0; j < input_node_dims_.size(); ++j) { int64_t i = input_node_dims_[j];
	for (int64_t i : input_node_dims_) {
		input_tensor_size_ *= i;
		out_tensor_size_ *= i;
	}

	// 得到输入和输出维度大小
	//std::cout << input_tensor_size_ << std::endl;
	// 设置线程数
	session_options_.SetIntraOpNumThreads(num_threads);
	// 设置参数的开关
	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	// 这段代码尝试创建一个 ONNX 模型的会话（session）对象，并使用给定的环境(env_)、模型文件路径(model_path)和会话选项(session_options_)进行初始化
	// 如果出错则会捕获catch的内容
	try {
		session_ = Ort::Session(env_, model_path.c_str(), session_options_);
	}
	catch (...) {
	}

	// 定义了一个 ort::AllocatorWithDefaultOptions 类型的变量 allocator，该变量用于分配内存用于 ONNX 运行时（ORT）中的张量/缓冲区等数据结构
	Ort::AllocatorWithDefaultOptions allocator;
	//获取输入name
	const char* input_name = session_.GetInputName(0, allocator);
	input_node_names_ = { input_name };
	//std::cout << "input name:" << input_name << std::endl;
	const char* output_name = session_.GetOutputName(0, allocator);
	out_node_names_ = { output_name };
	//std::cout << "output name:" << output_name << std::endl;
}

// 传入的是刚刚resize之后的dst_image 
// 并且也是经过float之后的矩阵
cv::Mat MODNet::normalize(cv::Mat& image) {

	// 这里的image = dst_image 

	// 定义了两个变量channels 和 normalized_image，它们都是 std::vector 类型的对象
	// 其中vector存放的类型为Mat类型的
	std::vector<cv::Mat> channels, normalized_image;

	// 利用cv的split操作将image的三个通道分割出来并且复制到channel这个mat类型的vector中
	cv::split(image, channels);

	// 定义三个mat类型的变量bgr
	cv::Mat r, g, b;
	// 因为读入的时候就是按照bgr的方式存储的所以索引0是b 1是g 2是r

	b = channels.at(0);
	g = channels.at(1);
	r = channels.at(2);
	// 归一化的操作！
	// 化简一下就是(b-127.5)/127.5
	// 255.就是浮点的操作
	b = (b / 255. - 0.5) / 0.5;
	g = (g / 255. - 0.5) / 0.5;
	r = (r / 255. - 0.5) / 0.5;

	// 将归一化后的bgr加入到normalized_image里面
	// 注意这里的操作这里push_back的时候他push的是rgb！！！
	normalized_image.push_back(r);
	normalized_image.push_back(g);
	normalized_image.push_back(b);

	// 创建一个新变量并且长和宽都是512 因为这里的image就是传入进来的dst_image 并且创建的为cv32f为浮点类型的
	// 这里没有指定通道数 只是指定长宽以及数值为浮点数类型
	cv::Mat out = cv::Mat(image.rows, image.cols, CV_32F);
	// merge操作如何进行有什么作用 为什么不能直接赋值呢
	// merge操作的含义是将多个单通道合成一个多通道的图片
	// normalization_image为归一化之后的矩阵 out为输出
	cv::merge(normalized_image, out);
	return out;
}

/*
* preprocess: resize -> normalize
*/
cv::Mat MODNet::preprocess(cv::Mat image) {
	
	// 这里的image = src_image 

	// 读入图片的h和w
	image_h = image.rows;
	image_w = image.cols;
	// 定义三个mat变量 dst dst_float 以及normalize_image
	// 第三个应该是归一化之后的图片
	cv::Mat dst, dst_float, normalized_image;
	// int(input_node_dims_[3]) = 512 
	// int(input_node_dims_[3]) = 512
	// 为什么非要reszie成512的呢？
	// 这里是不是之前转换成onnx的那个思想 我放一个tensor进去让他了解其信息
	// 后面的两个参数0 第一个代表插值使用的是双线插值算法
	// 第二个参数0代表不包含横纵比 也就是不保持横纵比
	cv::resize(image, dst, cv::Size(int(input_node_dims_[3]), int(input_node_dims_[2])), 0, 0);
	// 上面这一行输出的是dst
	// 将dst转化为浮点类型的 dst_float后面的参数为cv_32f
	dst.convertTo(dst_float, CV_32F);
	// 转化为浮点之后就需要normazitaion操作 
	// 这个地方是我上次做不出来的！
	normalized_image = normalize(dst_float);
	// 做完归一化操作之后返回normalizedimage
	return normalized_image;
}

/*
* 前处理 推理 后处理
* postprocess: preprocessed image -> infer -> postprocess
*/
cv::Mat MODNet::predict_image(cv::Mat& src) {

	// 给读入的src_image进行前处理
	// 这里是经过前处理后的preprocess image 也就是前处理的图片
	cv::Mat preprocessed_image = preprocess(src);

	// 这里可能比较重要！！
	// cv::dnn::blobFromImage函数是用于将图像转换为深度学习模型所需的Blob格式的函数
	// 这个操作的目的是将输入图像转换为适合于深度学习模型输入的形式，以便进行下一步的预测或推理 类型create_tensor操作？？？

	// 第一个参数是要转换的图像为预处理之后的preprocess_image
	// 第二个参数是指定Blob对象的个数，这里为1 那后面得到的维度tensor为(1,3,512,512) 其中3可能是其通道数
	// 第三个参数是指定Blob对象的大小，即（width，height）。它的值由input_node_dims_数组中的第3和第2个元素决定，
	// 这两个元素分别表示网络的输入图像的宽度和高度为512和512
	// 第四个参数是指定通道的平均值，这里设置为（0，0，0），表示不进行减均值操作。
	// 第五个参数是一个布尔值，用于控制Blob对象是否需要缩放。如果设置为false，则不进行缩放；
	// 如果设置为true，则将图像缩小到指定大小，并对其进行插值，以便适应网络的输入尺寸。
	// 第六个参数也是一个布尔值，用于控制Blob对象是否需要交换通道顺序。如果设置为true，则交换通道，将BGR格式转换为RGB格式
	cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size(int(input_node_dims_[3]), int(input_node_dims_[2])), cv::Scalar(0, 0, 0), false, true);
	
	/*std::cout << "blob's shape" << blob.cols << " " << blob.rows << " " << blob.channels() << endl;*/
	
	//std::cout << "load image success." << std::endl;
	// create input tensor

	// API中的memoryinfo类和createcpu静态方法来创建一个在CPU上进行内存分配的内存信息对象
	// 这个代码没怎么看懂
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	// 有个疑问这里的input_node_dims_.data(), input_node_dims_.size()哪来的 solved!
	// 上面的MODNet::MODNet实例化的时候会进行赋值  input_node_dims_ = input_node_dims;

	// 下面一行代码有多个参数
	// 其中Ort::Value::CreateTensor是onnxruntime的API 创建一个float类型的Tensor并且加入input_tensor_的尾部
	// 其中参数1 memory_info参数创建了一个新的ORT内存块（ORT Memory Block）
	// 参数2和参数3一起 blob.ptr<float>(), blob.total()创建了一个包含所有元素的一维数组，该数组存储了该张量中所有元素的值，即张量数据
	// 参数4和参数5创建一个包含输入张量的维度信息的一维数组 
	// 简单理解就是一个是给输入数据一个是给输入维度
	input_tensors_.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_node_dims_.data(), input_node_dims_.size()));
	//得到input_tensor_


	// 这个是在modnet.h中定义了一个名为session_的变量，类型为ort::session的指针，并将其初始化为nullptr，表示当前该指针不指向任何有效的内存地址。
	// ort::runoptions{ nullptr }：表示使用默认的运行选项。
	// input_node_names_.data()和input_tensors_.data()：表示输入节点名称和输入张量数据
	// nput_node_names_.data()返回的是一个指向input_node_names_数组第一个元素的指针
	// input_tensors_.data()返回的是一个指向input_tensors_数组第一个元素的指针
	// input_node_names_.size()：表示输入节点的数量。
	// out_node_names_.data()和out_node_names_.size()：表示输出节点的名称和数量均返回的是指针和上面一样
	
	std::vector<Ort::Value> output_tensors_ = session_.Run(
		Ort::RunOptions{ nullptr },
		input_node_names_.data(),
		input_tensors_.data(),
		input_node_names_.size(),
		out_node_names_.data(),
		out_node_names_.size()
	);

	// floatarr 指向了第一个张量的浮点数据缓冲区，并且可以用于读取或写入数据
	// 简单讲就是可以让floatarr指向刚刚生成的outputtensor
	float* floatarr = output_tensors_[0].GetTensorMutableData<float>();

	// decoder 解码
	// 生成一个mat类型的mask掩码变量定义为全0
	cv::Mat mask = cv::Mat::zeros(static_cast<int>(input_node_dims_[2]), static_cast<int>(input_node_dims_[3]), CV_8UC1);

	// 这循环代码要好好看看
	// "uchar"是"C++语言中的一种数据类型，代表unsigned char(无符号字符)，其范围在0到255之间
	for (int i{ 0 }; i < static_cast<int>(input_node_dims_[2]); i++) {
		for (int j{ 0 }; j < static_cast<int>(input_node_dims_[3]); ++j) {
			// 将floatarr数组中的元素值与0.5进行比较，如果大于0.5则将其转换为uchar类型的1，否则转换为uchar类型的0
			// 并且赋值给mask对应位置上的 其实就是顺序来赋值一开始i=0 j=0~511
			mask.at<uchar>(i, j) = static_cast<uchar>(floatarr[i * static_cast<int>(input_node_dims_[3]) + j] > 0.5);

		}
	}

	// image_h = image.rows
	// 将mask也重新resize一下 参数之前讲过了 也是0，0  
	cv::resize(mask, mask, cv::Size(image_w, image_h), 0, 0);
	input_tensors_.clear();
	//cv::imshow("mask", mask);
	//cv::waitKey(0);
	return mask;
}



void MODNet::predict_image(const std::string& src_path, const std::string& dst_path) {
	cv::Mat image = cv::imread(src_path);
	// 得到mask
	cv::Mat mask = predict_image(image);
	cv::Mat predict_image;
	// 这个是与操作 为什么是要输入两个image

	// 两个输入都是image则可以代表为为想截取自己的mask
	cv::bitwise_and(image, image, predict_image, mask = mask);
	// 保持预测图片
	cv::imwrite(dst_path, predict_image);
	// 我这里可以多保存一个mask
	std::cout << "predict image over" << std::endl;
}




// 这个目前不用看
void MODNet::predict_camera() {
	cv::Mat frame;
	cv::VideoCapture cap;
	int deviceID{ 0 };
	int apiID{ cv::CAP_ANY };
	cap.open(deviceID, apiID);
	if (!cap.isOpened()) {
		std::cout << "Error, cannot open camera!" << std::endl;
		return;
	}
	//--- GRAB AND WRITE LOOP
	std::cout << "Start grabbing" << std::endl << "Press any key to terminate" << std::endl;
	int count{ 0 };
	clock_t start{ clock() }, end;
	double fps{ 0 };
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			std::cout << "ERROR! blank frame grabbed" << std::endl;
			break;
		}
		cv::Mat mask = predict_image(frame);
		cv::Mat segFrame;
		cv::bitwise_and(frame, frame, segFrame, mask = mask);
		// fps
		end = clock();
		++count;
		fps = count / (float(end - start) / CLOCKS_PER_SEC);
		if (count >= 100) {
			count = 0;
			start = clock();
		}
		std::cout << fps << "  " << count << "   " << end - start << std::endl;
		//设置绘制文本的相关参数
		std::string text{ std::to_string(fps) };
		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 1;
		int thickness = 2;
		int baseline;
		cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

		//将文本框居中绘制
		cv::Point origin;
		origin.x = 20;
		origin.y = 20;
		cv::putText(segFrame, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

		// show live and wait for a key with timeout long enough to show images
		cv::imshow("Live", segFrame);
		if (cv::waitKey(5) >= 0)
			break;

	}
	cap.release();
	cv::destroyWindow("Live");

	return;
}
