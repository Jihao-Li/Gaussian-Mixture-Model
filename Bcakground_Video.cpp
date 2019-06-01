
/******************头文件*******************/
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

/******************高斯混合模型参数*******************/
#define g_gaussianNum  3      // 混合的高斯函数的个数，一般取值在3到5之间，经验证取3个效果较好
#define g_init_StdDev  6      // 高斯函数的标准差
#define g_alpha  0.01         // 学习率
#define g_deviationMult  2.5  // 标准差的倍数
#define g_threshold  0.75     // 权重阈值，一般取值在0.25到0.75之间
double g_rho = 0.0;           // 用于更新权重、均值、方差的参数，初始化为0
int g_trainFrames = 0;        // 用于训练高斯混合模型用的帧数

/******************屏幕要显示的窗口的名称*******************/
std::string videoPath = "监控视频\\Walk3.mpeg";
std::string foreName = "前景";
std::string backName = "背景";
std::string videoName = "原视频";

/******************视频参数*******************/
double g_rate;   // 帧率
int g_frames;    // 总帧数
int g_height;    // 视频的高度
int g_width;     // 视频的宽度

/******************函数声明*******************/
void trainGMM( cv::Mat gaussianMean[ g_gaussianNum ], cv::Mat gaussianStdDev[ g_gaussianNum ], cv::Mat gaussianWeight[ g_gaussianNum ] );      // 训练高斯混合模型
//void trainGMM_version2();      // 训练高斯混合模型


/******************主函数********************/
int main()
{
	/*******************变量定义************************/
	int countGuass = 0;       // 对高斯分布的个数进行计数
	int height = 0;           // 帧高的计数变量
	int width = 0;            // 帧宽的计数变量
	int sortTemp = 0;         // 用于排序的临时变量
	float valueTemp = 0.0;    // 排序时的数值交换的临时变量；每个高斯分布权重的累加和临时变量
	int rank[ g_gaussianNum ] = { 0 };   // 高斯分布会经过排序，rank数组是原始高斯分布的索引
	int indexTemp = 0;        // 前多少个高斯分布可以表示背景
	bool match = 0;           // 判断当前帧的像素是否是背景，match = 1则为背景，match = 0则为前景
	char savePath[ 30 ];      // 保存目标检测结果 
	int countFrames = 0;      // 保存帧的计数变量


	/*******************高斯混合模型的训练***********************/
	cv::Mat gaussianMean[ g_gaussianNum ];              // 创建4个高斯分布的均值矩阵，g_gaussianNum为4
	cv::Mat gaussianStdDev[ g_gaussianNum ];            // 创建4个高斯分布的标准差矩阵，g_gaussianNum为4
	// 初始化均值时，第1个高斯分布的均值为第一帧的像素值
	// 初始化标准差时，所有高斯分布赋予相等的适当标准差即可，由经验得6比较合适
	// 若使用彩色图像时，GMM方法假设RGB三个通道的像素互不相关，所以协方差矩阵为一个对角阵，可以简化为一个向量处理
	// OpenCV中的三通道排列顺序为BGR
	cv::Mat gaussianWeight[ g_gaussianNum ];    // 创建4个权重矩阵
	trainGMM( gaussianMean, gaussianStdDev, gaussianWeight );  
	// 使用高斯混合模型对背景建模，通过训练，获取高斯分布的均值、标准差、权重；用于训练的帧也会被存储


	/*******************计算权重和标准差的商**********************/
	cv::Mat divWeiStDe[ g_gaussianNum ];   // 创建4个存储权重和标准差的商的矩阵
	for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
	{
		cv::divide( gaussianWeight[ countGuass ], gaussianStdDev[ countGuass ], divWeiStDe[ countGuass ] ); 
		// 计算各个高斯分布和标准差的商
	}
	countGuass = 0;        // 高斯分布的计数变量清零
	
	/***********************进行前景检测*************************/
	cv::Mat videoFrame, videoTempFrame;    // 存储视频帧的Mat对象
	cv::Mat subMat[ g_gaussianNum ];       // 存储当前帧和各个高斯分布的均值之差
	cv::Mat backTemp( g_height, g_width, CV_32FC1 );   // 背景图片像素值的临时变量，但该变量是float型的
	cv::Mat fore( g_height, g_width, CV_8UC1 );   // 前景灰度图片
	cv::Mat back( g_height, g_width, CV_8UC1 );   // 背景灰度图片
	cv::Mat element;                              // 腐蚀和膨胀的结构元素
	element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 5, 5 ) );

	cv::VideoCapture cap( videoPath );     // 打开视频
	if( !cap.isOpened() )                  // 判断视频是否被打开
	{
		std::cout << "视频打开失败！" << std::endl;
		return 1;
	}
	cv::namedWindow( videoName, CV_WINDOW_NORMAL );   // 创建原视频窗口
	cv::namedWindow( backName, CV_WINDOW_NORMAL );    // 创建背景窗口
	cv::namedWindow( foreName, CV_WINDOW_NORMAL );    // 创建前景窗口
	
	std::cout << "模型训练完毕，正在对每一帧视频进行检测，如不想完整观看视频，请按ESC键退出。" << std::endl;
	while( cap.read( videoTempFrame ) )     // 开始读取视频帧，直到视频播放完毕
	{
		cv::imshow( videoName, videoTempFrame );                   // 显示读取的该帧视频
		cv::cvtColor( videoTempFrame, videoFrame, CV_BGR2GRAY );   // 转换到灰度空间
		videoFrame.convertTo( videoFrame, CV_32FC1 );   // 必须要将videoFrame转换为float类型，否则指针无法正常工作

		// 计算当前帧和各个高斯分布均值之差
		for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
		{
			cv::subtract( videoFrame, gaussianMean[ countGuass ], subMat[ countGuass ] );
			subMat[ countGuass ] = cv::abs( subMat[ countGuass ] );
		}

		// 对权重和标准差的商进行降序排序
		for( height = 0; height < g_height; height++ )     // 遍历高度，即矩阵的行
		{
			for( width = 0; width < g_width; width++ )     // 遍历宽度，即矩阵的列
			{
				for( countGuass = 1; countGuass < g_gaussianNum; countGuass++ )   // 遍历各个高斯分布
				{
					for( sortTemp = 0; sortTemp < countGuass; sortTemp++ )        // 冒泡法排序
					{
						if( ( *divWeiStDe[ countGuass ].ptr<float>( height, width ) ) > ( *divWeiStDe[ sortTemp ].ptr<float>( height, width ) ) )
						{
							// 交换divWeiStDe内变量的值，大的放到前边，小的放到后边
							valueTemp = *divWeiStDe[ sortTemp ].ptr<float>( height, width );
							*divWeiStDe[ sortTemp ].ptr<float>( height, width ) = *divWeiStDe[ countGuass ].ptr<float>( height, width );
							*divWeiStDe[ countGuass ].ptr<float>( height, width ) = valueTemp;

							// 交换索引值
							rank[ sortTemp ] = countGuass;
							rank[ countGuass ] = sortTemp;
						}
					}
				}
				countGuass = 0;      // 高斯分布的计数变量清零
				// 至此，对于一个像素值的高斯分布的排序完毕

				// 判断多少个高斯分布可以作为背景像素的模型
				valueTemp = 0.0;     // 临时变量清零
				*backTemp.ptr<float>( height, width ) = 0.0;   // 背景图片的临时像素值清零
				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
				{
					*backTemp.ptr<float>( height, width ) = ( *backTemp.ptr<float>( height, width ) ) + ( *gaussianMean[ countGuass ].ptr<float>( height, width ) ) * ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) );
					// 背景的像素值即该点各个高斯分布的权重和均值乘积的累加和
					valueTemp = valueTemp + ( *gaussianWeight[ rank[ countGuass ] ].ptr<float>( height, width ) );
					if( valueTemp > g_threshold )
					{
						indexTemp = countGuass;    // indexTemp表示前几个高斯分布可以表示背景
						break;                     // 跳出循环 
					}
				}
				countGuass = 0;     // 高斯分布的计数变量清零
				*back.ptr<uchar>( height, width ) = *backTemp.ptr<float>( height, width );  // 由float背景转换为uchar背景，之后就可以用imshow显示了

				// 判断当前帧是否和前indexTemp个高斯分布之一匹配
				match = 0;           // 默认为前景
				for( countGuass = 0; countGuass <= indexTemp; countGuass++ )
				{
					if( ( *subMat[ countGuass ].ptr<float>( height, width ) ) < ( *gaussianStdDev[ countGuass ].ptr<float>( height, width ) ) * ( g_deviationMult ) )
					{
						match = 1;  // 当前帧像素值和各个高斯分布有一个匹配，则为背景
						break;      // 跳出循环
					}
				}

				// 更新fore图片中的像素值
				if( match == 0 )    // 如果为前景
					*fore.ptr<uchar>( height, width ) = 255;  // 前景图片中，该点最亮
				else                // 如果为背景
					*fore.ptr<uchar>( height, width ) = 0;    // 前景图片中，该点最暗
			}
		}

		cv::erode( fore, fore, element );    // 腐蚀参数分别是：输入帧，输出帧，结构元素
		cv::dilate( fore, fore, element );   // 膨胀参数分别是：输入帧， 输出帧，结构元素

		cv::imshow( foreName, fore );        // 显示前景图片
		cv::imshow( backName, back );        // 显示背景图片

		sprintf( savePath, "%s%d%s", "前景帧\\frame", countFrames++, ".jpg" );
		cv::imwrite( savePath, fore );       // 保存前景帧

		if( cv::waitKey( 100 / g_rate ) == 27 )   // 帧率可以比原视频的帧率快一点，程序有其他语句也要占用时间
		{
			break;    // 如果按下ESC键，则退出播放
		}
	}

	cap.release();                    // 关闭视频，释放内存
	cv::destroyWindow( videoName );   // 销毁原视频窗口
	cv::destroyWindow( backName );    // 销毁背景窗口
	cv::destroyWindow( foreName );    // 销毁前景窗口

	system( "pause" );  // 让屏幕显示按任意键继续
}


/**********************函数内容**************************/
void trainGMM( cv::Mat gaussianMean[ g_gaussianNum ], cv::Mat gaussianStdDev[ g_gaussianNum ], cv::Mat gaussianWeight[ g_gaussianNum ] )
{
	/***************变量定义********************/
	int countGuass = 0;      // 对高斯分布的个数进行计数
	int countFrames = 0;     // 对训练的帧进行计数
	int height = 0;          // 帧高的计数变量
	int width = 0;           // 帧宽的计数变量
	bool matchFlag = 0;      // 更新高斯分布权重的标志
	double tempWeight = 0.0; // 临时变量，存储权重累加和
	int minIndex = 0;        // 高斯分布权重最小者的索引
	char fileName[ 30 ] ;    // 存储训练帧的路径名和文件名字符串


	/***************打开视频并读取视频信息****************/
	cv::VideoCapture trainCap( videoPath );   // 读取视频
	if( !trainCap.isOpened() )                // 判断视频是否被打开
	{
		std::cout << "视频打开失败！" << std::endl;
		//return 1;
	}
	g_rate = trainCap.get( CV_CAP_PROP_FPS );             // 读取帧率
	g_frames = trainCap.get( CV_CAP_PROP_FRAME_COUNT );   // 读取帧总数
	g_height = trainCap.get( CV_CAP_PROP_FRAME_HEIGHT );  // 读取视频的高度
	g_width = trainCap.get( CV_CAP_PROP_FRAME_WIDTH );    // 读取视频的宽度
	g_trainFrames = 0.01 * ( g_frames + 1 );              
	// 将前几帧用于训练高斯混合模型，g_frames是从0开始的，所以要加1
	// 这个数字不要过大，0.01比较合适；否则检测效果不好，可能是一种过拟合

	// 输出读取的视频信息
	std::cout << "输出视频信息" << std::endl;
	std::cout << "帧率为：" << std::fixed <<  g_rate << std::endl;
	std::cout << "总帧数为：" << g_frames + 1 << std::endl;
	std::cout << "帧高为：" << g_height << std::endl;
	std::cout << "帧宽为：" << g_width << std::endl;
	std::cout << "被训练的帧数为：" << g_trainFrames << std::endl;
	std::cout << "正在训练中..." << std::endl;


	/***************GMM模型初始化****************/
	cv::Mat tempFrame;                                  // 创建Mat对象，临时存放彩色帧
	cv::Mat trainFrame( g_height, g_width, CV_32FC1 );  // 创建Mat对象，用来存储训练帧
	// 初始化均值时，第1个高斯分布的均值为第一帧的像素值
	// 初始化标准差时，所有高斯分布赋予相等的适当标准差即可，由经验得6比较合适
	// 若使用彩色图像时，GMM方法假设RGB三个通道的像素互不相关，所以协方差矩阵为一个对角阵，可以简化为一个向量处理
	// OpenCV中的三通道排列顺序为BGR
	cv::Mat subMat[ g_gaussianNum ];            // 存储每一帧像素和各个高斯分布均值的差

	for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
	{
		gaussianMean[ countGuass ] = cv::Mat::zeros( g_height, g_width, CV_32FC1 ); 
		// 先将各个高斯分布均值矩阵全置为0，之后再操作第一个均值矩阵
		gaussianStdDev[ countGuass ] = cv::Mat::ones( g_height, g_width, CV_32FC1 ) * g_init_StdDev;
		// ones()函数只能是每个通道的第一列全1，其他列全0
		// 使用Mat( g_height, g_width, CV_32FC3, cv::Scalar( 1, 1, 1 ) );
		// 可以创建一个3通道全1的矩阵，再乘6，即可得到想要的矩阵
		gaussianWeight[ countGuass ] = cv::Mat::ones( g_height, g_width, CV_32FC1 ) * ( 1 / g_gaussianNum );
		// 这里的权重矩阵为单通道即可，将权重初始化为都相等
		subMat[ countGuass ] = cv::Mat::zeros( g_height, g_width, CV_32FC1 );
		// 像素值和高斯分布均值的差的存储矩阵
	}
	countGuass = 0;                  // 高斯分布个数的计数变量清零，方便下一次使用

	trainCap.read( trainFrame );     // 读取视频的第一帧
	cv::cvtColor( trainFrame, trainFrame, CV_BGR2GRAY );   // 转换到灰度空间
	gaussianMean[ 0 ] = trainFrame;                        // 第一个高斯分布的均值为第一帧的像素值
	gaussianMean[ 0 ].convertTo( gaussianMean[ 0 ], CV_32FC1 );
	trainCap.set( CV_CAP_PROP_POS_FRAMES, 0 );             // 读取完一帧后，视频还要退回到第一帧


	/***************GMM模型的训练****************/
	for( countFrames = 0; countFrames < g_trainFrames; countFrames++ )
	{
		trainCap.read( tempFrame );     // 读取每一帧
		cv::cvtColor( tempFrame, trainFrame, CV_BGR2GRAY );  // 将每一帧转换到灰度空间	
		trainFrame.convertTo( trainFrame, CV_32FC1 );        
		// 必须要将trainFrame转换为float类型，否则指针无法正常工作
		// convertTo()函数中第一个参数是转换完毕的矩阵，第二个参数是要转换的类型
		// 第三第四个参数可以对转换完毕后Mat元素的值进行调整，可以缺省

		for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )   // 分别计算被训练帧的像素值和各个高斯分布之差
		{
			cv::subtract( gaussianMean[ countGuass ], trainFrame, subMat[ countGuass ] );
			// 第一个参数是被减数；第二个参数是减数；第三个参数是差
			subMat[ countGuass ] = cv::abs( subMat[ countGuass ] );
			// 取绝对值，得到像素值和高斯分布的均值之间的距离
			// 如果使用RGB则要用平方和，cv::pow( subMat[ countGuass ], 2, subMat[ countGuass ] );
			// 第一个参数是底数；第二个参数是指数；第三个参数是幂
		}
		countGuass = 0;    // 高斯分布个数的计数变量清零，方便下一次使用

		// 高斯分布的均值、标准差、权重进行更新
		for( height = 0; height < g_height; height++ )     // 遍历高度，即矩阵的行
		{
			for( width = 0; width < g_width; width++ )     // 遍历宽度，即矩阵的列
			{
				matchFlag = 0;      // 匹配标志位置回0
				tempWeight = 0.0;   // 临时存储权重的变量清零，方便下一次使用
				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )  // 遍历各个高斯分布
				{
					// 下面的if判断中，是指针指向元素的乘法，尽量使用括号隔离
					// ptr遍历Mat矩阵元素的方式比at更为高效
					if( ( *subMat[ countGuass ].ptr<float>( height, width ) ) <= ( *gaussianStdDev[ countGuass ].ptr<float>( height, width ) ) * ( g_deviationMult ) )
					{
						matchFlag = 1;     // 至少有一个高斯分布和该像素值匹配

						g_rho = g_alpha / ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) );
						*gaussianMean[ countGuass ].ptr<float>( height, width ) = ( 1 - g_rho ) * ( *gaussianMean[ countGuass ].ptr<float>( height, width ) ) + g_rho * ( *trainFrame.ptr<float>( height, width ) );
						*gaussianStdDev[ countGuass ].ptr<float>( height, width ) = sqrt( ( 1 - g_rho ) * std::pow( *gaussianWeight[ countGuass ].ptr<float>( height, width ), 2 ) + g_rho * pow( ( *trainFrame.ptr<float>( height, width ) - *gaussianMean[ countGuass ].ptr<float>( height, width ) ), 2  ) );
						*gaussianWeight[ countGuass ].ptr<float>( height, width ) = ( 1 - g_alpha ) * ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) ) + g_alpha * matchFlag;	
						// 更新rho参数，更新均值参数，更新标准差参数，更新权重参数
						// rho = 学习率 / 权重；  均值 = ( 1 - rho ) * 均值 + rho * 当前像素
						// 标准差 = sqrt( ( 1 - rho ) * 方差 + rho * ( 当前像素 - 均值 )^2 )
						// 权重 = ( 1 - rho ) * 权重 + rho * matchFlag； 匹配则matchFlag = 1，不匹配则matchFlag = 0
					}
					else
					{
						*gaussianWeight[ countGuass ].ptr<float>( height, width ) = ( 1 - g_alpha ) * ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) ) + g_alpha * matchFlag;	
						// 对于不匹配的高斯分布，只更新权重；均值、方差、rho都不变；并且这里matchFlag = 0
					}
					tempWeight = tempWeight + ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) );
					// 所有权重累加起来，方便做归一化
				}
				countGuass = 0;       // 高斯分布个数的计数变量清零，方便下一次使用

				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )     // 遍历各个高斯分布
				{
					*gaussianWeight[ countGuass ].ptr<float>( height, width ) = *gaussianWeight[ countGuass ].ptr<float>( height, width ) / tempWeight;
					// 权重归一化
				}
				countGuass = 0;       // 高斯分布个数的计数变量清零，方便下一次使用
				// 至此，对于匹配上的高斯分布，各个参数更新完毕

				// 找到各个高斯分布权重中的最小者，记录权重值和位置
				tempWeight = *gaussianWeight[ 0 ].ptr<float>( height, width );  
				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
				{
					if( ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) ) <= tempWeight )
					{
						minIndex = countGuass;    // 记录权重最小的高斯分布的索引
						tempWeight = *gaussianWeight[ countGuass ].ptr<float>( height, width );
						// 将权重最小的高斯分布的值赋给tempWeight
					}
				}
				countGuass = 0;         // 高斯分布个数的计数变量清零，方便下一次使用

				if( matchFlag == 0 )    // 此像素值没有任何高斯分布与其匹配时
				{
					*gaussianMean[ minIndex ].ptr<float>( height, width ) = *trainFrame.ptr<float>( height, width );
					*gaussianStdDev[ minIndex ].ptr<float>( height, width ) = g_init_StdDev;
					// 没有任何高斯分布和当前像素值匹配时，均值即当前像素值
					// 标准差选择一个比任何高斯分布的标准差都要大的，可以选择初始化的标准差
				}
			}
		}

		sprintf( fileName, "%s%d%s", "训练帧\\frame", countFrames, ".jpg" );
		// 将文件路径名进行粘贴
		cv::imwrite( fileName, trainFrame );   // 保存截取的图片
	}

	trainCap.release();   // 关闭视频文件，释放内存
}