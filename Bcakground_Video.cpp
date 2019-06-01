
/******************ͷ�ļ�*******************/
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

/******************��˹���ģ�Ͳ���*******************/
#define g_gaussianNum  3      // ��ϵĸ�˹�����ĸ�����һ��ȡֵ��3��5֮�䣬����֤ȡ3��Ч���Ϻ�
#define g_init_StdDev  6      // ��˹�����ı�׼��
#define g_alpha  0.01         // ѧϰ��
#define g_deviationMult  2.5  // ��׼��ı���
#define g_threshold  0.75     // Ȩ����ֵ��һ��ȡֵ��0.25��0.75֮��
double g_rho = 0.0;           // ���ڸ���Ȩ�ء���ֵ������Ĳ�������ʼ��Ϊ0
int g_trainFrames = 0;        // ����ѵ����˹���ģ���õ�֡��

/******************��ĻҪ��ʾ�Ĵ��ڵ�����*******************/
std::string videoPath = "�����Ƶ\\Walk3.mpeg";
std::string foreName = "ǰ��";
std::string backName = "����";
std::string videoName = "ԭ��Ƶ";

/******************��Ƶ����*******************/
double g_rate;   // ֡��
int g_frames;    // ��֡��
int g_height;    // ��Ƶ�ĸ߶�
int g_width;     // ��Ƶ�Ŀ��

/******************��������*******************/
void trainGMM( cv::Mat gaussianMean[ g_gaussianNum ], cv::Mat gaussianStdDev[ g_gaussianNum ], cv::Mat gaussianWeight[ g_gaussianNum ] );      // ѵ����˹���ģ��
//void trainGMM_version2();      // ѵ����˹���ģ��


/******************������********************/
int main()
{
	/*******************��������************************/
	int countGuass = 0;       // �Ը�˹�ֲ��ĸ������м���
	int height = 0;           // ֡�ߵļ�������
	int width = 0;            // ֡��ļ�������
	int sortTemp = 0;         // �����������ʱ����
	float valueTemp = 0.0;    // ����ʱ����ֵ��������ʱ������ÿ����˹�ֲ�Ȩ�ص��ۼӺ���ʱ����
	int rank[ g_gaussianNum ] = { 0 };   // ��˹�ֲ��ᾭ������rank������ԭʼ��˹�ֲ�������
	int indexTemp = 0;        // ǰ���ٸ���˹�ֲ����Ա�ʾ����
	bool match = 0;           // �жϵ�ǰ֡�������Ƿ��Ǳ�����match = 1��Ϊ������match = 0��Ϊǰ��
	char savePath[ 30 ];      // ����Ŀ������ 
	int countFrames = 0;      // ����֡�ļ�������


	/*******************��˹���ģ�͵�ѵ��***********************/
	cv::Mat gaussianMean[ g_gaussianNum ];              // ����4����˹�ֲ��ľ�ֵ����g_gaussianNumΪ4
	cv::Mat gaussianStdDev[ g_gaussianNum ];            // ����4����˹�ֲ��ı�׼�����g_gaussianNumΪ4
	// ��ʼ����ֵʱ����1����˹�ֲ��ľ�ֵΪ��һ֡������ֵ
	// ��ʼ����׼��ʱ�����и�˹�ֲ�������ȵ��ʵ���׼��ɣ��ɾ����6�ȽϺ���
	// ��ʹ�ò�ɫͼ��ʱ��GMM��������RGB����ͨ�������ػ�����أ�����Э�������Ϊһ���Խ��󣬿��Լ�Ϊһ����������
	// OpenCV�е���ͨ������˳��ΪBGR
	cv::Mat gaussianWeight[ g_gaussianNum ];    // ����4��Ȩ�ؾ���
	trainGMM( gaussianMean, gaussianStdDev, gaussianWeight );  
	// ʹ�ø�˹���ģ�ͶԱ�����ģ��ͨ��ѵ������ȡ��˹�ֲ��ľ�ֵ����׼�Ȩ�أ�����ѵ����֡Ҳ�ᱻ�洢


	/*******************����Ȩ�غͱ�׼�����**********************/
	cv::Mat divWeiStDe[ g_gaussianNum ];   // ����4���洢Ȩ�غͱ�׼����̵ľ���
	for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
	{
		cv::divide( gaussianWeight[ countGuass ], gaussianStdDev[ countGuass ], divWeiStDe[ countGuass ] ); 
		// ���������˹�ֲ��ͱ�׼�����
	}
	countGuass = 0;        // ��˹�ֲ��ļ�����������
	
	/***********************����ǰ�����*************************/
	cv::Mat videoFrame, videoTempFrame;    // �洢��Ƶ֡��Mat����
	cv::Mat subMat[ g_gaussianNum ];       // �洢��ǰ֡�͸�����˹�ֲ��ľ�ֵ֮��
	cv::Mat backTemp( g_height, g_width, CV_32FC1 );   // ����ͼƬ����ֵ����ʱ���������ñ�����float�͵�
	cv::Mat fore( g_height, g_width, CV_8UC1 );   // ǰ���Ҷ�ͼƬ
	cv::Mat back( g_height, g_width, CV_8UC1 );   // �����Ҷ�ͼƬ
	cv::Mat element;                              // ��ʴ�����͵ĽṹԪ��
	element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 5, 5 ) );

	cv::VideoCapture cap( videoPath );     // ����Ƶ
	if( !cap.isOpened() )                  // �ж���Ƶ�Ƿ񱻴�
	{
		std::cout << "��Ƶ��ʧ�ܣ�" << std::endl;
		return 1;
	}
	cv::namedWindow( videoName, CV_WINDOW_NORMAL );   // ����ԭ��Ƶ����
	cv::namedWindow( backName, CV_WINDOW_NORMAL );    // ������������
	cv::namedWindow( foreName, CV_WINDOW_NORMAL );    // ����ǰ������
	
	std::cout << "ģ��ѵ����ϣ����ڶ�ÿһ֡��Ƶ���м�⣬�粻�������ۿ���Ƶ���밴ESC���˳���" << std::endl;
	while( cap.read( videoTempFrame ) )     // ��ʼ��ȡ��Ƶ֡��ֱ����Ƶ�������
	{
		cv::imshow( videoName, videoTempFrame );                   // ��ʾ��ȡ�ĸ�֡��Ƶ
		cv::cvtColor( videoTempFrame, videoFrame, CV_BGR2GRAY );   // ת�����Ҷȿռ�
		videoFrame.convertTo( videoFrame, CV_32FC1 );   // ����Ҫ��videoFrameת��Ϊfloat���ͣ�����ָ���޷���������

		// ���㵱ǰ֡�͸�����˹�ֲ���ֵ֮��
		for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
		{
			cv::subtract( videoFrame, gaussianMean[ countGuass ], subMat[ countGuass ] );
			subMat[ countGuass ] = cv::abs( subMat[ countGuass ] );
		}

		// ��Ȩ�غͱ�׼����̽��н�������
		for( height = 0; height < g_height; height++ )     // �����߶ȣ����������
		{
			for( width = 0; width < g_width; width++ )     // ������ȣ����������
			{
				for( countGuass = 1; countGuass < g_gaussianNum; countGuass++ )   // ����������˹�ֲ�
				{
					for( sortTemp = 0; sortTemp < countGuass; sortTemp++ )        // ð�ݷ�����
					{
						if( ( *divWeiStDe[ countGuass ].ptr<float>( height, width ) ) > ( *divWeiStDe[ sortTemp ].ptr<float>( height, width ) ) )
						{
							// ����divWeiStDe�ڱ�����ֵ����ķŵ�ǰ�ߣ�С�ķŵ����
							valueTemp = *divWeiStDe[ sortTemp ].ptr<float>( height, width );
							*divWeiStDe[ sortTemp ].ptr<float>( height, width ) = *divWeiStDe[ countGuass ].ptr<float>( height, width );
							*divWeiStDe[ countGuass ].ptr<float>( height, width ) = valueTemp;

							// ��������ֵ
							rank[ sortTemp ] = countGuass;
							rank[ countGuass ] = sortTemp;
						}
					}
				}
				countGuass = 0;      // ��˹�ֲ��ļ�����������
				// ���ˣ�����һ������ֵ�ĸ�˹�ֲ����������

				// �ж϶��ٸ���˹�ֲ�������Ϊ�������ص�ģ��
				valueTemp = 0.0;     // ��ʱ��������
				*backTemp.ptr<float>( height, width ) = 0.0;   // ����ͼƬ����ʱ����ֵ����
				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
				{
					*backTemp.ptr<float>( height, width ) = ( *backTemp.ptr<float>( height, width ) ) + ( *gaussianMean[ countGuass ].ptr<float>( height, width ) ) * ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) );
					// ����������ֵ���õ������˹�ֲ���Ȩ�غ;�ֵ�˻����ۼӺ�
					valueTemp = valueTemp + ( *gaussianWeight[ rank[ countGuass ] ].ptr<float>( height, width ) );
					if( valueTemp > g_threshold )
					{
						indexTemp = countGuass;    // indexTemp��ʾǰ������˹�ֲ����Ա�ʾ����
						break;                     // ����ѭ�� 
					}
				}
				countGuass = 0;     // ��˹�ֲ��ļ�����������
				*back.ptr<uchar>( height, width ) = *backTemp.ptr<float>( height, width );  // ��float����ת��Ϊuchar������֮��Ϳ�����imshow��ʾ��

				// �жϵ�ǰ֡�Ƿ��ǰindexTemp����˹�ֲ�֮һƥ��
				match = 0;           // Ĭ��Ϊǰ��
				for( countGuass = 0; countGuass <= indexTemp; countGuass++ )
				{
					if( ( *subMat[ countGuass ].ptr<float>( height, width ) ) < ( *gaussianStdDev[ countGuass ].ptr<float>( height, width ) ) * ( g_deviationMult ) )
					{
						match = 1;  // ��ǰ֡����ֵ�͸�����˹�ֲ���һ��ƥ�䣬��Ϊ����
						break;      // ����ѭ��
					}
				}

				// ����foreͼƬ�е�����ֵ
				if( match == 0 )    // ���Ϊǰ��
					*fore.ptr<uchar>( height, width ) = 255;  // ǰ��ͼƬ�У��õ�����
				else                // ���Ϊ����
					*fore.ptr<uchar>( height, width ) = 0;    // ǰ��ͼƬ�У��õ��
			}
		}

		cv::erode( fore, fore, element );    // ��ʴ�����ֱ��ǣ�����֡�����֡���ṹԪ��
		cv::dilate( fore, fore, element );   // ���Ͳ����ֱ��ǣ�����֡�� ���֡���ṹԪ��

		cv::imshow( foreName, fore );        // ��ʾǰ��ͼƬ
		cv::imshow( backName, back );        // ��ʾ����ͼƬ

		sprintf( savePath, "%s%d%s", "ǰ��֡\\frame", countFrames++, ".jpg" );
		cv::imwrite( savePath, fore );       // ����ǰ��֡

		if( cv::waitKey( 100 / g_rate ) == 27 )   // ֡�ʿ��Ա�ԭ��Ƶ��֡�ʿ�һ�㣬�������������ҲҪռ��ʱ��
		{
			break;    // �������ESC�������˳�����
		}
	}

	cap.release();                    // �ر���Ƶ���ͷ��ڴ�
	cv::destroyWindow( videoName );   // ����ԭ��Ƶ����
	cv::destroyWindow( backName );    // ���ٱ�������
	cv::destroyWindow( foreName );    // ����ǰ������

	system( "pause" );  // ����Ļ��ʾ�����������
}


/**********************��������**************************/
void trainGMM( cv::Mat gaussianMean[ g_gaussianNum ], cv::Mat gaussianStdDev[ g_gaussianNum ], cv::Mat gaussianWeight[ g_gaussianNum ] )
{
	/***************��������********************/
	int countGuass = 0;      // �Ը�˹�ֲ��ĸ������м���
	int countFrames = 0;     // ��ѵ����֡���м���
	int height = 0;          // ֡�ߵļ�������
	int width = 0;           // ֡��ļ�������
	bool matchFlag = 0;      // ���¸�˹�ֲ�Ȩ�صı�־
	double tempWeight = 0.0; // ��ʱ�������洢Ȩ���ۼӺ�
	int minIndex = 0;        // ��˹�ֲ�Ȩ����С�ߵ�����
	char fileName[ 30 ] ;    // �洢ѵ��֡��·�������ļ����ַ���


	/***************����Ƶ����ȡ��Ƶ��Ϣ****************/
	cv::VideoCapture trainCap( videoPath );   // ��ȡ��Ƶ
	if( !trainCap.isOpened() )                // �ж���Ƶ�Ƿ񱻴�
	{
		std::cout << "��Ƶ��ʧ�ܣ�" << std::endl;
		//return 1;
	}
	g_rate = trainCap.get( CV_CAP_PROP_FPS );             // ��ȡ֡��
	g_frames = trainCap.get( CV_CAP_PROP_FRAME_COUNT );   // ��ȡ֡����
	g_height = trainCap.get( CV_CAP_PROP_FRAME_HEIGHT );  // ��ȡ��Ƶ�ĸ߶�
	g_width = trainCap.get( CV_CAP_PROP_FRAME_WIDTH );    // ��ȡ��Ƶ�Ŀ��
	g_trainFrames = 0.01 * ( g_frames + 1 );              
	// ��ǰ��֡����ѵ����˹���ģ�ͣ�g_frames�Ǵ�0��ʼ�ģ�����Ҫ��1
	// ������ֲ�Ҫ����0.01�ȽϺ��ʣ�������Ч�����ã�������һ�ֹ����

	// �����ȡ����Ƶ��Ϣ
	std::cout << "�����Ƶ��Ϣ" << std::endl;
	std::cout << "֡��Ϊ��" << std::fixed <<  g_rate << std::endl;
	std::cout << "��֡��Ϊ��" << g_frames + 1 << std::endl;
	std::cout << "֡��Ϊ��" << g_height << std::endl;
	std::cout << "֡��Ϊ��" << g_width << std::endl;
	std::cout << "��ѵ����֡��Ϊ��" << g_trainFrames << std::endl;
	std::cout << "����ѵ����..." << std::endl;


	/***************GMMģ�ͳ�ʼ��****************/
	cv::Mat tempFrame;                                  // ����Mat������ʱ��Ų�ɫ֡
	cv::Mat trainFrame( g_height, g_width, CV_32FC1 );  // ����Mat���������洢ѵ��֡
	// ��ʼ����ֵʱ����1����˹�ֲ��ľ�ֵΪ��һ֡������ֵ
	// ��ʼ����׼��ʱ�����и�˹�ֲ�������ȵ��ʵ���׼��ɣ��ɾ����6�ȽϺ���
	// ��ʹ�ò�ɫͼ��ʱ��GMM��������RGB����ͨ�������ػ�����أ�����Э�������Ϊһ���Խ��󣬿��Լ�Ϊһ����������
	// OpenCV�е���ͨ������˳��ΪBGR
	cv::Mat subMat[ g_gaussianNum ];            // �洢ÿһ֡���غ͸�����˹�ֲ���ֵ�Ĳ�

	for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
	{
		gaussianMean[ countGuass ] = cv::Mat::zeros( g_height, g_width, CV_32FC1 ); 
		// �Ƚ�������˹�ֲ���ֵ����ȫ��Ϊ0��֮���ٲ�����һ����ֵ����
		gaussianStdDev[ countGuass ] = cv::Mat::ones( g_height, g_width, CV_32FC1 ) * g_init_StdDev;
		// ones()����ֻ����ÿ��ͨ���ĵ�һ��ȫ1��������ȫ0
		// ʹ��Mat( g_height, g_width, CV_32FC3, cv::Scalar( 1, 1, 1 ) );
		// ���Դ���һ��3ͨ��ȫ1�ľ����ٳ�6�����ɵõ���Ҫ�ľ���
		gaussianWeight[ countGuass ] = cv::Mat::ones( g_height, g_width, CV_32FC1 ) * ( 1 / g_gaussianNum );
		// �����Ȩ�ؾ���Ϊ��ͨ�����ɣ���Ȩ�س�ʼ��Ϊ�����
		subMat[ countGuass ] = cv::Mat::zeros( g_height, g_width, CV_32FC1 );
		// ����ֵ�͸�˹�ֲ���ֵ�Ĳ�Ĵ洢����
	}
	countGuass = 0;                  // ��˹�ֲ������ļ����������㣬������һ��ʹ��

	trainCap.read( trainFrame );     // ��ȡ��Ƶ�ĵ�һ֡
	cv::cvtColor( trainFrame, trainFrame, CV_BGR2GRAY );   // ת�����Ҷȿռ�
	gaussianMean[ 0 ] = trainFrame;                        // ��һ����˹�ֲ��ľ�ֵΪ��һ֡������ֵ
	gaussianMean[ 0 ].convertTo( gaussianMean[ 0 ], CV_32FC1 );
	trainCap.set( CV_CAP_PROP_POS_FRAMES, 0 );             // ��ȡ��һ֡����Ƶ��Ҫ�˻ص���һ֡


	/***************GMMģ�͵�ѵ��****************/
	for( countFrames = 0; countFrames < g_trainFrames; countFrames++ )
	{
		trainCap.read( tempFrame );     // ��ȡÿһ֡
		cv::cvtColor( tempFrame, trainFrame, CV_BGR2GRAY );  // ��ÿһ֡ת�����Ҷȿռ�	
		trainFrame.convertTo( trainFrame, CV_32FC1 );        
		// ����Ҫ��trainFrameת��Ϊfloat���ͣ�����ָ���޷���������
		// convertTo()�����е�һ��������ת����ϵľ��󣬵ڶ���������Ҫת��������
		// �������ĸ��������Զ�ת����Ϻ�MatԪ�ص�ֵ���е���������ȱʡ

		for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )   // �ֱ���㱻ѵ��֡������ֵ�͸�����˹�ֲ�֮��
		{
			cv::subtract( gaussianMean[ countGuass ], trainFrame, subMat[ countGuass ] );
			// ��һ�������Ǳ��������ڶ��������Ǽ����������������ǲ�
			subMat[ countGuass ] = cv::abs( subMat[ countGuass ] );
			// ȡ����ֵ���õ�����ֵ�͸�˹�ֲ��ľ�ֵ֮��ľ���
			// ���ʹ��RGB��Ҫ��ƽ���ͣ�cv::pow( subMat[ countGuass ], 2, subMat[ countGuass ] );
			// ��һ�������ǵ������ڶ���������ָ������������������
		}
		countGuass = 0;    // ��˹�ֲ������ļ����������㣬������һ��ʹ��

		// ��˹�ֲ��ľ�ֵ����׼�Ȩ�ؽ��и���
		for( height = 0; height < g_height; height++ )     // �����߶ȣ����������
		{
			for( width = 0; width < g_width; width++ )     // ������ȣ����������
			{
				matchFlag = 0;      // ƥ���־λ�û�0
				tempWeight = 0.0;   // ��ʱ�洢Ȩ�صı������㣬������һ��ʹ��
				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )  // ����������˹�ֲ�
				{
					// �����if�ж��У���ָ��ָ��Ԫ�صĳ˷�������ʹ�����Ÿ���
					// ptr����Mat����Ԫ�صķ�ʽ��at��Ϊ��Ч
					if( ( *subMat[ countGuass ].ptr<float>( height, width ) ) <= ( *gaussianStdDev[ countGuass ].ptr<float>( height, width ) ) * ( g_deviationMult ) )
					{
						matchFlag = 1;     // ������һ����˹�ֲ��͸�����ֵƥ��

						g_rho = g_alpha / ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) );
						*gaussianMean[ countGuass ].ptr<float>( height, width ) = ( 1 - g_rho ) * ( *gaussianMean[ countGuass ].ptr<float>( height, width ) ) + g_rho * ( *trainFrame.ptr<float>( height, width ) );
						*gaussianStdDev[ countGuass ].ptr<float>( height, width ) = sqrt( ( 1 - g_rho ) * std::pow( *gaussianWeight[ countGuass ].ptr<float>( height, width ), 2 ) + g_rho * pow( ( *trainFrame.ptr<float>( height, width ) - *gaussianMean[ countGuass ].ptr<float>( height, width ) ), 2  ) );
						*gaussianWeight[ countGuass ].ptr<float>( height, width ) = ( 1 - g_alpha ) * ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) ) + g_alpha * matchFlag;	
						// ����rho���������¾�ֵ���������±�׼�����������Ȩ�ز���
						// rho = ѧϰ�� / Ȩ�أ�  ��ֵ = ( 1 - rho ) * ��ֵ + rho * ��ǰ����
						// ��׼�� = sqrt( ( 1 - rho ) * ���� + rho * ( ��ǰ���� - ��ֵ )^2 )
						// Ȩ�� = ( 1 - rho ) * Ȩ�� + rho * matchFlag�� ƥ����matchFlag = 1����ƥ����matchFlag = 0
					}
					else
					{
						*gaussianWeight[ countGuass ].ptr<float>( height, width ) = ( 1 - g_alpha ) * ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) ) + g_alpha * matchFlag;	
						// ���ڲ�ƥ��ĸ�˹�ֲ���ֻ����Ȩ�أ���ֵ�����rho�����䣻��������matchFlag = 0
					}
					tempWeight = tempWeight + ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) );
					// ����Ȩ���ۼ���������������һ��
				}
				countGuass = 0;       // ��˹�ֲ������ļ����������㣬������һ��ʹ��

				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )     // ����������˹�ֲ�
				{
					*gaussianWeight[ countGuass ].ptr<float>( height, width ) = *gaussianWeight[ countGuass ].ptr<float>( height, width ) / tempWeight;
					// Ȩ�ع�һ��
				}
				countGuass = 0;       // ��˹�ֲ������ļ����������㣬������һ��ʹ��
				// ���ˣ�����ƥ���ϵĸ�˹�ֲ������������������

				// �ҵ�������˹�ֲ�Ȩ���е���С�ߣ���¼Ȩ��ֵ��λ��
				tempWeight = *gaussianWeight[ 0 ].ptr<float>( height, width );  
				for( countGuass = 0; countGuass < g_gaussianNum; countGuass++ )
				{
					if( ( *gaussianWeight[ countGuass ].ptr<float>( height, width ) ) <= tempWeight )
					{
						minIndex = countGuass;    // ��¼Ȩ����С�ĸ�˹�ֲ�������
						tempWeight = *gaussianWeight[ countGuass ].ptr<float>( height, width );
						// ��Ȩ����С�ĸ�˹�ֲ���ֵ����tempWeight
					}
				}
				countGuass = 0;         // ��˹�ֲ������ļ����������㣬������һ��ʹ��

				if( matchFlag == 0 )    // ������ֵû���κθ�˹�ֲ�����ƥ��ʱ
				{
					*gaussianMean[ minIndex ].ptr<float>( height, width ) = *trainFrame.ptr<float>( height, width );
					*gaussianStdDev[ minIndex ].ptr<float>( height, width ) = g_init_StdDev;
					// û���κθ�˹�ֲ��͵�ǰ����ֵƥ��ʱ����ֵ����ǰ����ֵ
					// ��׼��ѡ��һ�����κθ�˹�ֲ��ı�׼�Ҫ��ģ�����ѡ���ʼ���ı�׼��
				}
			}
		}

		sprintf( fileName, "%s%d%s", "ѵ��֡\\frame", countFrames, ".jpg" );
		// ���ļ�·��������ճ��
		cv::imwrite( fileName, trainFrame );   // �����ȡ��ͼƬ
	}

	trainCap.release();   // �ر���Ƶ�ļ����ͷ��ڴ�
}