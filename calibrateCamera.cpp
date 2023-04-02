#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include "opencv4/opencv2/calib3d/calib3d.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv2/viz/viz3d.hpp"
#include <iostream>
#include <fstream>
 
using namespace cv;
using namespace std;
 
int main()
{
	ofstream fout("/home/ros/slam_test_vscode/calibrateCamera/caliberation_result.txt");  /* 保存标定结果的文件 */	
	//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	cout<<"开始提取角点………………"<<endl;
	int image_count=0;  /* 图像数量 */
	Size image_size;  /* 图像的尺寸 */
	Size board_size = Size(9,6);    /* 标定板上每行、列的角点数 */
	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
    string filename = "/mnt/hgfs/shared_ubuntu/picture_of_calibrate_camera/";
    for(int i=1; i<12; i++){
        Mat imageInput = imread(filename + to_string(i) + ".jpg", 1); //这里必须是彩色图像
        image_count++;
        cout << "第" << i << "张图:" << endl;
        image_size.width = imageInput.cols;
        image_size.height =imageInput.rows;			
        cout<<"image_size.width = "<<image_size.width<<endl;
        cout<<"image_size.height = "<<image_size.height<<endl;
		
		/* 提取角点 */
		if (0 == findChessboardCorners(imageInput,board_size,image_points_buf))
		{			
			cout<<"can not find chessboard corners!\n"; //找不到角点
			exit(1);
		} 
		else 
		{
			Mat view_gray;
			cvtColor(imageInput,view_gray,COLOR_RGB2GRAY);
			/* 亚像素精确化 */
			find4QuadCornerSubpix(view_gray,image_points_buf,Size(5,5)); //对粗提取的角点进行精确化
			//cornerSubPix(view_gray,image_points_buf,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点
			/* 在图像上显示角点位置 */
			drawChessboardCorners(imageInput,board_size,image_points_buf,true); //用于在图片中标记角点
            namedWindow("Camera Calibration",WINDOW_FREERATIO);
			imshow("Camera Calibration",imageInput);//显示图片
			// waitKey(500); //0.5秒
		}
	}

    int m=1;
    for(vector<vector<Point2f>>::iterator it = image_points_seq.begin(); it != image_points_seq.end();it++){
        int n=0; 
        cout << "第" << m << "张图片的角点坐标数据:" << endl;
    
        for(vector<Point2f>::iterator vit = (*it).begin(); vit != (*it).end();vit++){           
            n++;
            cout <<  *vit;
            if(n == 9){ //每9个数据一行
                cout << endl;
                n=0;
            }
        }
        m++;
    }	
	cout<<"角点提取完成！\n";
 

	//以下是摄像机标定
	cout<<"开始标定………………";
	/*棋盘三维信息*/
	Size square_size = Size(32,32);  /* 实际测量得到的标定板上每个棋盘格的大小，单位是mm*/
	vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
	/*内外参数*/
	Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); /* 摄像机内参数矩阵 */
	vector<int> point_counts;  // 每幅图像中角点的数量
	Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,k3,p1,p2, */
	vector<Mat> tvecsMat;  /* 每幅图像的平移向量 */
	vector<Mat> rvecsMat; /* 每幅图像的旋转向量 */
	/* 初始化标定板上角点的三维坐标 */
	int i,j,t;
	for (t=0;t<image_count;t++) 
	{
		vector<Point3f> tempPointSet;
		for (i=0;i<board_size.height;i++) 
		{
			for (j=0;j<board_size.width;j++) 
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = j*square_size.width;
				realPoint.y = i*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i=0;i<image_count;i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	/* 开始标定 */
	calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs,rvecsMat,tvecsMat,0);
	cout<<"标定完成"<<endl;

	//对标定结果进行评价
	cout<<"开始评价标定结果………………\n";
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
	cout<<"每幅图像的标定误差：\n";
	fout<<"每幅图像的标定误差：\n";
	for (i=0;i<image_count;i++)
	{
		vector<Point3f> tempPointSet=object_points[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet,rvecsMat[i],tvecsMat[i],cameraMatrix,distCoeffs,image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
		Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
		for (int j = 0 ; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err/=  point_counts[i];   
		std::cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
		fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
	}   
	std::cout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl;   
	fout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;   
	std::cout<<"评价完成！"<<endl;

	//保存定标结果  	
	std::cout<<"开始保存定标结果………………"<<endl;       
	Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fout<<"相机内参数矩阵："<<endl;   
	fout<<cameraMatrix<<endl<<endl;   
	fout<<"畸变系数：\n";   
	fout<<distCoeffs<<endl<<endl<<endl;   
	for (int i=0; i<image_count; i++) 
	{ 
		fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;   
		fout<<rvecsMat[i]<<endl;    
		/* 将旋转向量转换为相对应的旋转矩阵 */   
		Rodrigues(rvecsMat[i],rotation_matrix);   
		fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;   
		fout<<rotation_matrix<<endl;   
		fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;   
		fout<<tvecsMat[i]<<endl<<endl;   
	}   
	std::cout<<"完成保存"<<endl; 
	fout<<endl;

	/************************************************************************  
	显示定标结果  
	*************************************************************************/
	
	Mat mapx = Mat(image_size,CV_32FC1);
	Mat mapy = Mat(image_size,CV_32FC1);
	std::cout<<"保存矫正图像"<<endl;
	string numFileName;
	std::stringstream StrStm;
	for (int i = 0 ; i != image_count ; i++)
	{
		std::cout<<"Frame #"<<i+1<<"..."<<endl;
		initUndistortRectifyMap(cameraMatrix,distCoeffs,Mat(),Mat(),image_size,CV_32FC1,mapx,mapy);		
		StrStm.clear();
		numFileName.clear();
		string filePath="/home/ros/slam_test_vscode/calibrateCamera/picture_d/";
		StrStm<<i+1;
		StrStm>>numFileName;
		filePath+=numFileName;
		filePath+="_d.jpg";
		Mat imageSource = imread(filePath);
		Mat newimage = imageSource.clone();
		//另一种不需要转换矩阵的方式
		//undistort(imageSource,newimage,cameraMatrix,distCoeffs);
		remap(imageSource,newimage,mapx, mapy, INTER_LINEAR);		
		StrStm.clear();
		filePath.clear();
		filePath = "/home/ros/slam_test_vscode/calibrateCamera/picture_d_another/";
		StrStm<<i+1;
		StrStm>>numFileName;
		filePath+=numFileName;
		filePath += "_d.jpg";
		imwrite(filePath,newimage);
	}
	std::cout<<"保存结束"<<endl;	
	
	return 0;
}