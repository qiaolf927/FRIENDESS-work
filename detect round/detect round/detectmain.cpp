#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace std;
using namespace cv;


Mat reSizeImage(Mat im, double scale)    
 { 
 unsigned int i,j;   
 int w = im.cols*scale;   
 int h = im.rows*scale;    
 Mat imnew = cvCreateMat(h, w, CV_8UC1); 
 imnew.step = im.step*scale; 
 for ( j = 0; j < h; j++)    
 for ( i = 0; i < w; i++)    
	 imnew.at<uchar>(j,i)=im.at<uchar>(j/scale, i/scale);   
 return imnew;  
}

double overlapregion(vector<double> a,vector<double> b)
{
	double ap1r = a[0];
	double ap1c = a[1];
	double ap2r = a[2];
	double ap2c = a[3];
	double bp1r = b[0];
	double bp1c = b[1];
	double bp2r = b[2];
	double bp2c = b[3];
	double ratio = 0;

	return ratio;
}


int main( int argc, char** argv )
{
 

 double radius = 100.0;
 double scale = 0.5;

 //打开图像
 char* imageName = "D:/工作/照片测试/2015-06-26 桌上视觉测试台/[DEV0]-ID009534-BW.bmp";

 Mat image;
 image = imread( imageName,0);

 if( !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 //降采样
 Mat imgrough = reSizeImage(image,scale);

 namedWindow("imagerough", CV_WINDOW_AUTOSIZE);
 imshow("imagerough", imgrough);

 double radiusrough = radius*scale;

 Mat imbwrough;
 threshold(imgrough,imbwrough,0,255,CV_THRESH_OTSU); 
 namedWindow("imagebwrough", CV_WINDOW_AUTOSIZE);
 imshow("imagebwrough", imbwrough);

 

 //轮廓检测
 Mat imedge;
 vector<vector<Point>> contours;
 vector<Vec4i> hierarchy;

 Canny( imbwrough, imedge, 0.05, 0.12 , 3 );
 findContours( imedge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0) );

 vector<vector<Point>> contoursfilt;
 vector<Point> onecontour;
 for(int i = 0;i<contours.size();i++){
	 onecontour = contours[i];
	 if (onecontour.size() > 2*radiusrough){
		 contoursfilt.push_back(onecontour);
	}
 }


 Mat imcontour = Mat::zeros( imedge.size(), CV_8U);


 drawContours(imcontour,contoursfilt,-1,Scalar(255),2);

  /// 在窗体中显示结果
 namedWindow( "Canny detect", CV_WINDOW_AUTOSIZE );
 imshow( "Canny detect", imedge );
 namedWindow( "imcontour", CV_WINDOW_AUTOSIZE );
 imshow( "imcontour", imcontour );

 //Hough变换 粗定位
 double r = imcontour.rows;
 double c = imcontour.cols;

 double abstep = 1;//Hough圆变换位置间隔
 int thetanum = 180;
 double thetastep = 3.14/thetanum;
 double theta;
 int aa;
 int bb;
 vector<int> hough_sum;
 for(int i = 0; i<(floor(r/abstep)*floor(c/abstep)) ; i++){
	 hough_sum.push_back(0);
 }
 
 for (int i = 0; i<r ; i++)
 {
	 for (int j = 0; j<c ; j++)
	 {
		 if (imcontour.at<uchar>(i,j)>100)
		 {
			 for(int k = 0; k<2*thetanum; k++)
			 {
				 theta = thetastep*k;
				 aa = floor((radiusrough*cos(theta)+j)/abstep);
				 bb = floor((radiusrough*sin(theta)+i)/abstep);
				 if(aa>-1 & aa<floor(c/abstep) & bb>-1 & bb<floor(r/abstep))
				 {
					hough_sum[aa + bb*floor(c/abstep)]++;//一行一行串接
				 }
			 }
		 }
	 }
 }
 int pos = (int) ( max_element(hough_sum.begin(),hough_sum.end()) - hough_sum.begin() );//最大值位置
 int maxvalue = hough_sum[pos];
 double Trough = 0.9*maxvalue;
 int centerr;
 int centerc;

 Mat imroughresult = Mat::zeros( imedge.size(), CV_8UC3);
 for (int i = 0; i<r ; i++)
 {
	 for (int j = 0; j<c ; j++)
	 {
		 imroughresult.at<Vec3b>(i,j)[0] = imgrough.at<uchar>(i,j);
		 imroughresult.at<Vec3b>(i,j)[1] = imgrough.at<uchar>(i,j);
		 imroughresult.at<Vec3b>(i,j)[2] = imgrough.at<uchar>(i,j);
	 }
 }


 vector<vector<int>> bbox;
 vector<int> bboxtemp;
 int boxindex=0;

 for (int i = 0; i<floor(r/abstep) ; i++)
 {
	 for (int j = 0; j<floor(c/abstep) ; j++)
	 {
		 if (hough_sum[j + i*floor(c/abstep)]>Trough)
		 {
			 centerr = floor(i*abstep);
			 centerc = floor(j*abstep);
			 imroughresult.at<Vec3b>(i,j)[0] = 0;
			 imroughresult.at<Vec3b>(i,j)[1] = 0;
			 imroughresult.at<Vec3b>(i,j)[2] = 255;
			 for(int k = 0; k<2*thetanum; k++)
			 {
				 theta = thetastep*k;
				 aa = floor(radiusrough*cos(theta))+centerc;
				 bb = floor(radiusrough*sin(theta))+centerr;
				 imroughresult.at<Vec3b>(bb,aa)[0] = 0;
				 imroughresult.at<Vec3b>(bb,aa)[1] = 0;
				 imroughresult.at<Vec3b>(bb,aa)[2] = 255;
			 }
			 bboxtemp.push_back(centerr-radiusrough);
			 bboxtemp.push_back(centerc-radiusrough);
			 bboxtemp.push_back(centerr+radiusrough);
			 bboxtemp.push_back(centerc+radiusrough);
			 bbox.push_back(bboxtemp);
			 bboxtemp.clear();
			 boxindex++;
		 }
	 }
 }

 namedWindow("imageroughresult", CV_WINDOW_AUTOSIZE);
 imshow("imageroughresult", imroughresult);

 cout << Trough << endl;

 //回到高分辨率
 for (int i=0;i<boxindex;i++)
 {
	 bbox[i][0] = floor(bbox[i][0]/scale);
	 bbox[i][1] = floor(bbox[i][1]/scale);
	 bbox[i][2] = floor(bbox[i][2]/scale);
	 bbox[i][3] = floor(bbox[i][3]/scale);
 }

 vector<vector<double>> region;
 vector<double> regiontemp;
 vector<double> compareregion;
 vector<vector<double>> rcenter;
 vector<double> rcentertemp;
 vector<int> overlapnum;
 bool flagpool;
 int k;
 double ratio;

 //划定区域
 for (int i=0;i<boxindex;i++)
 {
	 if (i == 0)
	 {
		 regiontemp.push_back(bbox[i][0]-30.0);
		 regiontemp.push_back(bbox[i][1]-30.0);
		 regiontemp.push_back(bbox[i][2]+30.0);
		 regiontemp.push_back(bbox[i][3]+30.0);
		 region.push_back(regiontemp);
		 regiontemp.clear();
		 rcentertemp.push_back((bbox[i][0]+bbox[i][2])/2.0);
		 rcentertemp.push_back((bbox[i][1]+bbox[i][3])/2.0);
		 rcenter.push_back(rcentertemp);
		 rcentertemp.clear();
		 overlapnum.push_back(1);
	 }else{
		 regiontemp.push_back(bbox[i][0]-30.0);
		 regiontemp.push_back(bbox[i][1]-30.0);
		 regiontemp.push_back(bbox[i][2]+30.0);
		 regiontemp.push_back(bbox[i][3]+30.0);		 
		 flagpool = true;
		 for (int j = 0; j<region.size();j++)
		 {
			 compareregion.push_back(rcenter[j][0]-radius-30.0);
			 compareregion.push_back(rcenter[j][1]-radius-30.0);
			 compareregion.push_back(rcenter[j][0]+radius+30.0);
			 compareregion.push_back(rcenter[j][1]+radius+30.0);
			 ratio = overlapregion(regiontemp,compareregion);

			 compareregion.clear();
		 }
		 regiontemp.clear();
	 }
 }
 //imwrite( "../../images/Gray_Image.jpg", gray_image );

 //Mat edgeim = edgelink(imedge,100);
 waitKey(0);

 return 0;
}
