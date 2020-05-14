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
	int overlaparea = 0;
	double allarea = 0.0;
	double ratio = 0.0;
	for (double i = ap1r; i<ap2r; i++)
	{
		for (double j = ap1c; j<ap2c; j++)
		{
			if (i>=bp1r&i<=bp2r&j>=bp1c&j<=bp2c)
			{
				overlaparea++;
			}
			allarea++;
		}
	}
	ratio = double(overlaparea/allarea);

	return ratio;
}

vector<vector<int>> extractfittedpoint(Mat im,vector<double> region,double radius)
{
	vector<int> certainnonzeropoint;
	vector<vector<int>> nonzeropoint;
	double squaredistance;
	for (int i=floor(region[0]);i<ceil(region[2]);i++)
	{
		for (int j=floor(region[1]);j<ceil(region[3]);j++)
		{
			if (i>=0&i<im.rows&j>=0&j<im.cols)
			{
				if (im.at<uchar>(i,j)>100)
				{
					certainnonzeropoint.push_back(i);
					certainnonzeropoint.push_back(j);
					squaredistance = (i - (region[0]+region[2])/2.0)*(i - (region[0]+region[2])/2.0) + (j - (region[1]+region[3])/2.0)*(j - (region[1]+region[3])/2.0);
					if (squaredistance>((radius-30)*(radius-30))&squaredistance<((radius+30)*(radius+30)))
					{
						certainnonzeropoint.push_back(1);
					}else
						{
							certainnonzeropoint.push_back(0);
						}
					nonzeropoint.push_back(certainnonzeropoint);
					certainnonzeropoint.clear();
				}
			}
		}
	}
	return nonzeropoint;
}

vector<vector<int>> getfittedpoint(vector<vector<int>> nonzeropoint)
{
	int pointsize = nonzeropoint.size();
	vector<int> certainnonzeropoint;
	vector<vector<int>> fittedpoint;
	for (int i =0;i<pointsize;i++)
	{
		if (nonzeropoint[i][2]==1)
		{
			certainnonzeropoint.push_back(nonzeropoint[i][0]);
			certainnonzeropoint.push_back(nonzeropoint[i][1]);
			fittedpoint.push_back(certainnonzeropoint);
			certainnonzeropoint.clear();
		}
	}
	return fittedpoint;
}

vector<vector<int>> getevalpoint(vector<vector<int>> nonzeropoint,vector<double> model)
{
	int pointsize = nonzeropoint.size();
	vector<int> certainnonzeropoint;
	vector<vector<int>> evalpoint;
	double a = model[0];
	double b = model[1];
	double R = model[2];

	for (int i =0;i<pointsize;i++)
	{
		if ((nonzeropoint[i][0]-a)*(nonzeropoint[i][0]-a) + (nonzeropoint[i][1]-b)*(nonzeropoint[i][1]-b) < (2*R)*(2*R))
		{
			certainnonzeropoint.push_back(nonzeropoint[i][0]);
			certainnonzeropoint.push_back(nonzeropoint[i][1]);
			evalpoint.push_back(certainnonzeropoint);
			certainnonzeropoint.clear();
		}
	}
	return evalpoint;
}

vector<vector<int>> randperm(int num,int firstn)
{
	vector<int> numlist;
	vector<int> randlist;
	vector<int> wholelist;
	vector<vector<int>> difflist;
	int randnum;
	int randchosen;
	vector<int>::iterator veciter;


	for (int i = 0;i<num;i++)
	{
		numlist.push_back(i);
		wholelist.push_back(i);
	}
	for (int i = 0;i<firstn;i++)
	{
		randnum = (rand() % (numlist.size()));
		randchosen = numlist[randnum];
		randlist.push_back(randchosen);
		veciter = numlist.begin();
		veciter = remove(numlist.begin(),numlist.end(),randnum);
	}

	difflist.push_back(randlist);
	difflist.push_back(wholelist);
	return difflist;
}

vector<double> lseround(vector<int> inliers,vector<vector<int>> fittedpoint)
{
	double resolution = 1000.0;
	
	int indnumber;
	double suchx;
	double suchy;
	double X1 = 0.0;
	double Y1 = 0.0;
	double X2 = 0.0;
	double Y2 = 0.0;
	double X3 = 0.0;
	double Y3 = 0.0;
	double X1Y1 = 0.0;
	double X1Y2 = 0.0;
	double X2Y1 = 0.0;
	int num = inliers.size();

	for (int i = 0 ; i<num ; i++)
	{
		indnumber = inliers[i];
		suchx = fittedpoint[indnumber][1]/resolution;
		suchy = fittedpoint[indnumber][0]/resolution;
		X1 = X1 + suchx;
		Y1 = Y1 + suchy;
		X2 = X2 + suchx*suchx;
		Y2 = Y2 + suchy*suchy;
		X3 = X3 + suchx*suchx*suchx;
		Y3 = Y3 + suchy*suchy*suchy;
		X1Y1 = X1Y1 + suchx*suchy;
		X1Y2 = X1Y2 + suchx*suchy*suchy;
		X2Y1 = X2Y1 + suchx*suchx*suchy;
	}

	double CC = num*X2 - X1*X1;
	double DD = num*X1Y1 - X1*Y1;
	double EE = num*X3 + num*X1Y2 - (X2+Y2)*X1;
	double GG = num*Y2 - Y1*Y1;
	double HH = num*X2Y1 + num*Y3 - (X2+Y2)*Y1;
	double aa = (HH*DD - EE*GG)/(CC*GG - DD*DD);
	double bb = (HH*CC - EE*DD)/(DD*DD - GG*CC);
	double cc = -(aa*X1 + bb*Y1 + X2 + Y2)/num;
	double R;
	R = sqrt(aa*aa + bb*bb - 4*cc)/2;

	vector<double> model;
	model.push_back(bb/(-2)*resolution);
	model.push_back(aa/(-2)*resolution);
	model.push_back(R*resolution);

	return model;
}

double pointerror(vector<double> model,int i,vector<vector<int>> fittedpoint)
{
	double a = model[0];
	double b = model[1];
	double R = model[2];
	int x = fittedpoint[i][1];
	int y = fittedpoint[i][0];

	double errorvalue = abs(R-sqrt((x-b)*(x-b)+(y-a)*(y-a)));

	return errorvalue;
}

double modelerrormeasure(vector<double> model,vector<int> pointlist,vector<vector<int>> fittedpoint)
{
	double a = model[0];
	double b = model[1];
	double R = model[2];
	double errorvalue = 0.0;
	int i;
	int x;
	int y;

	for (int j = 0; j<pointlist.size(); j++)
	{
		i = pointlist[j];
		x = fittedpoint[i][1];
		y = fittedpoint[i][0];
		errorvalue = errorvalue + abs(R-sqrt((x-b)*(x-b)+(y-a)*(y-a)));
	}
	
	errorvalue = errorvalue/pointlist.size();
	
	return errorvalue;
}

int main( int argc, char** argv )
{
 

 double radius = 230.0;
 double scale = 0.5;
 cin>>radius;

 clock_t start,finish;

 //打开图像


 char* imageName = argv[1];
 //char* imageName = "D:/testimage/[DEV0]-ID020536-BW.bmp";
 Mat image;
 image = imread( imageName,0);

 namedWindow("image", CV_WINDOW_AUTOSIZE);
 imshow("image", image);

 if( !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }
 start = clock();
 //降采样
 Mat imgrough = reSizeImage(image,scale);

 //namedWindow("imagerough", CV_WINDOW_AUTOSIZE);
 //imshow("imagerough", imgrough);

 double radiusrough = radius*scale;

 Mat imbwrough;
 threshold(imgrough,imbwrough,0,255,CV_THRESH_OTSU); 
 //namedWindow("imagebwrough", CV_WINDOW_AUTOSIZE);
 //imshow("imagebwrough", imbwrough);

 

 //轮廓检测
 Mat imedge;
 vector<vector<Point>> contours;
 vector<Vec4i> hierarchy;

 GaussianBlur(imbwrough,imedge,Size(11,11),3);
 Canny( imedge, imedge, 50, 150 , 3 );
 /*
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
 */
 Mat imcontour = imedge.clone();
  /// 在窗体中显示结果
 //namedWindow( "Canny detect", CV_WINDOW_AUTOSIZE );
 //imshow( "Canny detect", imedge );
 //namedWindow( "imcontour", CV_WINDOW_AUTOSIZE );
 //imshow( "imcontour", imcontour );

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
 /*
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
*/

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
			 /*
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
			 */
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

 //namedWindow("imageroughresult", CV_WINDOW_AUTOSIZE);
 //imshow("imageroughresult", imroughresult);

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
			 if (ratio > 0.7)
			 {
				 rcenter[j][0] = ((overlapnum[j]/(overlapnum[j]+1.0))*(rcenter[j][0])+(1/(overlapnum[j]+1.0))*((bbox[i][0]+bbox[i][2])/2.0));
				 rcenter[j][1] = ((overlapnum[j]/(overlapnum[j]+1.0))*(rcenter[j][1])+(1/(overlapnum[j]+1.0))*((bbox[i][1]+bbox[i][3])/2.0));
				 overlapnum[j]++;
				 region[j][0] = rcenter[j][0] - radius - 10*(5<(2+overlapnum[j])?5:(2+overlapnum[j]));
				 region[j][1] = rcenter[j][1] - radius - 10*(5<(2+overlapnum[j])?5:(2+overlapnum[j]));
				 region[j][2] = rcenter[j][0] + radius + 10*(5<(2+overlapnum[j])?5:(2+overlapnum[j]));
				 region[j][3] = rcenter[j][1] + radius + 10*(5<(2+overlapnum[j])?5:(2+overlapnum[j]));
				 flagpool = false;
				 compareregion.clear();
				 break;
			 }
			 compareregion.clear();
		 }
		 if (flagpool)
		 {
			 region.push_back(regiontemp);
			 rcentertemp.push_back((bbox[i][0]+bbox[i][2])/2.0);
			 rcentertemp.push_back((bbox[i][1]+bbox[i][3])/2.0);
			 rcenter.push_back(rcentertemp);
			 rcentertemp.clear();
			 overlapnum.push_back(1);
		 }
		 regiontemp.clear();
	 }
 }

 //精定位
 //imwrite( "../../images/Gray_Image.jpg", gray_image );

 //Mat edgeim = edgelink(imedge,100);
 Mat imbwo;
 threshold(image,imbwo,0,255,CV_THRESH_OTSU); 

 Mat imedgeo;
 vector<vector<Point>> contourso;
 vector<Vec4i> hierarchyo;

 GaussianBlur(imbwo,imedgeo,Size(11,11),3);
 Canny( imedgeo, imedgeo, 0.05, 0.12 , 3 );
 findContours( imedgeo, contourso, hierarchyo, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0) );

 vector<vector<Point>> contoursfilto;
 vector<Point> onecontouro;
 for(int i = 0;i<contourso.size();i++){
	 onecontouro = contourso[i];
	 if (onecontouro.size() > 2*radius){
		 contoursfilto.push_back(onecontouro);
	 }
 }


 Mat edgeimo = Mat::zeros( imedgeo.size(), CV_8U);
 drawContours(edgeimo,contoursfilto,-1,Scalar(255),2);
 
 vector<bool> regionjudge;
 for (int i = 0;i<region.size();i++)
 {
	 regionjudge.push_back(true);
 }

 finish = clock();
 double totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
 cout << totaltime << endl;

 vector<vector<int>> nonzeropoint; //[点行，点列，是否用来拟合圆]
 vector<vector<int>> fittedpoint;
 vector<vector<int>> nonzeroevalpoint;
 vector<int> evalind;

 int data_num;
 int ransac_n;
 int ransac_k;
 double ransac_t;
 int ransac_d;

 int iteration;
 vector<double> maybe_model;
 vector<double> better_model;
 vector<double> best_model;
 vector<int> best_consensus_set;
 vector<int> consensus_set;
 double best_error = 20;
 double this_error = 20;
 double certainerror;
 int pointx;
 int pointy;

 vector<int> maybe_inliers;
 vector<int> maybe_outliers;
 vector<vector<int>> diffliers;

 vector<vector<double>> modelrecord;
 vector<double> errorrecord;
 vector<int> consensus_setnumrecord;

 for (int i = 0;i<region.size();i++)
 {
	 modelrecord.push_back(best_model);
	 errorrecord.push_back(-1.0);
	 consensus_setnumrecord.push_back(0);
	 
	 nonzeropoint = extractfittedpoint(edgeimo,region[i],radius);
	 fittedpoint = getfittedpoint(nonzeropoint);

	 //ransac
	 data_num = fittedpoint.size();
	 ransac_n = 10;
	 ransac_k = 50;
	 ransac_t = 5.0;
	 ransac_d = 0.2*nonzeropoint.size();

	 iteration = 0;

	 best_error = 20;
	 this_error = 20;
	 if (data_num<ransac_n)
	 {
		 regionjudge[i]=false;
		 printf( " region=%d ; judge: %s ; few points\n " ,i+1,regionjudge[i]==true?"true":"false");
		 continue;
	 }

	 while (iteration < ransac_k)
	 {
		 diffliers = randperm(data_num,ransac_n);
		 maybe_inliers = diffliers[0];
		 maybe_outliers = diffliers[1];
		 maybe_model = lseround(maybe_inliers,fittedpoint);
		 consensus_set.clear();
		 for (int j = 0 ; j < maybe_outliers.size() ; j++)
		 {
			 pointx = fittedpoint[maybe_outliers[j]][1];
			 pointy = fittedpoint[maybe_outliers[j]][0];
			 certainerror = abs(maybe_model[2]-sqrt((pointx-maybe_model[1])*(pointx-maybe_model[1])+(pointy-maybe_model[0])*(pointy-maybe_model[0])));
			 if (certainerror < ransac_t)
			 {
				 consensus_set.push_back(maybe_outliers[j]);
			 }
		 }

		 if (consensus_set.size() > ransac_d)
		 {
			 better_model = lseround(consensus_set,fittedpoint);
			 this_error = modelerrormeasure(better_model,consensus_set,fittedpoint);
			 if (this_error < best_error)
			 {
				 best_model = better_model;
				 best_consensus_set = consensus_set;
				 best_error = this_error;
			 }
		 }
		 iteration++;
		 if (this_error < 5)
		 {
			 break;
		 }
	 }
	 if (best_model.size()<1)
	 {
		 regionjudge[i] = false;
		 printf( " region=%d ; judge: %s ; no good model\n " ,i+1,regionjudge[i]==true?"true":"false");
		 continue;
	 }
	 
	 nonzeroevalpoint = getevalpoint(nonzeropoint,best_model);
	 for (int j=0 ; j<nonzeroevalpoint.size() ; j++)
	 {
		 evalind.push_back(j);
	 }
	 errorrecord[i] = modelerrormeasure(best_model,evalind,nonzeroevalpoint);
	 evalind.clear();
	 modelrecord[i] = best_model;
	 consensus_setnumrecord[i] = best_consensus_set.size();

	 if (errorrecord[i] > (15.0 < best_model[2]/6 ? 15.0:best_model[2]/6) | consensus_setnumrecord[i] < 4*(250 < (int) best_model[2] ? 250:(int) best_model[2]))
	 {
		 regionjudge[i] = false;
	 }
	 if (errorrecord[i] < (10.0 < best_model[2]/20 ? 10.0:best_model[2]/20))
	 {
		 regionjudge[i] = true;
	 }
	 if (best_model[2]>radius*1.3 | best_model[2]<radius*0.7)
	 {
		 regionjudge[i] = false;
	 }
	 printf( " region=%d ; error value=%f ; consensus number=%d ; Center (%f,%f) ; Radius %f ; judge: %s\n " ,i+1,errorrecord[i],consensus_setnumrecord[i],modelrecord[i][0],modelrecord[i][1],modelrecord[i][2],regionjudge[i]==true?"true":"false");

	 maybe_model.clear();
	 better_model.clear();
	 best_model.clear();
	 best_consensus_set.clear();
	 consensus_set.clear();
	 best_error = 20;
	 this_error = 20;
	 maybe_inliers.clear();
	 maybe_outliers.clear();
	 diffliers.clear();
 }

 //输出结果
 Mat imresult = Mat::zeros( image.size(), CV_8UC3);

 for (int i = 0; i<image.rows ; i++)
 {
	 for (int j = 0; j<image.cols ; j++)
	 {
		 imresult.at<Vec3b>(i,j)[0] = image.at<uchar>(i,j);
		 imresult.at<Vec3b>(i,j)[1] = image.at<uchar>(i,j);
		 imresult.at<Vec3b>(i,j)[2] = image.at<uchar>(i,j);
	 }
 }


 int detectnum = 0;

 for (int i = 0; i< region.size(); i++)
 {
	 if (regionjudge[i])
	 {
		 detectnum++;
		 circle(imresult,Point((int) modelrecord[i][1],(int) modelrecord[i][0]),(int) modelrecord[i][2], Scalar(0,0,255),5);
		 printf( " num=%d   Center (%f,%f)   Radius %f \n " ,detectnum,modelrecord[i][0],modelrecord[i][1],modelrecord[i][2]);
	 }
 }
 finish = clock();
 totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
 cout << totaltime << endl;

 namedWindow("detected round", CV_WINDOW_AUTOSIZE);
 imshow("detected round", imresult);


 
 
 waitKey(0);

 return 0;
}
