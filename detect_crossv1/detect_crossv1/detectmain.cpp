#include <cv.h>
#include <highgui.h>
#include <iostream>


using namespace std;
using namespace cv;




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

vector<int> randperm2(vector<Point> contour,double tt)
{
	vector<int> numlist;
	vector<int> canlist;

	int randnum;

	int suchx;
	int suchy;

	randnum = (rand() % (contour.size()));
	suchx = contour[randnum].x;
	suchy = contour[randnum].y;
	numlist.push_back(randnum);

	for (int i = 0 ; i<contour.size() ; i++)
	{
		if (((contour[i].x-suchx)^2 + (contour[i].y-suchy)^2) > tt^2)
		{
			canlist.push_back(i);
		}
	}

	randnum = (rand() % (canlist.size()));
	
	numlist.push_back(canlist[randnum]);

	return numlist;
}

vector<double> lsline(vector<Point> inpoint)
{
	vector<double> para;
	double A = 0.0;
	double B = 0.0;
	double C = 0.0;
	double D = 0.0;
	int N = inpoint.size();
	double suchx;
	double suchy;

	for (int i = 0 ; i<N ; i++)
	{
		suchx = inpoint[i].x/1000.0;
		suchy = inpoint[i].y/1000.0;
		A = A + suchx*suchx;
		B = B + suchx;
		C = C + suchx*suchy;
		D = D + suchy;
	}

	double k,b;
	double rou,theta;
	if (A*N - B*B == 0)
	{
		theta = 0;
		rou = B/N;
	}else
	{
		k = (C*N-B*D)/(A*N - B*B);
		b = (A*D-C*B)/(A*N - B*B);
		if (k == 0)
		{
			theta = CV_PI/2;
		}else
		{
			theta = atan(-1/k);
		}
		if (k > 0)
		{
			rou = -b/sqrt(1+k*k);
		}else
		{
			rou = b/sqrt(1+k*k);
		}
	}
	rou = rou*1000.0;

	para.push_back(rou);
	para.push_back(theta);
	
	return para;

}

double modelerrormeasure(vector<double> model,vector<Point> inpoint)
{
	double rou = model[0];
	double theta = model[1];
	double errorvalue = 0.0;
	int x;
	int y;

	for (int j = 0; j<inpoint.size(); j++)
	{
		x = inpoint[j].x;
		y = inpoint[j].y;
		errorvalue = errorvalue + abs(rou-x*cos(theta) - y*sin(theta));
	}

	errorvalue = errorvalue/inpoint.size();

	return errorvalue;
}

vector<double> ransaclsline(vector<Point> inpoint,int size)
{
	double theta;
	double rou;
	double temprou;
	int x;
	int y;

	int data_num;
	int ransac_n;
	int ransac_k;
	double ransac_t;
	int ransac_d;

	int iteration;
	vector<double> maybe_model;
	vector<double> better_model;
	vector<double> best_model;
	vector<Point> best_consensus_set;
	vector<Point> consensus_set;
	double best_error = 20;
	double this_error = 20;
	
	data_num = inpoint.size();
	ransac_n = 2;
	ransac_k = 100;
	ransac_t = size>300?2.5:1.5;/////////////////////////////////
	ransac_d = 0.5*size;/////////////////////////////////

	iteration = 0;

	best_error = 20;
	this_error = 20;

	vector<int> randlist;
	vector<vector<int>> diffliers;

	while (iteration < ransac_k)
	{
		//diffliers = randperm(data_num,2);
		//randlist = diffliers[0];
		randlist = randperm2(inpoint,0.5*size);
		if (randlist.size()<2)
		{
			break;
		}
		int x1 = inpoint[randlist[0]].x;
		int y1 = inpoint[randlist[0]].y;
		int x2 = inpoint[randlist[1]].x;
		int y2 = inpoint[randlist[1]].y;


		if (y1 == y2)
		{
			theta = CV_PI/2;
		}else
		{
			theta = atan((x1-x2+0.0)/(y2-y1));
		}

		rou = x1*cos(theta) + y1*sin(theta);
		maybe_model.push_back(rou);
		maybe_model.push_back(theta);
		consensus_set.clear();
		for (int j = 0 ; j < inpoint.size() ; j++)
		{
			x = inpoint[j].x;
			y = inpoint[j].y;
			temprou = x*cos(theta) + y*sin(theta);
			if (abs(temprou-rou) < ransac_t)
			{
				consensus_set.push_back(inpoint[j]);
			}
		}

		if (consensus_set.size() > ransac_d)
		{
			better_model = lsline(consensus_set);
			this_error = modelerrormeasure(better_model,consensus_set);
			if (this_error < best_error)
			{
				best_model = better_model;
				best_consensus_set = consensus_set;
				best_error = this_error;
			}
		}
		iteration++;
		maybe_model.clear();
		if (this_error < 0.5)/////////
		{
			break;
		}
	}
	return best_model;

}

vector<vector<double>> RHT(vector<Point> contour,int size)
{
	vector<vector<double>> linepara;
	vector<vector<double>> pararecord;
	vector<int> votenum;
	vector<double> suchpara;
	int num;
	double theta;
	double rou;
	int judgeline = -1;
	vector<vector<int>> difflist;
	vector<int> randlist;
	bool flagcreate;

	vector<Point> outpointold;

	outpointold = contour;

	for (int k=0; k< 1500 ; k++)
	{

		if (outpointold.size()<0.5*size)
		{
			break;
		}

		//num = outpointold.size();
		//difflist = randperm(num,2);
		//randlist = difflist[0];

		randlist = randperm2(outpointold,0.5*size);
		if (randlist.size()<2)
		{
			break;
		}
		
		int x1 = outpointold[randlist[0]].x;
		int y1 = outpointold[randlist[0]].y;
		int x2 = outpointold[randlist[1]].x;
		int y2 = outpointold[randlist[1]].y;
		

		if (y1 == y2)
		{
			theta = CV_PI/2;
		}else
		{
			theta = atan((x1-x2+0.0)/(y2-y1));
		}

		rou = x1*cos(theta) + y1*sin(theta);
		flagcreate = true;

		for (int kk = 0; kk<pararecord.size(); kk++)
		{
			if (abs(rou-pararecord[kk][0])< (size>300?5:3) && abs(theta-pararecord[kk][1])< CV_PI/36)
			{
				flagcreate = false;
				votenum[kk]++;
				if (votenum[kk] > 10)
				{
					judgeline = kk;
				}
				break;
			}
		}

		if (flagcreate)
		{
			suchpara.push_back(rou);
			suchpara.push_back(theta);
			pararecord.push_back(suchpara);
			votenum.push_back(1);
		}
		

		double temprou;
		vector<Point> inpoint;
		vector<Point> outpoint;
		vector<double> fittedpara;


		if (judgeline > -1)
		{
			rou = pararecord[judgeline][0];
			theta = pararecord[judgeline][1];
			for (int i=0 ; i<outpointold.size() ; i++)
			{
				temprou = outpointold[i].x*cos(theta) + outpointold[i].y*sin(theta);
				if (abs(temprou-rou) < (size>300?5:3))
				{	
					inpoint.push_back(outpointold[i]);
				}else
				{
					outpoint.push_back(outpointold[i]);
				}
			}
			if (inpoint.size()>0.5*size)
			{
				//cout<<1<<endl;
				fittedpara = ransaclsline(inpoint,size);
				if (fittedpara.size() <2)
				{
					votenum[judgeline] = 1;
				}else
				{
					outpointold = outpoint;
					linepara.push_back(fittedpara);
				}
				
				//suchpara.push_back(rou);
				//suchpara.push_back(theta);
				//linepara.push_back(suchpara);

			}else
			{
				votenum[judgeline] = 1;
			}
		}
		
		suchpara.clear();
		inpoint.clear();
		outpoint.clear();
		judgeline = -1;


	}


	return linepara;
}


vector<vector<double>> HT(vector<Point> contour)
{
	vector<vector<double>> linepara;
	int thetaresolutionnum = 72;
	double thetaresolution = 2*CV_PI/thetaresolutionnum;
	double rouresolution = 2.0;
	int rouresolutionnum = cvRound(1650/rouresolution);

	vector<vector<int>> houghspace;//[theta,rou],theta从pi/2 到 -pi/2
	vector<int> houghspacesuchtheta;

	vector<Point> outpointold;

	outpointold = contour;

	for (int i = 0 ; i<thetaresolutionnum ; i++)
	{
		for (int j = 0 ; j<rouresolutionnum ; j++)
		{
			houghspacesuchtheta.push_back(0);
		}
		houghspace.push_back(houghspacesuchtheta);
		houghspacesuchtheta.clear();
	}

	int x;
	int y;
	double suchrou;
	int maxvalue;
	int maxthetapo;
	int maxroupo;

	double temprou;
	vector<Point> inpoint;
	vector<Point> outpoint;
	vector<double> fittedpara;
	double rou;
	double theta;

	for (int num = 0 ; num<7 ; num++)
	{
		maxvalue = 100;
		maxthetapo = -1;
		maxroupo = -1;
		if (outpointold.size()<100)
		{
			break;
		}
	
		for (int i = 0 ; i<outpointold.size() ; i++)
		{
			x = outpointold[i].x;
			y = outpointold[i].y;
	
			for (int j = 0 ; j<thetaresolutionnum ; j++)
			{
				suchrou = x*cos(j*thetaresolution) + y*sin(j*thetaresolution);
				if (floor(suchrou/rouresolution)>=0 & floor(suchrou/rouresolution) < rouresolutionnum)
				{
					houghspace[j][floor(suchrou/rouresolution)]++;
					if (houghspace[j][floor(suchrou/rouresolution)] > maxvalue)
					{
						maxvalue = houghspace[j][floor(suchrou/rouresolution)];
						maxthetapo = j;
						maxroupo = floor(suchrou/rouresolution);
					}
				}
				
			}
		}
		if (maxroupo<0)
		{
			break;
		}

		rou = maxroupo*rouresolution+rouresolution/2;
		theta = maxthetapo*thetaresolution;
		for (int i=0 ; i<outpointold.size() ; i++)
		{
			temprou = outpointold[i].x*cos(theta) + outpointold[i].y*sin(theta);
			if (abs(temprou-rou) < 3)
			{	
				inpoint.push_back(outpointold[i]);
			}else
			{
				outpoint.push_back(outpointold[i]);
			}
		}
		if (inpoint.size()>100)
		{

			outpointold = outpoint;
			fittedpara = lsline(inpoint);
			linepara.push_back(fittedpara);

		}else
		{
			break;
		}

		inpoint.clear();
		outpoint.clear();

		for (int i = 0 ; i<thetaresolutionnum ; i++)
		{
			for (int j = 0 ; j<rouresolutionnum ; j++)
			{
				houghspace[i][j] = 0;
			}
		}
	}

	return linepara;

}

vector<double> calculate_center(vector<double> p1,vector<double> p2,vector<double> p3,vector<double> p4)
{
	//p1,p2为一组平行线
	//p3,p4为一组平行线
	double rou1 = p1[0];
	double theta1 = p1[1];
	double rou2 = p2[0];
	double theta2 = p2[1];
	double rou3 = p3[0];
	double theta3 = p3[1];
	double rou4 = p4[0];
	double theta4 = p4[1];
	double interp1x;
	double interp1y;
	double interp2x;
	double interp2y;
	double interp3x;
	double interp3y;
	double interp4x;
	double interp4y;
	vector<double> centerpo;
	//intersection point of p1,p3
	interp1x = (rou1*sin(theta3)-rou3*sin(theta1))/sin(theta3-theta1);
	interp1y = (rou1*cos(theta3)-rou3*cos(theta1))/sin(theta1-theta3);
	//intersection point of p1,p4
	interp2x = (rou1*sin(theta4)-rou4*sin(theta1))/sin(theta4-theta1);
	interp2y = (rou1*cos(theta4)-rou4*cos(theta1))/sin(theta1-theta4);
	//intersection point of p2,p3
	interp3x = (rou2*sin(theta3)-rou3*sin(theta2))/sin(theta3-theta2);
	interp3y = (rou2*cos(theta3)-rou3*cos(theta2))/sin(theta2-theta3);
	//intersection point of p2,p4
	interp4x = (rou2*sin(theta4)-rou4*sin(theta2))/sin(theta4-theta2);
	interp4y = (rou2*cos(theta4)-rou4*cos(theta2))/sin(theta2-theta4);

	centerpo.push_back((interp1x+interp2x+interp3x+interp4x)/4);
	centerpo.push_back((interp1y+interp2y+interp3y+interp4y)/4);

	return centerpo;
}

int main( int argc, char** argv )
{

	//打开图像

	//char* imageName = argv[1];
	double crosslength = 800.0;//能包含十字的正方形框边长
	clock_t start,finish;

	char* imageName = argv[1];
	//char* imageName = "D:/feiQ自动接收/杨撷成(10.1.2.22)/2015-08-14 10_08_23/Capture/4_Image.bmp";
	//char* imageName = "D:/工作/照片测试/2015-06-26 桌上视觉测试台/[DEV0]-ID002172-BW.bmp";
	//char* imageName = "D:/工作/照片测试/2015-08-21 测试照片/[DEV1]-ID014029-BW.bmp";
	Mat image;
	image = imread( imageName,0);
	if( !image.data )
	{
		printf( " No image data \n " );
		return -1;
	}

	//namedWindow("image", CV_WINDOW_AUTOSIZE);
	//imshow("image", image);

	
/*
	equalizeHist(image,image);

	

	Mat imgwhite(image.rows,image.cols,CV_8UC1,255);
	Mat imagebar;
	absdiff(imgwhite,image,imagebar);

	Mat imagepre;



	morphologyEx(imagebar,imagepre,MORPH_OPEN,getStructuringElement(MORPH_RECT,Size(3,3)));
	morphologyEx(imagepre,imagepre,MORPH_CLOSE,getStructuringElement(MORPH_RECT,Size(5,5)));

	//absdiff(imagepre,imagebar,imagepre);

	namedWindow("imagepre", CV_WINDOW_AUTOSIZE);
	imshow("imagepre", imagepre);


	Mat imagebw;
	threshold(imagepre,imagebw,0,255,CV_THRESH_OTSU);
	namedWindow("imagebw", CV_WINDOW_AUTOSIZE);
	imshow("imagebw", imagebw);
*/
	start = clock();

	Mat imedge,imedge2;
	GaussianBlur(image,imedge,Size(11,11),2);
	Canny( imedge, imedge, 50, 150 , 3 );
	
	morphologyEx(imedge,imedge,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(30,30)));
	//namedWindow("imedge", CV_WINDOW_AUTOSIZE);
	//imshow("imedge", imedge);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours( imedge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0) );

	int contournum = 0;

	Mat imcontour = Mat::zeros( imedge.size(), CV_8U);

	vector<vector<Point>> contoursfilt;
	vector<Point> onecontour;
	for(int i = 0;i<contours.size();i++){
		onecontour = contours[i];
		if (onecontour.size() > 400){
			contoursfilt.push_back(onecontour);
			contournum++;
		}
	}

	drawContours(imcontour,contoursfilt,-1,Scalar(255),2);
	//namedWindow("imcontour", CV_WINDOW_AUTOSIZE);
	//imshow("imcontour", imcontour);


	Mat imagec3 = Mat::zeros( image.size(), CV_8UC3);
	for (int i = 0; i<image.rows ; i++)
	{
		for (int j = 0; j<image.cols ; j++)
		{
			imagec3.at<Vec3b>(i,j)[0] = image.at<uchar>(i,j);
			imagec3.at<Vec3b>(i,j)[1] = image.at<uchar>(i,j);
			imagec3.at<Vec3b>(i,j)[2] = image.at<uchar>(i,j);
		}
	}
	Mat imresult = imagec3.clone();
	double rou;
	double theta;
	double x0,y0;
	Point pt1,pt2;
	
	finish = clock();
	double totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
	cout << totaltime << endl;

	//筛选轮廓 十字轮廓已在contourfilt中
	//识别结果参数
	vector<double> centerx;
	vector<double> centery;
	vector<double> crosscentertheta;
	vector<double> crosscenter;
	vector<int> preciselevel;

	int tempx;
	int tempy;
	vector<bool> regionjudge;
	vector<vector<double>> lineparameter;//[[rou,theta]]
	vector<vector<int>> thetacluster;
	vector<int> suchcluster;
	vector<double> clusterthetavalue;
	vector<double> clusterrouvalue;
	vector<vector<int>> thetaclusterrough;
	vector<double> clusterthetavaluerough;
	vector<double> clusterrouvaluerough;
	double orththeta;

	bool thetanew;

	for (int i = 0;i<contoursfilt.size();i++)
	{
		centerx.push_back(-1);
		centery.push_back(-1);
		crosscentertheta.push_back(0);
		regionjudge.push_back(true);
		preciselevel.push_back(0);
		int maxx = 0;
		int maxy = 0;
		int minx = image.cols;
		int miny = image.rows;
		onecontour = contoursfilt[i];
		for (int j = 0; j<onecontour.size(); j++)
		{
			tempx = onecontour[j].x;
			tempy = onecontour[j].y;
			if (tempx > maxx)
			{
				maxx = tempx;
			}
			if (tempx < minx)
			{
				minx = tempx;
			}
			if (tempy > maxy)
			{
				maxy = tempy;
			}
			if (tempy < miny)
			{
				miny = tempy;
			}
			if (maxx-minx>crosslength | maxy-miny>crosslength)
			{
				regionjudge[i] = false;
				break;
			}
		}
		if (!regionjudge[i])
		{
			continue;
		}
		if (abs((maxx-minx)-(maxy-miny))>100)
		{
			regionjudge[i] = false;
			continue;
		}

		lineparameter = RHT(onecontour,(maxx-minx)<(maxy-miny)?(maxx-minx):(maxy-miny));
		//lineparameter = HT(onecontour);
		
		for (int k = 0;k<lineparameter.size();k++)
		{
			rou = lineparameter[k][0];
			theta = lineparameter[k][1];
			x0 = rou*cos(theta);
			y0 = rou*sin(theta);
			pt1.x = cvRound(x0+2000*(-sin(theta)));
			pt1.y = cvRound(y0+2000*(cos(theta)));
			pt2.x = cvRound(x0-2000*(-sin(theta)));
			pt2.y = cvRound(y0-2000*(cos(theta)));
			line(imresult,pt1,pt2,Scalar(0,0,255),1,CV_AA);
		}
		
		if (lineparameter.size()>3)
		{
			for (int k = 0;k<lineparameter.size();k++)
			{
				rou = lineparameter[k][0];
				theta = lineparameter[k][1];
				int kt=k+1;
				while (kt<lineparameter.size())
				{
					if ((abs(theta-lineparameter[kt][1])<CV_PI/180&abs(rou-lineparameter[kt][0])<2)|((CV_PI-abs(theta-lineparameter[kt][1]))<CV_PI/180&abs(rou+lineparameter[kt][0])<2))
					{
						vector<vector<double>>::iterator it = lineparameter.begin()+kt;
						lineparameter.erase(it);
						kt--;
					}
					kt++;
				}
			}
		}

		if (lineparameter.size()!=4)
		{
			regionjudge[i] = false;
			continue;
		}

		for (int k = 0;k<lineparameter.size();k++)
		{
			rou = lineparameter[k][0];
			theta = lineparameter[k][1];

			thetanew = true;
			for (int kk = 0;kk<thetacluster.size();kk++)
			{
				if ((abs(theta-clusterthetavalue[kk])<CV_PI/60&abs(rou-clusterrouvalue[kk])<40)|((CV_PI-abs(theta-clusterthetavalue[kk]))<CV_PI/60&abs(rou+clusterrouvalue[kk])<40))
				{
					thetacluster[kk].push_back(k);
					thetanew = false;
					break;
				}
			}
			if (thetanew)
			{
				suchcluster.push_back(k);
				thetacluster.push_back(suchcluster);
				suchcluster.clear();
				clusterthetavalue.push_back(theta);
				clusterrouvalue.push_back(rou);
			}
			//rough theta
			thetanew = true;
			for (int kk = 0;kk<thetaclusterrough.size();kk++)
			{
				if ((abs(theta-clusterthetavaluerough[kk])<CV_PI/18&abs(rou-clusterrouvaluerough[kk])<60)|((CV_PI-abs(theta-clusterthetavaluerough[kk]))<CV_PI/18&abs(rou+clusterrouvaluerough[kk])<60))
				{
					thetaclusterrough[kk].push_back(k);
					thetanew = false;
					break;
				}
			}
			if (thetanew)
			{
				suchcluster.push_back(k);
				thetaclusterrough.push_back(suchcluster);
				suchcluster.clear();
				clusterthetavaluerough.push_back(theta);
				clusterrouvaluerough.push_back(rou);
			}
		}

		//judge 4 lines of cross
		for (int kk = 0;kk<thetacluster.size();kk++)
		{
			if (thetacluster[kk].size() != 2)
			{
				continue;
			}
			//计算候选线的法向角度
			if (abs(lineparameter[thetacluster[kk][0]][1]-lineparameter[thetacluster[kk][1]][1])<CV_PI/2)
			{
				orththeta = (lineparameter[thetacluster[kk][0]][1]+lineparameter[thetacluster[kk][1]][1])/2 + CV_PI/2;
			}else
			{
				orththeta = (lineparameter[thetacluster[kk][0]][1]+lineparameter[thetacluster[kk][1]][1])/2 + CV_PI;
			}
			
			if (orththeta > CV_PI/2)
			{
				orththeta = orththeta - CV_PI;
			}

			for (int kkk = 0; kkk<thetacluster.size();kkk++)
			{					
				if (abs(clusterthetavalue[kkk] - orththeta)<CV_PI/36|(CV_PI-abs(clusterthetavalue[kkk] - orththeta))<CV_PI/36)
				{
					if (thetacluster[kkk].size() == 2)
					{
						if (abs(lineparameter[thetacluster[kkk][0]][1]-lineparameter[thetacluster[kkk][1]][1])<CV_PI/2)
						{
							crosscentertheta[i] = (lineparameter[thetacluster[kkk][0]][1]+lineparameter[thetacluster[kkk][1]][1])/2;
						}else
						{
							crosscentertheta[i] = (lineparameter[thetacluster[kkk][0]][1]+lineparameter[thetacluster[kkk][1]][1]+CV_PI)/2;
							if (crosscentertheta[i]>CV_PI/2)
							{
								crosscentertheta[i] = crosscentertheta[i]-CV_PI;
							}
						}

						if (abs(crosscentertheta[i]-orththeta)<CV_PI/2)
						{
							crosscentertheta[i] = (crosscentertheta[i]+orththeta)/2;
						}else
						{
							crosscentertheta[i] = (crosscentertheta[i]+orththeta+CV_PI)/2;
							if (crosscentertheta[i]>CV_PI/2)
							{
								crosscentertheta[i] = crosscentertheta[i]-CV_PI;
							}
						}
						crosscenter = calculate_center(lineparameter[thetacluster[kk][0]],lineparameter[thetacluster[kk][1]],lineparameter[thetacluster[kkk][0]],lineparameter[thetacluster[kkk][1]]);
						centerx[i] = crosscenter[0];
						centery[i] = crosscenter[1];
						//最好情况，计算中心，旋转角
						preciselevel[i] = 1;
						break;
					}
				}
			}
			if (centerx[i]<0)
			{
				for (int kkk = 0; kkk<thetaclusterrough.size();kkk++)
				{					
					if (abs(clusterthetavaluerough[kkk] - orththeta)<CV_PI/18|(CV_PI-abs(clusterthetavaluerough[kkk] - orththeta))<CV_PI/18)
					{
						if (thetaclusterrough[kkk].size() == 2)
						{
							crosscentertheta[i] = orththeta;
							crosscenter = calculate_center(lineparameter[thetacluster[kk][0]],lineparameter[thetacluster[kk][1]],lineparameter[thetaclusterrough[kkk][0]],lineparameter[thetaclusterrough[kkk][1]]);
							centerx[i] = crosscenter[0];
							centery[i] = crosscenter[1];
							//次好情况，计算中心，旋转角
							preciselevel[i] = 2;
							break;
						}
					}
				}
			}
			if (centerx[i]>0)
			{
				break;
			}
		}
		if (centerx[i]<0)
		{
			for (int kk = 0;kk<thetaclusterrough.size();kk++)
			{
				if (thetaclusterrough[kk].size() != 2)
				{
					continue;
				}

				
				if (abs(lineparameter[thetaclusterrough[kk][0]][1]-lineparameter[thetaclusterrough[kk][1]][1])<CV_PI/2)
				{
					orththeta = (lineparameter[thetaclusterrough[kk][0]][1]+lineparameter[thetaclusterrough[kk][1]][1])/2 + CV_PI/2;
				}else
				{
					orththeta = (lineparameter[thetaclusterrough[kk][0]][1]+lineparameter[thetaclusterrough[kk][1]][1])/2 + CV_PI;
				}

				if (orththeta > CV_PI/2)
				{
					orththeta = orththeta - CV_PI;
				}

				for (int kkk = 0; kkk<thetaclusterrough.size();kkk++)
				{					
					if (abs(clusterthetavaluerough[kkk] - orththeta)<CV_PI/18|(CV_PI-abs(clusterthetavaluerough[kkk] - orththeta))<CV_PI/18)
					{
						if (thetaclusterrough[kkk].size() == 2)
						{
							if (abs(lineparameter[thetaclusterrough[kkk][0]][1]-lineparameter[thetaclusterrough[kkk][1]][1])<CV_PI/2)
							{
								crosscentertheta[i] = (lineparameter[thetaclusterrough[kkk][0]][1]+lineparameter[thetaclusterrough[kkk][1]][1])/2;
							}else
							{
								crosscentertheta[i] = (lineparameter[thetaclusterrough[kkk][0]][1]+lineparameter[thetaclusterrough[kkk][1]][1]+CV_PI)/2;
								if (crosscentertheta[i]>CV_PI/2)
								{
									crosscentertheta[i] = crosscentertheta[i]-CV_PI;
								}
							}

							if (abs(crosscentertheta[i]-orththeta)<CV_PI/2)
							{
								crosscentertheta[i] = (crosscentertheta[i]+orththeta)/2;
							}else
							{
								crosscentertheta[i] = (crosscentertheta[i]+orththeta+CV_PI)/2;
								if (crosscentertheta[i]>CV_PI/2)
								{
									crosscentertheta[i] = crosscentertheta[i]-CV_PI;
								}
							}
							crosscenter = calculate_center(lineparameter[thetaclusterrough[kk][0]],lineparameter[thetaclusterrough[kk][1]],lineparameter[thetaclusterrough[kkk][0]],lineparameter[thetaclusterrough[kkk][1]]);
							centerx[i] = crosscenter[0];
							centery[i] = crosscenter[1];
							//粗略情况，计算中心，旋转角
							preciselevel[i] = 3;
							break;
						}
					}
				}
				if (centerx[i]>0)
				{
					break;
				}
			}
		}
		if (centerx[i] < 0)
		{
			regionjudge[i] = false;
		}
		thetacluster.clear();
		clusterthetavalue.clear();
		clusterrouvalue.clear();
		thetaclusterrough.clear();
		clusterthetavaluerough.clear();
		clusterrouvaluerough.clear();
	}
	cout<<contournum<<endl;


	int detectnum = 0;

	for (int i = 0; i< contoursfilt.size(); i++)
	{
		if (regionjudge[i])
		{
			detectnum++;
			line(imresult,Point(centerx[i]-100*sin(-crosscentertheta[i]),centery[i]-100*cos(-crosscentertheta[i])),Point(centerx[i]+100*sin(-crosscentertheta[i]),centery[i]+100*cos(-crosscentertheta[i])),Scalar(255,0,0),2);
			line(imresult,Point(centerx[i]-100*sin(-crosscentertheta[i]-CV_PI/2),centery[i]-100*cos(-crosscentertheta[i]-CV_PI/2)),Point(centerx[i]+100*sin(-crosscentertheta[i]-CV_PI/2),centery[i]+100*cos(-crosscentertheta[i]-CV_PI/2)),Scalar(255,0,0),2);
			printf( " num=%d   Center (%f,%f)   theta %f preciselevel %d \n" ,detectnum,centery[i],centerx[i],crosscentertheta[i]>0?crosscentertheta[i]:crosscentertheta[i]+CV_PI/2,preciselevel[i]);
		}
	}

	finish = clock();
	totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
	cout << totaltime << endl;

	namedWindow("imresult", CV_WINDOW_AUTOSIZE);
	imshow("imresult", imresult);


	waitKey(0);

	return 0;
}