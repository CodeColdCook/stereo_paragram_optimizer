#include <iostream>
#include <algorithm>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
///BA
#include "ORBextractor.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include "Converter.h"
//#include "Optimizer.h"
#include "BA.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

///EuRoC
//#define CONFIG_PATH "./config/EuRoC.yaml"
//#define IMG_PATH_R "./data/1403638561927829504.png"
//#define IMG_PATH_L "./data/1403638562177829376.png"
///zhijiang
#define CONFIG_PATH "./config/zhijiang.yaml"
//#define IMG_PATH_R "./data/frame0040_R.jpg"
static string IMG_PATH_R;
//#define IMG_PATH_L "./data/frame0040_L.jpg"
static string IMG_PATH_L;
#define OUT_FILE_PATH "./config/record.yaml"
#define FRAME_GRID_ROWS 48 // 图像网格数量
#define FRAME_GRID_COLS 64 // 图像网格数量
typedef pair<int,int> Match;
/// frame
float fx, fy, cx, cy;
float fx_r, fy_r, cx_r, cy_r; /// stereo
static Mat K,DistCoef(4,1,CV_32F);
static Mat K_r,DistCoef_r(4,1,CV_32F); /// stereo
vector<size_t> mGrid_l[FRAME_GRID_COLS][FRAME_GRID_ROWS];
vector<size_t> mGrid_r[FRAME_GRID_COLS][FRAME_GRID_ROWS];
static float mfGridElementWidthInv; //每个网格宽度的逆 
static float mfGridElementHeightInv; //每个网格高度的逆
static float mnMinX,mnMinY,mnmaxX,mnmaxY;
/// ORB
vector<float> mvInvLevelSigma2;
static int N_l, N_r;
Mat mDescriptors_l, mDescriptors_r;
vector<KeyPoint> mvKeys_l, mvKeys_r;
vector<KeyPoint> mvKeysun_l,mvKeysun_r;
vector<Point2f> mvbPrevMatched;
vector<int> mvIniMatches;
/// matches
static const int HISTO_LENGTH = 30;
const int TH_HIGH = 100;
const int TH_LOW = 50;
static float windowSize = 100;
vector<int> vMatchedDistance;
vector<int> vnMatches21;
int nmatches=0;
float mfNNratio = 0.9; // 第一最优距离/第二最优距离  阈值  此阈值判别该特征点只与第一距离的描述子距离近,而与其他描述子距离远  
bool mbCheckOrientation = true; // 匹配过程中是否需要判别旋转的标志量
vector<int> vnMatches12;
vector<int> rotHist[HISTO_LENGTH];
/// Initialization
int min_th_n_Triangulates = 25;
float min_th_parallax = 0.1;
float mSigma = 1.0, mSigma2 = 1.0;
vector< vector<size_t> > mvSets;
const int mMaxIterations = 100;
vector<Match> mvinit_Matches12;
vector<bool> mvbinit_Matched1;
vector<Point2f> vPn1, vPn2;
Mat T1, T2, T2inv, T2t, R21, t21;
static int N;
vector<Point3f> vP3D;   // 最终形式：vP3D,reserve(N_l)  vP3D[mvinit_Matches12[i].first]=Point3f 能三角化，且是内点
vector<bool> vbTriangulated; // 最终形式：size=N_l，储存着左图每个特征点对应的  是否能三角化
/// Map
Mat Tc2w; // T21[R21|t21]
Mat Tc1w;
vector<pair<int,Mat>> MapPoints; // 最终形式：大小为最终的nmatches，vector[i]=pair<左图特征点idx，Mat（3,3,CV32F）>



bool PosInGrid(const KeyPoint &kp, int &posX, int &posY);
void AssignFeatureToGrid(vector<KeyPoint> &kp_un, vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS]);
void KPundistort(vector<KeyPoint> &kps, vector<KeyPoint> &kps_un, const int N, const Mat &K_, const Mat &DistCoef_);
void ComputeImageBounds(Mat img);
int DescriptorDistance(const Mat &a, const Mat &b);
void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
void SearchForInitialization();
vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS]);
void Normalize(const vector<KeyPoint> &vKeys, vector<Point2f> &vNormalizedPoints, Mat &T);
Mat ComputeH21(const vector<Point2f> &vP1, const vector<Point2f> &vP2);
Mat ComputeF21(const vector<Point2f> &vP1,const vector<Point2f> &vP2);
float CheckHomography(const Mat &H21, const Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
float CheckFundamental(const Mat &F21, vector<bool> &vbMatchesInliers, float sigma);
void FindHomography(vector<bool> &vbMatchesInliers, float &score, Mat &H21);
void FindFundamental(vector<bool> &vbMatchesInliers, float &score, Mat &F21);
void Get_mvSets();
void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
int CheckRT(const cv::Mat &R, const cv::Mat &t, vector<bool> &vbMatchesInliers
                , vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);
bool ReconstructF(vector<bool> &vbMatchesInliers, Mat F21, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

void Draw_keypoints(const Mat img_l_, const Mat img_r_,
                    vector<KeyPoint> vKeypointsDraw_l_, vector<KeyPoint> vKeypointsDraw_r_, 
                    vector<int> vnMatches_lr);


int main(int argc, char** argv)
{   
    cout << "welecome " << endl;   
    FileStorage fSettings(CONFIG_PATH, FileStorage::READ);
    min_th_parallax = fSettings["min_th_parallax"];
    min_th_n_Triangulates = fSettings["min_th_n_Triangulates"];

    fx = fSettings["Camera.fx"];
    fy = fSettings["Camera.fy"];
    cx = fSettings["Camera.cx"];
    cy = fSettings["Camera.cy"];
    K = Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(0,2) = cx;
    K.at<float>(1,1) = fy;
    K.at<float>(1,2) = cy;
    float invfx = 1.0f/fx; float invfy = 1.0f/fy;

    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    // 右目
    fx_r = fSettings["Camera.fx_r"];
    fy_r = fSettings["Camera.fy_r"];
    cx_r = fSettings["Camera.cx_r"];
    cy_r = fSettings["Camera.cy_r"];
    K_r = Mat::eye(3,3,CV_32F);
    K_r.at<float>(0,0) = fx_r;
    K_r.at<float>(0,2) = cx_r;
    K_r.at<float>(1,1) = fy_r;
    K_r.at<float>(1,2) = cy_r;
    float invfx_r = 1.0f/fx_r; float invfy_r = 1.0f/fy_r;

    DistCoef_r.at<float>(0) = fSettings["Camera.k1_r"];
    DistCoef_r.at<float>(1) = fSettings["Camera.k2_r"];
    DistCoef_r.at<float>(2) = fSettings["Camera.p1_r"];
    DistCoef_r.at<float>(3) = fSettings["Camera.p2_r"];

    /*
    float k3 = fSettings["Camera.k3"];
    if(k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }*/

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];                 //提取fast特征点的默认阈值 20
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];            //如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    ORBextractor *mpORBextractorLeft, *mpIniORBextractor;
    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    mpIniORBextractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    
    Mat img_l, img_r;
    IMG_PATH_L = static_cast<string>(fSettings["IMG_PATH_L"]);
    IMG_PATH_R = static_cast<string>(fSettings["IMG_PATH_R"]);
    img_l = imread(IMG_PATH_L, CV_LOAD_IMAGE_ANYCOLOR);
    img_r = imread(IMG_PATH_R, CV_LOAD_IMAGE_ANYCOLOR);
    imshow("img_l", img_l);
    waitKey();
    if(img_r.channels() == 3)
    {
        cout << "Converting imgs to gray..." << endl; 
        cvtColor(img_l, img_l, CV_RGB2GRAY);
        cvtColor(img_r, img_r, CV_RGB2GRAY);
    }
    
    (*mpORBextractorLeft)(img_l,Mat(),mvKeys_l,mDescriptors_l);
    (*mpORBextractorLeft)(img_r,Mat(),mvKeys_r,mDescriptors_r);
    N_l = mvKeys_l.size();
    N_r = mvKeys_r.size();
    cout << "feature number of img_1: " << N_l<<endl;
    cout << "feature number of img_2: " << N_r<<endl;

    // 关键点的失真矫正
    Mat K_1 = K.clone(); Mat K_2 = K_r.clone();
    K_1.convertTo(K_1,CV_32F); K_2.convertTo(K_2,CV_32F);
    Mat DistCoef_1 = DistCoef.clone(); Mat DistCoef_2 = DistCoef_r.clone();  
    KPundistort(mvKeys_l, mvKeysun_l, N_l, K, DistCoef);
    KPundistort(mvKeys_r, mvKeysun_r, N_r, K_r, DistCoef_r);
    ComputeImageBounds(img_l);
    AssignFeatureToGrid(mvKeysun_l,mGrid_l);
    AssignFeatureToGrid(mvKeysun_r,mGrid_r);
                                            cout << "AssignFeatureToGrid is ok...  " << endl;
    // 初始化第一帧数据，准备与下一帧进行单目初始化
    // vbPrevMatched 储存着 frame1的特征点坐标
    if(N_l >100)
    {
        mvbPrevMatched.resize(N_l);
        for(int i=0;i<N_l;i++)
            mvbPrevMatched[i] = mvKeysun_l[i].pt;
        fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
        // mpInitializer =  new Initializer(mCurrentFrame,1.0,200); 
    }
    // 处理第二帧数据
    if(N_r < 100)
        cout << "Too difficult to track..." << endl;
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    // 初始化下的特征匹配
    vnMatches12 = vector<int>(N_l, -1);
    vMatchedDistance = vector<int>(N_r, INT_MAX);
    vnMatches21= vector<int>(N_r, -1);
    
    SearchForInitialization(); 
cout << "SearchForInitialization() is ok... and matches is: " << nmatches<< endl;

    //cout << nmatches << endl;
    //Update prev matched  更新前一帧的匹配点对vbPrevMatched容器
    for(size_t i1=0,iend1=vnMatches12.size();i1<iend1;i1++)
        if(vnMatches12[i1]>=0)
            mvbPrevMatched[i1] = mvKeysun_r[i1].pt;
Draw_keypoints(img_l, img_r,
                    mvKeys_l, mvKeys_r, 
                    vnMatches12);
    // 取出随机匹配点对，存放在容器 mvSets[i][j] 中，i为左图特征点索引（mvKeysun_l）,j为右图索引（mvKeysun_r）
    // 过程vector：mvinit_Matches12储存着匹配点对的索引，vector：mvbinit_Matched1记录左图特征点是否有匹配
    Get_mvSets();       
cout << "Get_mvSets() is ok..." << endl;

    /// 迭代计算H和F
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    vbMatchesInliersH = vbMatchesInliersF = vector<bool>(N,false);
    float SH, SF;
    Mat H, F, H21, F21;
    Normalize(mvKeysun_l,vPn1, T1);
    Normalize(mvKeysun_r,vPn2, T2);
    T2inv = T2.inv();
    T2t = T2.t();
cout << "It is going to find H ..." << endl;
    FindHomography(ref(vbMatchesInliersH), ref(SH), ref(H21));     
cout << "FindHomography is ok..." << endl;
cout << "It is going to find F..." << endl;
    FindFundamental(ref(vbMatchesInliersF), ref(SF), ref(F21));     
cout << "FindFundamental is ok..." << endl;
cout << "SH: " << SH << endl << "SF: " << SF << endl;
    // cout << "F21: " << endl << F21 << endl;
    float RH = SH/(SH+SF);
    //ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    bool ReconstructF_isok = ReconstructF(vbMatchesInliersF, F21, vbTriangulated, min_th_parallax, min_th_n_Triangulates);
    if(!ReconstructF_isok)
        cout << "ReconstructF is not ok ...  Please make another shoot" << endl;
    if(ReconstructF_isok)
    {
        cout << "ReconstructF is ok " << endl;
        cout <<"nmatches before Triangulated: " << nmatches << endl;
        for(size_t i=0, iend = vnMatches12.size();i!=iend;i++ )
        {
            if(vnMatches12[i] >= 0 && !vbTriangulated[i])
            {
                vnMatches12[i] = -1;
                nmatches--;
            }
        }
        cout <<"nmatches after Triangulated: " << nmatches << endl;
        Tc1w = Mat::eye(4,4,CV_32F);
        Tc2w = Mat::eye(4,4,CV_32F);
        R21.copyTo(Tc2w.rowRange(0,3).colRange(0,3)); cout << "R21" << R21 << endl;
        t21.copyTo(Tc2w.rowRange(0,3).col(3));        cout << "t21" << t21 << endl;
        MapPoints.reserve(nmatches);
        for(size_t i=0; i<vnMatches12.size(); i++)
        {
            if(vnMatches12[i]<0)
                continue;
            else
            {
                cout<< vP3D[i] << endl;
                Mat Mat3Dpoint(vP3D[i]);
                MapPoints.push_back(make_pair(i,Mat3Dpoint));
            } 
        }
        cout << "MapPoints.size(): " <<MapPoints.size() << endl;
        // 提取对应的3d点 2d点 R21, t21
        vector<Point3f> points_3d;
        vector<Point2f> points_2d;
        points_2d.reserve(MapPoints.size());
        points_3d.reserve(MapPoints.size());
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
        for(size_t i=0;i<MapPoints.size();i++)
        {   
            size_t idx_l = MapPoints[i].first;
            size_t idx_r = vnMatches12[idx_l];
            Point3f point_3d = Point3f(MapPoints[i].second.at<float>(0,0),MapPoints[i].second.at<float>(1,0),MapPoints[i].second.at<float>(2,0));
            Point2f point_2d = mvKeysun_r[idx_r].pt;
            points_3d.push_back(point_3d);
            points_2d.push_back(point_2d);
        }
        K_r.convertTo(K_r,CV_64F);
        R21.convertTo(R21,CV_64F);
        t21.convertTo(t21,CV_64F);
        cout << "R21" << R21 << endl; cout << "t21" << t21 << endl;
        Mat T21;
        vector<Mat> Mat_Points_3d;
        /// BA

        Optimizer::bundleAdjustment(points_3d, points_2d, T21, R21, t21,K_r, Mat_Points_3d);
        cout << "T21 after BA: " << T21 << endl;
    }
    
    return 0;
}

bool PosInGrid(const KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY)*mfGridElementHeightInv);
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>= FRAME_GRID_ROWS)
        return false;
    return true;
}

void AssignFeatureToGrid(vector<KeyPoint> &kp_un, vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS])
{
    int nReserve = 0.5f*N_l/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0;i<FRAME_GRID_COLS;i++)
        for(unsigned int j=0;j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);
    for(int i=0;i<N_l;i++)
    {
        KeyPoint kp = kp_un[i];
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i); // i 为特征点的索引（在mvKeysun_l中）
    }
}

void KPundistort(vector<KeyPoint> &kps, vector<KeyPoint> &kps_un, const int N, const Mat &K_, const Mat &DistCoef_)
// 调整了K_和DistCoef_
{
    Mat mat_p(N,2,CV_32F);
    for(int i=0;i<N;i++)
    {
        mat_p.at<float>(i,0) = kps[i].pt.x;
        mat_p.at<float>(i,1) = kps[i].pt.y;
    }
    mat_p = mat_p.reshape(2); // 先将定义好的mat—_p转为2通道，供矫正使用，矫正后可以还原，对2列的数据进行取值
    undistortPoints(mat_p,mat_p,K_,DistCoef_,Mat(),K_);
    mat_p = mat_p.reshape(1);
    kps_un.resize(N);
    for(int i=0;i<N;i++)
    {
        KeyPoint kp = kps[i];
        kp.pt.x = mat_p.at<float>(i,0);
        kp.pt.y = mat_p.at<float>(i,1);
        kps_un[i] = kp;
    }
}

// 计算矫正之后的图像边界 左上（0,0）右上（1,0）左下（1,0）右下（1,1）// stereo(暂时没有设置双目的边界)
void ComputeImageBounds(Mat img)
{
    Mat mat_b(4,2,CV_32F);
    mat_b.at<float>(0,0)=0.0; mat_b.at<float>(0,1)=0.0;
    mat_b.at<float>(1,0)=img.cols; mat_b.at<float>(1,1)=0.0;
    mat_b.at<float>(2,0)=0.0; mat_b.at<float>(2,1)=img.rows;
    mat_b.at<float>(3,0)=img.cols; mat_b.at<float>(3,1)=img.rows;
    //cout << "undistort before: " <<mat_b << endl;
    mat_b=mat_b.reshape(2);
    undistortPoints(mat_b,mat_b,K,DistCoef,Mat(),K);
    mat_b=mat_b.reshape(1);
    //cout << "after undistort: " <<mat_b << endl;
    mnMinX=min(mat_b.at<float>(0,0),mat_b.at<float>(2,0));
    mnmaxX=max(mat_b.at<float>(1,0),mat_b.at<float>(3,0));
    mnMinY=min(mat_b.at<float>(0,1),mat_b.at<float>(1,1));
    mnmaxY=max(mat_b.at<float>(2,1),mat_b.at<float>(3,1));
    
    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS)/(mnmaxX-mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS)/(mnmaxY-mnMinY);
}

vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS])
{
    vector<size_t> vIndices;
    vIndices.reserve(N_r);
    //计算该区域位于的最小网格横坐标
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;
    //计算该区域位于的最大网格横坐标
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;
    //计算该区域位于的最小网格纵坐标
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;
    //计算该区域位于的最大网格纵坐标
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);
    //查找这些（通过上述计算得到的）网格中的特征点，找出在层数范围内并且也在区域范围内的特征点
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {  
            const vector<size_t> vCell = mGrid[ix][iy];    //第(ix,iy)个网格特征点序号的集合
            if(vCell.empty())
                continue;
            //遍历这个（第(ix,iy)）网格中所有特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const KeyPoint &kpUn = mvKeysun_r[vCell[j]];   //得到具体的特征点
                if(bCheckLevels)
                {
		  //kpUn.octave表示这个特征点所在的层
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }
                //剔除在区域范围外的点
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

int DescriptorDistance(const Mat &a, const Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;  // 在直方图区间内特征点数量最大值
    int max2=0;  // 在直方图区间内特征点数量第二最大值
    int max3=0;  // 在直方图区间内特征点数量第三最大值

    for(int i=0; i<L; i++)
    {
      // 在该直方图区间内特征点数量
        const int s = histo[i].size();   
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)  // 如果max2/max1<0.1  那么证明第二第三方向不具有区分性,则将其索引置位初值
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)// 如果max3/max1<0.1  那么证明第三方向不具有区分性,则将其索引置位初值
    {
        ind3=-1;
    }
}

void SearchForInitialization()
{
    //cout << "------------------------------------SearchForInitialization()-------------------------------" << endl;
    const float factor = 1.0f/HISTO_LENGTH;
    for(int i1=0;i1<N_l;i1++)
    {
        KeyPoint kp1 = mvKeysun_l[i1];
        int level1 = kp1.octave; // 特征点被搜索到的金字塔层数
        if(level1>0)
            continue;
        //得到2帧对应区域的特征点索引
        //cout << "----GetFeaturesInArea-----" << endl;
        vector<size_t> vIndices2 = GetFeaturesInArea(mvbPrevMatched[i1].x,mvbPrevMatched[i1].y, windowSize, level1,level1, mGrid_r);
        //cout << "----GetFeaturesInArea is ok -----" << endl;
        if (vIndices2.empty())
            continue;

        // 描述子距离计算
        Mat d1 = mDescriptors_l.row(i1);
        int bestDist=INT_MAX, bestDist2=INT_MAX, bestIdx2 = -1;
        // 遍历所有F2中对应区域的特征点 寻找最优特征描述子距离和第二优特征描述子距离
        for(vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
        {
            size_t i2 = *vit;
            Mat d2 = mDescriptors_r.row(i2);
            int dist = DescriptorDistance(d1,d2);
            if (vMatchedDistance[i2] < dist)
                continue;
            if(dist < bestDist) // 最近
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2) // 次近
            {
                bestDist2=dist;
            }
        }
        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0) // 若第二帧中的特征点已经被匹配到了，先清除之前的匹配信息，用当前匹配信息取而代之
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1; 
                    nmatches--;
                }
                // 将最优距离对应的匹配点在帧1和帧2中的索引存储
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                // 两个特征点之间的旋转角度
                if(mbCheckOrientation)  //如果需要旋转则构建旋转角度的直方图
                {
                    float rot = mvKeysun_l[i1].angle-mvKeysun_r[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    //if(nmatches < 100)
                    //    cout << bin << endl;
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                    
                }
            }
        }
        
    }
    cout << "----CheckOrientation-----" << endl;
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }
    }
}

void Normalize(const vector<KeyPoint> &vKeys, vector<Point2f> &vNormalizedPoints, Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;
    // 将所有vKeys点减去中心坐标，使x坐标和y坐标均值分别为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;
    // 将x坐标和y坐标分别进行尺度缩放，使得x坐标和y坐标的一阶绝对矩分别为1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }
    // T=( sX ,0 ,-meanY*sY ;0 ,sY ,-meanX*sX ;0 ,0 ,1  ) T为坐标变换的矩阵   X'=T*X   ,X为归一化之前的像素坐标  X'为归一化之后的坐标
    T = Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

void Get_mvSets()
{
    mvinit_Matches12.reserve(N_r); // 所有匹配上的集合，关键点的index没有变
    mvbinit_Matched1.resize(N_l);
    for(size_t i=0,iend=vnMatches12.size();i<iend;i++)
    {
        if(vnMatches12[i]>=0)
        {
            mvinit_Matches12.push_back(make_pair(i,vnMatches12[i]));
            mvbinit_Matched1[i] = true;
        }
        else 
            mvbinit_Matched1[i] = false;
    }
    N = mvinit_Matches12.size();
    vector<size_t> vAllIndices, vAvailableIndices;//[vnMatches12.size()]
    vAllIndices.reserve(N); // vector::resize() 使用array index;  vector::reserve()使用 push_back();
    for(int i=0;i<N;i++)
        vAllIndices.push_back(i);
    mvSets = vector<vector<size_t> >(mMaxIterations, vector<size_t>(8,0));
    DUtils::Random::SeedRandOnce(0);
    for(int it=0;it<mMaxIterations;it++) 
    {
        vAvailableIndices = vAllIndices;
        for(size_t j=0;j<8;j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];
            mvSets[it][j] = idx;
            vAvailableIndices[randi] = vAvailableIndices.back(); // 把随机到的数取走
            vAvailableIndices.pop_back();
        }
    }
}

Mat ComputeH21(const vector<Point2f> &vP1, const vector<Point2f> &vP2)
{
    const int N = vP1.size();

    Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    Mat u,w,vt;
    SVDecomp(A,w,u,vt,SVD::MODIFY_A | SVD::FULL_UV);
    return vt.row(8).reshape(0, 3);  // 将向量形式变为3*3的矩阵
}

Mat ComputeF21(const vector<Point2f> &vP1,const vector<Point2f> &vP2)
{
    const int N = vP1.size();

    Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    Mat u,w,vt;

    SVDecomp(A,w,u,vt,SVD::MODIFY_A | SVD::FULL_UV);
    //F矩阵为A最小奇异值对应的奇异向量
    Mat Fpre = vt.row(8).reshape(0, 3);

    SVDecomp(Fpre,w,u,vt,SVD::MODIFY_A | SVD::FULL_UV);
    // 将本质矩阵的奇异值最小值硬性设为0,使得本质矩阵满足内在要求  详情请见高翔slam P151
    w.at<float>(2)=0;

    return  u*Mat::diag(w)*vt;
}

float CheckHomography(const Mat &H21, const Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);
     // 循环所有的特征点,计算对称传输误差 1->2的几何误差   2->1的几何误差  
    //  得分是所有特征点的(阈值th-几何误差)和
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const KeyPoint &kp1 = mvKeys_l[mvinit_Matches12[i].first];
        const KeyPoint &kp2 = mvKeys_r[mvinit_Matches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image  计算第一幅图像的几何误差
        // x2in1 = H12*x2

        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image  计算第二幅图像的几何误差
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)  // 如果重投影误差满足阈值 则是内点  否则为外点
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

float CheckFundamental(const Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const KeyPoint &kp1 = mvKeysun_l[mvinit_Matches12[i].first];
        const KeyPoint &kp2 = mvKeysun_r[mvinit_Matches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)    点到极线的距离

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image  基础矩阵的性质:(P,P')的基础矩阵是(P',P)基础矩阵的转置  
	// 因此存在l1 =F21.t*x2=(a1,b1,c1)
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

void FindHomography(vector<bool> &vbMatchesInliers, float &score, Mat &H21)
{
    float currentScore;
    Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    vector<Point2f> vPn1i(8);
    vector<Point2f> vPn2i(8); 
    for(int i=0;i<mMaxIterations;i++)
    {
        for(int j=0;j<8;j++)
        {
            int idx = mvSets[i][j]; // 取出8对归一化特征点
            vPn1i[j] = vPn1[mvinit_Matches12[idx].first];
            vPn2i[j] = vPn2[mvinit_Matches12[idx].second];
        }
        //cout << "It is going to ComputeH21..." << endl;
        Mat Hn = ComputeH21(vPn1i,vPn2i);
        H21i = T2inv * Hn * T1;
        H12i = H21i.inv();
        //cout << "It is Checking Homography..." << endl;
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
        if(currentScore>score)  // 找到所有RANSAC迭代中最大得分
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

void FindFundamental(vector<bool> &vbMatchesInliers, float &score, Mat &F21)
{
    float currentScore;
    Mat F21i, F12i;
    vector<bool> vbCurrentInliers(N,false);
    vector<Point2f> vPn1i(8);
    vector<Point2f> vPn2i(8);
    for(int i=0;i<mMaxIterations;i++)
    {
        for(int j=0;j<8;j++)
        {
            int idx = mvSets[i][j]; // 取出8对归一化特征点
            vPn1i[j] = vPn1[mvinit_Matches12[idx].first];
            vPn2i[j] = vPn2[mvinit_Matches12[idx].second];
        }
        //cout << "It is going to ComputeH21..." << endl;
        Mat Fn = ComputeF21(vPn1i,vPn2i);
        F21i = T2t*Fn*T1;
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);
        if(currentScore>score)  // 找到所有RANSAC迭代中最大得分
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

bool ReconstructF(vector<bool> &vbMatchesInliers, Mat F21, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    //ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    //ReconstructF(vbMatchesInliersF, F21, vP3D, vbTriangulated, 1.0, 50);
    int n=0;
    for(size_t i=0;i<vbMatchesInliers.size();i++)
        if(vbMatchesInliers[i])
            n++;
    int minGood = max(static_cast<int>(0.45*n), minTriangulated);
    cout << "good matches: " << n <<endl;
    cout << "min good Triangulate number: " << minGood << endl;
    Mat E21 = K_r.t() * F21 * K;
    Mat R1, R2, t;
    DecomposeE(E21, R1, R2, t);
    t = (Mat_<float>(3,1) << -0.1963650, 0.0004219,  -0.001686);
    //cout << "R1: " << endl << R1 << endl;
    //cout << "R2: " << endl << R2 << endl;
    //cout << "t: " << endl << t << endl;
    Mat t1 = t, t2 = -t;
    vector<Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;
    //int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood1 = CheckRT(R1, t1, vbMatchesInliers, vP3D1,4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2, t1, vbMatchesInliers, vP3D2,4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1, t2, vbMatchesInliers, vP3D3,4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2, t2, vbMatchesInliers, vP3D4,4.0*mSigma2, vbTriangulated4, parallax4);
    cout << "parallax1: " << parallax1 << "         nGood1: " << nGood1 <<endl;
    cout << "parallax2: " << parallax2 << "         nGood2: " << nGood2 <<endl;
    cout << "parallax3: " << parallax3 << "         nGood3: " << nGood3 <<endl;
    cout << "parallax4: " << parallax4 << "         nGood4: " << nGood4 <<endl;
    int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));
    cout << "max good Triangulate number: " << maxGood << endl;
    int nsimilar = 0;
    if(nGood1>0.6*maxGood)
        nsimilar++;
    if(nGood2>0.6*maxGood)
        nsimilar++;
    if(nGood3>0.6*maxGood)
        nsimilar++;
    if(nGood4>0.6*maxGood)
        nsimilar++;
    if(maxGood < minGood || nsimilar > 1)
        return false;
    //cout << "its going to find the right R t..." <<endl;
    if(maxGood == nGood1 && parallax1 > minParallax)
    {
        vP3D = vP3D1;
        vbTriangulated = vbTriangulated1;
        R21 = R1.clone();
        t21 = t1.clone();
        return true;
    }
    else if(maxGood == nGood2 && parallax2 > minParallax)
    {
        vP3D = vP3D2;
        vbTriangulated = vbTriangulated2;
        R21 = R2.clone();
        t21 = t1.clone();
        return true;
    }
    else if(maxGood == nGood3 && parallax3 > minParallax)
    {
        vP3D = vP3D3;
        vbTriangulated = vbTriangulated3;
        R21 = R1.clone();
        t21 = t2.clone();
        return true;
    }
    else if(maxGood == nGood4 && parallax4 > minParallax)
    {
        cout << "Find the 4 is the right choice..." << endl;
        vP3D = vP3D4;
        vbTriangulated = vbTriangulated4;
        R21 = R2.clone();
        t21 = t2.clone();
        return true;
    }
    else
    {
        cout << "This caculate is wrong ... " << endl; 
        return false;
    }  
}

void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    // 将E进行SVD分解
    cv::SVD::compute(E,w,u,vt);
    // 将特征向量附给t
    u.col(2).copyTo(t);
    // 将向量t进行归一化
    t=t/cv::norm(t);
    // w=Rz(pi/2)
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;  //R1=u*W*vt
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt; //R2=u*W'*vt
    if(cv::determinant(R2)<0)
        R2=-R2;
}

void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

int CheckRT(const cv::Mat &R, const cv::Mat &t, vector<bool> &vbMatchesInliers,
                vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    vbGood = vector<bool>(mvKeysun_l.size(),false);
    vP3D.resize(mvKeysun_l.size());
    vector<float> vCosParallax;
    vCosParallax.reserve(mvKeysun_l.size());
    Mat P1(3,4,CV_32F,Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3)); // 相机1为世界坐标初始点，不许进行T变换，直接为相机内参矩阵
    Mat camO1 = cv::Mat::zeros(3,1,CV_32F); // 初始化相机1的位姿
    Mat P2(3,4,CV_32F,Scalar(0));
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K_r * P2;
    Mat camO2 = -R.t()*t;
    int nGood=0;
    for(size_t i=0;i<vbMatchesInliers.size();i++)
    {
        if(!vbMatchesInliers[i])
            continue;
        const cv::KeyPoint &kp1 = mvKeysun_l[mvinit_Matches12[i].first];
        const cv::KeyPoint &kp2 = mvKeysun_r[mvinit_Matches12[i].second];
        cv::Mat p3dC1;
        Triangulate(kp1,kp2,P1,P2,p3dC1);
        // 
        //if(i<100)
           // cout << p3dC1.at<float>(0) << "  "<< p3dC1.at<float>(1) << "  "<< p3dC1.at<float>(2) << "  " << endl;
        // 
        if(!isfinite(p3dC1.at<float>(0)) ||!isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)) )
        {
            cout << "wrong match ... " << p3dC1 << endl;
            continue;
        }
        Mat normal1 = p3dC1 - camO1, normal2 = p3dC1 - camO2;
        float dist1 = norm(normal1), dist2 = norm(normal2);
        float cosParallax = normal1.dot(normal2)/(dist1 * dist2); // 将右相机光心camO2和P之间的向量转到cam1坐标系中，那两个向量之间夹角为视差角
        // cout << "cosParallax: " << cosParallax << endl;
        if(p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998) //cos(0.3624)=0.99998
            continue;
        Mat p3dC2 = R*p3dC1+t;
        if(p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998) 
            continue;
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
        if(squareError1>th2)  //重投影误差大于阈值
            continue;
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx_r*p3dC2.at<float>(0)*invZ2+cx_r; // stereo 将归一化处坐标返回像素坐标（用于计算重投影误差，误差阈值可以和H F 的评分相同）
        im2y = fy_r*p3dC2.at<float>(1)*invZ2+cy_r;
        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);
        if(squareError2>th2)
            continue;
        vCosParallax.push_back(cosParallax);
        vP3D[mvinit_Matches12[i].first] = Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        nGood++;
        if(cosParallax < 0.99998)
            vbGood[mvinit_Matches12[i].first] = true;
    }
    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());
        size_t idx = min(50 ,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else 
        parallax = 0;
    return nGood;    
}

void Draw_keypoints(const Mat img_l_, const Mat img_r_, vector<KeyPoint> vKeypointsDraw_l_, vector<KeyPoint> vKeypointsDraw_r_, vector<int> vnMatches_lr)
// 目前设置为显示矫正前的特征点
{
    Mat img_l = img_l_.clone();
    Mat img_r = img_r_.clone();
    vector<KeyPoint> vKeypointsDraw_l;
    vector<KeyPoint> vKeypointsDraw_r;
    int nFs = 0;
    for(int i=0;i<vnMatches12.size();i++)
    {
        if(vnMatches12[i]<0)
            continue;
        nFs++;
    }
    vKeypointsDraw_l.resize(nFs+1);
    vKeypointsDraw_r.resize(nFs+1);
    int n=0;
    for(int i=0;i<vnMatches12.size();i++)
    {
        if(vnMatches12[i]<0)
            continue;
        int idx_l = i;
        int idx_r = vnMatches12[i];
        vKeypointsDraw_l[n] = vKeypointsDraw_l_[idx_l];
        vKeypointsDraw_r[n] = vKeypointsDraw_r_[idx_r];
        n++;
    }
    
    Mat outImg_l, outImg_r;
    drawKeypoints(img_l,vKeypointsDraw_l,outImg_l,Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img_r,vKeypointsDraw_r,outImg_r,Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("kp_img_l", outImg_l);
    imshow("kp_img_r", outImg_r);
    waitKey();
}

