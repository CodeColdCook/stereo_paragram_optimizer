#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include <algorithm>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

using namespace std;
using namespace cv;

namespace ORB_SLAM2
{
class Optimizer
{
public:

    void static BundleAdjustment(vector<pair<int,Mat>> MapPoints, Mat& Tc1w, Mat& Tc2w, vector<int> vnMatches12, Mat K,
                                    vector<KeyPoint> mvKeysun_l, vector<KeyPoint> mvKeysun_r, vector<float> mvInvLevelSigma2,
                                    int nIterations = 20, bool *pbStopFlag=NULL, const bool bRobust = true);

    //void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
    //                            int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
    //                            const bool bRobust = true);
                                
    //void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
    //                                   const unsigned long nLoopKF=0, const bool bRobust = true);


    //vector<pair<int,Mat>> MapPoints; // 最终形式：大小为最终的nmatches，vector[i]=pair<左图特征点idx，Mat（3,3,CV32F）>
    //Mat Tc1w = Mat::eye(4,4,CV_32F);
    //Mat Tc2w = Mat::eye(4,4,CV_32F);
    //vector<int> vnMatches12; // 筛选关键点，联系3D点和特征点
    //vector<KeyPoint> mvKeysun_l,mvKeysun_r;
    //vector<Point2f> vPn1, vPn2;


};


}//namespace ORB_SLAM
#endif // OPTIMIZER_H