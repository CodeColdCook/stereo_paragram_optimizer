#ifndef BA_H
#define BA_H

#include <iostream>
#include <opencv2/core/core.hpp>



using namespace std;
using namespace cv;

namespace ORB_SLAM2
{
class Optimizer
{
public:

    void static bundleAdjustment (
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    Mat& T21,
    Mat& R, Mat& t,
    const Mat K,
    vector<Mat> Mat_Points_3d);

};


}//namespace ORB_SLAM
#endif // OPTIMIZER_H