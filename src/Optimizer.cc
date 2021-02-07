#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"



#include<Eigen/StdVector>

#include "Converter.h"

namespace ORB_SLAM2
{
    //void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    //{
    //    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    //    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    //    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
    //}
    void Optimizer::BundleAdjustment(vector<pair<int,Mat>> MapPoints, Mat& Tc1w, Mat& Tc2w, vector<int> vnMatches12, Mat K,
                                    vector<KeyPoint> mvKeysun_l, vector<KeyPoint> mvKeysun_r,vector<float> mvInvLevelSigma2,
                                    int nIterations, bool *pbStopFlag, const bool bRobust)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *LinearSover;
        LinearSover = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(LinearSover);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);
        
        // 设置了两帧的位姿，定点序号为0,1
        for(size_t i=0;i<2;i++)
        {
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            if(i==0)
            {
                vSE3->setEstimate(Converter::toSE3Quat(Tc1w));
                vSE3->setFixed(true);
            }
            if(i==1)
            {
                vSE3->setEstimate(Converter::toSE3Quat(Tc2w));
                vSE3->setFixed(false);
            }   
            vSE3->setId(i);
            optimizer.addVertex(vSE3);
        }
        
        const float thHuber2D = sqrt(5.99);
        for(size_t i = 0; i < MapPoints.size(); i++)
        {
            // 3D点
            int idx_l = MapPoints[i].first;
            int idx_r = vnMatches12[idx_l];
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(MapPoints[i].second));
            int id = 2+ idx_l;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            int nedges = 0;
            for(size_t id_frame = 1;id_frame < 2; id_frame++)
            {
                nedges ++;
                // 提取观测值
                KeyPoint kpun;
                if(id_frame==0)
                    kpun = mvKeysun_l[idx_l];
                if(id_frame==1)
                    kpun = mvKeysun_r[idx_r];
                Eigen::Matrix<double,2,1> obs;
                obs << kpun.pt.x, kpun.pt.y;
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id_frame)));
                e->setMeasurement(obs);
                //float invSigma2 = mvInvLevelSigma2[kpun.octave];
                //cout << invSigma2 << endl;
                //Eigen::Matrix2d infor_m = Eigen::Matrix2d::Identity();
                //cout << infor_m << endl;
                //e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                e->setInformation(Eigen::Matrix2d::Identity());
                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }
                e->fx = K.at<float>(0,0);
                e->fy = K.at<float>(1,1);
                e->cx = K.at<float>(0,2);
                e->cy = K.at<float>(1,2);

                optimizer.addEdge(e);
            }
            
        }
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);

    }






} //namespace ORB_SLAM