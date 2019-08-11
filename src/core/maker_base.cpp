#include "maker_base.h"
#include "config.h"

using namespace cv;
using namespace std;

namespace calibcamodo {

MakerBase::MakerBase(Dataset *_pDataset): mpDataset(_pDataset) {

    // load odometry error configure
    mOdoLinErrR     = Config::CALIB_ODOLIN_ERRR;
    mOdoLinErrMin   = Config::CALIB_ODOLIN_ERRMIN;
    mOdoRotErrR     = Config::CALIB_ODOLIN_ERRR;
    mOdoRotErrRLin  = Config::CALIB_ODOROT_ERRRLIN;
    mOdoRotErrMin   = Config::CALIB_ODOROT_ERRMIN;
}

void MakerBase::MakeMsrOdo()
{
    mpDataset->ClearMsrOdo();
    const std::map<int, PtrKeyFrame>& mmapId2pKf = mpDataset->GetKfMap();

    // 遍历所有关键帧，按时间顺序排列
    for(auto iter1 = mmapId2pKf.cbegin(), iter2 = iter1++;
        iter1 != mmapId2pKf.cend();
        ++iter1, ++iter2)
    {
        PtrKeyFrame pKfHead = (*iter1).second;
        PtrKeyFrame pKfTail = (*iter2).second;
        Se2 dodo = pKfTail->GetOdo() - pKfHead->GetOdo();
        Mat info = Mat::eye(3,3,CV_32FC1);

        double dist = dodo.dist();
        /** @todo 计算协方差，注意信息矩阵*/

        // stdlin: 线性标准差(直线运动的标准差)
        // 运动的位移越大，标准差越大
        double stdlin = max(dist*mOdoLinErrR, mOdoLinErrMin);

        double theta = dodo.theta;
        // stdrot: 旋转的标准差
        double stdrot = max(max(abs(theta)*mOdoRotErrR, mOdoRotErrMin), dist*mOdoRotErrRLin);

        // 生成信息矩阵!!，g2o要用到
        info.at<float>(0,0) = 1/stdlin/stdlin;
        info.at<float>(1,1) = 1/stdlin/stdlin;
        info.at<float>(2,2) = 1/stdrot/stdrot;

        PtrMsrSe2Kf2Kf pMeasureOdo = make_shared<MeasureSe2Kf2Kf>(dodo, info, pKfHead, pKfTail);
        mpDataset->AddMsrOdo(pMeasureOdo);
    }
}

void MakerBase::InitKfPose()
{
    for(auto ptr : mpDataset->GetKfSet()) {
        PtrKeyFrame pKf = ptr;
        Se2 se2odo = pKf->GetOdo();
        //里程计的测量值即是，机器人坐标系到世界坐标系变换矩阵，对
        Se2 se2wb = se2odo;
        Se3 se3wb = Se3(se2wb); //转换为3维变换
        //相机坐标系到机器人坐标系的转换矩阵
        Se3 se3bc = mpDataset->GetCamOffset();
        Se3 se3wc = se3wb + se3bc; //相机坐标系到世界坐标系的转换矩阵

        pKf->SetPoseBase(se2odo);
        pKf->SetPoseCamera(se3wc);
    }
}

}



