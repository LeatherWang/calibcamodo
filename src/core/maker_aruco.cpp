#include "maker_aruco.h"
#include "config.h"

using namespace cv;
using namespace std;

namespace calibcamodo {

MakerAruco::MakerAruco(DatasetAruco* _pDatasetAruco):
    MakerBase(_pDatasetAruco), mpDatasetAruco(_pDatasetAruco) {

    // set aruco error configure
    mAmkZErrRZ      = Config::CALIB_AMKZ_ERRRZ;
    mAmkZErrMin     = Config::CALIB_AMKZ_ERRMIN;
    mAmkXYErrRZ     = Config::CALIB_AMKXY_ERRRZ;
    mAmkXYErrMin    = Config::CALIB_AMKXY_ERRMIN;
}

void MakerAruco::MakeMkAndMsrMk()
{
    const set<PtrKeyFrameAruco> setpKfAruco = mpDatasetAruco->GetKfArucoSet(); //由创建关键帧函数生成

    // Create aruco marks and mark measurements
    for (auto ptr : setpKfAruco)
    {
        PtrKeyFrameAruco pKfAruco = ptr;

        // 当前关键帧看到的所有aruco
        const std::vector<aruco::Marker>& vecAruco = pKfAruco->GetMsrAruco(); //get aruco measurements in this KF

        for (auto measure_aruco : vecAruco)
        {
            int id = measure_aruco.id; //aruco 的id号(aruco的属性之一，注意与关键帧的id号的区别)
            Mat tvec = measure_aruco.Tvec; //aruco在相机坐标系中的坐标
            double marksize = measure_aruco.ssize; //aruco的米制大小

            /// @attention 以marker距离相机的距离(Z方向)为方差确定的依据
            double z = abs(tvec.at<float>(2));
            double stdxy = max(z*mAmkXYErrRZ, mAmkXYErrMin);
            double stdz = max(z*mAmkZErrRZ, mAmkZErrMin);

            Mat info = Mat::eye(3,3,CV_32FC1);
            info.at<float>(0,0) = 1/stdxy/stdxy;
            info.at<float>(1,1) = 1/stdxy/stdxy;
            info.at<float>(2,2) = 1/stdz/stdz;

            // add new aruco mark into dataset
            PtrMapMarkAruco pMkAruco = make_shared<MapMarkAruco>(id, id, marksize); //第一个id是MapMarker的id，第二个是MapMarkAruco的id，二者相同

            //! 增加aruco mark到数据库，包括几种不同类型的数据库，主要是为了使用方便
            if (!mpDatasetAruco->AddMkAruco(pMkAruco))
                pMkAruco = mpDatasetAruco->GetMkAruco(id); //使用数据库中的marker

            // add new measurement into dataset
            // 增加到数据库: mmsrplMk
            /** PtrMsrPt3Kf2Mk，包含:
             * @param tvec      marker在关键帧中的3D位置
             * @param info      信息矩阵
             * @param pKfAruco  关键帧
             * @param pMkAruco  ArucoMarker
             */
            PtrMsrPt3Kf2Mk pMsrMk = make_shared<MeasurePt3Kf2Mk>(tvec, info, pKfAruco, pMkAruco);
            mpDatasetAruco->AddMsrMk(pMsrMk);
        }
    }
}

void MakerAruco::InitMkPose()
{
    // 遍历每一个aruco marker，
    // 根据<第一个包含它的关键帧的位姿>与<该marker在该关键帧中的3D位置>，
    // 设置marker的世界坐标
    for(auto ptr : mpDatasetAruco->GetMkSet())
    {
        PtrMapMark pMk = ptr;

        // 包含该aruco的所有PtrMsrPt3Kf2Mk
        /** @attention setpMsr已经按照ID号排序过??*/
        set<PtrMsrPt3Kf2Mk> setpMsr = mpDatasetAruco->GetMsrMkByMk(pMk);
        if(!setpMsr.empty())
        {
            // 选择所有PtrMsrPt3Kf2Mk中升序排列的第一个对应的关键帧
            PtrKeyFrame pKf = (*setpMsr.cbegin())->pKf;
            Se3 se3wc = pKf->GetPoseCamera();
            Se3 se3cm;

            // 只用到了aruco marker的平移，即将aruco当作类似orb特征点，只有3D位置，没有方向(质点)
            se3cm.tvec = (*setpMsr.cbegin())->pt3.tvec();
            Se3 se3wm = se3wc + se3cm;
            pMk->SetPose(se3wm);
        }
    }
}

void MakerAruco::InitKfMkPose() {
    InitKfPose(); //使用估计的外参重新初始化关键帧的位姿
    InitMkPose(); //使用估计的外参更新Marker的位姿
}



}
