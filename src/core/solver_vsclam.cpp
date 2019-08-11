#include "solver_vsclam.h"
#include "adapter.h"

using namespace cv;
using namespace std;
using namespace g2o;

namespace calibcamodo {

SolverVsclam::SolverVsclam(Dataset* _pDataset):
    SolverBase(_pDataset) {}

void SolverVsclam::DoCalib() {
    OptimizeSlam();
    //OptimizeSclam();
}

void SolverVsclam::DoCalib_2() {
    //OptimizeSlam();
    OptimizeSclam();
}

// 这一步仅仅是优化<相机位姿>与<MapPoints>位置
void SolverVsclam::OptimizeSlam()
{
    // Init optimizer
    SparseOptimizer optimizer;
    bool bOptVerbose = true;
    InitOptimizerSlam(optimizer, bOptVerbose);

    // Add Parameters
    /*【步骤1】: 加入顶点: <相机内参>和se3bc*/
    Se3 se3bc = mpDataset->GetCamOffset();
    std::cout<<"se3bc:"<<se3bc<<std::endl;
    int idParamCamera = 0;

    /** @attention 仅仅是加入参数并没有优化??*/
    AddParaCamera(optimizer, mpDataset->GetCamMat(), toG2oIsometry3D(se3bc), idParamCamera);

    int idVertexMax = 0;
    // Add keyframe vertices
    /*【步骤2】: 顶点是mSe2wb: <关键帧对应的>机器人位姿相对世界坐标系的坐标*/
    map<PtrKeyFrame, int> mapKf2IdOpt;
    for (auto ptr : mpDataset->GetKfSet()) {
        PtrKeyFrame pKf = ptr;
        SE2 pose = toG2oSE2(pKf->GetPoseBase());
        if(!pKf->GetId())
            AddVertexSE2(optimizer, pose, idVertexMax, true);
        else
            AddVertexSE2(optimizer, pose, idVertexMax, false);
        mapKf2IdOpt[pKf] = idVertexMax++;
    }

    // Add mappoint vertices
    /*【步骤3】: 加入MapPoints顶点*/
    map<PtrMapPoint,int> mapMp2IdOpt;
    for (auto ptr : mpDataset->GetMpSet()) {
        PtrMapPoint pMp = ptr;
        Vector3D pose = toG2oVector3D(pMp->GetPos().tvec());
        AddVertexPointXYZ(optimizer, pose, idVertexMax, true);
        mapMp2IdOpt[pMp] = idVertexMax++;
    }

    // Add odometry edges
    /*【步骤4】: 里程计测量构成的边*/
    vector<g2o::EdgeSE2*> vecpEdgeOdo;
    for (auto ptr : mpDataset->GetMsrOdoSet())
    {
        PtrMsrSe2Kf2Kf pMsrOdo = ptr;
        PtrKeyFrame pKf0 = pMsrOdo->pKfHead;
        PtrKeyFrame pKf1 = pMsrOdo->pKfTail;
        int id0 = mapKf2IdOpt[pKf0];
        int id1 = mapKf2IdOpt[pKf1];
        // 测量
        g2o::SE2 measure = toG2oSE2(pMsrOdo->se2);
        // 信息矩阵
        g2o::Matrix3D info = toEigenMatrixXd(pMsrOdo->info); /** @done */

        /** @attention 顶点和边都是由测量值得到????*/
        g2o::EdgeSE2* pEdgeOdo = AddEdgeSE2(optimizer, id0, id1, measure, info);
        vecpEdgeOdo.push_back(pEdgeOdo);
    }

    // Set mark measurement edges
    /*【步骤5】: 相机观测*/
    vector<g2o::EdgeVSlam*> vecpEdgeVSlam;
    for (auto ptr : mpDataset->GetMsrMpAll()) {
        PtrMsrUVKf2Mp pMsrMp = ptr;
        PtrKeyFrame pKf = pMsrMp->pKf;
        PtrMapPoint pMp = pMsrMp->pMp;
        int idKf = mapKf2IdOpt[pKf]; //得到关键帧的ID号
        int idMp = mapMp2IdOpt[pMp]; //得到MapPoint的ID号
        g2o::Vector2D measure = toG2oVector2D(pMsrMp->measure);
        // 信息矩阵
        g2o::Matrix2D info = toEigenMatrixXd(pMsrMp->info); /** @todo 根据提取所在的<层>再搞一搞???? */

        /** @attention 使用g2o自动求导*/
        g2o::EdgeVSlam* pEdgeVSlam = AddEdgeVSlam(optimizer, idKf, idMp, idParamCamera, measure, info);
        vecpEdgeVSlam.push_back(pEdgeVSlam);
    }

    // Do optimize
    /*【步骤6】: 优化*/
    optimizer.initializeOptimization();
    optimizer.optimize(15); //迭代次数:30

    /*【步骤7】: 去除outliner*/
    for(size_t i=0, iend=vecpEdgeVSlam.size(); i<iend;i++)
    {
        g2o::EdgeVSlam* e = vecpEdgeVSlam[i];

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if(e->chi2()>5.991)
        {
            std::cout<<e->chi2()<<endl;
            e->setLevel(1);// 不优化
        }
    }

    /*【步骤8】: 再次优化*/
    optimizer.initializeOptimization();
    optimizer.optimize(15);

    // Refresh all keyframes
    for (auto pair : mapKf2IdOpt) {
        PtrKeyFrame pKf = pair.first;
        int idOpt = pair.second;
        VertexSE2* pVertex = static_cast<VertexSE2*>(optimizer.vertex(idOpt));
        // 设置关键帧的世界坐标
        pKf->SetPoseAllbyB(toSe2(pVertex->estimate()), se3bc);
    }

    // Refresh all mappoints
    for (auto pair : mapMp2IdOpt) {
        PtrMapPoint pMp = pair.first;
        int idOpt = pair.second;
        VertexPointXYZ* pVertex = static_cast<VertexPointXYZ*>(optimizer.vertex(idOpt));
        Mat tvec_wm = toCvMatf(pVertex->estimate());
        pMp->SetPos(Pt3(tvec_wm));
    }
}

void SolverVsclam::OptimizeSclam()
{
    // Init optimizer
    SparseOptimizer optimizer;
    bool bOptVerbose = true;
    InitOptimizerCalib(optimizer);
    optimizer.setVerbose(bOptVerbose);

    // Add Parameters
    // 顶点: 加入相机的内参
    /// @attention 这里不再设置Camoffset，仅仅设置相机内参，Camoffset在下面单独设置??
    int idParamCamera = 0;
    AddCamPara(optimizer, mpDataset->GetCamMat(), idParamCamera);

    int idVertexMax = 0;
    // Add camera extrinsic vertex
    /** 顶点: 里程计与相机的外参*/
    Se3 se3cb = mpDataset->GetCamOffset(); /** @todo 函数应该的应该是se3bc*/
    int idVertexCamOffset = idVertexMax++;
    AddVertexSE3(optimizer, toG2oIsometry3D(se3cb), idVertexCamOffset, false);

    // Add keyframe vertices
    map<PtrKeyFrame, int> mapKf2IdOpt;
    for (auto ptr : mpDataset->GetKfSet()) {
        PtrKeyFrame pKf = ptr;
        SE2 pose = toG2oSE2(pKf->GetPoseBase());
        if(!pKf->GetId())
            AddVertexSE2(optimizer, pose, idVertexMax, true); /** @attention 这里设置为固定了*/
        else
            AddVertexSE2(optimizer, pose, idVertexMax, false);
        mapKf2IdOpt[pKf] = idVertexMax++;
    }

    // Add mappoint vertices
    map<PtrMapPoint,int> mapMp2IdOpt;
    for (auto ptr : mpDataset->GetMpSet()) {
        PtrMapPoint pMp = ptr;
        Vector3D pose = toG2oVector3D(pMp->GetPos().tvec());
        AddVertexPointXYZ(optimizer, pose, idVertexMax, true);
        mapMp2IdOpt[pMp] = idVertexMax++;
    }

    // Add odometry edges
    vector<g2o::EdgeSE2*> vecpEdgeOdo;
    for (auto ptr : mpDataset->GetMsrOdoSet()) {
        PtrMsrSe2Kf2Kf pMsrOdo = ptr;
        PtrKeyFrame pKf0 = pMsrOdo->pKfHead;
        PtrKeyFrame pKf1 = pMsrOdo->pKfTail;
        int id0 = mapKf2IdOpt[pKf0];
        int id1 = mapKf2IdOpt[pKf1];
        g2o::SE2 measure = toG2oSE2(pMsrOdo->se2);
        g2o::Matrix3D info = toEigenMatrixXd(pMsrOdo->info);
        g2o::EdgeSE2* pEdgeOdo = AddEdgeSE2(optimizer, id0, id1, measure, info);
        vecpEdgeOdo.push_back(pEdgeOdo);
    }

    // Set mark measurement edges
    /** @attention 多元边*/
    vector<g2o::EdgeVSclam*> vecpEdgeVSclam;
    for (auto ptr : mpDataset->GetMsrMpAll()) {
        PtrMsrUVKf2Mp pMsrMp = ptr;
        PtrKeyFrame pKf = pMsrMp->pKf;
        PtrMapPoint pMp = pMsrMp->pMp;
        int idKf = mapKf2IdOpt[pKf];
        int idMp = mapMp2IdOpt[pMp];
        g2o::Vector2D measure = toG2oVector2D(pMsrMp->measure);
        g2o::Matrix2D info = toEigenMatrixXd(pMsrMp->info);

        /** @todo 这边怎么构造的，如何求导?*/
        g2o::EdgeVSclam* pEdgeVSclam = AddEdgeVSclam(optimizer, idKf, idMp, idVertexCamOffset, idParamCamera, measure, info);
        vecpEdgeVSclam.push_back(pEdgeVSclam);
    }

    // Do optimize
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    // Renew camera offset
    VertexSE3* pVertexCamOffset = static_cast<VertexSE3*>(optimizer.vertex(idVertexCamOffset));
    mpDataset->SetCamOffset(toSe3(pVertexCamOffset->estimate()));

    // Refresh all keyframes
    for (auto pair : mapKf2IdOpt) {
        PtrKeyFrame pKf = pair.first;
        int idOpt = pair.second;
        VertexSE2* pVertex = static_cast<VertexSE2*>(optimizer.vertex(idOpt));
        pKf->SetPoseAllbyB(toSe2(pVertex->estimate()), se3cb);
    }

    // Refresh all mappoints
    for (auto pair : mapMp2IdOpt) {
        PtrMapPoint pMp = pair.first;
        int idOpt = pair.second;
        VertexPointXYZ* pVertex = static_cast<VertexPointXYZ*>(optimizer.vertex(idOpt));
        Mat tvec_wm = toCvMatf(pVertex->estimate());
        pMp->SetPos(Pt3(tvec_wm));
    }
}

void SolverVsclam::optimize_mappoints_keyframe()
{
    SparseOptimizer optimizer;
    bool bOptVerbose = true;
    InitOptimizerSlam(optimizer, bOptVerbose);

    int idParamCamera = 0;
    AddCamPara(optimizer, mpDataset->GetCamMat(), idParamCamera);

    int idVertexMax = 0;
    map<PtrKeyFrame, int> mapKf2IdOpt;
    for (auto ptr : mpDataset->GetKfSet())
    {
        PtrKeyFrame pKf = ptr;
        Se3 Se3wc = pKf->GetPoseCamera();
        g2o::Isometry3D Twc = toG2oIsometry3D(Se3wc);
        g2o::Isometry3D Tcw = Twc.inverse(); /** @attention 转换成Tcw*/
        if(idVertexMax != 0)
            AddVertexSE3(optimizer, Tcw, idVertexMax, false);
        else
            AddVertexSE3(optimizer, Tcw, idVertexMax, true); //第一帧固定

cout<<idVertexMax<<": "<<Se3wc.tvec.t()<<endl;

        mapKf2IdOpt[pKf] = idVertexMax++;
    }

    map<PtrMapPoint,int> mapMp2IdOpt;
    for (auto ptr : mpDataset->GetMpSet())
    {
        PtrMapPoint pMp = ptr;
        Vector3D pose = toG2oVector3D(pMp->GetPos().tvec());
        AddVertexPointXYZ(optimizer, pose, idVertexMax, true);
        mapMp2IdOpt[pMp] = idVertexMax++;
    }

    vector<g2o::EdgeUV_SE3_XYZ*> vecpEdgeUV_SE3_XYZ;
    vector<PtrMapPoint> vecPtrMp;
    for (auto ptr : mpDataset->GetMsrMpAll())
    {
        PtrMsrUVKf2Mp pMsrMp = ptr;
        PtrKeyFrame pKf = pMsrMp->pKf;
        PtrMapPoint pMp = pMsrMp->pMp;
        /** @todo leather add*/
        if(pMp->mbBad)
            continue;
        int idKf = mapKf2IdOpt[pKf]; //得到关键帧的ID号
        int idMp = mapMp2IdOpt[pMp]; //得到MapPoint的ID号
        g2o::Vector2D measure = toG2oVector2D(pMsrMp->measure);
        g2o::Matrix2D info = toEigenMatrixXd(pMsrMp->info); /** @todo 根据提取所在的<层>再搞一搞???? */

        /** @attention 使用g2o自动求导*/
        g2o::EdgeUV_SE3_XYZ* pEdgeUV_SE3_XYZ = AddEdgeUV_SE3_XYZ(optimizer, idKf, idMp, idParamCamera, measure, info);
        vecpEdgeUV_SE3_XYZ.push_back(pEdgeUV_SE3_XYZ);
        vecPtrMp.push_back(pMp); /// leather add
    }

    optimizer.initializeOptimization();
    optimizer.optimize(30);
/*
//    for(size_t i=0, iend=vecpEdgeUV_SE3_XYZ.size(); i<iend;i++)
//    {
//        g2o::EdgeUV_SE3_XYZ* e = vecpEdgeUV_SE3_XYZ[i];

//        if(e->chi2()>5.991)
//        {
//            std::cout<<e->chi2()<<endl;
//            std::cout<<vecPtrMp[1]->GetPos().tvec()<<endl;
//            std::cout<<"--------------------------------"<<endl;
//            e->setLevel(1);// 不优化
//        }
//    }

//    optimizer.initializeOptimization();
//    optimizer.optimize(15);

//    for(size_t i=0, iend=vecpEdgeUV_SE3_XYZ.size(); i<iend;i++)
//    {
//        g2o::EdgeUV_SE3_XYZ* e = vecpEdgeUV_SE3_XYZ[i];

//        if(e->chi2()>5.991)
//        {
//            //std::cout<<e->chi2()<<endl;
//            vecPtrMp[i]->mbBad = true; /// @attention 剔除该MapPoint
//        }
//    }
*/
    // Refresh all keyframes
    for (auto pair : mapKf2IdOpt) {
        PtrKeyFrame pKf = pair.first;
        int idOpt = pair.second;
        VertexSE3* pVertex = static_cast<VertexSE3*>(optimizer.vertex(idOpt));
        //pKf->SetPoseAllbyB(toSe3(pVertex->estimate()), se3bc);
        /** @attention 注意这里不能更改对应的里程计的测量得到的位姿变换值*/
        pKf->SetPoseCamera(toSe3(pVertex->estimate().inverse())); /// @attention 求逆!!!!!!!!!
cout<<idOpt<<": "<<pKf->GetPoseCamera().tvec.t()<<endl;
    }

    // Refresh all mappoints
    for (auto pair : mapMp2IdOpt) {
        PtrMapPoint pMp = pair.first;
        int idOpt = pair.second;
        VertexPointXYZ* pVertex = static_cast<VertexPointXYZ*>(optimizer.vertex(idOpt));
        Mat tvec_wm = toCvMatf(pVertex->estimate());
        pMp->SetPos(Pt3(tvec_wm));
    }
}

void SolverVsclam::optimize_mappoints_keyframe_plane()
{
    SparseOptimizer optimizer;
    bool bOptVerbose = true;
    InitOptimizerSlam(optimizer, bOptVerbose);

    int idParamCamera = 0;
    AddCamPara(optimizer, mpDataset->GetCamMat(), idParamCamera);

    int idVertexMax = 0;

    // 顶点: 相机pose
    map<PtrKeyFrame, int> mapKf2IdOpt;
    for (auto ptr : mpDataset->GetKfSet())
    {
        PtrKeyFrame pKf = ptr;
        Se3 Se3wc = pKf->GetPoseCamera();
        Matrix3D Rwc=toEigenMatrix3d(Se3wc.R());
        Vector3D twc=toG2oVector3D(Se3wc.tvec);
        g2o::SE3Quat Twc=g2o::SE3Quat(Rwc, twc);
        g2o::SE3Quat Tcw = Twc.inverse();

        if(idVertexMax != 0) /* fix a bug */
            AddVertexSE3Expmap(optimizer, Tcw, idVertexMax, false);
        else
            AddVertexSE3Expmap(optimizer, Tcw, idVertexMax, true); //第一帧固定
        mapKf2IdOpt[pKf] = idVertexMax++;
    }

    // 顶点: mapPoints
    map<PtrMapPoint,int> mapMp2IdOpt;
    for (auto ptr : mpDataset->GetMpSet())
    {
        PtrMapPoint pMp = ptr;
        Vector3D pose = toG2oVector3D(pMp->GetPos().tvec());
        AddVertexSBAPointXYZ(optimizer, pose, idVertexMax, true);
        mapMp2IdOpt[pMp] = idVertexMax++;
    }

    // 边: SE3-XYZ
    vector<g2o::EdgeProjectXYZ2UV*> vecpEdgeProjectXYZ2UV;
    const float thHuberMono = sqrt(5.991);
    for (auto ptr : mpDataset->GetMsrMpAll())
    {
        PtrMsrUVKf2Mp pMsrMp = ptr;
        PtrKeyFrame pKf = pMsrMp->pKf;
        PtrMapPoint pMp = pMsrMp->pMp;
        /** @todo leather add*/
        if(pMp->mbBad)
            continue;
        int idKf = mapKf2IdOpt[pKf]; //得到关键帧的ID号
        int idMp = mapMp2IdOpt[pMp]; //得到MapPoint的ID号
        g2o::Vector2D measure = toG2oVector2D(pMsrMp->measure);
        g2o::Matrix2D info = toEigenMatrixXd(pMsrMp->info); /** @todo 根据提取所在的<层>再搞一搞???? */

        g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();//MapPoints在前，SE3在后
        e->vertices()[0] = optimizer.vertex(idMp);
        e->vertices()[1] = optimizer.vertex(idKf);
        e->setMeasurement(measure);
        e->setInformation(info);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(thHuberMono);
        e->setRobustKernel(rk);
        e->setParameterId(0, idParamCamera);
        optimizer.addEdge(e);

        vecpEdgeProjectXYZ2UV.push_back(e);
    }

    // 一元边:平面约束
    map<PtrKeyFrame, int>::iterator iter;
    for (iter=mapKf2IdOpt.begin(); iter!=mapKf2IdOpt.end(); iter++)
    {
        PtrKeyFrame pKf = iter->first;
        int idKf = iter->second;

        Se3 Se3wc = pKf->GetPoseCamera();
        Matrix3D Rwc=toEigenMatrix3d(Se3wc.R());
        Vector3D twc=toG2oVector3D(Se3wc.tvec);
        twc(2) = 0; /** @attention 投影到平面运动空间，令z=0, pitch=roll=0????*/
        g2o::SE3Quat Twc=g2o::SE3Quat(Rwc, twc);
        g2o::SE3Quat Tcw = Twc.inverse();

        g2o::Matrix6d covariance_6d = g2o::Matrix6d::Identity();
        covariance_6d(0,0) = 1.0;
        covariance_6d(1,1) = 1.0;
        covariance_6d(2,2) = 1.0;
        covariance_6d(3,3) = 1.0;
        covariance_6d(4,4) = 1.0;
        covariance_6d(5,5) = 0.0001;

        g2o::Matrix6d Info = g2o::Matrix6d::Identity();
        Info = covariance_6d.inverse();

        g2o::EdgePlaneConstraint* e = new g2o::EdgePlaneConstraint;
        e->vertices()[0] = optimizer.vertex(idKf);
        e->setMeasurement(Tcw);
        e->setInformation(Info);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(30);

    // Refresh all keyframes
    for (auto pair : mapKf2IdOpt) {
        PtrKeyFrame pKf = pair.first;
        int idOpt = pair.second;
        VertexSE3Expmap* pVertex = static_cast<VertexSE3Expmap*>(optimizer.vertex(idOpt));
        //pKf->SetPoseAllbyB(toSe3(pVertex->estimate()), se3bc);
        /** @attention 注意这里不能更改对应的里程计的测量得到的位姿变换值*/
//SE3Quat->Se3
        pKf->SetPoseCamera(Se3(toCvMatf(pVertex->estimate().inverse()))); /// @attention 求逆!!!!!!!!!
//cout<<idOpt<<": "<<pKf->GetPoseCamera().tvec.t()<<endl;
    }

    // Refresh all mappoints
    for (auto pair : mapMp2IdOpt) {
        PtrMapPoint pMp = pair.first;
        int idOpt = pair.second;
        VertexSBAPointXYZ* pVertex = static_cast<VertexSBAPointXYZ*>(optimizer.vertex(idOpt));
        Mat tvec_wm = toCvMatf(pVertex->estimate());
        pMp->SetPos(Pt3(tvec_wm));
    }
}

void SolverVsclam::optimize_extrinsic()
{
    SparseOptimizer optimizer;
    bool bOptVerbose = true;
    InitOptimizerSlam(optimizer, bOptVerbose);

    int idParamCamera = 0;
    AddCamPara(optimizer, mpDataset->GetCamMat(), idParamCamera);

    int idVertexMax = 0;
    // 顶点: 外参
    Se3 se3bc = mpDataset->GetCamOffset();
    int idVertexCamOffset = idVertexMax++;
//Se3<->SE3Quat
    Matrix3D Rbc=toEigenMatrix3d(se3bc.R());
    Vector3D tbc=toG2oVector3D(se3bc.tvec);
    g2o::SE3Quat Tbc=g2o::SE3Quat(Rbc, tbc);
    std::cout<<"Tbc: "<<Tbc<<std::endl;
    AddVertexSE3Expmap(optimizer, Tbc, idVertexCamOffset, false);

    // 顶点: 相机pose
    map<PtrKeyFrame, int> mapKf2IdOpt;
    for (auto ptr : mpDataset->GetKfSet())
    {
        PtrKeyFrame pKf = ptr;
        Se3 Se3wc = pKf->GetPoseCamera();
        Matrix3D Rwc=toEigenMatrix3d(Se3wc.R());
        Vector3D twc=toG2oVector3D(Se3wc.tvec);
        g2o::SE3Quat Twc=g2o::SE3Quat(Rwc, twc);
        g2o::SE3Quat Tcw = Twc.inverse();

        if(idVertexMax != 1)
            AddVertexSE3Expmap(optimizer, Tcw, idVertexMax, false);
        else
            AddVertexSE3Expmap(optimizer, Tcw, idVertexMax, true); //第一帧固定
        mapKf2IdOpt[pKf] = idVertexMax++;
    }

    // 顶点: mapPoints
    map<PtrMapPoint,int> mapMp2IdOpt;
    for (auto ptr : mpDataset->GetMpSet())
    {
        PtrMapPoint pMp = ptr;
        Vector3D pose = toG2oVector3D(pMp->GetPos().tvec());
        AddVertexSBAPointXYZ(optimizer, pose, idVertexMax, true);
        mapMp2IdOpt[pMp] = idVertexMax++;
    }

    // 一元边:平面约束
    map<PtrKeyFrame, int>::iterator iter;
    for (iter=mapKf2IdOpt.begin(); iter!=mapKf2IdOpt.end(); iter++)
    {
        PtrKeyFrame pKf = iter->first;
        int idKf = iter->second;

        Se3 Se3wc = pKf->GetPoseCamera();
        Matrix3D Rwc=toEigenMatrix3d(Se3wc.R());
        Vector3D twc=toG2oVector3D(Se3wc.tvec);
        twc(2) = 0; /** @attention 投影到平面运动空间，令z=0, pitch=roll=0????*/
        g2o::SE3Quat Twc=g2o::SE3Quat(Rwc, twc);
        g2o::SE3Quat Tcw = Twc.inverse();

        g2o::Matrix6d covariance_6d = g2o::Matrix6d::Identity();
        covariance_6d(0,0) = 1.0;
        covariance_6d(1,1) = 1.0;
        covariance_6d(2,2) = 1.0;
        covariance_6d(3,3) = 1.0;
        covariance_6d(4,4) = 1.0;
        covariance_6d(5,5) = 0.0001;

        g2o::Matrix6d Info = g2o::Matrix6d::Identity();
        Info = covariance_6d.inverse();

        g2o::EdgePlaneConstraint* e = new g2o::EdgePlaneConstraint;
        e->vertices()[0] = optimizer.vertex(idKf);
        e->setMeasurement(Tcw);
        e->setInformation(Info);
        optimizer.addEdge(e);
    }

    // 边: SE3-XYZ
    vector<g2o::EdgeProjectXYZ2UV*> vecpEdgeProjectXYZ2UV;
    const float thHuberMono = sqrt(5.991);
    for (auto ptr : mpDataset->GetMsrMpAll())
    {
        PtrMsrUVKf2Mp pMsrMp = ptr;
        PtrKeyFrame pKf = pMsrMp->pKf;
        PtrMapPoint pMp = pMsrMp->pMp;
        /** @todo leather add*/
        if(pMp->mbBad)
            continue;
        int idKf = mapKf2IdOpt[pKf]; //得到关键帧的ID号
        int idMp = mapMp2IdOpt[pMp]; //得到MapPoint的ID号
        g2o::Vector2D measure = toG2oVector2D(pMsrMp->measure);
        g2o::Matrix2D info = toEigenMatrixXd(pMsrMp->info); /** @todo 根据提取所在的<层>再搞一搞???? */

        g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();//MapPoints在前，SE3在后
        e->vertices()[0] = optimizer.vertex(idMp);
        e->vertices()[1] = optimizer.vertex(idKf);
        e->setMeasurement(measure);
        e->setInformation(info);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(thHuberMono);
        e->setRobustKernel(rk);
        e->setParameterId(0, idParamCamera);
        optimizer.addEdge(e);

        vecpEdgeProjectXYZ2UV.push_back(e);
    }

    // 边: SE3-SE3
    vector<g2o::EdgeOnlineCalibration*> vecEdgeOnlineCalibration;
    for (auto ptr : mpDataset->GetMsrOdoSet())
    {
        PtrMsrSe2Kf2Kf pMsrOdo = ptr;
        PtrKeyFrame pKf0 = pMsrOdo->pKfHead;
        PtrKeyFrame pKf1 = pMsrOdo->pKfTail;
        int id0 = mapKf2IdOpt[pKf0];
        int id1 = mapKf2IdOpt[pKf1];

        Se2 measure_se2 = pMsrOdo->se2;
        g2o::Matrix3D covariance = toEigenMatrixXd(pMsrOdo->info).inverse(); /// 注意这里又转换成了协方差
//Se2<->SE3Quat
        Eigen::AngleAxisd rotz(measure_se2.theta, Eigen::Vector3d::UnitZ());
        g2o::SE3Quat relativePose_SE3Quat(rotz.toRotationMatrix(), Eigen::Vector3d(measure_se2.x, measure_se2.y, 0));

        g2o::Matrix6d covariance_6d = g2o::Matrix6d::Identity();
        covariance_6d(0,0) = covariance(2,2);
        covariance_6d(0,4) = covariance(2,0); covariance_6d(0,5) = covariance(2,1);
        covariance_6d(4,0) = covariance(0,2); covariance_6d(5,0) = covariance(1,2);

        covariance_6d(3,3) = covariance(0,0);
        //covariance_6d(4,5) = covariance(0,1);covariance_6d(5,4) = covariance(1,0);

        covariance_6d(4,4) = covariance(1,1);

        covariance_6d(1,1) = 0.00001;
        covariance_6d(2,2) = 0.01;
        covariance_6d(5,5) = 0.0001; //平移分量:Z

//        std::cout<<relativePose_SE3Quat.to_homogeneous_matrix()<<std::endl;
//        std::cout<<covariance_6d<<std::endl;
//        std::cout<<"--------------------------------"<<endl;

        g2o::Matrix6d Info = g2o::Matrix6d::Identity();
        Info = covariance_6d.inverse();

        g2o::EdgeOnlineCalibration* e = new g2o::EdgeOnlineCalibration;
        e->vertices()[0] = optimizer.vertex(id0);
        e->vertices()[1] = optimizer.vertex(id1);
        e->vertices()[2] = optimizer.vertex(idVertexCamOffset);
        e->setMeasurement(relativePose_SE3Quat);
        e->setInformation(Info);
        optimizer.addEdge(e);
        vecEdgeOnlineCalibration.push_back(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(50);

    // Refresh all keyframes
    for (auto pair : mapKf2IdOpt) {
        PtrKeyFrame pKf = pair.first;
        int idOpt = pair.second;
        VertexSE3Expmap* pVertex = static_cast<VertexSE3Expmap*>(optimizer.vertex(idOpt));
        //pKf->SetPoseAllbyB(toSe3(pVertex->estimate()), se3bc);
        /** @attention 注意这里不能更改对应的里程计的测量得到的位姿变换值*/
//SE3Quat->Se3
        pKf->SetPoseCamera(Se3(toCvMatf(pVertex->estimate().inverse()))); /// @attention 求逆!!!!!!!!!
cout<<idOpt<<": "<<pKf->GetPoseCamera().tvec.t()<<endl;
    }

    // Refresh all mappoints
    for (auto pair : mapMp2IdOpt) {
        PtrMapPoint pMp = pair.first;
        int idOpt = pair.second;
        VertexSBAPointXYZ* pVertex = static_cast<VertexSBAPointXYZ*>(optimizer.vertex(idOpt));
        Mat tvec_wm = toCvMatf(pVertex->estimate());
        pMp->SetPos(Pt3(tvec_wm));
    }

    VertexSE3Expmap* pVertex = static_cast<VertexSE3Expmap*>(optimizer.vertex(idVertexCamOffset));
    std::cout<<"Tbc:"<<pVertex->estimate().to_homogeneous_matrix()<<std::endl;
}

}




















