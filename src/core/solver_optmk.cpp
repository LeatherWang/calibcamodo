#include "solver_optmk.h"
#include "g2o/g2o_api.h"
#include "adapter.h"

using namespace cv;
using namespace std;
using namespace g2o;

namespace calibcamodo {

SolverOptMk::SolverOptMk(Dataset* _pDataset):
    SolverBase(_pDataset) {

}

void SolverOptMk::DoCalib() {
    //【1】Set optimizer
    SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    InitOptimizerCalib(optimizer);


    int idParamCamera = 0;
    AddCamPara(optimizer, mpDataset->GetCamMat(), idParamCamera);


    //【2】Set extrinsic vertex
    int idVertexMax = 0;
    Isometry3D Iso3_bc = toG2oIsometry3D(mpDataset->GetCamOffset());
    int idVertexCamOffset = idVertexMax++;
    AddVertexSE3(optimizer, Iso3_bc, idVertexCamOffset);

    //【3】Set keyframe vertices
    map<PtrKeyFrame,int> mappKf2IdOpt;
    for (auto ptr : mpDataset->GetKfSet()) {
        PtrKeyFrame pKf = ptr;
        if(!pKf->GetId()) //id=0，表示第一个关键帧
            AddVertexSE2(optimizer, toG2oSE2(pKf->GetPoseBase()), idVertexMax, true); /** @attention 此编号并非按照关键帧的时间顺序排列，而是set类型自己排列*/
        else
            AddVertexSE2(optimizer, toG2oSE2(pKf->GetPoseBase()), idVertexMax, false);
        mappKf2IdOpt[pKf] = idVertexMax++;
    }

    //【4】Set mark vertices
    map<PtrMapMark,int> mappMk2IdOpt;
    for (auto ptr : mpDataset->GetMkSet()) {
        PtrMapMark pMk = ptr;
        //! NEED TO ADD INIT MK POSE HERE !!!
        g2o::Vector3D pose = toG2oVector3D(pMk->GetPose().tvec); //在世界坐标系下的坐标

        AddVertexPointXYZ(optimizer, pose, idVertexMax, true); /** @attention 设置边缘化*/
        mappMk2IdOpt[pMk] = idVertexMax++;
        // DEBUG
//                cerr << "mkId: " << pMk->GetId() << endl;
//                cerr << "mkTvec: " << pMk->GetPose().tvec << endl;
//                cerr << "pose: " << pose << endl;
    }

    //【5】Set odometry edges
    for (auto ptr : mpDataset->GetMsrOdoSet()) {
        PtrMsrSe2Kf2Kf pMsrOdo = ptr;
        PtrKeyFrame pKf0 = pMsrOdo->pKfHead;
        PtrKeyFrame pKf1 = pMsrOdo->pKfTail;
        int id0 = mappKf2IdOpt[pKf0];
        int id1 = mappKf2IdOpt[pKf1];
        g2o::SE2 measure = toG2oSE2(pMsrOdo->se2);
        g2o::Matrix3D info = toEigenMatrixXd(pMsrOdo->info);
        AddEdgeSE2(optimizer, id0, id1, measure, info);

//        cout<<pKf0->GetId()<<": "<<pKf0->GetPoseBase().x<<", "<<pKf0->GetPoseBase().y<<endl;
//        cout<<pKf1->GetId()<<": "<<pKf1->GetPoseBase().x<<", "<<pKf1->GetPoseBase().y<<endl;
//        cout<<"---------------------"<<endl;

        // DEBUG
//                cerr <<"info: "<<info << endl;
//                cerr << pMsrOdo->info << endl;
    }

    //【6】Set mark measurement edges
    // 三元边，观测是三维的
    for (auto ptr : mpDataset->GetMsrMkAll()) {
        PtrMsrPt3Kf2Mk pMsrMk = ptr;
        PtrKeyFrame pKf = pMsrMk->pKf;
        PtrMapMark pMk = pMsrMk->pMk;

        int idKf = mappKf2IdOpt[pKf];
        int idMk = mappMk2IdOpt[pMk];

//        g2o::Vector3D measure = toG2oVector3D(pMsrMk->measure);
//        g2o::Matrix3D info = toEigenMatrixXd(pMsrMk->info);

//        AddEdgeOptMk(optimizer, idKf, idMk, 0, measure, info);

        Eigen::Vector3d vec3MsrPt = toG2oVector3D(pMsrMk->measure);
        Eigen::Vector3d vec3KP = toEigenMatrixXd(mpDataset->GetCamMat())*vec3MsrPt;
        g2o::Vector2D measure;
        measure(0) = vec3KP(0)/vec3KP(2);
        measure(1) = vec3KP(1)/vec3KP(2);
        g2o::Matrix2D info;
        info << 1, 0, 0, 1;
        AddEdgeVSclam(optimizer, idKf, idMk, idVertexCamOffset, idParamCamera, measure, info);

        // DEBUG
//                cerr <<"info: " <<info << endl;
        //        cerr << pMsrMk->measure << endl;
        //        cerr << measure << endl;
        //        cerr << pMsrMk->info << endl;
    }

    //【7】Do optimize
    optimizer.initializeOptimization();
    optimizer.optimize(50);

    //【8】Refresh calibration results
    g2o::VertexSE3* v = static_cast<g2o::VertexSE3*>(optimizer.vertex(0));
    Isometry3D Iso3_bc_opt = v->estimate();

    Se3 se3bc = toSe3(Iso3_bc_opt);
    mpDataset->SetCamOffset(se3bc);

    //【9】Refresh keyframe
    for (auto pair : mappKf2IdOpt) {
        PtrKeyFrame pKf = pair.first;
        int idOpt = pair.second;
        VertexSE2* pVertex = static_cast<VertexSE2*>(optimizer.vertex(idOpt));

//        cout<<pKf->GetId()<<": "<<pKf->GetPoseCamera().tvec.t()<<endl;

        pKf->SetPoseAllbyB(toSe2(pVertex->estimate()), se3bc);

//        cout<<idOpt<<": "<<pKf->GetPoseCamera().tvec.t()<<endl;
        if(!pKf->GetId())
        {
            cout<<"---------------------"<<endl;
            cout<<pKf->GetId()<<": "<<pKf->GetPoseBase().x<<", "<<pKf->GetPoseBase().y<<endl;
            cout<<"---------------------"<<endl;
        }
    }

    //【10】Refresh landmark
    for (auto pair : mappMk2IdOpt) {
        PtrMapMark pMk = pair.first;
        int idOpt = pair.second;
        VertexPointXYZ* pVertex = static_cast<VertexPointXYZ*>(optimizer.vertex(idOpt));
        Mat tvec_wm = toCvMatf(pVertex->estimate());
        pMk->SetPoseTvec(tvec_wm);

        // DEBUG:
        //        cerr << "tvec_wm: " << tvec_wm.t() << endl;
    }
}

}
