
#include "slide_optimation.h"
#include "g2o/g2o_api.h"
#include "core/adapter.h"

using namespace std;
using namespace g2o;
using namespace cv;

namespace calibcamodo
{
SlideOptimation::SlideOptimation(DatasetAruco *_pDataSetAruco, unsigned int _nLocalWindowSize):
    mbAcceptKeyFrames(true),is_thread_exit(false)
{
    // 初始化互斥量，使用默认的互斥量属性
    int res_1 = pthread_mutex_init(&mMutexNewKFs, NULL);
    int res_2 = pthread_mutex_init(&mMutexAccept, NULL);
    if(res_1 || res_2)
    {
        cerr<<"pthread_mutex_init failed\n"<<endl;
        exit(EXIT_FAILURE);
    }

    mpDatasetAruco = _pDataSetAruco;
    mnLocalWindowSize = _nLocalWindowSize;
    mnFixedCamerasSize = 4;
}

SlideOptimation::~SlideOptimation()
{
    // distory mutex
    pthread_mutex_destroy(&mMutexNewKFs);
    pthread_mutex_destroy(&mMutexAccept);
}

void * SlideOptimation::Run(void *__this)
{
    pthread_detach(pthread_self()); //将joined线程重设置为分离线程，省去资源回收的麻烦，不使用pthread_join进行资源回收，将导致僵尸线程，占据资源不释放
    SlideOptimation * _this =(SlideOptimation *)__this;
    std::cout<<"thread start"<<endl;
    while(!_this->is_thread_exit)
    {
        _this->SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if (_this->CheckNewKeyFrames())
        {
            _this->ProcessNewKeyFrame();
            //cerr<<"local windows size: "<<_this->mlLocalKeyFrames.size()<<endl;
            if(_this->mpDatasetAruco->GetKfSet().size() > (_this->mnLocalWindowSize+_this->mnFixedCamerasSize))
            {
                //_this->DoOptimation();
                _this->calibExtrinsic();
            }
        }
        _this->SetAcceptKeyFrames(true);

        usleep(1000);

    }
    std::cout<<"thread halted!"<<endl;
    return 0;
}

void SlideOptimation::start_optimation_thread()
{
    is_thread_exit = false;
    pthread_t pth;
    pthread_create(&pth,NULL,Run,(void*)this);
}

void SlideOptimation::AddToLocalWindow(PtrKeyFrameAruco pKF)
{
    mlLocalKeyFrames.push_back(pKF);
    if (mlLocalKeyFrames.size() > mnLocalWindowSize) {
        mlLocalKeyFrames.pop_front();
    }
}

void SlideOptimation::InsertKeyFrame(PtrKeyFrameAruco pKF)
{
    pthread_mutex_lock (&mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    pthread_mutex_unlock(&mMutexNewKFs);
}

bool SlideOptimation::CheckNewKeyFrames()
{
    pthread_mutex_lock (&mMutexNewKFs);
    bool bNonEmpty = (!mlNewKeyFrames.empty());
    pthread_mutex_unlock(&mMutexNewKFs);

    return bNonEmpty;
}

void SlideOptimation::SetAcceptKeyFrames(bool flag)
{
    pthread_mutex_lock (&mMutexAccept);
    mbAcceptKeyFrames = flag;
    pthread_mutex_unlock (&mMutexAccept);
}

bool SlideOptimation::AcceptKeyFrames()
{
    pthread_mutex_lock (&mMutexAccept);
    bool bAcceptKeyFrames = mbAcceptKeyFrames;
    pthread_mutex_unlock (&mMutexAccept);

    return bAcceptKeyFrames;
}

void SlideOptimation::ProcessNewKeyFrame()
{
    {
        pthread_mutex_lock (&mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();// 从列表中获得一个等待被插入的关键帧
        mlNewKeyFrames.pop_front(); //从队列中移除
        pthread_mutex_unlock(&mMutexNewKFs);
    }

    AddToLocalWindow(mpCurrentKeyFrame);
}

bool isRotationMatrix(const Eigen::Matrix3d &R)
{
    Eigen::Matrix3d Rt;
    Rt = R.transpose();
    Eigen::Matrix3d shouldBeIdentity = Rt * R;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    return (shouldBeIdentity - I).norm() < 1e-6;
}

Eigen::Vector3d rotationMatrixToEulerAngles(const Eigen::Matrix3d &R)
{
    assert(isRotationMatrix(R));
    double sy = sqrt(R(0,0) * R(0,0) +  R(1,0) * R(1,0));
    bool singular = sy < 1e-6; // true: `Y`方向旋转为`+/-90`度
    double x, y, z;
    if (!singular)
    {
        x = atan2(R(2,1) , R(2,2));
        y = atan2(-R(2,0), sy);
        z = atan2(R(1,0), R(0,0));
    }
    else
    {
        x = atan2(-R(1,2), R(1,1));
        y = atan2(-R(2,0), sy);
        z = 0;
    }
    return Eigen::Vector3d(x, y, z);
}

Eigen::Matrix4d inversePose(const Eigen::Matrix4d &pose) {
    Eigen::Matrix4d ret = pose;
    ret.block<3, 3>(0, 0) = pose.block<3, 3>(0, 0).transpose();
    ret.block<3, 1>(0, 3) = -ret.block<3, 3>(0, 0) * pose.block<3, 1>(0, 3);
    return ret;
}


Eigen::Matrix3d hat( const Eigen::Vector3d &omega )
{
    Eigen::Matrix3d Omega;
    Omega <<  0, -omega( 2 ),  omega( 1 )
          ,  omega( 2 ),     0, -omega( 0 )
          , -omega( 1 ),  omega( 0 ),    0;
    return Omega;
}

Eigen::Matrix3d JacobianRInv ( const Eigen::Vector3d& w )
{
    Eigen::Matrix3d Jrinv = Eigen::Matrix3d::Identity();
    double theta = w.norm();
    if ( theta < 1e-10 )
    {
        return Jrinv;
    }
    else
    {
        Eigen::Vector3d k = w.normalized();  // k - unit direction vector of w
        Eigen::Matrix3d K = hat ( k );
        Jrinv = Jrinv
                + 0.5*hat ( k )
                + ( 1.0 - ( 1.0+std::cos ( theta ) ) *theta / ( 2.0*std::sin ( theta ) ) ) *K*K;
    }

    return Jrinv;
}

Eigen::Matrix3d JacobianLInv( const Eigen::Vector3d& w )
{
    return JacobianRInv(-w);
}

void SlideOptimation::calibExtrinsic()
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>(); //!@attention
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    Eigen::Matrix4d Toc; //!@todo
//    Toc<<  0.,  0., 1., 10.325,
//          -1.,  0., 0., 0.066,
//           0., -1., 0., 0.,
//           0.,  0., 0., 1.;
    Toc = toEigenMatrixXd(mpDatasetAruco->GetCamOffset().T());

    cout<<"__r: "<< rotationMatrixToEulerAngles(Toc.block<3,3>(0,0)).transpose()*180/M_PI <<endl;
    cout<<"t: "<< Toc.block<3,1>(0,3).transpose() <<endl;


    // 外参顶点
    g2o::VertexSE3Expmap* vCalib = new g2o::VertexSE3Expmap;
    int id_calib = 0;
    {
        g2o::SE3Quat SE3Q_Tbc = g2o::SE3Quat( Toc.block<3,3>(0,0), Toc.block<3,1>(0,3));
        vCalib->setEstimate(SE3Q_Tbc);
        vCalib->setId(id_calib);
        vCalib->setFixed(false);
        optimizer.addVertex(vCalib);
    }

    std::vector<g2o::VertexSE3Expmap*> vecKFPoseVertex;
    std::vector<g2o::EdgeOnlineCalibration*> vecEdge;
    int KFId = id_calib + 1;

    int markerId = mlLocalKeyFrames.size() + 1;

    // 起始顶点(固定)
    PtrKeyFrameAruco lastKF = NULL;
    auto iter = mlLocalKeyFrames.begin();
    {
        PtrKeyFrameAruco curKF = *iter;
        cv::Mat Twc_ = curKF->GetPoseCamera().T();
        Eigen::Matrix4d Twc = toEigenMatrixXd(Twc_);
        Eigen::Matrix4d Tcw = inversePose(Twc);

        g2o::VertexSE3Expmap * curKFPoseVertex = new g2o::VertexSE3Expmap();
        curKFPoseVertex->setEstimate(g2o::SE3Quat(Tcw.block<3,3>(0,0), Tcw.block<3,1>(0,3)));
        curKFPoseVertex->setId(KFId++);
        curKFPoseVertex->setFixed(true); //固定优化窗口第一帧Pose
        optimizer.addVertex(curKFPoseVertex);
        vecKFPoseVertex.push_back(curKFPoseVertex);

        lastKF = curKF;
    }

    map<PtrKeyFrameAruco,int> mappKf2IdOpt;
    map<int, g2o::VertexSE3Expmap*> mapIdMarker;
    for(iter++; iter != mlLocalKeyFrames.end(); iter++)
    {
        PtrKeyFrameAruco curKF = *iter;
        cv::Mat Twc_ = curKF->GetPoseCamera().T();
        Eigen::Matrix4d Twc = toEigenMatrixXd(Twc_);
        Eigen::Matrix4d Tcw = inversePose(Twc);

        mappKf2IdOpt.insert(make_pair(curKF, KFId));
        g2o::VertexSE3Expmap* curKFPoseVertex = new g2o::VertexSE3Expmap();
        curKFPoseVertex->setEstimate(g2o::SE3Quat(Tcw.block<3,3>(0,0), Tcw.block<3,1>(0,3)));
        curKFPoseVertex->setId(KFId++);
        curKFPoseVertex->setFixed(false);
        optimizer.addVertex(curKFPoseVertex);

        //添加一元边
        {
            const std::vector<PtrMapMarkAruco> &vecpMapMark = curKF->mvpMapMark; //当前关键帧的所有观测到Marker
            for(auto mapMark : vecpMapMark)
            {
                int id = mapMark->GetId();

                // marker顶点
                g2o::VertexSE3Expmap* markerPoseVertex = new g2o::VertexSE3Expmap();
                auto iter = mapIdMarker.find(id);
                if(iter == mapIdMarker.end())
                {
                    Eigen::Matrix4d Twm = toEigenMatrixXd(mapMark->GetPose().T());
                    Eigen::Matrix4d Tmw = inversePose(Twm);
                    markerPoseVertex->setEstimate(g2o::SE3Quat(Tmw.block<3,3>(0,0), Tmw.block<3,1>(0,3)));
                    markerPoseVertex->setId(markerId++); //!@attention
                    markerPoseVertex->setFixed(true); //不优化
                    optimizer.addVertex(markerPoseVertex);
                    mapIdMarker.insert(make_pair(id, markerPoseVertex));
                }
                else
                    markerPoseVertex = iter->second;

                // 测量
                Eigen::Matrix4d Tcm = curKF->mmapId2ArucoMsrPose[id];
                g2o::SE3Quat SE3QTcm(Tcm.block<3,3>(0,0), Tcm.block<3,1>(0,3));

                g2o::Matrix6d info_se3_Tcm = g2o::Matrix6d::Identity(); //!@todo

                g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap;
                e->vertices()[0] = markerPoseVertex; //Tmw
                e->vertices()[1] = curKFPoseVertex; //Tcw
                e->setMeasurement(SE3QTcm);
                e->setInformation(info_se3_Tcm);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(sqrt(16.812));

                //        e->computeError();
                //        g2o::Vector6d error = e->error();
                //        cout<<"1 error: " <<error.transpose()<<endl;
                optimizer.addEdge(e);
            }
        }

        // 添加三元边
//        {
//            Se2 odo_se2_bibj = curKF->GetPoseBase() - lastKF->GetPoseBase();
//            double dist = sqrt(odo_se2_bibj.x*odo_se2_bibj.x + odo_se2_bibj.y*odo_se2_bibj.y);
//            cout <<"odom: "<<odo_se2_bibj.x << "  " << odo_se2_bibj.y << "  " << odo_se2_bibj.theta << endl;

//            Eigen::Vector3f covRaw;
//            double stdlin = max(dist*mpDatasetAruco->mOdoLinErrR, mpDatasetAruco->mOdoLinErrMin);
//            double theta = odo_se2_bibj.theta;
//            double stdrot = max(max(fabs(theta)*mpDatasetAruco->mOdoRotErrR, mpDatasetAruco->mOdoRotErrMin), dist*mpDatasetAruco->mOdoRotErrRLin);
//            covRaw(0) = stdlin*stdlin;
//            covRaw(1) = stdlin*stdlin;
//            covRaw(2) = stdrot*stdrot;

//            g2o::Matrix6d info_se3_bTb = g2o::Matrix6d::Identity();
////            info_se3_bTb(0,0)=1e4;
////            info_se3_bTb(1,1)=1e4;
////            info_se3_bTb(2,2)=1/covRaw(2);

//            Eigen::AngleAxisd rotz(odo_se2_bibj.theta, Eigen::Vector3d::UnitZ());
//            g2o::SE3Quat SE3Tbibj(rotz.toRotationMatrix(), Eigen::Vector3d(odo_se2_bibj.x, odo_se2_bibj.y, 0));

////            Eigen::Matrix3d cov_tranlation = Eigen::Matrix3d::Identity();
////            cov_tranlation(0,0) = covRaw(0);
////            cov_tranlation(0,0) = covRaw(1);
////            cov_tranlation(2,2) = 0.001;
////            Eigen::Matrix<double, 6, 1, Eigen::ColMajor> lie = SE3Tbibj.log();
////            Eigen::Matrix3d JLInv = JacobianLInv(lie.block<3,1>(0,0));
////            Eigen::Matrix3d cov_lie_translation = JLInv*cov_tranlation*(JLInv.transpose()); //将t的协方差转换到其对应的李代数的协方差
////            info_se3_bTb.block<3,3>(3,3) = cov_lie_translation.inverse();

//            g2o::EdgeOnlineCalibration* e = new g2o::EdgeOnlineCalibration;
//            e->vertices()[0] = vecKFPoseVertex.back();
//            e->vertices()[1] = curKFPoseVertex;
//            e->vertices()[2] = vCalib;
//            e->setMeasurement(SE3Tbibj);
//            e->setInformation(info_se3_bTb);

//            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//            e->setRobustKernel(rk);
//            rk->setDelta(sqrt(16.812));

//                    e->computeError();
//                    g2o::Vector6d error = e->error();
//                    cout<<"1 error: " <<error.transpose()<<endl;
//            optimizer.addEdge(e);

//            vecKFPoseVertex.push_back(curKFPoseVertex);
//            vecEdge.push_back(e);
//            lastKF = curKF;
//        }
    }


    optimizer.initializeOptimization();
    optimizer.optimize(20);

    g2o::VertexSE3Expmap* vSE3Tbc = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(id_calib));
    Eigen::Matrix4d Tbc = vSE3Tbc->estimate().to_homogeneous_matrix();

    //Eigen::Vector3d euler_angles = Tbc.block<3,3>(0,0).eulerAngles( 2,1,0 );//ZYX顺序
    //cout<<"euler angles: Z-Y-X= "<<euler_angles.transpose()*180/M_PI<<endl<<endl;
    cout<<"r: "<< rotationMatrixToEulerAngles(Tbc.block<3,3>(0,0)).transpose()*180/M_PI <<endl;
    cout<<"t: "<< Tbc.block<3,1>(0,3).transpose() <<endl;
    cout << endl<<endl;
}

void SlideOptimation::DoOptimation()
{/*
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    InitOptimizerCalib(optimizer);

    //【1】相机内参
    int idParamCamera = 0;
    AddCamPara(optimizer, mpDatasetAruco->GetCamMat(), idParamCamera);

    //【2】Set extrinsic vertex
    int idVertexMax = 0;
    Isometry3D Iso3_bc = toG2oIsometry3D(mpDatasetAruco->GetCamOffset());
    int idVertexCamOffset = idVertexMax++;
    AddVertexSE3(optimizer, Iso3_bc, idVertexCamOffset);

    //【3】Set keyframe vertices
    map<PtrKeyFrameAruco,int> mappKf2IdOpt;
    std::list<PtrKeyFrameAruco> lFixedCameras;
    {
        PtrKeyFrameAruco pFixedKF=this->mlLocalKeyFrames.front()->mpParent; //! @attention lFixedCameras中的关键帧是倒序排列
        lFixedCameras.push_back(pFixedKF);
        if(!pFixedKF) {
            cerr<<FRED("[ERROR:]")<<"this KF has no parent."<<endl;
            return;
        }
//        cerr<<"pFixedKF size: "<<pFixedKF->GetId()<<endl;
        for(int i=1; i<mnFixedCamerasSize; i++)
        {
            PtrKeyFrameAruco pKFPrevLocal = pFixedKF->mpParent;
            if(pKFPrevLocal) {
                lFixedCameras.push_back(pKFPrevLocal);
                pFixedKF = pKFPrevLocal;
            }
            else {
                cerr << FRED("[ERROR:]") << "pKFPrevLocal is NULL?" << endl;
                return;
            }
//            cerr<<"pFixedKF size: "<<pKFPrevLocal->GetId()<<endl;
        }
    }
//    cerr<<"lFixedCameras size: "<<lFixedCameras.size()<<endl;
    for (list<PtrKeyFrameAruco>::const_iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end();
         lit != lend; lit++)
    {
        PtrKeyFrameAruco pFixedKFi = *lit;
        AddVertexSE2(optimizer, toG2oSE2(pFixedKFi->GetPoseBase()), idVertexMax, true);
        mappKf2IdOpt[pFixedKFi] = idVertexMax++;
    }
    for (list<PtrKeyFrameAruco>::const_iterator lit = mlLocalKeyFrames.begin(), lend = mlLocalKeyFrames.end();
         lit != lend; lit++)
    {
        PtrKeyFrameAruco pKFi = *lit;
        AddVertexSE2(optimizer, toG2oSE2(pKFi->GetPoseBase()), idVertexMax, false);
        mappKf2IdOpt[pKFi] = idVertexMax++;
    }

    //【4】Set mark vertices
    map<PtrMapMarkAruco, int> mappMk2IdOpt;
    for (list<PtrKeyFrameAruco>::const_iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end();
         lit != lend; lit++)
    {
        PtrKeyFrameAruco pFixedKFi = *lit;
        for(auto pMapMark : pFixedKFi->mvpMapMark)
        {
            // 防止重复添加
            if(pMapMark->mnBAOptForKF != mpCurrentKeyFrame->GetId()) //! @todo debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            {
                pMapMark->mnBAOptForKF = mpCurrentKeyFrame->GetId();
                g2o::Vector3D pose = toG2oVector3D(pMapMark->GetPose().tvec); //在世界坐标系下的坐标
                AddVertexPointXYZ(optimizer, pose, idVertexMax, false); //! @attention 设置边缘化
                mappMk2IdOpt[pMapMark] = idVertexMax++;
            }
        }
    }
    for (list<PtrKeyFrameAruco>::const_iterator lit = mlLocalKeyFrames.begin(), lend = mlLocalKeyFrames.end();
         lit != lend; lit++)
    {
        PtrKeyFrameAruco pKFi = *lit;
        for(auto pMapMark : pKFi->mvpMapMark)
        {
            if(pMapMark->mnBAOptForKF != mpCurrentKeyFrame->GetId())
            {
                pMapMark->mnBAOptForKF = mpCurrentKeyFrame->GetId();
                g2o::Vector3D pose = toG2oVector3D(pMapMark->GetPose().tvec); //在世界坐标系下的坐标
                AddVertexPointXYZ(optimizer, pose, idVertexMax, false);
                mappMk2IdOpt[pMapMark] = idVertexMax++;
            }
        }
    }

    //【5】Set odometry edges
    {
        list<PtrKeyFrameAruco>::const_iterator lend = lFixedCameras.end();
        lend--;
        for (list<PtrKeyFrameAruco>::const_iterator lit = lFixedCameras.begin(); lit != lend; lit++)
        {
            PtrKeyFrameAruco pFixedKFi = *lit;
            PtrKeyFrameAruco pFixedKFPrev = pFixedKFi->mpParent;
            int id0 = mappKf2IdOpt[pFixedKFPrev];
            int id1 = mappKf2IdOpt[pFixedKFi];
//            cout<<"id0: "<<pFixedKFPrev->GetId()<<" id1: "<<pFixedKFi->GetId()<<endl;
            g2o::SE2 measure = toG2oSE2(pFixedKFi->dodo);
            g2o::Matrix3D info = toEigenMatrixXd(pFixedKFi->GetCov());
            AddEdgeSE2(optimizer, id0, id1, measure, info);
        }
    }
    for (list<PtrKeyFrameAruco>::const_iterator lit = mlLocalKeyFrames.begin(), lend = mlLocalKeyFrames.end();
         lit != lend; lit++)
    {
        PtrKeyFrameAruco pKFi = *lit;
        PtrKeyFrameAruco pKFPrev = pKFi->mpParent;
        int id0 = mappKf2IdOpt[pKFPrev];
        int id1 = mappKf2IdOpt[pKFi];
        g2o::SE2 measure = toG2oSE2(pKFi->dodo);
        g2o::Matrix3D info = toEigenMatrixXd(pKFi->GetCov());
        AddEdgeSE2(optimizer, id0, id1, measure, info);
//        cout<<pKf0->GetId()<<": "<<pKf0->GetPoseBase().x<<", "<<pKf0->GetPoseBase().y<<endl;
//        cout<<pKf1->GetId()<<": "<<pKf1->GetPoseBase().x<<", "<<pKf1->GetPoseBase().y<<endl;
//        cout<<"---------------------"<<endl;

        // DEBUG
//                cerr <<"info: "<<info << endl;
//                cerr << pMsrOdo->info << endl;
    }

    cerr<<"maker num: "<<mappMk2IdOpt.size()<<", KF num: "<<mappKf2IdOpt.size()<<endl;
    //【6】Set mark measurement edges
    // 三元边，观测是三维的
    for (list<PtrKeyFrameAruco>::const_iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end();
         lit != lend; lit++)
    {
        PtrKeyFrameAruco pFixedKFi = *lit;
        for(auto pMapMark : pFixedKFi->mvpMapMark)
        {
            int idKf = mappKf2IdOpt[pFixedKFi];
            int idMk = mappMk2IdOpt[pMapMark];

            g2o::Vector3D measure = pFixedKFi->mmapId2ArucoMsrPose[pMapMark->GetId()];
            g2o::Matrix3D info = pFixedKFi->mmapId2ArucoMsrInfo[pMapMark->GetId()];
            AddEdgeOptMk(optimizer, idKf, idMk, 0, measure, info);

            // 重投影误差
//            Eigen::Vector3d vec3MsrPt = pFixedKFi->mmapId2ArucoMsrPose[pMapMark->GetId()];
//            Eigen::Vector3d vec3KP = toEigenMatrixXd(mpDatasetAruco->GetCamMat())*vec3MsrPt;
//            g2o::Vector2D measure;
//            measure(0) = vec3KP(0)/vec3KP(2);
//            measure(1) = vec3KP(1)/vec3KP(2);
//            g2o::Matrix2D info;
//            info << 1, 0, 1, 0;
//            AddEdgeVSclam(optimizer, idKf, idMk, idVertexCamOffset, idParamCamera, measure, info);
        }
    }
    for (list<PtrKeyFrameAruco>::const_iterator lit = mlLocalKeyFrames.begin(), lend = mlLocalKeyFrames.end();
         lit != lend; lit++)
    {
        PtrKeyFrameAruco pKFi = *lit;
        for(auto pMapMark : pKFi->mvpMapMark)
        {
            int idKf = mappKf2IdOpt[pKFi];
            int idMk = mappMk2IdOpt[pMapMark];

//            Eigen::Vector3d vec3MsrPt = pKFi->mmapId2ArucoMsrPose[pMapMark->GetId()];
//            Eigen::Vector3d vec3KP = toEigenMatrixXd(mpDatasetAruco->GetCamMat())*vec3MsrPt;
//            g2o::Vector2D measure;
//            measure(0) = vec3KP(0)/vec3KP(2);
//            measure(1) = vec3KP(1)/vec3KP(2);
//            g2o::Matrix2D info;
//            info << 1, 0, 0, 1;
//            AddEdgeVSclam(optimizer, idKf, idMk, idVertexCamOffset, idParamCamera, measure, info);

            g2o::Vector3D measure = pKFi->mmapId2ArucoMsrPose[pMapMark->GetId()];
            g2o::Matrix3D info = pKFi->mmapId2ArucoMsrInfo[pMapMark->GetId()];
            AddEdgeOptMk(optimizer, idKf, idMk, 0, measure, info);

            // DEBUG
//        cerr <<"info: " <<info << endl;
//        cerr << pMsrMk->measure << endl;
//        cerr << measure << endl;
//        cerr << pMsrMk->info << endl;
        }
    }

    //【7】Do optimize
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    //【8】Refresh calibration results
    g2o::VertexSE3* v = static_cast<g2o::VertexSE3*>(optimizer.vertex(0));
    Isometry3D Iso3_bc_opt = v->estimate();
    Se3 se3bc = toSe3(Iso3_bc_opt);
    mpDatasetAruco->SetCamOffset(se3bc);

    //【9】Refresh keyframe
    for (auto pair : mappKf2IdOpt)
    {
        PtrKeyFrame pKf = pair.first;
        int idOpt = pair.second;
        VertexSE2* pVertex = static_cast<VertexSE2*>(optimizer.vertex(idOpt));

//        cout<<pKf->GetId()<<": "<<pKf->GetPoseCamera().tvec.t()<<endl;

        pKf->SetPoseAllbyB(toSe2(pVertex->estimate()), se3bc);

//        cout<<idOpt<<": "<<pKf->GetPoseCamera().tvec.t()<<endl;
//        if(!pKf->GetId())
//        {
//            cout<<"---------------------"<<endl;
//            cout<<pKf->GetId()<<": "<<pKf->GetPoseBase().x<<", "<<pKf->GetPoseBase().y<<endl;
//            cout<<"---------------------"<<endl;
//        }
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

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    cerr << "Rbc" << mpDatasetAruco->GetCamOffset() << endl << endl;*/
}

} //namespace




