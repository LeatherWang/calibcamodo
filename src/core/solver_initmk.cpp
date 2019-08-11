#include "solver_initmk.h"
#include "adapter.h"
#include <cmath>

using namespace cv;
using namespace std;

namespace calibcamodo {

SolverInitMk::SolverInitMk(Dataset* _pDataset):
    SolverBase(_pDataset) {}

bool SolverInitMk::isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

cv::Vec3f SolverInitMk::rotationMatrixToEulerAngles(cv::Mat &R)
{
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<float>(0,0) * R.at<float>(0,0) +  R.at<float>(1,0) * R.at<float>(1,0));
    bool singular = sy < 1e-6; // true: `Y`方向旋转为`+/-90`度
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<float>(2,1) , R.at<float>(2,2));
        y = atan2(-R.at<float>(2,0), sy);
        z = atan2(R.at<float>(1,0), R.at<float>(0,0));
    }
    else
    {
        x = atan2(-R.at<float>(1,2), R.at<float>(1,1));
        y = atan2(-R.at<float>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}

void SolverInitMk::DoCalib() {

    // calibrate the ground plane, return 3-by-1 norm vector in camera frame
    // 标定地平面，得到相机坐标系下的归一化的法向量
    Mat nvec_cg;
    ComputeGrndPlane(nvec_cg);

    // compute camera projection frame, with 2 solutions on 2 direction of ground
    // 构造辅助参考系<camera projection frame>
    // 计算<camera projection frame>与<camera frame>之间的位姿变换
    Mat rvec_dc_1, tvec_dc_1, rvec_dc_2, tvec_dc_2;
    ComputeCamProjFrame( nvec_cg, rvec_dc_1, tvec_dc_1); // tvec_dc_1是零向量??这是因为<camera projection frame>与<camera frame>原点重合
    ComputeCamProjFrame(-nvec_cg, rvec_dc_2, tvec_dc_2); /** @attention 反方向计算一次*/

    cerr << "【1】nvec_cg" << nvec_cg.t() << endl;
//    cerr << "rvec_dc_1" << rvec_dc_1 << endl;
//    cerr << "rvec_dc_2" << rvec_dc_2 << endl;

    // compute xyyaw between based frame and camera projection frame,
    // choose the solution with smaller residual
    // 计算<based frame>与<camera projection frame>之间的位姿变换
    Mat rvec_bd_1, tvec_bd_1, rvec_bd_2, tvec_bd_2;
    double norm_res_1 = Compute2DExtrinsic(rvec_dc_1, tvec_dc_1, rvec_bd_1, tvec_bd_1);
    double norm_res_2 = Compute2DExtrinsic(rvec_dc_2, tvec_dc_2, rvec_bd_2, tvec_bd_2);
    Mat T_dc, T_bd, T_bc;
    /** @attention 取残差最小*/
    if (norm_res_1 < norm_res_2) {
        Vec2MatSe3(rvec_dc_1, tvec_dc_1, T_dc);
        Vec2MatSe3(rvec_bd_1, tvec_bd_1, T_bd);

        Mat rotation_matrinx;
        cv::Rodrigues(rvec_dc_1, rotation_matrinx);
        cout<<"【2】rvec_dc_1 to euler:"<<rotationMatrixToEulerAngles(rotation_matrinx)*180/M_PI<<endl;
        cout<<"【3.1】tvec_bd_1:"<<tvec_bd_1.t()<<endl;
        cv::Rodrigues(rvec_bd_1, rotation_matrinx);
        cout<<"【3.2】rvec_bd_1 to euler:"<<rotationMatrixToEulerAngles(rotation_matrinx)*180/M_PI<<endl;
    }
    else {
        Vec2MatSe3(rvec_dc_2, tvec_dc_2, T_dc);
        Vec2MatSe3(rvec_bd_2, tvec_bd_2, T_bd);

        Mat rotation_matrinx;
        cv::Rodrigues(rvec_dc_2, rotation_matrinx);
        cout<<"【2】rvec_dc_2 to euler:"<<rotationMatrixToEulerAngles(rotation_matrinx)*180/M_PI<<endl;
        cout<<"【3.1】tvec_bd_2:"<<tvec_bd_2.t()<<endl;
        cv::Rodrigues(rvec_bd_2, rotation_matrinx);
        cout<<"【3.2】rvec_bd_2 to euler:"<<rotationMatrixToEulerAngles(rotation_matrinx)*180/M_PI<<endl;
    }
    T_bc = T_bd*T_dc;
    mpDataset->SetCamOffset(Se3(T_bc));

    Mat R_bc = T_bc.rowRange(0,3).colRange(0,3).clone();
    cout<<"【4】R_bc to euler:"<<rotationMatrixToEulerAngles(R_bc)*180/M_PI<<endl;
}

// 参考文献: Automatic Simultaneous Extrinsic-Odometric Calibration for Camera-Odometry System
void SolverInitMk::ComputeGrndPlane(Mat &nvec_cg)
{
    // 获取所有的<关键帧>与该关键帧观测到的所有<Marker>对应关系
    const set<PtrMsrPt3Kf2Mk> & setMsrMk = mpDataset->GetMsrMkAll();

    int numLclIdMk = 0;
    int numLclIdKf = 0;
    map<PtrMapMark, int> mapMk2LclId;
    map<PtrKeyFrame, int> mapKf2LclId;

    for (auto ptrmeasure : setMsrMk) {
        PtrKeyFrame pKf = ptrmeasure->pKf;
        PtrMapMark pMk = ptrmeasure->pMk;
        if(!mapMk2LclId.count(pMk)) //寻找共有多少个Marker
            mapMk2LclId[pMk] = numLclIdMk++;
        if(!mapKf2LclId.count(pKf)) //寻找共有多少个关键帧
            mapKf2LclId[pKf] = numLclIdKf++;
    }

    const int dimrow = numLclIdKf;
    const int dimcol = 3+numLclIdMk;
    /** 公式14*/
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimrow, dimcol);

    for (auto ptr : setMsrMk)
    {
        PtrMsrPt3Kf2Mk pMsrMk = ptr;
        PtrKeyFrame pKf = pMsrMk->pKf;
        PtrMapMark pMk = pMsrMk->pMk;

        int lclIdMk = mapMk2LclId[pMk];
        int lclIdKf = mapKf2LclId[pKf];

        // 以关键帧编号为行号，此行是唯一的
        Mat tvec = pMsrMk->pt3.tvec(); //marker在相机坐标系下的位置
        A(lclIdKf,0) = tvec.at<float>(0);
        A(lclIdKf,1) = tvec.at<float>(1);
        A(lclIdKf,2) = tvec.at<float>(2);
        A(lclIdKf,3+lclIdMk) = 1; //以maker的编号为列号
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::VectorXd singular = svd.singularValues(); //奇异值
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd norm = singular;
    Eigen::VectorXd singularnorm = singular;
    Eigen::VectorXd vbest; //最小奇异值对应的奇异向量
    double singularnormbest = INFINITY;

    for (int i = 0; i < V.rows(); i++) //V是方阵，所以V.rows()==V.cols()
    {
        Eigen::Vector3d vecn = V.block(0,i,3,1); //相当于matrix.block<p,q>(i,j);，获取(0,i)位置3x1向量
                                                 /// @todo 为什么只取前三行的向量
        norm(i) = sqrt(vecn(0)*vecn(0)+vecn(1)*vecn(1)+vecn(2)*vecn(2));
        singularnorm(i) = singular(i)/norm(i); //奇异值归一化: (奇异值)/(奇异值对应的奇异向量的模)

        if (singularnorm(i) < singularnormbest)
        {
            singularnormbest = singularnorm(i);
            vbest = V.block(0,i,3,1)/norm(i);
        }
    }

    Mat nvec = Mat::zeros(3,1,CV_32FC1);
    nvec.at<float>(0) = vbest(0);
    nvec.at<float>(1) = vbest(1);
    nvec.at<float>(2) = vbest(2);

    //    cerr << "A:" << endl << A << endl;
    //    cerr << "singular:" << endl << singular << endl;
    //    cerr << "V:" << endl << V << endl;
    //    cerr << "norm:" << endl << norm << endl;
    //    cerr << "singularnorm:" << endl << singularnorm << endl;
    //    cerr << "vbest:" << endl << vbest << endl;

    nvec.copyTo(nvec_cg);
}

void SolverInitMk::ComputeCamProjFrame(const Mat &nvec_cg, Mat &rvec_dc, Mat &tvec_dc)
{
    /** 论文: 公式15*/
    // define an approximate norm vector "nvecApprox" with a large angle with ground norm
    // 选择一个与地面法向量夹角比较大的单位向量，以便使用二者的叉积计算rx
    Mat nvecApprox;
    Mat rz = nvec_cg; //【1】得到rz
    float rz0 = rz.at<float>(0);
    float rz1 = rz.at<float>(1);
    float rz2 = rz.at<float>(2);
    // 根据nvec_cg最小分量所处的位置，选一个夹角最大的向量
    if (abs(rz0) < abs(rz1) && abs(rz0) < abs(rz2)) {
        nvecApprox = (Mat_<float>(3,1) << 1, 0, 0);
    }
    else if (abs(rz1) < abs(rz2)) {
        nvecApprox = (Mat_<float>(3,1) << 0, 1, 0);
    }
    else {
        nvecApprox = (Mat_<float>(3,1) << 0, 0, 1);
    }

    // create the roation matrix
    // 计算旋转矩阵
    Mat rx = rz.cross(nvecApprox); //【2】计算叉积得到rx
    rx = rx/norm(rx); //归一化

    /** @attention 真机智!!!!*/
    Mat ry = rz.cross(rx); //【3】rz与rx的叉积是ry

    Mat Rcd = Mat::zeros(3,3,CV_32FC1);
    rx.copyTo(Rcd.colRange(0,1));
    ry.copyTo(Rcd.colRange(1,2));
    rz.copyTo(Rcd.colRange(2,3)); //d的z轴与地面的法向量平行
    Rodrigues(Rcd.t(), rvec_dc); //转换成李代数

    tvec_dc = Mat::zeros(3,1,CV_32FC1); //<camera projection frame>与<camera frame>原点重合，故设置为0

    //    cerr << "nvec_cg" << endl << nvec_cg << endl;
    //    cerr << "Rdc" << endl << Rcd.t() << endl;
    //    cerr << "rvec_dc" << rvec_dc << endl;
    //    cerr << "tvec_dc" << tvec_dc << endl;
}


double SolverInitMk::Compute2DExtrinsic(const Mat &rvec_dc, const Mat &tvec_dc, Mat &rvec_bd, Mat &tvec_bd)
{
    const set<PtrMsrSe2Kf2Kf>& setMsrOdo = mpDataset->GetMsrOdoSet();
    const set<PtrMsrPt3Kf2Mk>& setMsrMk = mpDataset->GetMsrMkAll();

    double threshSmallRotation = 1.0/5000; /** @todo */

    vector<HyperEdgeOdoMk> vecHyperEdge;
    vector<HyperEdgeOdoMk> vecHyperEdgeSmallRot;
    vector<HyperEdgeOdoMk> vecHyperEdgeLargeRot;

    for(auto ptrmsrodo : setMsrOdo) {
        PtrMsrSe2Kf2Kf pMsrOdo = ptrmsrodo;
        double odo_ratio = pMsrOdo->se2.ratio();

        PtrKeyFrame pKf1 = pMsrOdo->pKfHead;
        PtrKeyFrame pKf2 = pMsrOdo->pKfTail;

        set<pair<PtrMsrPt3Kf2Mk, PtrMsrPt3Kf2Mk>> setpairMsrMk;
        // 寻找相邻关键帧之间的共视的Markers
        FindCovisMark(pKf1, pKf2, setpairMsrMk);

        for (auto pairMsrMk : setpairMsrMk) {
            HyperEdgeOdoMk edge(pMsrOdo, pairMsrMk.first, pairMsrMk.second);
            vecHyperEdge.push_back(edge);
/** 论文: 公式20 ，注意这里有与论文不一致的地方*/
            if (abs(odo_ratio) < threshSmallRotation) {
                vecHyperEdgeSmallRot.push_back(edge);
            }
            else {
                vecHyperEdgeLargeRot.push_back(edge);
            }
        }
    }

    //    cerr << "Number of hyper edges: " << vecHyperEdge.size() << endl;
    //    cerr << "Number of hyper edges with small rotation: " << vecHyperEdgeSmallRot.size() << endl;
    //    cerr << "Number of hyper edges with large rotation: " << vecHyperEdgeLargeRot.size() << endl;

    // COMPUTE YAW ANGLE
    Mat R_dc;
    Rodrigues(rvec_dc, R_dc);

    double yawsum = 0;
    int yawcount = 0;
    for(auto edge : vecHyperEdgeSmallRot) {
        HyperEdgeOdoMk hyperEdge = edge;
        PtrMsrSe2Kf2Kf pMsrOdo = hyperEdge.pMsrOdo;
        PtrMsrPt3Kf2Mk pMsrMk1 = hyperEdge.pMsrMk1;
        PtrMsrPt3Kf2Mk pMsrMk2 = hyperEdge.pMsrMk2;

        Se3 se3_b1b2 = Se3(pMsrOdo->se2); /// 直接使用Se2初始化Se3

        Mat R_b1b2 = se3_b1b2.R();
        Mat tvec_b1b2 = se3_b1b2.tvec; //里程计测量值

        Mat tvec_c1m = pMsrMk1->pt3.tvec();
        Mat tvec_c2m = pMsrMk2->pt3.tvec();
        Mat tvec_b1b2_bar = R_dc*tvec_c1m - R_b1b2*R_dc*tvec_c2m; //公式19，使用landmarks得到的测量值

        double xb = tvec_b1b2.at<float>(0);
        double yb = tvec_b1b2.at<float>(1);
        double xbbar = tvec_b1b2_bar.at<float>(0);
        double ybbar = tvec_b1b2_bar.at<float>(1);
        double yaw = atan2(yb,xb) - atan2(ybbar,xbbar); //公式21
        yaw = Period(yaw, PI, -PI);

        yawsum += yaw;
        yawcount++;

        // DEBUG:
        //        cerr << "R_dc" << endl << R_dc << endl;
        //        cerr << "tvec_c1m" << endl << tvec_c1m.t() << endl;
        //        cerr << "tvec_c2m" << endl << tvec_c2m.t() << endl;
        //        cerr << "tvec_b1b2" << endl << tvec_b1b2.t() << endl;
        //        cerr << "tvec_b1b2_bar" << endl << tvec_b1b2_bar.t() << endl;
        //        cerr << xb << " " << yb << " " << xbbar << " " << ybbar << " " << yaw << endl;
        //        cerr << endl;
    }
    double yawavr = yawsum/yawcount;
    //    cerr << "Yaw: " << yawavr << endl;
    rvec_bd = ( Mat_<float>(3,1) << 0, 0, yawavr); //求均值

    // COMPUTE XY TRANSLATION
    Mat R_bd;
    Rodrigues(rvec_bd, R_bd);
    Mat R_bc = R_bd * R_dc; ///下面使用生成的R_bc

    const int numHyperEdge = vecHyperEdgeLargeRot.size(); //满足论文公式25下面的那个条件
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(numHyperEdge*2, 2);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(numHyperEdge*2, 1);

    int countEdge = 0;
    for(auto edge : vecHyperEdgeLargeRot) {

        HyperEdgeOdoMk hyperEdge = edge;
        PtrMsrSe2Kf2Kf pMsrOdo = hyperEdge.pMsrOdo;
        PtrMsrPt3Kf2Mk pMsrMk1 = hyperEdge.pMsrMk1;
        PtrMsrPt3Kf2Mk pMsrMk2 = hyperEdge.pMsrMk2;

        Se3 se3_b1b2 = pMsrOdo->se2;
        Mat R_b1b2 = se3_b1b2.R();
        Mat tvec_b1b2 = se3_b1b2.tvec;
        Mat tvec_c1m = pMsrMk1->pt3.tvec();
        Mat tvec_c2m = pMsrMk2->pt3.tvec();

        Mat A_blk = Mat::eye(3,3,CV_32FC1) - R_b1b2;
        Mat b_blk = R_b1b2*R_bc*tvec_c2m - R_bc*tvec_c1m + tvec_b1b2;

        Mat A_blk_trim = A_blk.rowRange(0,2).colRange(0,2);
        Mat b_blk_trim = b_blk.rowRange(0,2);

        Eigen::MatrixXd A_blk_trim_eigen = toEigenMatrixXd(A_blk_trim);
        Eigen::MatrixXd b_blk_trim_eigen = toEigenMatrixXd(b_blk_trim);

        A.block(countEdge*2,0,2,2) = A_blk_trim_eigen;
        b.block(countEdge*2,0,2,1) = b_blk_trim_eigen;

        countEdge++;
    }

    // 使用SVD求解线性最小二乘超定方程
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd x = svd.solve(b);
    Eigen::VectorXd residual = A*x - b; /** @attention 残差*/
    tvec_bd = ( Mat_<float>(3,1) << x(0), x(1), 0 );

    // DEBUG
    //    Mat rvec_bc;
    //    Rodrigues(R_bc, rvec_bc);
    //    cerr << "rvec_bc:" << endl << rvec_bc << endl;
    //    cerr << "x:" << endl << x << endl;
    //    cerr << "A:" << endl << A << endl;
    //    cerr << "b:" << endl << b << endl;
    //    cerr << "residual:" << endl << residual << endl;

    return residual.norm();
}

// 寻找相邻关键帧之间的共视marker
int SolverInitMk::FindCovisMark(const PtrKeyFrame _pKf1, const PtrKeyFrame _pKf2, set<pair<PtrMsrPt3Kf2Mk, PtrMsrPt3Kf2Mk>> &_setpairMsrMk) {
    // Find convisible mark from two keyframe, consider the ordered set
    // set类型已经按照marker的ID号排序好了
    _setpairMsrMk.clear();
    set<PtrMapMark> setpMk1 = mpDataset->GetMkByKf(_pKf1);
    set<PtrMapMark> setpMk2 = mpDataset->GetMkByKf(_pKf2);
    set<PtrMapMark> setpMkCovis;
    for (auto iterMk1 = setpMk1.begin(), iterMk2 = setpMk2.begin();
         iterMk1 != setpMk1.end() && iterMk2 != setpMk2.end(); ) {
        if (*iterMk1 == *iterMk2) {
            PtrMapMark pMkCovis = *iterMk1;
            setpMkCovis.insert(pMkCovis);
            iterMk1++;
            iterMk2++;
        }
        else if (*iterMk1 < *iterMk2) {
            iterMk1++;
        }
        else {
            iterMk2++;
        }
    }

    for(auto pMkCovis : setpMkCovis) {
        PtrMsrPt3Kf2Mk pMsrMk1 = mpDataset->GetMsrMkByKfMk(_pKf1, pMkCovis);
        PtrMsrPt3Kf2Mk pMsrMk2 = mpDataset->GetMsrMkByKfMk(_pKf2, pMkCovis);
        assert(pMsrMk1 && pMsrMk2);
        _setpairMsrMk.insert(make_pair(pMsrMk1,pMsrMk2));
    }

    return 0;
}

}













