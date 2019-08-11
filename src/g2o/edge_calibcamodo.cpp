#include "edge_calibcamodo.h"
#include "g2o_math.h"

#include "core/adapter.h"

#include <Eigen/LU>

using namespace Eigen;
using namespace std;
using namespace calibcamodo;

namespace g2o {

EdgeOptMk::EdgeOptMk() :
    BaseMultiEdge<3, Vector3D>() {
    resize(3);
}

void EdgeOptMk::computeError()
{
    const VertexSE2* baseFrame          = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexPointXYZ* markPoint     = static_cast<const VertexPointXYZ*>(_vertices[1]);
    const VertexSE3* cameraOffset       = static_cast<const VertexSE3*>(_vertices[2]);

    SE2 se2_wb = baseFrame->estimate();
    Vector3D xyz_wm = markPoint->estimate();
    Isometry3D iso3_bc = cameraOffset->estimate();
    Vector3D xyz_cm_measure = _measurement;
    Isometry2D iso2_wb = se2_wb.toIsometry();

    Matrix4D T3_bc = iso3_bc.matrix();
    Matrix3D T2_wb = iso2_wb.matrix();
    Matrix4D T3_wb;
    T3_wb << T2_wb(0,0), T2_wb(0,1), 0, T2_wb(0,2),
            T2_wb(1,0), T2_wb(1,1), 0, T2_wb(1,2),
            0, 0, 1, 0,
            0, 0, 0, 1;

    Vector4D xyz1_wm;
    xyz1_wm << xyz_wm(0), xyz_wm(1), xyz_wm(2), 1;
    Vector4D xyz1_cm_measure;
    xyz1_cm_measure << xyz_cm_measure(0), xyz_cm_measure(1), xyz_cm_measure(2), 1;
    Vector4D delta = (T3_wb*T3_bc).inverse() * xyz1_wm - xyz1_cm_measure;
    _error << delta(0), delta(1), delta(2); //赋值
}

void EdgeOptMk::linearizeOplus()
{
    // use numeric Jacobians
    BaseMultiEdge<3, Vector3D>::linearizeOplus();
    return;

//    const VertexSE2* base_pose = static_cast<const VertexSE2*>(vertex(0));
//    const VertexPointXYZ* point = static_cast<const VertexPointXYZ*>(vertex(1));
//    const VertexSE3* cam_offset = static_cast<const VertexSE3*>(vertex(2));
//    typedef ceres::internal::AutoDiff<EdgeOptMk, double, VertexSE2::Dimension, VertexPointXYZ::Dimension,
//                                      VertexSE3::Dimension> BalAutoDiff;

//    Matrix<double, Dimension, VertexSE2::Dimension, Eigen::RowMajor> dError_dBase;
//    Matrix<double, Dimension, VertexPointXYZ::Dimension, Eigen::RowMajor> dError_dPoint;
//    Matrix<double, Dimension, VertexSE3::Dimension, Eigen::RowMajor> dError_dOffset;

//    double *parameters[] = { const_cast<double*>(base_pose->estimate().toVector().data()), const_cast<double*>(point->estimate().data()),
//                           const_cast<double*>(cam_offset->estimate().data())};
//    double *jacobians[] = { dError_dBase.data(), dError_dPoint.data(), dError_dOffset.data()};
//    double value[Dimension];
//    bool diffState = BalAutoDiff::Differentiate(*this, parameters, Dimension, value, jacobians);

//    // copy over the Jacobians (convert row-major -> column-major)
//    if (diffState) {
//      _jacobianOplus[0] = dError_dBase;
//      _jacobianOplus[1] = dError_dPoint;
//      _jacobianOplus[2] = dError_dOffset;
//    } else {
//      assert(0 && "Error while differentiating");
//      _jacobianOplus[0].setZero();
//      _jacobianOplus[1].setZero();
//      _jacobianOplus[2].setZero();
//    }
}

}

namespace g2o {

EdgeVSlam::EdgeVSlam() :
    BaseBinaryEdge<2, Vector2D, VertexSE2, VertexPointXYZ>() {
    resize(2);
    paramCam = 0;
    resizeParameters(1);
    installParameter(paramCam, 0);
}

void EdgeVSlam::computeError() {
    const ParameterCamera* paramCam
            = static_cast<const ParameterCamera*>(parameter(0));
    const VertexSE2* baseFrame = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexPointXYZ* markPoint = static_cast<const VertexPointXYZ*>(_vertices[1]);

    Matrix3D K = paramCam->Kcam();
    Isometry3D Iso3_bc = paramCam->offset();

    SE2 se2_wb = baseFrame->estimate();
    Isometry2D iso2_wb = se2_wb.toIsometry();
    Matrix3D T2_wb = iso2_wb.matrix();
    Matrix4D T3_wb;
    T3_wb << T2_wb(0,0), T2_wb(0,1), 0, T2_wb(0,2),
            T2_wb(1,0), T2_wb(1,1), 0, T2_wb(1,2),
            0, 0, 1, 0,
            0, 0, 0, 1;

    Matrix4D T3_bc = Iso3_bc.matrix();
    Vector3D xyz_wm = markPoint->estimate();

    Matrix4D T3_cw = T3_bc.inverse() * T3_wb.inverse();
    Matrix3D R3_cw = T3_cw.block<3,3>(0,0);
    Vector3D xyz_cw = T3_cw.block<3,1>(0,3);

    Vector3D uvbar_cm = K * (R3_cw*xyz_wm + xyz_cw);
    Vector2D uv_cm;
    uv_cm << uvbar_cm(0)/uvbar_cm(2), uvbar_cm(1)/uvbar_cm(2);
    Vector2D uv_cm_measure = _measurement;

    _error = uv_cm - uv_cm_measure;
}

void EdgeVSlam::linearizeOplus() {

    /** @todo 使用了自动求导吗????*/
    BaseBinaryEdge<2, Vector2D, VertexSE2, VertexPointXYZ>::linearizeOplus();
/*
//    cerr << "base::_jacobianOplusXi" << endl << _jacobianOplusXi << endl;
//    cerr << "base::_jacobianOplusXj" << endl << _jacobianOplusXj << endl;

//    const ParameterCamera* paramCam
//            = static_cast<const ParameterCamera*>(parameter(0));
//    const VertexSE2* baseFrame = static_cast<const VertexSE2*>(_vertices[0]);
//    const VertexPointXYZ* markPoint = static_cast<const VertexPointXYZ*>(_vertices[1]);

//    SE3Quat se3bc = toG2oSE3Quat(paramCam->offset());
//    SE3Quat se3cb = se3bc.inverse();
//    Matrix4D Tcb = toTransMat(se3cb);

//    Vector3D pwm = markPoint->estimate();
//    SE2 se2wb = baseFrame->estimate();
//    SE3Quat se3wb = toG2oSE3Quat(se2wb);
//    SE3Quat se3bw = se3wb.inverse();

//    Matrix<double,4,6,Eigen::ColMajor> J_pcm_xiwb_bar = - Tcb * dcircle(se3bw*pwm) * JJl(se3bw);
//    Matrix<double,3,6,Eigen::ColMajor> J_pcm_xiwb = J_pcm_xiwb_bar.topRows(3);
//    Matrix<double,3,3,Eigen::ColMajor> J_pcm_v3wb = J_pcm_xiwb * JacobianSE3SE2(se3wb);


////    cerr << "Tcb" << endl << Tcb << endl;
////    cerr << "dcircle(se3bw*pwm)" << endl << dcircle(se3bw*pwm) << endl;
////    cerr << "JJl(se3bw)" << endl << JJl(se3bw) << endl;
////    cerr << "J_pcm_xiwb" << endl << J_pcm_xiwb << endl;
////    cerr << "J_pcm_v3wb" << endl << J_pcm_v3wb << endl;
////    cerr << "JacobianSE3SE2(se3wb)" << endl << JacobianSE3SE2(se3wb) << endl;

//    Vector3D pcm = se3cb*se3bw*pwm;
//    Matrix3D K = paramCam->Kcam();
//    Matrix<double,2,3,Eigen::ColMajor> J_uv_pcm = JacobianUV2XYZ(pcm, K);

//    SE3Quat se3cw = se3cb*se3bw;
//    Matrix3D Rcw = toG2oIsometry3D(se3cw).rotation();

//    _jacobianOplusXi = J_uv_pcm*J_pcm_v3wb;
//    _jacobianOplusXj = J_uv_pcm*Rcw;

////    double tmp = _jacobianOplusXi(0,0);
////    if(std::isnan(tmp)) {
////        cerr << "nan!" << endl;
////    }
////    cerr << "derived::_jacobianOplusXi" << endl << _jacobianOplusXi << endl;
////    cerr << "derived::_jacobianOplusXj" << endl << _jacobianOplusXj << endl;
*/
    return;
}

}

namespace g2o {

EdgeVSclam::EdgeVSclam() :
    BaseMultiEdge<2, Vector2D>() {
    resize(3);
    _camParam = 0;
    resizeParameters(1);
    installParameter(_camParam, 0);
}

void EdgeVSclam::computeError() {
    const CameraParameters* camParam
            = static_cast<const CameraParameters*>(parameter(0));
    const VertexSE2* baseFrame = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexPointXYZ* markPoint = static_cast<const VertexPointXYZ*>(_vertices[1]);
    const VertexSE3* cameraOffset = static_cast<const VertexSE3*>(_vertices[2]);

    SE3Quat se3wb = toG2oSE3Quat(baseFrame->estimate());
    SE3Quat se3bc = toG2oSE3Quat(cameraOffset->estimate());
    Vector3D pwm = markPoint->estimate();
    SE3Quat se3cw = (se3wb*se3bc).inverse();
    _error = camParam->cam_map(se3cw.map(pwm)) - _measurement;

    return;
}

}

namespace g2o {

EdgeUV_SE3_XYZ::EdgeUV_SE3_XYZ() : BaseBinaryEdge<2, Vector2D, VertexSE3, VertexPointXYZ>()
{
    resize(2); /** @todo 顶点的数目*/
    _camParam = 0;
    resizeParameters(1);
    installParameter(_camParam, 0);
}

void EdgeUV_SE3_XYZ::computeError()
{
    const CameraParameters* camParam
            = static_cast<const CameraParameters*>(parameter(0));
    const VertexSE3* Tcw = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexPointXYZ* markPoint = static_cast<const VertexPointXYZ*>(_vertices[1]);

    Isometry3D g2o_Tcw = Tcw->estimate();
    Vector3D xyz_wm = markPoint->estimate();

    Matrix4D T3_cw = g2o_Tcw.matrix();
    Matrix3D R3_cw = T3_cw.block<3,3>(0,0);
    Vector3D xyz_cw = T3_cw.block<3,1>(0,3);
    _error = camParam->cam_map(R3_cw*xyz_wm + xyz_cw) - _measurement;
}

void EdgeUV_SE3_XYZ::linearizeOplus()
{
    BaseBinaryEdge<2, Vector2D, VertexSE3, VertexPointXYZ>::linearizeOplus();
}


EdgeOnlineCalibration::EdgeOnlineCalibration() : BaseMultiEdge<6,SE3Quat>() {
    resize(3);
}

void EdgeOnlineCalibration::computeError()  {
  const VertexSE3Expmap* prev = static_cast<const VertexSE3Expmap*>(_vertices[0]);
  const VertexSE3Expmap* cur = static_cast<const VertexSE3Expmap*>(_vertices[1]);
  const VertexSE3Expmap* calib = static_cast<const VertexSE3Expmap*>(_vertices[2]);

  SE3Quat obs(_measurement);

  //SE3Quat manifold_error = obs.inverse()*calib->estimate().inverse()*prev->estimate()*cur->estimate().inverse()*calib->estimate();

  // Tbibj^(-1) * Tbc * Tciw * Tcjw^-1 * Tbc^-1
  //! @attention 出过一次bug，obs没有求逆啊!!
  SE3Quat manifold_error = obs.inverse() * calib->estimate() *
                           prev->estimate() * cur->estimate().inverse() *
                           calib->estimate().inverse();

  _error = manifold_error.log(); //旋转在前，平移在后
}


// 给定误差求J_R^{-1}的近似
Matrix6d JRInv( SE3Quat Se3_ )
{
    Matrix6d J;
    J.block(0,0,3,3) = skew(Se3_.log().head(3));
    J.block(0,3,3,3) = Eigen::Matrix3d::Zero(3,3);
    J.block(3,0,3,3) = skew(Se3_.log().tail(3)); //!@attention 因为g2o中李代数顺序不同，所以J.block(3,0,3,3)与J.block(3,3,3,3)交换
    J.block(3,3,3,3) = J.block(0,0,3,3);
    J = J*0.5 + Matrix6d::Identity();
    return J;
}

void EdgeOnlineCalibration::linearizeOplus()
{
//    BaseMultiEdge<6, SE3Quat>::linearizeOplus();
    const VertexSE3Expmap* prev = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    const VertexSE3Expmap* cur = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSE3Expmap* calib = static_cast<const VertexSE3Expmap*>(_vertices[2]);

    SE3Quat obs(_measurement);
    SE3Quat errorSE3 = SE3Quat::exp(_error);
    Matrix6d Jri = JRInv(errorSE3);

    Matrix6d JPi, JPj, JPex;

    // Tx = Tbibj^(-1) * Tbc * Tciw * Tcjw^-1 * Tbc^-1
    // JRInv(Tx) * Ad(Tbc*Tcjw*Tciw^-1)
    JPi = Jri * (calib->estimate()*cur->estimate()*prev->estimate().inverse()).adj();

    // -JRInv(Tx) * Ad(Tbc)
    JPj = -Jri * calib->estimate().adj();

    // JRInv(Tx) * (Ad((Tbc * Tciw * Tcjw^-1 * Tbc^-1)^-1) - I)
    JPex = Jri * (((obs*errorSE3).inverse()).adj() - Matrix6d::Identity());

    _jacobianOplus[0] = JPi;
    _jacobianOplus[1] = JPj;
    _jacobianOplus[2] = JPex;
}
}

namespace g2o {
EdgePlaneConstraint::EdgePlaneConstraint() : BaseUnaryEdge<6, SE3Quat, VertexSE3Expmap>(){
    resize(1);
}

void EdgePlaneConstraint::computeError()
{
    const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    SE3Quat err = _measurement * v->estimate().inverse();
    _error = err.log(); //旋转在前，平移在后
}

void EdgePlaneConstraint::setMeasurement(const SE3Quat &m)
{
    _measurement = m;
}

void EdgePlaneConstraint::linearizeOplus()
{
    BaseUnaryEdge<6, SE3Quat, VertexSE3Expmap>::linearizeOplus();
///@todo
/*    const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector6d err = (_measurement * v->estimate().inverse()).log();
    _jacobianOplusXi = -invJJl(-err);
    JJlInv(); //平移在前，旋转在后
    */
}

}

