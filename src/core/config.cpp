#include "config.h"
#include "type.h"

#include <iostream>


using namespace std;
using namespace cv;

namespace calibcamodo {

//! IO
int Config::NUM_FRAME;
std::string Config::STR_FOLDERPATH_MAIN;
std::string Config::STR_FOlDERPATH_IMG;
std::string Config::STR_FILEPATH_ODO;
std::string Config::STR_FILEPATH_CAM;
std::string Config::STR_FILEPATH_CALIB;
std::string Config::STR_FILEPATH_ORBVOC;

//!Camera Intrinsics
int Config::IMAGE_WIDTH;
int Config::IMAGE_HEIGHT;
cv::Mat Config::CAMERA_MATRIX;
cv::Mat Config::DISTORTION_COEFFICIENTS;

//!Camera Extrinsic
cv::Mat Config::RVEC_BC = (Mat_<float>(3,1) << 0,0,0);
cv::Mat Config::TVEC_BC = (Mat_<float>(3,1) << 0,0,0);

//! Dataset
double Config::DATASET_THRESH_KF_ODOLIN;
double Config::DATASET_THRESH_KF_ODOROT;
double Config::MARK_SIZE;

//! Solver
double Config::CALIB_ODOLIN_ERRR;
double Config::CALIB_ODOLIN_ERRMIN;
double Config::CALIB_ODOROT_ERRR;
double Config::CALIB_ODOROT_ERRRLIN;
double Config::CALIB_ODOROT_ERRMIN;

double Config::CALIB_AMKZ_ERRRZ;
double Config::CALIB_AMKZ_ERRMIN;
double Config::CALIB_AMKXY_ERRRZ;
double Config::CALIB_AMKXY_ERRMIN;

//! ROS PUBLISHER
double Config::MAPPUB_SCALE_RATIO;

void Config::InitConfig(std::string _strfolderpathmain) {

    STR_FOLDERPATH_MAIN = _strfolderpathmain;
    // frame文件路径
    STR_FOlDERPATH_IMG  = _strfolderpathmain+"/image/";
    // 里程计数据
    STR_FILEPATH_ODO    = _strfolderpathmain+"/rec/Odo.rec";
    // 相机内参，没用到?
    STR_FILEPATH_CAM    = _strfolderpathmain+"/config/CamConfig.yml";
    // 标定配置文件，未知
    STR_FILEPATH_CALIB  = _strfolderpathmain+"/config/CalibConfig.yml";

    cout<<STR_FILEPATH_CALIB<<endl;

    FileStorage file(STR_FILEPATH_CALIB, cv::FileStorage::READ);
    file["NUM_FRAME"] >> NUM_FRAME; /** @todo 根据结果确定*/

    file["IMAGE_WIDTH"] >> IMAGE_WIDTH; /** @todo 根据自己的相机确定*/
    file["IMAGE_HEIGHT"] >> IMAGE_HEIGHT;
    Mat cameramatrix;
    file["CAMERA_MATRIX"] >> cameramatrix;
    cameramatrix.convertTo(CAMERA_MATRIX, CV_32FC1);
    cout<<"CAMERA_MATRIX: "<<endl<<cameramatrix<<endl<<endl;
    Mat distortion;
    file["DISTORTION_COEFFICIENTS"] >> distortion;
    distortion.convertTo(DISTORTION_COEFFICIENTS, CV_32FC1);
    cout<<"DISTORTION_COEFFICIENTS: "<<endl<<distortion<<endl<<endl;

    Mat rvec_bc, tvec_bc;
    file["RVEC_BC"] >> rvec_bc;
    if(!rvec_bc.empty())
        rvec_bc.convertTo(RVEC_BC, CV_32FC1); /** @todo 初始值，自己手测计算*/
    // 单位: 毫米(mm)
    file["TVEC_BC"] >> tvec_bc;
    if(!tvec_bc.empty())
        tvec_bc.convertTo(TVEC_BC, CV_32FC1); /** @todo 初始值，自己手测计算*/

    // Mark的尺寸，使用calib_orb用不到
    file["MARK_SIZE"] >> MARK_SIZE;

    // 阈值
    file["DATASET_THRESH_KF_ODOLIN"] >> DATASET_THRESH_KF_ODOLIN; /** @todo 500*/
    file["DATASET_THRESH_KF_ODOROT"] >> DATASET_THRESH_KF_ODOROT; /** @todo 0.2*/

    // 里程计的标准差
    file["CALIB_ODOLIN_ERRR"] >> CALIB_ODOLIN_ERRR; /** @todo 根据实际情况填写*/
    file["CALIB_ODOLIN_ERRMIN"] >> CALIB_ODOLIN_ERRMIN;
    file["CALIB_ODOROT_ERRR"] >> CALIB_ODOROT_ERRR;
    file["CALIB_ODOROT_ERRRLIN"] >> CALIB_ODOROT_ERRRLIN; /** @todo 未知!!!!*/
    file["CALIB_ODOROT_ERRMIN"] >> CALIB_ODOROT_ERRMIN;

    // Mark的标准差，使用calib_orb用不到
    file["CALIB_AMKZ_ERRRZ"] >> CALIB_AMKZ_ERRRZ;
    file["CALIB_AMKZ_ERRMIN"] >> CALIB_AMKZ_ERRMIN;
    file["CALIB_AMKXY_ERRRZ"] >> CALIB_AMKXY_ERRRZ;
    file["CALIB_AMKXY_ERRMIN"] >> CALIB_AMKXY_ERRMIN;

    // 地图尺度比例
    file["MAPPUB_SCALE_RATIO"] >> MAPPUB_SCALE_RATIO; /** @attention 将mm转换为m在rviz中显示*/

    // ORB词典路径
    file["STR_FILEPATH_ORBVOC"] >> STR_FILEPATH_ORBVOC;
}

}
