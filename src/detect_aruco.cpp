#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "core/dataset.h"
#include "core/frame.h"
#include "core/measure.h"
#include "core/mapmark.h"
#include "core/maker_aruco.h"
#include "core/solver_initmk.h"
#include "core/solver_optmk.h"
#include "core/adapter.h"
#include "core/type.h"
#include "core/config.h"

#include "ros/mappublish.h"

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <math.h>
//#include "slam_car/stm_to_pc.h"
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <glog/logging.h>

using namespace std;
using namespace cv;
using namespace aruco;
using namespace calibcamodo;

// 段错误回调函数
// 将信息输出到单独的文件和 LOG(ERROR)
void SignalHandle(const char* data, int size)
{
    std::ofstream fs("/home/leather/leather_temp/OpenSourceORBVIO/glog_dump.log",std::ios::app);
    std::string str = std::string(data,size);
    fs<<str;
    fs.close();
    LOG(ERROR)<<str;
}

class GLogHelper
{
public:
    GLogHelper(char* program)
    {
        google::InitGoogleLogging(program);

        FLAGS_minloglevel=google::INFO; //限制输出到 stderr 的部分信息，包括此错误级别和更高错误级别的日志信息
        FLAGS_alsologtostderr = true; //是否同时将日志输出到文件和stderr
        FLAGS_colorlogtostderr=true; //设置输出到屏幕的日志显示相应颜色
        // 除了将日志输出到文件之外，还将此错误级别和更高错误级别的日志同时输出到 stderr
        //FLAGS_stderrthreshold=google::INFO;

        FLAGS_stop_logging_if_full_disk = true; //当磁盘被写满时，停止日志输出

        FLAGS_log_dir="/home/leather/leather_temp/OpenSourceORBVIO/"; //设置日志文件输出目录
        //google::SetLogDestination(google::ERROR,"log/prefix_");   //第一个参数为日志级别，第二个参数表示输出目录及日志文件名前缀。

        google::InstallFailureSignalHandler();
        //默认捕捉 SIGSEGV 信号信息输出会输出到 stderr，可以通过下面的方法自定义输出方式：
        google::InstallFailureWriter(&SignalHandle);
    }
    ~GLogHelper()
    {
        google::ShutdownGoogleLogging();
    }
};

bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
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



template<class T>
std::string toString(const T &value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

class Detect_Aruco
{
public:
    Detect_Aruco();
    ~Detect_Aruco();
    void spin();

private:
    ros::NodeHandle n;

    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image,
      slam_car::stm_to_pc> ImagesSyncPolicy;

    message_filters::Subscriber<sensor_msgs::Image>  *image_for_sync_sub_;
    message_filters::Subscriber<slam_car::stm_to_pc> *odom_for_sync_sub_;
    message_filters::Synchronizer<ImagesSyncPolicy> *sync_;

    const static int rate = 100; /** @attention */

    double pose_x;
    double pose_y;
    double pose_th;
    double velocity_th;

    aruco::MarkerDetector mMDetector;
    int ThePyrDownLevel = 0;
    int ThresParam1 = 19;
    int ThresParam2 = 15;

    aruco::CameraParameters mCamParam;
    double mThreshOdoLin;
    double mThreshOdoRot;

    PtrKeyFrameAruco pCurrentKeyFrame, pLastKeyFrame;

    void sync_callback(const sensor_msgs::ImageConstPtr& image_msg, const slam_car::stm_to_pcConstPtr& odom_msg);
};

Detect_Aruco::Detect_Aruco()
{
    // 初始化变量
    pose_x = pose_y = pose_th = 0.0;
    velocity_th = 0.0;

    mMDetector.pyrDown(ThePyrDownLevel);
    mMDetector.setCornerRefinementMethod(MarkerDetector::LINES);
    mMDetector.setThresholdParams(ThresParam1, ThresParam2);

    mCamParam.CameraMatrix = Config::CAMERA_MATRIX.clone();
    mCamParam.Distorsion = Config::DISTORTION_COEFFICIENTS.clone();
    mCamParam.CamSize.width = Config::IMAGE_WIDTH;
    mCamParam.CamSize.height = Config::IMAGE_HEIGHT;

    mThreshOdoLin = Config::DATASET_THRESH_KF_ODOLIN;
    mThreshOdoRot = Config::DATASET_THRESH_KF_ODOROT;

    // 时间同步
    image_for_sync_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(n, "/usb_cam/image_raw", 1);
    odom_for_sync_sub_ = new message_filters::Subscriber<slam_car::stm_to_pc>(n, "/odomtry_from_stm", 1);
    sync_ = new message_filters::Synchronizer<ImagesSyncPolicy>(ImagesSyncPolicy(10),*image_for_sync_sub_, *odom_for_sync_sub_);
    sync_->registerCallback(boost::bind(&Detect_Aruco::sync_callback, this, _1, _2));
}

Detect_Aruco::~Detect_Aruco()
{
    // 删除指针
    delete image_for_sync_sub_;
    delete odom_for_sync_sub_;
    delete sync_;
}

void Detect_Aruco::sync_callback(const sensor_msgs::ImageConstPtr& image_msg, const slam_car::stm_to_pcConstPtr &odom_msg)
{
    static int frame_counter=0;
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);//RGB8

    Mat image = cv_ptr->image;


    // 为了实时现实检查结果，如不需要可屏蔽
    Mat mImgAruco;
    std::vector<aruco::Marker> mvecAruco;
    mMDetector.detect(image, mvecAruco, Config::CAMERA_MATRIX, Config::DISTORTION_COEFFICIENTS, Config::MARK_SIZE);
    image.copyTo(mImgAruco);
//    // 绘制出包含aruco标记的框
    for (auto mk : mvecAruco) {
        mk.draw(mImgAruco, Scalar(0,0,255), 2);
        Mat Rotation_Matrinx;
        cv::Rodrigues(mk.Rvec, Rotation_Matrinx);
//        cout<<"Rvec: "<<rotationMatrixToEulerAngles(Rotation_Matrinx)*180/M_PI<<endl<<"                                     "
//                                                                                      "Tvec: "<<mk.Tvec.t()<<endl;
    }
    cv::imshow("mImgAruco", mImgAruco);
    waitKey(1);


    Se2 odom_from_stm(odom_msg->coord_x_to_pc*1000.0, odom_msg->coord_y_to_pc*1000.0, odom_msg->z_angle_to_pc);
    frame_counter++;
    PtrFrame pCurrentFrame = make_shared<Frame>(odom_from_stm, frame_counter);
    if(!pLastKeyFrame)
    {
        pCurrentFrame->SetImg(image);
        PtrKeyFrameAruco pKFrame = make_shared<KeyFrameAruco>(*pCurrentFrame); /// @todo 初始化第一帧为第一个上一关键帧
        pKFrame->ComputeAruco(mCamParam, mMDetector, Config::MARK_SIZE);
        const std::vector<aruco::Marker>& vecAruco = pKFrame->GetMsrAruco();
        if(vecAruco.size())
        {
            pLastKeyFrame = pKFrame;
            cout<<"init completed!!"<<endl;
        }
        return;
    }
    Se2 dodo = pCurrentFrame->GetOdo() - pLastKeyFrame->GetOdo();
    double dl = dodo.dist();
    double dr = abs(dodo.theta);

    // 位移大于阈值或者角度变换大于阈值，则为一个关键帧
    if (dl > mThreshOdoLin || dr > mThreshOdoRot)
    {
        pCurrentFrame->SetImg(image);
        //pCurrentKeyFrame = make_shared<KeyFrameAruco>(*pCurrentFrame);
        //! reset函数将旧对象的引用计数减1（当然，如果发现引用计数为0时，则析构旧对象），然后将新对象的指针交给智能指针保管
        pCurrentKeyFrame.reset(new KeyFrameAruco(*pCurrentFrame));
        pCurrentKeyFrame->ComputeAruco(mCamParam, mMDetector, Config::MARK_SIZE);

        const std::vector<aruco::Marker>& vecArucoCurrentKF = pCurrentKeyFrame->GetMsrAruco();

        // 寻找前后两帧共视的aruco marker，并计算相邻两关键帧之间的位姿变换
        for (auto measure_aruco_currentKF : vecArucoCurrentKF)
        {
            int id_marker_currentKF = measure_aruco_currentKF.id;
            const std::vector<aruco::Marker>& vecArucoLastKF = pLastKeyFrame->GetMsrAruco();
            for (auto measure_aruco_lastKF : vecArucoLastKF)
            {
                if(id_marker_currentKF == measure_aruco_lastKF.id)
                {
                    Se3 Tci_m(measure_aruco_lastKF.Rvec, measure_aruco_lastKF.Tvec);


                    Mat tcm = measure_aruco_currentKF.Tvec;
                    Mat Rvec = measure_aruco_currentKF.Rvec;
                    Mat Tcj_m = Mat::eye(4,4,Rvec.type());
                    Mat Rcm;
                    Rodrigues(Rvec, Rcm);
                    Rcm.copyTo(Tcj_m.rowRange(0,3).colRange(0,3));
                    tcm.copyTo(Tcj_m.rowRange(0,3).col(3));
                    Mat Rmc = Rcm.t();
                    Mat Om = -Rmc*tcm;

                    Rodrigues(Rmc, Rvec);
                    Se3 Tm_cj(Rvec, Om);

                    Se3 Tci_cj = Tci_m+Tm_cj;
                    Mat Rci_cj = Tci_cj.R();
                    cout<<"---------------------------"<<endl;
                    cout<<"Tci_cj: "<<endl<<
                          "euler: "<<rotationMatrixToEulerAngles(Rci_cj)*180/M_PI<<endl<<
                          "tvec:  "<<Tci_cj.tvec.t()<<endl;
                    cout<<"dodo: "<<dodo<<endl;

                    Mat Rm_ci = Tci_m.R().t();
                    cout<<"Tm_ci: "<<endl<<
                          "euler: "<<rotationMatrixToEulerAngles(Rm_ci)*180/M_PI<<endl<<
                          "tvec:  "<<(-Rm_ci*Tci_m.tvec).t()<<endl;

                    Mat Rm_cj = Tm_cj.R();
                    cout<<"Tm_cj: "<<endl<<
                          "euler: "<<rotationMatrixToEulerAngles(Rm_cj)*180/M_PI<<endl<<
                          "tvec:  "<<Tm_cj.tvec.t()<<endl;
                }
            }
        }

        pLastKeyFrame = pCurrentKeyFrame;  /// @todo 引用次数+1
    }
}

void Detect_Aruco::spin()
{
    ros::Rate loop_rate(rate);

    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}

int main(int argc,char **argv)
{
    GLogHelper gh(argv[0]);
    LOG(INFO)<<"starting.........."<<std::endl;
    LOG(WARNING)<<"starting.........."<<std::endl;

    if(argc < 2){
        cout<<"lack init config file"<<endl;
        return 0;
    }

    string strFolderPathMain = argv[1];

    // Init config
    Config::InitConfig(strFolderPathMain);

    ros::init(argc, argv, "detect_aruco");
    Detect_Aruco obj;
    obj.spin();

//    DatasetAruco datasetAruco;
//    cerr << "Dataset: creating frames ..." << endl;
//    datasetAruco.CreateFrames(); //与calib_orb相同
//    cerr << "Dataset: creating keyframes ..." << endl;
//    datasetAruco.CreateKeyFrames();

    return 0;
}

















