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
#include "slam_car/stm_to_pc.h"
#include <math.h>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <glog/logging.h>
#include "slide_opt/slide_optimation.h"
#include "slide_opt/color.h"

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

class Calib_Aruco_Online
{
public:
    Calib_Aruco_Online();
    ~Calib_Aruco_Online();
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

    DatasetAruco datasetAruco;
    PtrKeyFrameAruco pCurrentKeyFrame, pLastKeyFrame, pLastInsertedKF;
    SlideOptimation *mSlideOptiamtion;
    MapPublish *mappublish;

    void sync_callback(const sensor_msgs::ImageConstPtr& image_msg, const slam_car::stm_to_pcConstPtr& odom_msg);
};

Calib_Aruco_Online::Calib_Aruco_Online()
{
    // 初始化变量
    pose_x = pose_y = pose_th = 0.0;
    velocity_th = 0.0;

    // 时间同步
    image_for_sync_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(n, "/usb_cam/image_raw", 1);
    odom_for_sync_sub_ = new message_filters::Subscriber<slam_car::stm_to_pc>(n, "/odomtry_from_stm", 1);
    sync_ = new message_filters::Synchronizer<ImagesSyncPolicy>(ImagesSyncPolicy(10),*image_for_sync_sub_, *odom_for_sync_sub_);
    sync_->registerCallback(boost::bind(&Calib_Aruco_Online::sync_callback, this, _1, _2));

    // 开启优化线程
    mSlideOptiamtion = new SlideOptimation(&datasetAruco, 8); //!@todo
    mSlideOptiamtion->start_optimation_thread();
    mappublish = new MapPublish(&datasetAruco);
}

Calib_Aruco_Online::~Calib_Aruco_Online()
{
    // 关闭线程
    mSlideOptiamtion->is_thread_exit = true;

    // 删除指针
    delete image_for_sync_sub_;
    delete odom_for_sync_sub_;
    delete sync_;
}

void Calib_Aruco_Online::sync_callback(const sensor_msgs::ImageConstPtr& image_msg, const slam_car::stm_to_pcConstPtr &odom_msg)
{
    ros::Time before_time = ros::Time::now();
    static int frame_counter=0;
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);//RGB8

    Mat image = cv_ptr->image;

//    geometry_msgs::Quaternion odom_quat;
//    odom_quat = odom_msg->pose.pose.orientation;
//    Eigen::Quaterniond q;
//    q.x() = odom_quat.x;
//    q.y() = odom_quat.y;
//    q.z() = odom_quat.z;
//    q.w() = odom_quat.w;

//    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);

    //!@todo 单位
//    Se2 odom_from_stm(odom_msg->pose.pose.position.x *1000.0, odom_msg->pose.pose.position.y*1000.0,
//                      euler(0));
    Se2 odom_from_stm(odom_msg->coord_x_to_pc*1000.0, odom_msg->coord_y_to_pc*1000.0, odom_msg->z_angle_to_pc);

    frame_counter++;
    /*【1】设置当前普通帧的base_pose*/
    PtrFrame pCurrentFrame = make_shared<Frame>(odom_from_stm, frame_counter);

    /*【2】初始化*/
    if(!pLastKeyFrame)
    {
        pCurrentFrame->SetImg(image);
        PtrKeyFrameAruco pKFrame = make_shared<KeyFrameAruco>(*pCurrentFrame); /// @todo 初始化第一帧为第一个上一关键帧
        datasetAruco.IdentifyArucoMarker(pKFrame);
        const std::vector<aruco::Marker>& vecAruco = pKFrame->GetMsrAruco();

        // 检测到aruco时才算初始化成功
        if(vecAruco.empty())
            return;

    /*【4】设置关键帧位姿*/
        pLastKeyFrame = pKFrame;
        datasetAruco.AddKfAruco(pLastKeyFrame); //加入到数据库中
        Se2 se2wb = pLastKeyFrame->GetOdo();
        Se3 se3wc; //旋转为0,平移为0
        pLastKeyFrame->SetPoseCamera(se3wc); //!@attention 世界坐标系为第一帧相机坐标系
        pLastKeyFrame->SetPoseBase(se2wb); //设置里程计测量，此处的w是指里程计自己的起始坐标系，与上面的世界坐标系是不同的

    /*【6】提取marker*/
        const std::vector<aruco::Marker>& vecArucoCurrKF = pLastKeyFrame->GetMsrAruco();
        pLastKeyFrame->mvpMapMark = vector<PtrMapMarkAruco>(vecArucoCurrKF.size());
        int counter_aruco=0;
        for (auto measure_aruco : vecArucoCurrKF)
        {
            int id = measure_aruco.id;
            Mat tvec = measure_aruco.Tvec;
            Mat rvec = measure_aruco.Rvec;
            double marksize = measure_aruco.ssize;

            double z = abs(tvec.at<float>(2)); /// @attention 以marker距离相机的距离(Z方向)为方差确定的依据
            double stdxy = max(z*datasetAruco.mAmkXYErrRZ, datasetAruco.mAmkXYErrMin);
            double stdz = max(z*datasetAruco.mAmkZErrRZ, datasetAruco.mAmkZErrMin);

    /*【7】设置marker的协方差*/
            Mat info_aruco = Mat::eye(3,3,CV_32FC1);
            info_aruco.at<float>(0,0) = 1/stdxy/stdxy;
            info_aruco.at<float>(1,1) = 1/stdxy/stdxy;
            info_aruco.at<float>(2,2) = 1/stdz/stdz;

            PtrMapMarkAruco pMkAruco = make_shared<MapMarkAruco>(id, id, marksize);
            //增加aruco mark到数据库，包括几种不同类型的数据库，主要是为了使用方便

            if(datasetAruco.AddMkAruco(pMkAruco))
            {
                // 第一次观测到Marker
    /*【8】设置marker世界坐标*/
                Se3 se3wc = pLastKeyFrame->GetPoseCamera(); //初始化时，旋转为0,平移为0
                Se3 se3cm;
                se3cm.tvec = tvec;
                se3cm.rvec = rvec;
                Se3 se3wm = se3wc + se3cm;
                pMkAruco->SetPose(se3wm);
//                cerr<<"mark id: "<<pMkAruco->GetId()<<endl;
            }
            else
            {
                pMkAruco = datasetAruco.GetMkAruco(id); //使用之前存储过的
            }

            // 加入数据库中，仅仅是为了显示，无其它使用
            PtrMsrPt3Kf2Mk pMsrMk = make_shared<MeasurePt3Kf2Mk>(tvec, info_aruco, pLastKeyFrame, pMkAruco);
            datasetAruco.AddMsrMk(pMsrMk);

            pLastKeyFrame->mvpMapMark[counter_aruco] = pMkAruco; //加入观测
            counter_aruco++;
        }

        if(mSlideOptiamtion->AcceptKeyFrames()) {
            mSlideOptiamtion->InsertKeyFrame(pLastKeyFrame);
            pLastInsertedKF = pLastKeyFrame;
        }
        else {
            cerr<<FRED("[error:] ")<<"busy, insert error"<<endl;
            exit(EXIT_FAILURE);
        }

        mappublish->run(1,0);
        cerr<<"init complete! "<<"marker number: "<<vecArucoCurrKF.size()<<endl<<endl;
        return;
    }
    else
    {
    /*【3】位移大于阈值或者角度变换大于阈值，则为一个关键帧*/
        Se2 dodo = pCurrentFrame->GetOdo() - pLastKeyFrame->GetOdo();
        double dist = dodo.dist();
        double dtheta = abs(dodo.theta);
        if (dist > datasetAruco.mThreshOdoLin || dtheta > datasetAruco.mThreshOdoRot)
        {
            pCurrentFrame->SetImg(image);
            // reset函数将旧对象的引用计数减1（当然，如果发现引用计数为0时，则析构旧对象），然后将新对象的指针交给智能指针保管
            pCurrentKeyFrame.reset(new KeyFrameAruco(*pCurrentFrame));

            datasetAruco.IdentifyArucoMarker(pCurrentKeyFrame); //提取marker
            datasetAruco.AddKfAruco(pCurrentKeyFrame);
            const std::vector<aruco::Marker>& vecArucoCurrKF = pCurrentKeyFrame->GetMsrAruco();
            if(vecArucoCurrKF.empty())
            {
                cerr << FYEL("[Warning:] ") << "No Aruco detected!  " <<
                        "marker number: " << vecArucoCurrKF.size() <<endl<<endl;
                return;
            }


    /*【4】设置关键帧位姿*/
            {
                Se2 se2wb = pCurrentKeyFrame->GetOdo();
                pCurrentKeyFrame->SetPoseBase(se2wb); //设置里程计测量
                float minDis = 999999.9f;
                int numOfMatch = 0;
                for (auto measure_aruco : vecArucoCurrKF)
                {
                    int id = measure_aruco.id;
                    PtrMapMarkAruco pMkAruco = datasetAruco.GetMkAruco(id);
                    if(pMkAruco)
                    {
                        numOfMatch++;
                        Mat t_cm = measure_aruco.Tvec;
                        Mat r_cm = measure_aruco.Rvec;
                        if(fabs(t_cm.at<float>(2)) < minDis) //取最近的marker
                        {
                            minDis = fabs(t_cm.at<float>(2));
                            cv::Mat R_cm, R_mc;
                            cv::Rodrigues(r_cm, R_cm);
                            R_mc = R_cm.t();
                            cv::Mat r_mc, t_mc;
                            cv::Rodrigues(R_mc, r_mc);
                            t_mc = -R_mc*t_cm;
                            Se3 T_mc(r_mc, t_mc);

                            Se3 T_wm = pMkAruco->GetPose();//marker在世界坐标系的pose
                            Se3 T_wc = T_wm + T_mc;

                            pCurrentKeyFrame->SetPoseCamera(T_wc); //根据maker的世界坐标系下的pose,和当前相机与marker之间的位姿变换计算相机的pose
                        }
                    }
                }
                if(numOfMatch < 1)
                {
                    cerr << FYEL("[Error:] ") << "No match any Aruco! " << endl<<endl;
                    return;
                }
            }

    /*【5】设置base pose运动输入对应的协方差*/
            {
                Mat info = Mat::eye(3,3,CV_32FC1);
                double stdlin = max(dist*datasetAruco.mOdoLinErrR, datasetAruco.mOdoLinErrMin);
                double stdrot = max(max(dtheta*datasetAruco.mOdoRotErrR, datasetAruco.mOdoRotErrMin), dist*datasetAruco.mOdoRotErrRLin);
                info.at<float>(0,0) = 1/stdlin/stdlin;
                info.at<float>(1,1) = 1/stdlin/stdlin;
                info.at<float>(2,2) = 1/stdrot/stdrot;
                pCurrentKeyFrame->SetCov(info);

                // 加入数据库中，仅仅是为了显示，无其它使用
                PtrMsrSe2Kf2Kf pMeasureOdo = make_shared<MeasureSe2Kf2Kf>(dodo, info, pLastKeyFrame, pCurrentKeyFrame);
                datasetAruco.AddMsrOdo(pMeasureOdo);
            }

    /*【6】提取marker*/

            pCurrentKeyFrame->mvpMapMark = vector<PtrMapMarkAruco>(vecArucoCurrKF.size()); //分配的大小，调用了默认构造函数
            //cerr<<"size: "<<vecArucoCurrKF.size()<<endl;
            int counter_aruco=0;
            for (auto measure_aruco : vecArucoCurrKF)
            {
                int id = measure_aruco.id;
                Mat tvec = measure_aruco.Tvec;
                Mat rvec = measure_aruco.Rvec;
                double marksize = measure_aruco.ssize;

                double z = abs(tvec.at<float>(2)); /// @attention 以marker距离相机的距离(Z方向)为方差确定的依据
                double stdxy = max(z*datasetAruco.mAmkXYErrRZ, datasetAruco.mAmkXYErrMin);
                double stdz = max(z*datasetAruco.mAmkZErrRZ, datasetAruco.mAmkZErrMin);

    /*【7】设置marker的协方差*/
                Mat info_aruco = Mat::eye(3,3,CV_32FC1);
                info_aruco.at<float>(0,0) = 1/stdxy/stdxy;
                info_aruco.at<float>(1,1) = 1/stdxy/stdxy;
                info_aruco.at<float>(2,2) = 1/stdz/stdz;

                PtrMapMarkAruco pMkAruco = make_shared<MapMarkAruco>(id, id, marksize);
                //增加aruco mark到数据库，包括几种不同类型的数据库，主要是为了使用方便

                if(datasetAruco.AddMkAruco(pMkAruco))
                {
                    // 第一次观测到Marker
    /*【8】设置marker世界坐标*/
                    Se3 se3wc = pCurrentKeyFrame->GetPoseCamera();
                    Se3 se3cm;
                    se3cm.tvec = tvec; //只要平移
                    se3cm.rvec = rvec;
                    Se3 se3wm = se3wc + se3cm;
                    pMkAruco->SetPose(se3wm); //世界坐标系是相机第一帧所在的坐标系
//                    cerr<<"mark id: "<<pMkAruco->GetId()<<endl;
                }
                else
                {
                    pMkAruco = datasetAruco.GetMkAruco(id); //使用之前存储过的
                }

                // 加入数据库中，仅仅是为了显示，无其它使用
                PtrMsrPt3Kf2Mk pMsrMk = make_shared<MeasurePt3Kf2Mk>(tvec, info_aruco, pCurrentKeyFrame, pMkAruco);
                datasetAruco.AddMsrMk(pMsrMk);

                pCurrentKeyFrame->mvpMapMark[counter_aruco] = pMkAruco; /** @todo */
                pCurrentKeyFrame->mmapId2ArucoMsrInfo[id] = toEigenMatrixXd(info_aruco); /** @debug 测试*/

                {
                    Mat R = Mat::zeros(3,3,CV_32FC1);
                    Mat T = Mat::eye(4,4,CV_32FC1);
                    Rodrigues(rvec, R);
                    R.copyTo(T.colRange(0,3).rowRange(0,3));
                    tvec.copyTo(T.col(3).rowRange(0,3));
                    pCurrentKeyFrame->mmapId2ArucoMsrPose[id] = toEigenMatrixXd(T);
                }
                counter_aruco++;
            }

//            cerr<<"pose: ["<<pCurrentKeyFrame->GetPoseBase().x<<","
//                           <<pCurrentKeyFrame->GetPoseBase().y<<","
//                           <<pCurrentKeyFrame->GetPoseBase().theta<<"] "
//                           <<"number: "<<vecArucoCurrKF.size()<<endl;

    /*【9】设置前一个关键帧*/
            pCurrentKeyFrame->dodo = dodo;
            //pCurrentKeyFrame->mpParent = pLastKeyFrame;
            pLastKeyFrame = pCurrentKeyFrame; //更新

            if(mSlideOptiamtion->AcceptKeyFrames())
            {
                mSlideOptiamtion->InsertKeyFrame(pCurrentKeyFrame); /** @todo */
                pCurrentKeyFrame->mpParent = pLastInsertedKF;
                pLastInsertedKF = pCurrentKeyFrame;
            }
            else
                cerr<<FYEL("[warning:] ")<<"busy, insert error"<<endl;

            /*【10】发布，显示*/
            mappublish->run(1,0);

            //cout<<"time goes by: "<<(ros::Time::now()-before_time).toSec()<<endl;
        }
    }
}

void Calib_Aruco_Online::spin()
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
//    GLogHelper gh(argv[0]);
//    LOG(INFO)<<"starting.........."<<std::endl;
//    LOG(WARNING)<<"starting.........."<<std::endl;

//    if(argc < 2){
//        cout<<"lack init config file"<<endl;
//        return 0;
//    }

//    string strFolderPathMain = argv[1];

    // Init config
    Config::InitConfig(string("/home/leather/leather_catkin/src/calibcamodo"));

    ros::init(argc, argv, "Calib_Aruco_Online");
    Calib_Aruco_Online obj;
    obj.spin();

    return 0;
}

















