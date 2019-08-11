#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "core/dataset.h"
#include "core/frame.h"
#include "core/measure.h"
#include "core/mapmark.h"
#include "core/adapter.h"
#include "core/type.h"
#include "core/config.h"
#include "core/maker_orb.h"
#include "core/solver_vsclam.h"

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

#include <glog/logging.h>
/****************use datasheet********************/
//
//rosrun calibcamodo calib_orb /home/leather/leather_temp/data
//运行前检查:
//         修改CalibConfig.yml文件

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

int main(int argc, char **argv)
{
    // debug
    GLogHelper gh(argv[0]);
    LOG(INFO)<<"starting.........."<<std::endl;
    LOG(WARNING)<<"starting.........."<<std::endl;


    string strFolderPathMain = argv[1];
//    int numFrame = atoi(argv[2]);
//    double markerSize = atof(argv[3]);

    //! Init ros
    ros::init(argc, argv, "pub");
    ros::start();
    ros::NodeHandle nh;
    ros::Rate rate(100);
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/image_raw",1);

    //! Init config
    Config::InitConfig(strFolderPathMain);    

    //! Init dataset
    cerr << "DatasetOrb: init ..." << endl;
    // 初始化dataset
    DatasetOrb datasetOrb;
    cerr << "DatasetOrb: creating frames ..." << endl;
    datasetOrb.CreateFrames();
    cerr << "DatasetOrb: creating keyframes ..." << endl;
    // 创建关键帧
    datasetOrb.CreateKeyFrames();
    cerr << "DatasetOrb: dataset created." << endl << endl;

    //! Init mappublisher with ros rviz
    MapPublish mappublish(&datasetOrb);

    // DEBUG ...
    // 初始设置的里程计与相机的外参
    cerr << "solverOrb: result.init = " << datasetOrb.GetCamOffset() << endl;

    MakerOrb makerOrb(&datasetOrb);
    makerOrb.DoMake();
    mappublish.run(10, 1); //发布关键帧、里程计的边与orb Marker，在rviz中查看

    cerr << "ok " <<endl;
    SolverVsclam solverVsclam(&datasetOrb);
    //solverVsclam.optimize_mappoints_keyframe();
//    solverVsclam.optimize_mappoints_keyframe_plane();
//    mappublish.run(10, 1);
//    solverVsclam.optimize_extrinsic();
//    mappublish.run(10, 1);
    solverVsclam.DoCalib();
    mappublish.run(10, 1);
    solverVsclam.DoCalib_2();
    mappublish.run(10, 1);

    cerr << "solverOrb: result.vsclam = " << datasetOrb.GetCamOffset() << endl;


    return 0;
}
