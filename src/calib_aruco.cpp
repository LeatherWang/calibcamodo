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

int main(int argc, char **argv)
{
    // debug
    GLogHelper gh(argv[0]);
    LOG(WARNING)<<"starting.........."<<std::endl;

    string strFolderPathMain = argv[1];

    // Init ros
    ros::init(argc, argv, "pub");
    ros::start();
    ros::NodeHandle nh;
    ros::Rate rate(100);
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/image_raw",1);

    // Init config
    Config::InitConfig(strFolderPathMain);

    // Init dataset
    DatasetAruco datasetAruco;
    cerr << "Dataset: creating frames ..." << endl;
    datasetAruco.CreateFrames(); //与calib_orb相同
    cerr << "Dataset: creating keyframes ..." << endl;
    datasetAruco.CreateKeyFrames();
    cerr << "Dataset: dataset created." << endl << endl;

    // Init mappublisher with ros rviz
    MapPublish mappublish(&datasetAruco);


    // Set the problem in dataset by makerAruco
    MakerAruco makerAruco(&datasetAruco);
    makerAruco.DoMake(); //将aruco marker当作MapPoints
    mappublish.run(10, 0);

    // Calibrate by SolverInitMk
    SolverInitMk solverInitMk(&datasetAruco);
    solverInitMk.DoCalib();
    cerr << "solverInitMk: result = " << datasetAruco.GetCamOffset() << endl << endl;
    makerAruco.InitKfMkPose(); //使用估计的外参更新关键帧和marker的位姿
    mappublish.run(10, 0);

    //datasetAruco.SetCamOffset(datasetAruco.GetCamOffset());

    // Calibrate by SolverOptMk
    SolverOptMk solverOptMk(&datasetAruco);
    ros::Time before_opt=ros::Time::now();
    solverOptMk.DoCalib();
    cerr <<"time goes by: "<<(ros::Time::now()-before_opt).toSec()<<endl;
    cerr << "solverInitMk: result = " << datasetAruco.GetCamOffset() << endl << endl;
    mappublish.run(10, 0);

    Mat Rbc = datasetAruco.GetCamOffset().R();
    cout<<"R_bc to euler:"<<solverInitMk.rotationMatrixToEulerAngles(Rbc)*180/M_PI<<endl<<endl;
    return 0;
}














