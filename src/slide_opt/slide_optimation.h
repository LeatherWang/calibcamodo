/**
* @file serial_api.h
* @brief This is the API to use boost asio serial.
* @version 0.0
* @author LeatherWang
* @date 2018-2-9
*/

#ifndef SLIDE_OPTIMATION_H
#define SLIDE_OPTIMATION_H

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <string>
#include <math.h>
#include <stdio.h>
#include <iostream>

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
#include "core/solver_base.h"

#include <Eigen/Eigen>
#include <chrono>
#include "slide_opt/color.h"

using namespace std;
using namespace boost::asio;

typedef unsigned char uchar;
typedef unsigned short ushort;

namespace calibcamodo
{

class SlideOptimation
{
public:
    SlideOptimation(DatasetAruco* _pDataSetAruco, unsigned int _nLocalWindowSize=10);
    ~SlideOptimation();
    // start a thread
    void start_optimation_thread();
    void AddToLocalWindow(PtrKeyFrameAruco pKF);
    void InsertKeyFrame(PtrKeyFrameAruco pKF);
    bool AcceptKeyFrames();

public:
    bool is_thread_exit;
    unsigned int mnLocalWindowSize;
    unsigned int mnFixedCamerasSize;

private:
    static void *Run(void *__this);
    bool CheckNewKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    void ProcessNewKeyFrame();
    void DoOptimation();
    void calibExtrinsic();

private:
    std::list<PtrKeyFrameAruco> mlLocalKeyFrames;
    std::list<PtrKeyFrameAruco> mlNewKeyFrames; ///< 等待处理的关键帧列表
    pthread_mutex_t mMutexNewKFs;
    pthread_mutex_t mMutexAccept;
    bool mbAcceptKeyFrames;
    PtrKeyFrameAruco mpCurrentKeyFrame;
    DatasetAruco* mpDatasetAruco;
};
}//namespace
#endif
