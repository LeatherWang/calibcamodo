#ifndef MAKER_ARUCO_H
#define MAKER_ARUCO_H

#include "maker_base.h"

namespace calibcamodo {

class MakerAruco : public MakerBase {

public:
    MakerAruco(DatasetAruco* _pDatasetAruco);

    void MakeMkAndMsrMk();
    void InitMkPose();
    void InitKfMkPose();

    virtual void DoMake()
    {
        MakeMsrOdo();
        InitKfPose(); //使用粗略的外参初始化相机位姿

        MakeMkAndMsrMk();
        InitMkPose();
    }

private:
    DatasetAruco* mpDatasetAruco;

    double mAmkZErrRZ;
    double mAmkZErrMin;
    double mAmkXYErrRZ;
    double mAmkXYErrMin;
};

}



# endif
