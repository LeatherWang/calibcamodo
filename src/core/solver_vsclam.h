#ifndef SOLVER_VSCLAM_H
#define SOLVER_VSCLAM_H

#include "solver_base.h"
#include "g2o/g2o_api.h"

namespace calibcamodo {

class SolverVsclam : public SolverBase {
public:
    SolverVsclam(Dataset* _pDataset);

    virtual void DoCalib();
    virtual void DoCalib_2();

    void OptimizeSlam();
    void OptimizeSclam();
    void optimize_mappoints_keyframe();
    void optimize_extrinsic();
    void optimize_mappoints_keyframe_plane();

    // debug functions
    static void PrintEdgeInfoOdo(const std::vector<g2o::EdgeSE2*>& vecpEdgeOdo);
    static void PrintEdgeInfoVSlam(const std::vector<g2o::EdgeVSlam*>& vecpEdgeVSlam);


};

}
#endif // SOLVER_VSCLAM_H
