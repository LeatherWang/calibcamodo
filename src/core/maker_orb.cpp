#include "maker_orb.h"
#include "math_cv.h"

using namespace cv;
using namespace std;

namespace calibcamodo {

MakerOrb::MakerOrb(DatasetOrb* _pDatasetOrb):
    MakerBase(_pDatasetOrb), mpDatasetOrb(_pDatasetOrb) {
    mOrbMatcher = ORBmatcher();
}

void MakerOrb::MakeMpAndMsrMp()
{
    // match orb-keypoints between neigbour keyframes
    std::map<int, PtrKeyFrameOrb> mapId2pKfOrb = mpDatasetOrb->GetKfOrbMap();
    // 遍历所有的关键帧进行相邻帧之间的匹配
    for (auto iter2 = mapId2pKfOrb.cbegin(), iter1 = iter2++;
         iter2 != mapId2pKfOrb.cend();
         ++iter1, ++iter2)
    {
        PtrKeyFrameOrb pKf1 = iter1->second;
        PtrKeyFrameOrb pKf2 = iter2->second;

        map<int, int> mapOrbMatches;
        // 匹配orb特征点，orb-slam中的方法
        MatchKeyPointOrb(&mOrbMatcher, pKf1, pKf2, mapOrbMatches);

        // 根据特征点在前后pKf1与pKf2像素的欧式距离剔除
        map<int, int> mapOrbMatchesGood1;
        RejectOutlierDist(pKf1, pKf2, mapOrbMatches, mapOrbMatchesGood1);

        // 根据ransac剔除
        map<int, int> mapOrbMatchesGood2;
        RejectOutlierRansac(pKf1, pKf2, mapOrbMatchesGood1, mapOrbMatchesGood2);

        // debug ...
//                DrawMatches(pKf1, pKf2, mapOrbMatches, "raw-match");
//                DrawMatches(pKf1, pKf2, mapOrbMatchesGood1, "good-match-1");
//                DrawMatches(pKf1, pKf2, mapOrbMatchesGood2, "good-match-2");
        //        cerr << " -- number of raw matches: " << mapOrbMatches.size();
        //        cerr << " -- number of good matches 1: " << mapOrbMatchesGood1.size();
        //        cerr << " -- number of good matches 2: " << mapOrbMatchesGood2.size();
        //        cerr << endl;
        // debug end

        // 生成MapPoints，并加入数据库中
        InitMpByTrain(pKf1, pKf2, mapOrbMatchesGood2);
    }
}

void MakerOrb::InitMpByTrain(PtrKeyFrameOrb pKf1, PtrKeyFrameOrb pKf2,
                                  const std::map<int, int>& match)
{
    // 得到投影矩阵
    Mat matCamP1 = ComputeCamMatP(pKf1, mpDatasetOrb->GetCamMat());
    Mat matCamP2 = ComputeCamMatP(pKf2, mpDatasetOrb->GetCamMat());
    std::map<int, int> matchMpPlaxGood;
    // 遍历每一对匹配点，三角化生成MapPoints
    for(auto pair : match)
    {
        int id1 = pair.first;
        int id2 = pair.second;

        KeyPoint kp1Un = pKf1->mvecKeyPointUndist[id1];
        KeyPoint kp2Un = pKf2->mvecKeyPointUndist[id2];
        Point2f pt1Un = kp1Un.pt;
        Point2f pt2Un = kp2Un.pt;

        KeyPoint kp1 = pKf1->mvecKeyPoint[id1];
        KeyPoint kp2 = pKf2->mvecKeyPoint[id2];
        Point2f pt1 = kp1.pt;
        Point2f pt2 = kp2.pt;

        // do triangulation
        Mat x3D;
        // orb-slam中的方法
        Triangulate(pt1Un, pt2Un, matCamP1, matCamP2, x3D);
        Point3f pt3wp(x3D);
        Point3f pt3wo1(pKf1->GetPoseCamera().tvec);
        Point3f pt3wo2(pKf2->GetPoseCamera().tvec);

        // check if parallax good, and create mappoint
        // 检查视差是否大于阈值，大于返回true
        if (CheckParallax(pt3wo1, pt3wo2, pt3wp))
        {
            matchMpPlaxGood[id1] = id2;

            //检查该关键帧的该特征点是否已经对应特征点?
            /** @attention 对闭环不起作用*/
            PtrMapPoint pMp = mpDatasetOrb->GetMpByKfId(pKf1, id1);

            if (pMp) {
                // use old mappoint
//                 cerr << "find old mappoint!" << endl;
            }
            else {
                // add new mappoint
                PtrMapPointOrb pMpOrb = make_shared<MapPointOrb>(Pt3(pt3wp));
                // 加入到数据库中
                mpDatasetOrb->AddMpOrb(pMpOrb);
                pMp = pMpOrb;
            }

            // Msr是measure的简写
            Mat info = (Mat_<float>(2,2) << 1/pow(1.2,kp1.octave),0,0,1/pow(1.2,kp1.octave)); /** @todo */
            Mat info_2 = (Mat_<float>(2,2) << 1/pow(1.2,kp2.octave),0,0,1/pow(1.2,kp2.octave));
//            if(kp1.octave != kp2.octave)
//                cout<<"error!!"<<kp1.octave<<" "<<kp2.octave<<endl;

            PtrMsrUVKf2Mp pMsr1 = make_shared<MeasureUVKf2Mp>(
                        pt1, pt1Un, info, mpDatasetOrb->GetCamMat(),
                        mpDatasetOrb->GetCamDist(), pKf1, pMp, id1);
            PtrMsrUVKf2Mp pMsr2 = make_shared<MeasureUVKf2Mp>(
                        pt2, pt2Un, info_2, mpDatasetOrb->GetCamMat(),
                        mpDatasetOrb->GetCamDist(), pKf2, pMp, id2);

            // 加入数据库中
            mpDatasetOrb->AddMsrMp(pMsr1);
            mpDatasetOrb->AddMsrMp(pMsr2);
        }


        // debug
        //        cerr << "matCamP1" << endl << matCamP1 << endl;
        //        cerr << "matCamP2" << endl << matCamP2 << endl;
        //        cerr << "pt1" << endl << pt1 << endl;
        //        cerr << "pt2" << endl << pt2 << endl;
        //        cerr << "pt3wp" << endl << pt3wp << endl;
        //        cerr << endl;
    }

    // debug
        //cerr << " -- number of mappoint candidates: " << matchMpPlaxGood.size() << endl;

}

}
