/**
*    This file is part of OV²SLAM.
*    
*    Copyright (C) 2020 ONERA
*
*    For more information see <https://github.com/ov2slam/ov2slam>
*
*    OV²SLAM is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OV²SLAM is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
*
*    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
*             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
*/

#include <thread>
#include <chrono>
// #include <opencv2/highgui.hpp>

#include "ov2slam.hpp"


SlamManager::SlamManager(std::shared_ptr<SlamParams> pstate)
    : pslamstate_(pstate)
{
    // We first setup the calibration to init everything related
    // to the configuration of the current run
    std::cout << "\n SetupCalibration()\n";
    setupCalibration();

    // If no stereo rectification required (i.e. mono config or 
    // stereo w/o rectification) and image undistortion required
    if( pslamstate_->bdo_undist_ ) {
        std::cout << "\n Setup Image Undistortion\n";
        pcalib_model_left_->setUndistMap(pslamstate_->alpha_);
    }

    pcurframe_.reset( new Frame(pcalib_model_left_, pslamstate_->nmaxdist_) );

    // Create all objects to be used within OV²SLAM
    // =============================================
    int tilesize = 50;
    cv::Size clahe_tiles(pcalib_model_left_->img_w_ / tilesize
                        , pcalib_model_left_->img_h_ / tilesize);
                        
    cv::Ptr<cv::CLAHE> pclahe = cv::createCLAHE(pslamstate_->fclahe_val_, clahe_tiles);

    pfeatextract_.reset( new FeatureExtractor(
                                pslamstate_->nbmaxkps_, pslamstate_->nmaxdist_, 
                                pslamstate_->dmaxquality_, pslamstate_->nfast_th_
                            ) 
                        );

    ptracker_.reset( new FeatureTracker(pslamstate_->nmax_iter_, 
                            pslamstate_->fmax_px_precision_, pclahe
                        )
                    );

    // Map Manager will handle Keyframes / MapPoints
    pmap_.reset( new MapManager(pslamstate_, pcurframe_, pfeatextract_, ptracker_) );

    // Visual Front-End processes every incoming frames 
    pvisualfrontend_.reset( new VisualFrontEnd(pslamstate_, pcurframe_, 
                                    pmap_, ptracker_
                                )
                            );

    // Mapper thread handles Keyframes' processing
    // (i.e. triangulation, local map tracking, BA, LC)
    pmapper_.reset( new Mapper(pslamstate_, pmap_, pcurframe_) );
}


#ifdef MULTI_THREAD
void SlamManager::run()
{
    std::cout << "\nOV²SLAM is ready to process incoming images!\n";

    bis_on_ = true;

    std::thread mapper_thread(&Mapper::run, pmapper_);

    cv::Mat img;

    double time = -1.; // Current image timestamp

    // Main SLAM loop
    while( !bexit_required_ ) {

        // 0. Get New Images
        // =============================================
        if( getNewImage(img, time) )
        {
            // Update current frame
            frame_id_++;
            pcurframe_->updateFrame(frame_id_, time);

            // Display info on current frame state
            if( pslamstate_->debug_ )
                pcurframe_->displayFrameInfo();

            // 1. Send images to the FrontEnd
            // =============================================
            if( pslamstate_->debug_ )
                std::cout << "\n \t >>> [SLAM Node] New image send to Front-End\n";

            bool is_kf_req = pvisualfrontend_->visualTracking(img, time);


            if( pslamstate_->breset_req_ ) {
                reset();
                continue;
            }

            // 2. Create new KF if req. / Send new KF to Mapper
            // ================================================
            if( is_kf_req ) 
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n \t >>> [SLAM Node] New Keyframe send to Back-End\n";

                Keyframe kf(pcurframe_->kfid_, img);
                pmapper_->addNewKf(kf);

            }
        } 
        else {
            std::chrono::milliseconds dura(1);
            std::this_thread::sleep_for(dura);
        }
    }

    std::cout << "\nOV²SLAM is stopping!\n";

    pmapper_->bexit_required_ = true;
    
    mapper_thread.join();

    bis_on_ = false;
}

#else

void SlamManager::step()
{    
    bis_on_ = true;
    
    cv::Mat img;

    double time = -1.; // Current image timestamp

    if( getNewImage(img, time) )
    {
        // Update current frame
        frame_id_++;
        pcurframe_->updateFrame(frame_id_, time);

        // Display info on current frame state
        if( pslamstate_->debug_ )
            pcurframe_->displayFrameInfo();

        // 1. Send images to the FrontEnd
        // =============================================
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> [SLAM Node] New image send to Front-End\n";

        bool is_kf_req = pvisualfrontend_->visualTracking(img, time);


        if( pslamstate_->breset_req_ ) {
            reset();
            return;
        }

        // 2. Create new KF if req. / Send new KF to Mapper
        // ================================================
        if( is_kf_req ) 
        {
            if( pslamstate_->debug_ )
                std::cout << "\n \t >>> [SLAM Node] New Keyframe send to Back-End\n";

            Keyframe kf(pcurframe_->kfid_, img);
            pmapper_->addNewKf(kf);
            pmapper_->step();
        }
    } 
}

#endif

void SlamManager::addNewMonoImage(const double time, cv::Mat &im0)
{
    if( pslamstate_->bdo_undist_ ) {
        pcalib_model_left_->rectifyImage(im0, im0);
    }

    std::lock_guard<std::mutex> lock(img_mutex_);
    qimg_.push(im0);
    qimg_time_.push(time);

    bnew_img_available_ = true;
}


bool SlamManager::getNewImage(cv::Mat &im, double &time)
{
    std::lock_guard<std::mutex> lock(img_mutex_);

    if( !bnew_img_available_ ) {
        return false;
    }

    int k = 0;

    do {
        k++;

        im = qimg_.front();
        qimg_.pop();

        time = qimg_time_.front();
        qimg_time_.pop();

        if( !pslamstate_->bforce_realtime_ )
            break;

    } while( !qimg_.empty() );

    if( k > 1 ) {    
        if( pslamstate_->debug_ )
            std::cout << "\n SLAM is late!  Skipped " << k-1 << " frames...\n";
    }
    
    if( qimg_.empty() ) {
        bnew_img_available_ = false;
    }

    return true;
}

void SlamManager::setupCalibration()
{
    pcalib_model_left_.reset( 
                new CameraCalibration(
                        pslamstate_->cam_left_model_, 
                        pslamstate_->fxl_, pslamstate_->fyl_, 
                        pslamstate_->cxl_, pslamstate_->cyl_,
                        pslamstate_->k1l_, pslamstate_->k2l_, 
                        pslamstate_->p1l_, pslamstate_->p2l_,
                        pslamstate_->img_left_w_, 
                        pslamstate_->img_left_h_
                        ) 
                    );
}

void SlamManager::reset()
{
    std::cout << "\n=======================================\n";
    std::cout << "\t RESET REQUIRED!";
    std::cout << "\n=======================================\n";

    pcurframe_->reset();
    pvisualfrontend_->reset();
    pmap_->reset();
    pmapper_->reset();

    pslamstate_->reset();

    frame_id_ = -1;

    std::lock_guard<std::mutex> lock(img_mutex_);
    
    qimg_ = std::queue<cv::Mat>(); 
    qimg_time_ = std::queue<double>();

    bnew_img_available_ = false;
    
    std::cout << "\n=======================================\n";
    std::cout << "\t RESET APPLIED!";
    std::cout << "\n=======================================\n";
}


// ==========================
//   Visualization functions
// ==========================

cv::Mat SlamManager::visualFrame()
{
    // Display keypoints
    cv::Mat img = cv::Mat(pslamstate_->img_left_h_, pslamstate_->img_left_w_, CV_8UC3);
    cv::Mat img_clone;

    {
        std::lock_guard<std::mutex> lock(pvisualfrontend_->img_mutex_);
        img_clone = pvisualfrontend_->cur_img_.clone();
    }
    
    std::stringstream ss;

    if(! img_clone.empty()) {
        cv::cvtColor(img_clone, img, CV_GRAY2RGB);
        for( const auto &kp : pcurframe_->getKeypoints() ) {
            cv::Scalar col = kp.is_retracked_ ? ( kp.is3d_ ? cv::Scalar(0,255,0) : cv::Scalar(235, 235, 52) )
                : (kp.is3d_ ? cv::Scalar(255,0,0) : cv::Scalar(0,0,255));
            cv::circle(img, kp.px_, 4, col, -1);
        }

        if( pslamstate_->breset_req_ ) ss << "RESET REQUIRED";
        else ss << " SLAM ON  |  "
                << "KEY FRAMES : " << (pmap_->map_pkfs_.size()) <<", MAP POINTS : " << (pmap_->map_plms_.size());
    } else {
        ss << "WAITING FOR IMAGES";
    }

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);
    
    cv::Mat img_text = cv::Mat(img.rows + textSize.height + 10, img.cols, img.type());
    img.copyTo(img_text.rowRange(0, img.rows).colRange(0, img.cols));
    img_text.rowRange(img.rows, img_text.rows) =
        cv::Mat::zeros(textSize.height + 10, img.cols, img.type());
    cv::putText(img_text, ss.str(), cv::Point(5, img_text.rows - 5),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);

    return img_text;
}