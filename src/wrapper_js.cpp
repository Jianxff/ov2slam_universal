#include <thread>
#include <memory>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "ov2slam.hpp"
#include "slam_params.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

struct FeaturePoint{
    FeaturePoint(){}
    FeaturePoint(Keypoint& p) {
      u = p.px_.x, v = p.px_.y, is_3d = p.is3d_;
    }
    float u = 0, v = 0; bool is_3d = false;
};



class Session {
public:
    Session (emscripten::val Module) {
        _module = Module;
    }

    void init(int nwidth, int nheight){
        if(initialized) {
            puts("Already initialized!\n");
            return;
        }
        
        nw = nwidth, nh = nheight, nsize = nw * nh * 4;

        fpts.reserve(100);
        mem_ptr.reset(new uint8_t[nsize]);
        pparams.reset(new SlamParams(nw, nh));
        pslam.reset(new SlamManager(pparams));

#ifdef MULTI_THREAD
        slam_thread = std::thread(&SlamManager::run, pslam);
#endif
        
        initialized = true;
    }

    emscripten::val addImage(emscripten::val arrayBuffer){
        if(!initialized) {
            puts("Not initialized!\n");
            return emscripten::val(-1);
        }

        cv::Mat image = getImageRGBA(arrayBuffer);
        cv::cvtColor(image, image, cv::COLOR_RGBA2GRAY);

        auto ms = std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch()
        );
        double time_st = (double)ms.count();

        pslam->addNewMonoImage(time_st / 1000.0, image);

// #ifndef MULTI_THREAD
//         pslam->step();
// #endif

        // if(pslam->pslamstate_->breset_req_) {
        //     return emscripten::val(1);
        // }

        return emscripten::val(0);
    }

#ifndef MULTI_THREAD
    void update() {
        pslam->step();
    }
#endif

    emscripten::val getCameraPoseMatrix() {
        if(!initialized) {
            puts("Not initialized!\n");
            return emscripten::val(-1);
        }

        if(pslam->pslamstate_->breset_req_) {
            return emscripten::val(0);
        }

        auto twc = pslam->pcurframe_->getTwc().matrix();

        emscripten::val pose = emscripten::val::array();
        for(int i = 0; i < 4; i++) {
            emscripten::val row = emscripten::val::array();
            for(int j = 0; j < 4; j++)
                row.set(j, twc(i, j));
            pose.set(i, row);
        }

        return pose;
    }

    emscripten::val getFeaturePoints() {
        if(!initialized) {
            puts("Not initialized!\n");
            return emscripten::val(-1);
        }

        emscripten::val fparray = emscripten::val::array();
        int i = 0;
        for(auto& p : pslam->pcurframe_->getKeypoints()) {
            fparray.set(i++, emscripten::val(FeaturePoint(p)));
        }
        return fparray;
    }
    
    void stop(){
        if(initialized) {
            pslam->bexit_required_ = true;
#ifdef MULTI_THREAD
            slam_thread.join();
#endif
        }

        initialized = false;
        pparams.reset();
        pslam.reset();
        
        puts("[Session] Threads Stopped.");
    }

private:

    cv::Mat getImageRGBA(emscripten::val arrayBuffer) {
        int len = arrayBuffer["byteLength"].as<int>();
        if(len != nsize){
            puts("Error: image size not match\n");
            exit(-1);
        }
        int ptr = (int)mem_ptr.get() / sizeof(uint8_t);
        _module["HEAPU8"].call<emscripten::val>("set", arrayBuffer, emscripten::val(ptr));
        cv::Mat image(nh, nw, CV_8UC4, mem_ptr.get());
        return image;
    }
    
    int nw, nh, nsize;
    bool initialized = false;
    emscripten::val _module;

    std::vector<FeaturePoint> fpts;
    std::shared_ptr<uint8_t[]> mem_ptr;
    std::shared_ptr<SlamParams> pparams;
    std::shared_ptr<SlamManager> pslam;

#ifdef MULTI_THREAD
    std::thread slam_thread;
#endif

};


EMSCRIPTEN_BINDINGS(Slam){
    emscripten::value_object<FeaturePoint>("FeaturePoint")
    .field("u",&FeaturePoint::u).field("v",&FeaturePoint::v).field("is_3d",&FeaturePoint::is_3d);

    emscripten::class_<Session>("Session")
        .constructor<emscripten::val>()
        .function("init", &Session::init)
        .function("addImage", &Session::addImage)
#ifndef MULTI_THREAD
        .function("update", &Session::update)
#endif
        .function("getCameraPoseMatrix", &Session::getCameraPoseMatrix)
        .function("getFeaturePoints", &Session::getFeaturePoints)
        .function("stop", &Session::stop);

}
