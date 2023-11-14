#include <thread>
#include <memory>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "ov2slam.hpp"
#include "slam_params.hpp"
#include "visualize.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class Session {
public:
    Session(const std::string config_file) {
		const cv::FileStorage fsSettings(config_file.c_str(), cv::FileStorage::READ);
    	if(!fsSettings.isOpened()) {
        	std::cout << "Failed to open settings file";
        	exit(-1);
    	}

        pparams.reset(new SlamParams(fsSettings));
        pslam.reset(new SlamManager(pparams));
		slam_thread = std::thread(&SlamManager::run, pslam);
		initialized = true;
    }

    Session(const int imwidth, const int imheight, const bool debug) {
        pparams.reset(new SlamParams(imwidth, imheight, debug));
        pslam.reset(new SlamManager(pparams));
        slam_thread = std::thread(&SlamManager::run, pslam);
        initialized = true;
    }
    

	void startVisualize() {
		if(!initialized) {
			puts("Not initialized!\n");
			return;
		}
		
		pviz.reset( new Visualize(pslam) );
		viz_thread = std::thread(&Visualize::run, pviz);
		visualize = true;
	}

    void addTrack(py::array_t<uint8_t>& input, double time_ms = -1){
        if(!initialized) {
            puts("Not initialized!\n");
            return;
        }

        cv::Mat image = getImageBGR(input);
        cv::cvtColor(image, image, cv::COLOR_RGBA2GRAY);

        if(time_ms < 0) {
            auto ms = std::chrono::duration_cast< std::chrono::milliseconds >(
                std::chrono::system_clock::now().time_since_epoch()
            );
            time_ms = (double)ms.count();
        }

        pslam->addNewMonoImage(time_ms / 1000.0, image);
    }

    Eigen::Matrix4d getCameraPoseMatrix() {
        if(!initialized) {
            puts("Not initialized!\n");
            return Eigen::Matrix4d::Identity();
        }

        if(pslam->pslamstate_->breset_req_) {
            return Eigen::Matrix4d::Identity();
        }

        Eigen::Matrix4d twc = pslam->pcurframe_->getTwc().matrix();

        return twc;
    }

    // emscripten::val getFeaturePoints() {
    //     if(!initialized) {
    //         puts("Not initialized!\n");
    //         return emscripten::val(-1);
    //     }

    //     emscripten::val fparray = emscripten::val::array();
    //     int i = 0;
    //     for(auto& p : pslam->pcurframe_->getKeypoints()) {
    //         fparray.set(i++, emscripten::val(FeaturePoint(p)));
    //     }
    //     return fparray;
    // }
    
    void stop(){
		if(visualize) {
			pviz->bexit_required_ = true;
			viz_thread.join();
		}

        if(initialized) {
            pslam->bexit_required_ = true;
            slam_thread.join();
        }

        initialized = false;
		visualize = false;
        pparams.reset();
        pslam.reset();
		pviz.reset();
        
        puts("[Session] Threads Stopped.");
    }

private:

    cv::Mat getImageBGR(py::array_t<uint8_t>& input) {
        if(input.ndim() != 3) 
            throw std::runtime_error("get Image : number of dimensions must be 3");
        py::buffer_info buf = input.request();
		cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);
        return image;
    }

    bool initialized = false;
	bool visualize = false;

    std::shared_ptr<SlamParams> pparams;
    std::shared_ptr<SlamManager> pslam;
	std::shared_ptr<Visualize> pviz;
	std::thread viz_thread;
    std::thread slam_thread;

};

PYBIND11_MODULE(ov2slam, m) {
	m.doc() = "OV2SLAM Python Bindings";

	py::class_<Session>(m, "Session")
        .def(py::init<const std::string>(), py::arg("config_file"))
        .def(py::init<const int, const int, const bool>(), py::arg("imwidth"), py::arg("imheight"), py::arg("debug") = false)
        .def("enable_viewer", &Session::startVisualize)
		.def("add_track", &Session::addTrack, py::arg("image"), py::arg("time_ms") = -1)
		.def("camera_pose", &Session::getCameraPoseMatrix)
		// .def("getFeaturePoints", &Session::getFeaturePoints)
		.def("stop", &Session::stop);

}