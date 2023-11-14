/**
 * EuRoC dataset test
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "ov2slam.hpp"
#include "visualize.hpp"

void load_images(const std::string& img_folder, const std::string &img_ts_path, 
  std::vector<std::string>& img_path, std::vector<double>& img_ts) 
{
  std::ifstream ifs;
  ifs.open(img_ts_path.c_str());

  if( !ifs.is_open() ) {
    std::cerr << "Cannot open file: " << img_ts_path << std::endl;
    exit(-1);
  }

  img_path.reserve(5000);
  img_ts.reserve(5000);

  while( !ifs.eof() ) {
    std::string s;
    getline(ifs, s);
    if( !s.empty() ){
      std::stringstream ss;
      ss << s;
      img_path.push_back(img_folder + "/" + ss.str() + ".png");
      double t;
      ss >> t;
      img_ts.push_back(t / 1e9);
    }
  }
}

// ./mono_euroc ../parameters_files/fast/euroc/euroc_mono.yaml ~/dataset/mav0/cam0/data ../example/euroc_ts/V101.txt

int main(int argc, char** argv) {
    std::vector<std::string> img_path;
    std::vector<double> img_ts;

    load_images(std::string(argv[2]), std::string(argv[3]), img_path, img_ts);

    // Load the parameters
    std::string parameters_file = argv[1];

    const cv::FileStorage fsSettings(parameters_file.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
        std::cout << "Failed to open settings file";
        return 1;
    }

    std::shared_ptr<SlamParams> pparams;
    pparams.reset( new SlamParams(fsSettings) );

    std::shared_ptr<SlamManager> pslam;
    pslam.reset( new SlamManager(pparams) );

    std::shared_ptr<Visualize> pviz;
    pviz.reset( new Visualize(pslam) );

    std::thread slam_thread(&SlamManager::run, pslam);
    slam_thread.detach();
    
    std::thread viz_thread(&Visualize::run, pviz);
    viz_thread.detach();

    for(size_t i = 0; i < img_path.size(); ++i) {
        cv::Mat im = cv::imread(img_path[i], CV_LOAD_IMAGE_UNCHANGED);
        pslam->addNewMonoImage(img_ts[i], im);
        usleep(1000 * 1000 / 30);
    }

  return 0;
}