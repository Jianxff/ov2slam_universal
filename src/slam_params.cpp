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

#include "slam_params.hpp"

SlamParams::SlamParams(const cv::FileStorage &fsSettings) {

    std::cout << "\nSLAM Parameters are being setup...\n";

    // READ THE SETTINGS
    debug_ = static_cast<int>(fsSettings["debug"]);

    bforce_realtime_ = static_cast<int>(fsSettings["force_realtime"]);

    cam_left_topic_.assign(fsSettings["Camera.topic_left"]);
    cam_left_model_.assign(fsSettings["Camera.model_left"]);
    img_left_w_ = fsSettings["Camera.left_nwidth"];
    img_left_h_ = fsSettings["Camera.left_nheight"];

    fxl_ = fsSettings["Camera.fxl"];
    fyl_ = fsSettings["Camera.fyl"];
    cxl_ = fsSettings["Camera.cxl"];
    cyl_ = fsSettings["Camera.cyl"];

    k1l_ = fsSettings["Camera.k1l"];
    k2l_ = fsSettings["Camera.k2l"];
    p1l_ = fsSettings["Camera.p1l"];
    p2l_ = fsSettings["Camera.p2l"];

    finit_parallax_ = fsSettings["finit_parallax"];

    alpha_ = fsSettings["alpha"];

    bdo_undist_ = static_cast<int>(fsSettings["bdo_undist"]);
    
    bdo_random = static_cast<int>(fsSettings["bdo_random"]);

    use_shi_tomasi_ = static_cast<int>(fsSettings["use_shi_tomasi"]);
    use_fast_ = static_cast<int>(fsSettings["use_fast"]);
    use_brief_ = static_cast<int>(fsSettings["use_brief"]);
    use_singlescale_detector_ = static_cast<int>(fsSettings["use_singlescale_detector"]);

    nfast_th_ = fsSettings["nfast_th"];
    dmaxquality_ = fsSettings["dmaxquality"];

    nmaxdist_ = fsSettings["nmaxdist"];
    float nbwcells = ceil( (float)img_left_w_ / nmaxdist_ );
    float nbhcells = ceil( (float)img_left_h_ / nmaxdist_ );
    nbmaxkps_ = nbwcells * nbhcells;

    use_clahe_ = static_cast<int>(fsSettings["use_clahe"]);
    fclahe_val_ = fsSettings["fclahe_val"];

    do_klt_ = static_cast<int>(fsSettings["do_klt"]);
    klt_use_prior_ = static_cast<int>(fsSettings["klt_use_prior"]);

    btrack_keyframetoframe_ = static_cast<int>(fsSettings["btrack_keyframetoframe"]);
    
    nklt_win_size_ = fsSettings["nklt_win_size"];
    nklt_pyr_lvl_ = fsSettings["nklt_pyr_lvl"];

    klt_win_size_ = cv::Size(nklt_win_size_, nklt_win_size_);

    fmax_fbklt_dist_ = fsSettings["fmax_fbklt_dist"];
    nmax_iter_ = fsSettings["nmax_iter"];
    fmax_px_precision_ = fsSettings["fmax_px_precision"];

    
    nklt_err_ = fsSettings["nklt_err"];

    // Matching th.
    bdo_track_localmap_ = static_cast<int>(fsSettings["bdo_track_localmap"]);

    fmax_desc_dist_ = fsSettings["fmax_desc_dist"];
    fmax_proj_pxdist_ = fsSettings["fmax_proj_pxdist"];

    doepipolar_ = static_cast<int>(fsSettings["doepipolar"]);
    dop3p_ = static_cast<int>(fsSettings["dop3p"]);

    fransac_err_ = fsSettings["fransac_err"];
    fepi_th_ = fransac_err_;
    nransac_iter_ = fsSettings["nransac_iter"];

    fmax_reproj_err_ = fsSettings["fmax_reproj_err"];
    buse_inv_depth_ = static_cast<int>(fsSettings["buse_inv_depth"]);

    // Bundle Adjustment Parameters
    // (mostly related to Ceres options)
    robust_mono_th_ = fsSettings["robust_mono_th"];

    use_sparse_schur_ = static_cast<int>(fsSettings["use_sparse_schur"]);
    use_dogleg_ = static_cast<int>(fsSettings["use_dogleg"]);
    use_subspace_dogleg_ = static_cast<int>(fsSettings["use_subspace_dogleg"]);
    use_nonmonotic_step_ = static_cast<int>(fsSettings["use_nonmonotic_step"]);

    apply_l2_after_robust_ = static_cast<int>(fsSettings["apply_l2_after_robust"]);

    nmin_covscore_ = fsSettings["nmin_covscore"];

    // Map Filtering parameters
    fkf_filtering_ratio_ = fsSettings["fkf_filtering_ratio"]; 
}

SlamParams::SlamParams(int imwidth, int imheight, int fov, bool accurate){
    bforce_realtime_ = 1;

    img_left_w_ = imwidth;
    img_left_h_ = imheight;

    cxl_ = (double) imwidth * 0.5;
    cyl_ = (double) imheight * 0.5;

    // auto-calculate fx and fy
    double aspect = (double) imwidth / (double) imheight;
    double fovH = imwidth > imheight ? (fov * aspect) : fov; 
    double fovV = imwidth > imheight ? fov : (fov * aspect);
    
    double fx = cxl_ / std::tan(fovH * 0.5 * M_PI / 180);
    double fy = cyl_ / std::tan(fovV * 0.5 * M_PI / 180);
    
    fxl_ = std::min(fx, fy);
    fyl_ = fxl_;
    

    k1l_ = 0;
    k2l_ = 0;
    p1l_ = 0;
    p2l_ = 0;

    finit_parallax_ = 20.;

    // bdo_stereo_rect_ = 0;
    alpha_ = 0;

    bdo_undist_ = 0;
    
    bdo_random = 1;

    use_shi_tomasi_ = 0;

    use_fast_ = accurate ? 0 : 1;
    use_brief_ = 1;
    use_singlescale_detector_ = accurate ? 1 : 0;

    nfast_th_ = 10;
    dmaxquality_ = 0.001;

    nmaxdist_ = accurate ? 35 : 50;
    float nbwcells = ceil( (float)img_left_w_ / nmaxdist_ );
    float nbhcells = ceil( (float)img_left_h_ / nmaxdist_ );
    nbmaxkps_ = nbwcells * nbhcells;

    use_clahe_ = 0;
    fclahe_val_ = 3;

    do_klt_ = 1;
    klt_use_prior_ = 1;

    btrack_keyframetoframe_ = 0;

    nklt_win_size_ = 9;
    nklt_pyr_lvl_ = 3;

    klt_win_size_ = cv::Size(nklt_win_size_, nklt_win_size_);

    fmax_fbklt_dist_ = 0.5;
    nmax_iter_ = 30;
    fmax_px_precision_ =0.01;

    
    nklt_err_ = 30.;

    // Matching th.
    bdo_track_localmap_ = 1;

    fmax_desc_dist_ = 0.2;
    fmax_proj_pxdist_ = 2.;

    doepipolar_ = 1;
    dop3p_ = accurate ? 1 : 0;

    fransac_err_ = 3.;
    fepi_th_ = fransac_err_;
    nransac_iter_ = 100;

    fmax_reproj_err_ = 3.;
    buse_inv_depth_ = 1;

    // Bundle Adjustment Parameters
    // (mostly related to Ceres options)
    robust_mono_th_ = 5.9915;

    use_sparse_schur_ = 1;
    use_dogleg_ = 0;
    use_subspace_dogleg_ = 0;
    use_nonmonotic_step_ = 0;

    apply_l2_after_robust_ = 1;

    nmin_covscore_ = 25;

    // Map Filtering parameters
    fkf_filtering_ratio_ = 0.9;
    
}

void SlamParams::reset() {
    blocalba_is_on_ = false;
    bvision_init_ = false;
    breset_req_ = false;
}