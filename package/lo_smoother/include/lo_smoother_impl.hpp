#ifndef LO_SMOOTHER_IMPL_HPP_
#define LO_SMOOTHER_IMPL_HPP_

#include "lo_smoother.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <filesystem>

namespace lo_smoother {

template <typename PointT>
inline LoSmoother<PointT>::LoSmoother(const std::string& lo_config, const std::string& solver_config) {
    // 1. Defaults
    double lag_window = 2.0; 
    double isam_relinearize_thresh = 0.1;
    int isam_relinearize_skip = 1;
    int max_points_per_voxel = 20;

    // 2. Load Config
    std::ifstream file(lo_config);
    nlohmann::json j;
    if (file.is_open()) {
        try {
            file >> j;
            if (j.contains("smoother_lag")) lag_window = j["smoother_lag"];
            if (j.contains("origin_lla")) { ref_lat_ = j["origin_lla"][0]; ref_lon_ = j["origin_lla"][1]; ref_alt_ = j["origin_lla"][2]; }
            
            if (j.contains("isam2")) {
                if (j["isam2"].contains("relinearize_threshold")) isam_relinearize_thresh = j["isam2"]["relinearize_threshold"];
                if (j["isam2"].contains("relinearize_skip")) isam_relinearize_skip = j["isam2"]["relinearize_skip"];
            }
            
            // --- NEW: Constant Velocity Process Noise ---
            if (j.contains("constant_velocity")) {
                if (j["constant_velocity"].contains("process_noise_trans_sigma")) proc_noise_trans_sigma_ = j["constant_velocity"]["process_noise_trans_sigma"];
                if (j["constant_velocity"].contains("process_noise_rot_sigma")) proc_noise_rot_sigma_ = j["constant_velocity"]["process_noise_rot_sigma"];
                if (j["constant_velocity"].contains("process_noise_accel_sigma")) proc_noise_accel_sigma_ = j["constant_velocity"]["process_noise_accel_sigma"];
            }

            if (j.contains("map")) {
                if (j["map"].contains("leaf_size")) map_leaf_size_ = j["map"]["leaf_size"];
                if (j["map"].contains("radius")) map_radius_ = j["map"]["radius"];
                if (j["map"].contains("skip")) map_update_skip_ = j["map"]["skip"];
                if (j["map"].contains("enable_logging")) enable_disk_logging_ = j["map"]["enable_logging"];
                if (j["map"].contains("data_dir")) data_dir_ = j["map"]["data_dir"];
                if (j["map"].contains("keyframe_skip")) keyframe_skip_ = j["map"]["keyframe_skip"];
                if (j["map"].contains("max_points_per_voxel")) max_points_per_voxel = j["map"]["max_points_per_voxel"].get<int>();
            }

            if (j.contains("solver")) {
                std::string s_type = "SVN_NDT";
                if (j["solver"].contains("type")) s_type = j["solver"]["type"].get<std::string>();

                if (s_type == "SVN_ICP") solver_type_ = LidarSolverType::SVN_ICP;
                else if (s_type == "NDT_GENERIC") solver_type_ = LidarSolverType::NDT_GENERIC;
                else solver_type_ = LidarSolverType::SVN_NDT;
            }
        } catch (...) { std::cerr << "[LoSmoother] Config Parse Error.\n"; }
    }

    if (enable_disk_logging_) {
        std::filesystem::remove_all(data_dir_);
        std::filesystem::create_directories(data_dir_);
    }

    // 3. Initialize Solvers
    if (solver_type_ == LidarSolverType::SVN_ICP) {svn_icp_ptr_ = std::make_unique<svn_icp::SvnIterativeClosestPoint<PointT, PointT>>(solver_config);}
    else if (solver_type_ == LidarSolverType::NDT_GENERIC) {ndt_generic_ptr_ = std::make_unique<ndt_generic::NormalDistributionsTransform<PointT, PointT>>(solver_config);}
    else {svn_ndt_ptr_ = std::make_unique<svn_ndt::SvnNormalDistributionsTransform<PointT, PointT>>(solver_config);}

    // 4. Initialize ISAM2
    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = isam_relinearize_thresh;
    isam_params.relinearizeSkip = isam_relinearize_skip;
    isam_params.enableRelinearization = true; 
    smoother_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(lag_window, isam_params);

    // 5. Initialize Map
    map_cloud_.reset(new pcl::PointCloud<PointT>());
    global_map_ = std::make_unique<VoxelMap<PointT>>(map_leaf_size_, map_radius_, max_points_per_voxel);
}

template <typename PointT>
inline void LoSmoother<PointT>::setOrigin(double lat, double lon, double alt) {
    ref_lat_ = lat; ref_lon_ = lon; ref_alt_ = alt;
    std::cout << "[LO] Reference origin updated: " << ref_lat_ << ", " << ref_lon_ << ", " << ref_alt_ << std::endl;
}

template <typename PointT>
inline void LoSmoother<PointT>::update(double timestamp, 
                                const PointCloudConstPtr& scan)
{   
    if (!initialized_) {
        gtsam::Pose3 initial_pose = gtsam::Pose3::Identity();
        gtsam::Vector3 initial_velocity = gtsam::Vector3::Zero(); // Fallback
        
        // if (!imu_window.empty()) {
        //     const auto& init_imu = imu_window.back();

        //     // 1. Extract Initial Pose
        //     double q_norm = std::sqrt(init_imu.qw_20*init_imu.qw_20 + init_imu.qx_20*init_imu.qx_20 + 
        //                             init_imu.qy_20*init_imu.qy_20 + init_imu.qz_20*init_imu.qz_20);
        //     Eigen::Vector3d ned = GeoUtils::LLAtoNED_Exact(init_imu.latitude_20, init_imu.longitude_20, init_imu.altitude_20, ref_lat_, ref_lon_, ref_alt_);
        //     if (q_norm > 1e-6) {
        //         gtsam::Rot3 init_rot = gtsam::Rot3::Quaternion(
        //             init_imu.qw_20, init_imu.qx_20, init_imu.qy_20, init_imu.qz_20
        //         );
        //         initial_pose = gtsam::Pose3(init_rot, gtsam::Point3(ned.x(),ned.y(),ned.z()));
        //     } else {
        //         std::cout << "[LO] Warning: Invalid initial pose received. Using Identity." << std::endl;
        //     }

        //     // 2. Extract Initial Velocity (NED frame)
        //     initial_velocity = gtsam::Vector3(
        //         init_imu.velocityNorth_20,
        //         init_imu.velocityEast_20,
        //         init_imu.velocityDown_20
        //     );

        //     std::cout << "[LO] Initializing at Exact GT State: Pos=[" << ned.transpose() 
        //               << "], Vel=[" << initial_velocity.transpose() << "]" << std::endl;
        // }
        
        initialize_system(timestamp, initial_pose, initial_velocity);
        
        last_lidar_time_ = timestamp;
        manage_map(scan, initial_pose);
        if (enable_disk_logging_) save_keyframe(timestamp, scan, initial_pose);
        return;
    }
        
    step_counter_++;
    
    // 1. Calculate DT
    double dt = timestamp - last_lidar_time_;
    if (dt < 1e-4) dt = 0.1; 
    
    dt_avg_ = 0.9 * dt_avg_ + 0.1 * dt; 

    // 2. CONSTANT VELOCITY PREDICTION
    // Extrapolate initial guess using the explicit global velocity state
    gtsam::Point3 pred_trans = prev_pose_.translation() + prev_vel_ * dt;
    gtsam::Pose3 initial_guess = gtsam::Pose3(prev_pose_.rotation(), pred_trans);

    if (global_map_->empty()) manage_map(scan, initial_guess);

    // 3. REGISTRATION (NDT/ICP)
    gtsam::Pose3 measured_pose;
    Eigen::Matrix<double, 6, 6> measured_cov;

    auto t_start = std::chrono::high_resolution_clock::now();
    if (solver_type_ == LidarSolverType::SVN_NDT) {
        svn_ndt_ptr_->setInputTarget(map_cloud_);
        auto res = svn_ndt_ptr_->align(*scan, initial_guess);
        measured_pose = res.final_pose; 
        measured_cov = res.final_covariance;
    } else if (solver_type_ == LidarSolverType::SVN_ICP) {
        svn_icp_ptr_->setInputTarget(map_cloud_);
        auto res = svn_icp_ptr_->align(*scan, initial_guess);
        measured_pose = res.final_pose; 
        measured_cov = res.final_covariance;
    } else {
        ndt_generic_ptr_->setInputTarget(map_cloud_);
        auto res = ndt_generic_ptr_->align(*scan, initial_guess);
        measured_pose = res.final_pose; 
        measured_cov = res.final_covariance;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    last_alingment_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    last_measured_cov_ = measured_cov;

    // 4. GRAPH UPDATE
    // Factor A: Lidar Measurement Prior
    auto reg_noise = gtsam::noiseModel::Gaussian::Covariance(measured_cov);
    new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(step_counter_), measured_pose, reg_noise));

    // Factor B: Global Kinematic Integration (Links X_{k-1}, X_k, V_{k-1})
    // FIX: Integration uncertainty scales linearly with time
    auto kin_noise = gtsam::noiseModel::Isotropic::Sigma(3, proc_noise_trans_sigma_ * dt);
    new_factors_.add(std::make_shared<GlobalVelocityFactor>(
        X(step_counter_-1), X(step_counter_), V(step_counter_-1), dt, kin_noise
    ));

    // Factor C: Rotation Random Walk (Zero Angular Velocity Assumption)
    // FIX: Rotational drift uncertainty must scale with the square root of time
    gtsam::Vector6 rot_sigmas;
    double scaled_rot_sigma = proc_noise_rot_sigma_ * std::sqrt(dt);
    rot_sigmas << scaled_rot_sigma, scaled_rot_sigma, scaled_rot_sigma, 1e6, 1e6, 1e6;
    auto rot_noise = gtsam::noiseModel::Diagonal::Sigmas(rot_sigmas);
    
    gtsam::Pose3 identity_motion(gtsam::Rot3::Identity(), gtsam::Point3(0,0,0));
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        X(step_counter_-1), X(step_counter_), identity_motion, rot_noise
    ));

    // Factor D: Velocity Random Walk (Continuous Wiener Process)
    // ALREADY CORRECT: Limits linear acceleration using square root of time
    auto accel_noise = gtsam::noiseModel::Isotropic::Sigma(3, proc_noise_accel_sigma_ * std::sqrt(dt));
    new_factors_.add(gtsam::BetweenFactor<gtsam::Vector3>(
        V(step_counter_-1), V(step_counter_), gtsam::Vector3::Zero(), accel_noise
    ));

    // Insert Values and Timestamps for Lag Smoother
    new_values_.insert(X(step_counter_), measured_pose);
    new_values_.insert(V(step_counter_), prev_vel_); 
    
    new_timestamps_[X(step_counter_)] = timestamp;
    new_timestamps_[V(step_counter_)] = timestamp;

    try {
        smoother_->update(new_factors_, new_values_, new_timestamps_);
    } catch(std::exception& e) {
        std::cerr << "[LO] CRITICAL WARNING: Smoother exception at step " << step_counter_ << ": " << e.what() << std::endl;
    }
    
    // Update State History
    prev_pose_ = smoother_->calculateEstimate<gtsam::Pose3>(X(step_counter_)); 
    prev_vel_ = smoother_->calculateEstimate<gtsam::Vector3>(V(step_counter_));

    if (!prev_pose_.matrix().allFinite()) {
        prev_pose_ = gtsam::Pose3(gtsam::Rot3::Identity(), prev_pose_.translation()); 
        prev_vel_ = gtsam::Vector3::Zero();
    }

    if (step_counter_ % map_update_skip_ == 0) manage_map(scan, prev_pose_);
    if (step_counter_ % keyframe_skip_ == 0) save_keyframe(timestamp, scan, prev_pose_);

    new_factors_.resize(0); new_values_.clear(); new_timestamps_.clear();
    last_lidar_time_ = timestamp;
}

template <typename PointT>
inline void LoSmoother<PointT>::initialize_system(double timestamp, const gtsam::Pose3& initial_pose, const gtsam::Vector3& initial_velocity) {
    prev_pose_ = initial_pose;
    prev_vel_ = initial_velocity; // Seed the system with IMU velocity

    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished()); 
    
    // We can loosen the velocity prior noise slightly since the IMU velocity might have a small error, 
    // but 1e-4 is still a solid starting anchor.
    auto vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-4);

    new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), prev_pose_, pose_noise));
    new_values_.insert(X(0), prev_pose_);
    new_timestamps_[X(0)] = timestamp;

    new_factors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), prev_vel_, vel_noise));
    new_values_.insert(V(0), prev_vel_);
    new_timestamps_[V(0)] = timestamp;

    try {
        smoother_->update(new_factors_, new_values_, new_timestamps_);
    } catch(...) {
        std::cerr << "[LO] CRITICAL WARNING: Indeterminant Linear System at initial step!" << std::endl;
    }
    
    new_factors_.resize(0); new_values_.clear(); new_timestamps_.clear();
    
    last_lidar_time_ = timestamp;
    initialized_ = true;
    step_counter_ = 0;
}

template <typename PointT>
inline void LoSmoother<PointT>::manage_map(const PointCloudConstPtr& scan, const gtsam::Pose3& pose) {
    if (scan->empty()) return;
    global_map_->update(scan, pose.matrix().cast<float>());
    map_cloud_ = global_map_->getPointCloud();
}

template <typename PointT>
inline void LoSmoother<PointT>::save_keyframe(double timestamp, const PointCloudConstPtr& scan_body, const gtsam::Pose3& pose) {
    if (!enable_disk_logging_ || scan_body->empty()) return;
    std::stringstream ss;
    ss.imbue(std::locale::classic());
    ss << data_dir_ << std::fixed << std::setprecision(6) << timestamp << ".pcd";
    std::string filename = ss.str();
    typename pcl::PointCloud<PointT>::Ptr scan_map(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*scan_body, *scan_map, pose.matrix().cast<float>());
    try {
        pcl::io::savePCDFileBinaryCompressed(filename, *scan_map);
    } catch (...) {}
    keyframes_.push_back({(size_t)step_counter_, timestamp, pose, filename});
}

template <typename PointT>
inline void LoSmoother<PointT>::get_current_state(LoState& out_state) const {
    if (!initialized_) return;
    out_state.timestamp = last_lidar_time_;
    Eigen::Vector3d t = prev_pose_.translation();
    out_state.q_nb = prev_pose_.rotation().toQuaternion();
    
    Eigen::Vector3d lla = GeoUtils::NEDtoLLA_Exact(t.x(), t.y(), t.z(), ref_lat_, ref_lon_, ref_alt_);
    out_state.north = t.x(); out_state.east = t.y(); out_state.down = t.z();
    out_state.lat = lla.x(); out_state.lon = lla.y(); out_state.alt = lla.z();
    
    // Assign optimized velocity directly
    out_state.vel_global = prev_vel_; 

    try {
        out_state.covariance = smoother_->marginalCovariance(X(step_counter_));
    } catch (...) { out_state.covariance.setIdentity(); }
    out_state.alingment_time_ms = last_alingment_time_ms_;
    out_state.reg_covariance = last_measured_cov_;
}

} // namespace lo_smoother

#endif // LO_SMOOTHER_IMPL_HPP_