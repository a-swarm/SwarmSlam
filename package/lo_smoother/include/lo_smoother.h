#ifndef LO_SMOOTHER_H_
#define LO_SMOOTHER_H_

#include <vector>
#include <deque>
#include <memory>
#include <string>
#include <map>
#include <random>

#include <gtsamreference.hpp>
#include <pclreference.hpp>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <svn_ndt.h>
#include <svn_icp.h>
#include <ndt_generic.h>
#include <geoutils.hpp>
// #include <compassframeimu.hpp> 
#include <map.hpp> 

// Using standard symbols for Pose (X) and Global Velocity (V)
using gtsam::symbol_shorthand::X; 
using gtsam::symbol_shorthand::V; 

namespace lo_smoother {

// --- CUSTOM FACTOR: GLOBAL CONSTANT VELOCITY ---
// Enforces X_k.translation = X_{k-1}.translation + V_{k-1} * dt
class GlobalVelocityFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3> {
    double dt_;
public:
    GlobalVelocityFactor(gtsam::Key pose1, gtsam::Key pose2, gtsam::Key vel1, 
                         double dt, const gtsam::SharedNoiseModel& model) 
        : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3>(model, pose1, pose2, vel1), dt_(dt) {}

    // Modern GTSAM signature using raw pointers for optional Jacobians (No Boost)
    gtsam::Vector evaluateError(const gtsam::Pose3& p1, const gtsam::Pose3& p2, const gtsam::Vector3& v1,
                                gtsam::Matrix* H1 = nullptr,
                                gtsam::Matrix* H2 = nullptr,
                                gtsam::Matrix* H3 = nullptr) const override {
        
        gtsam::Matrix36 H_t1, H_t2;
        // translation() natively accepts gtsam::Matrix* in modern GTSAM
        gtsam::Point3 t1 = p1.translation(H1 ? &H_t1 : nullptr);
        gtsam::Point3 t2 = p2.translation(H2 ? &H_t2 : nullptr);
        
        if (H1) *H1 = -H_t1;
        if (H2) *H2 = H_t2;
        if (H3) *H3 = -gtsam::Matrix33::Identity() * dt_;
        
        return t2 - (t1 + v1 * dt_);
    }
};

struct LoState {
    double timestamp;
    double lat, lon, alt;  
    double north, east, down;    
    Eigen::Vector3d vel_global; 
    Eigen::Quaterniond q_nb;    
    Eigen::Matrix<double, 6, 6> covariance; 
    Eigen::Matrix<double, 6, 6> reg_covariance;
    double alingment_time_ms;

    LoState() : covariance(Eigen::Matrix<double, 6, 6>::Identity()),
                 reg_covariance(Eigen::Matrix<double, 6, 6>::Identity()) {}
};

struct Keyframe {
    size_t id;                  
    double timestamp;           
    gtsam::Pose3 pose;          
    std::string cloud_filename; 
};

enum class LidarSolverType {
    SVN_NDT,
    SVN_ICP,
    NDT_GENERIC
};

template <typename PointT>
class LoSmoother {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

    explicit LoSmoother(const std::string& lo_config, const std::string& solver_config);
    ~LoSmoother() = default;

    void update(double timestamp, 
                const PointCloudConstPtr& scan);
    void get_current_state(LoState& out_state) const;
    void setOrigin(double lat, double lon, double alt);
    
    PointCloudConstPtr get_map() const { return map_cloud_; }
    const std::vector<Keyframe>& get_keyframes() const { return keyframes_; }
    bool is_initialized() const { return initialized_; }

private:

    void initialize_system(double timestamp, const gtsam::Pose3& initial_pose, const gtsam::Vector3& initial_velocity);
    void manage_map(const PointCloudConstPtr& scan, const gtsam::Pose3& pose);
    void save_keyframe(double timestamp, const PointCloudConstPtr& scan_body, const gtsam::Pose3& pose);

    // GTSAM Components
    std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;
    
    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;
    std::map<gtsam::Key, double> new_timestamps_;

    // Process Noise Parameters
    double proc_noise_trans_sigma_ = 0.05; 
    double proc_noise_rot_sigma_ = 0.05;   
    double proc_noise_accel_sigma_ = 0.5;  

    bool initialized_ = false;
    long long step_counter_ = 0;
    double last_lidar_time_ = -1.0;
    double dt_avg_ = 0.1; 
    
    // Explicit State Tracking
    gtsam::Pose3 prev_pose_;      
    gtsam::Vector3 prev_vel_ = gtsam::Vector3::Zero(); 

    // Solvers
    LidarSolverType solver_type_ = LidarSolverType::SVN_NDT;
    
    std::unique_ptr<svn_ndt::SvnNormalDistributionsTransform<PointT, PointT>> svn_ndt_ptr_;
    std::unique_ptr<svn_icp::SvnIterativeClosestPoint<PointT, PointT>> svn_icp_ptr_;
    std::unique_ptr<ndt_generic::NormalDistributionsTransform<PointT, PointT>> ndt_generic_ptr_;

    // Map Management
    std::unique_ptr<VoxelMap<PointT>> global_map_;
    PointCloudPtr map_cloud_;
    
    double map_leaf_size_ = 0.5;   
    double map_radius_ = 100.0;    
    int map_update_skip_ = 1;      
    double ref_lat_ = 0.0, ref_lon_ = 0.0, ref_alt_ = 0.0; 

    // Logging
    bool enable_disk_logging_ = true;
    std::string data_dir_ = "lo_data";
    std::vector<Keyframe> keyframes_;
    int keyframe_skip_ = 1;

    double last_alingment_time_ms_ = 0.0;
    Eigen::Matrix<double, 6, 6> last_measured_cov_ = Eigen::Matrix<double, 6, 6>::Identity();
};

} // namespace lo_smoother

#include "lo_smoother_impl.hpp"

#endif // LO_SMOOTHER_H_