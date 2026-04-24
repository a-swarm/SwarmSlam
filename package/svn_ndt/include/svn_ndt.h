#ifndef SVN_NDT_H_
#define SVN_NDT_H_

// --- Standard/External Library Includes ---
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

// --- GTSAM Includes ---
#include <gtsam/geometry/Pose3.h> // For representing poses on SE(3) manifold

// --- PCL Includes ---
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// --- Custom Voxel Grid Include ---
#include <svn_voxel_grid_covariance.h> 

namespace svn_ndt
{

enum class NeighborSearchMethod
{   
    KDTREE,  
    DIRECT27,
    DIRECT7, 
    DIRECT1  
};

struct SvnNdtResult
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    gtsam::Pose3 final_pose;
    Eigen::Matrix<double, 6, 6> final_covariance;  
    bool converged = false;          
    int iterations = 0;   
    double ksd_score = 0.0;

    SvnNdtResult() : final_pose(gtsam::Pose3()), final_covariance(Eigen::Matrix<double, 6, 6>::Zero()) {}
};

template <typename PointSource, typename PointTarget>
class SvnNormalDistributionsTransform
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    using PointCloudSource = pcl::PointCloud<PointSource>;
    using PointCloudSourcePtr = typename PointCloudSource::Ptr;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
    using PointCloudTarget = pcl::PointCloud<PointTarget>;
    using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using TargetGrid = svn_ndt::VoxelGridCovariance<PointTarget>;
    using TargetGridLeafConstPtr = typename TargetGrid::LeafConstPtr;

    SvnNormalDistributionsTransform(const std::string& json_path);
    virtual ~SvnNormalDistributionsTransform() = default; 

    void setInputTarget(const PointCloudTargetConstPtr& cloud);
    void setMinPointPerVoxel(int min_points);
    void setResolution(float resolution);
    float getResolution() const { return resolution_; }

    void setNeighborhoodSearchMethod(NeighborSearchMethod method) { search_method_ = method; }
    NeighborSearchMethod getNeighborhoodSearchMethod() const { return search_method_; }

    void setNumThreads(int num_threads) { 
        num_threads_ = (num_threads > 0) ? num_threads : 1;
        target_cells_.setNumThreads(num_threads_);  
    }
    int getNumThreads() const { return num_threads_; }

    void setParticleCount(int k) { K_ = (k > 1) ? k : 1; }
    int getParticleCount() const { return K_; }

    void setMaxIterations(int max_iter) { max_iter_ = (max_iter > 0) ? max_iter : 1; }
    int getMaxIterations() const { return max_iter_; }

    void setKernelBandwidth(double kernel_h) { kernel_h_ = (kernel_h > 1e-9) ? kernel_h : 1e-9; }
    double getKernelBandwidth() const { return kernel_h_; }

    void setStepSize(double step_size_trans, double step_size_rot) { 
        step_size_trans_ = (step_size_trans > 0) ? step_size_trans : 1e-6; 
        step_size_rot_   = (step_size_rot > 0) ? step_size_rot : 1e-6;
    }

    void setTransformationEpsilon(double trans_eps, double rot_eps) { 
        trans_eps_ = (trans_eps >= 0) ? trans_eps : 1e-9;
        rot_eps_   = (rot_eps >= 0) ? rot_eps : 1e-9;
    }

    void setOutlierRatio(double outlier_ratio); 
    double getOutlierRatio() const { return outlier_ratio_; }

    void setUseGaussNewtonHessian(bool use_gn) { use_gauss_newton_hessian_ = use_gn; }
    void setInitialParticleSpread(const Vector6d& initial_particle_sigmas) {initial_particle_sigmas_ = initial_particle_sigmas;}
    void setParticleInflation(const Vector6d& inflation_sigmas) {inflation_sigmas_ = inflation_sigmas;}
    void setEnableWarmStart(bool enable_warm_start) { enable_warm_start_ = enable_warm_start; }

    SvnNdtResult align(const PointCloudSource& source_cloud, const gtsam::Pose3& prior_mean);

protected: 

    // --- UPDATED SIGNATURE: Added particle_idx and inner_threads ---
    double computeParticleDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const gtsam::Pose3& pose,
        bool compute_hessian,
        int particle_idx,
        int inner_threads);

    double updateDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const Eigen::Matrix<double, 3, 6>& point_jacobian, 
        const Eigen::Vector3d& x_trans,
        const Eigen::Matrix3d& c_inv,
        bool compute_hessian = true,
        bool use_gauss_newton_hessian = true) const;

    void computePointDerivatives(
        const Eigen::Vector3d& p_rot,
        const Eigen::Matrix3d& R,                  
        Eigen::Matrix<double, 3, 6>& point_jacobian) const;

    double rbf_kernel(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const; 
    Vector6d rbf_kernel_gradient(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const; 
    void updateNdtConstants();

protected: 

    // --- UPDATED BUFFERS: Converted to 2D vectors [particle_idx][point_idx] ---
    std::vector<std::vector<double>> scores_buffer_;
    std::vector<std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>>> score_gradients_buffer_;
    std::vector<std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>>> hessians_buffer_;

    TargetGrid target_cells_;                       
    float resolution_ = 1.0f;                       
    double outlier_ratio_ = 0.55;                   
    double gauss_d1_{}, gauss_d2_{}, gauss_d3_{};   

    Eigen::Matrix<float, 8, 4> j_ang_;              
    Eigen::Matrix<float, 16, 4> h_ang_;             

    NeighborSearchMethod search_method_ = NeighborSearchMethod::DIRECT7; 
    int num_threads_ = 1;                           

    int K_ = 30;                 
    int max_iter_ = 50;          
    double kernel_h_ = 0.1;      
    double step_size_trans_ = 0.1;
    double step_size_rot_ = 0.05;
    double trans_eps_ = 0.001;
    double rot_eps_ = 0.001;

    Matrix6d last_covariance_ = Matrix6d::Identity();
    bool has_last_covariance_ = false;
    bool enable_warm_start_ = false;

    Vector6d initial_particle_sigmas_ = (Vector6d() << 0.05, 0.05, 0.08, 0.1, 0.1, 0.1).finished();
    Vector6d inflation_sigmas_ = (Vector6d() << 0.01, 0.01, 0.01, 0.02, 0.02, 0.02).finished();
    PointCloudSourceConstPtr input_; 
    bool use_gauss_newton_hessian_ = true; 

}; 

} // namespace svn_ndt

#include <svn_ndt_impl.hpp>

#endif // SVN_NDT_H_