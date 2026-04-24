#ifndef SVN_ICP_H_
#define SVN_ICP_H_

// --- Standard/External Library Includes ---
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

// --- GTSAM Includes ---
#include <gtsam/geometry/Pose3.h> 

// --- PCL Includes ---
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h> // Replaces VoxelGridCovariance

namespace svn_icp
{

/**
 * @brief Structure to hold the results of the SVN-ICP alignment process.
 */
struct SvnIcpResult
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    gtsam::Pose3 final_pose;
    Eigen::Matrix<double, 6, 6> final_covariance;  
    bool converged = false;          
    int iterations = 0;   
    double ksd_score = 0.0;
    double fitness_score = 0.0; // Mean Squared Error

    SvnIcpResult() : final_pose(gtsam::Pose3()), final_covariance(Eigen::Matrix<double, 6, 6>::Zero()) {}
};

/**
 * @brief Stein Variational Newton (SVN) implementation for ICP Registration.
 *
 * @details
 * Approximates the posterior p(T|Z) using particles optimized via SVGD.
 * Replaces the NDT Log-Likelihood with the ICP Least Squares objective:
 * \f[ E(T) = \sum \| T \cdot p_s - p_{target} \|^2 \f]
 */
template <typename PointSource, typename PointTarget>
class SvnIterativeClosestPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    // --- Type Aliases ---
    using PointCloudSource = pcl::PointCloud<PointSource>;
    using PointCloudSourcePtr = typename PointCloudSource::Ptr;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
    using PointCloudTarget = pcl::PointCloud<PointTarget>;
    using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using KdTree = pcl::KdTreeFLANN<PointTarget>;

    // --- Constructor / Destructor ---
    SvnIterativeClosestPoint(const std::string& json_path);
    virtual ~SvnIterativeClosestPoint() = default; 

    // --- Core ICP Setup ---
    void setInputTarget(const PointCloudTargetConstPtr& cloud);

    /** @brief Sets the max distance for a valid point correspondence. */
    void setMaxCorrespondenceDistance(double dist);
    double getMaxCorrespondenceDistance() const { return max_corr_dist_; }

    // --- Parallelism Control ---
    void setNumThreads(int num_threads) { num_threads_ = (num_threads > 0) ? num_threads : 1; }
    int getNumThreads() const { return num_threads_; }

    // --- SVN Hyperparameters (Same as NDT) ---
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

    void setInitialParticleSpread(const Vector6d& initial_particle_sigmas) { initial_particle_sigmas_ = initial_particle_sigmas; }

    void setParticleInflation(const Vector6d& inflation_sigmas) {inflation_sigmas_ = inflation_sigmas;}

    void setEnableWarmStart(bool enable_warm_start) { enable_warm_start_ = enable_warm_start; }
    // --- Main Alignment Function ---
    SvnIcpResult align(const PointCloudSource& source_cloud, const gtsam::Pose3& prior_mean);

protected:

    // --- UPDATED SIGNATURE: Added particle_idx and inner_threads ---
    double computeParticleDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const gtsam::Pose3& pose,
        int particle_idx,
        int inner_threads);

    /**
     * @brief Computes the Point-to-Point Error Gradient and Hessian.
     */
    void updateDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const Eigen::Matrix<double, 3, 6>& point_jacobian, 
        const Eigen::Vector3d& error_vector) const;

    /** @brief Computes Left-Invariant Jacobian (Same as NDT version). */
    void computePointDerivatives(
        const Eigen::Vector3d& p_rot,
        const Eigen::Matrix3d& R,                  
        Eigen::Matrix<double, 3, 6>& point_jacobian) const;

    // --- SVN Helper Functions ---
    double rbf_kernel(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const;
    Vector6d rbf_kernel_gradient(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const;

    // --- Member Variables ---
protected: 
    // --- UPDATED BUFFERS: Converted to 2D vectors [particle_idx][point_idx] ---
    std::vector<std::vector<double>> scores_buffer_;
    std::vector<std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>>> score_gradients_buffer_;
    std::vector<std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>>> hessians_buffer_;

    // ICP State
    KdTree kdtree_;                                 //!< KD-Tree for NN search
    PointCloudTargetConstPtr target_cloud_;         //!< Target cloud ptr
    double max_corr_dist_ = 1.0;                    //!< Max correspondence distance
    double max_corr_dist_sq_ = 1.0;                 //!< Squared max distance (cache)

    // Configuration
    int num_threads_ = 1;

    // SVN Hyperparameters
    int K_ = 30;
    int max_iter_ = 50;
    double kernel_h_ = 0.1;
    double step_size_trans_ = 0.1;
    double step_size_rot_ = 0.05;
    double trans_eps_ = 0.001;
    double rot_eps_ = 0.001;

    // --- ADAPTIVE UNCERTAINTY VARS ---
    Matrix6d last_covariance_ = Matrix6d::Identity();
    bool has_last_covariance_ = false;
    bool enable_warm_start_ = false;

    Vector6d initial_particle_sigmas_ = (Vector6d() << 0.05, 0.05, 0.08, 0.1, 0.1, 0.1).finished();

    Vector6d inflation_sigmas_ = (Vector6d() << 0.01, 0.01, 0.01, 0.02, 0.02, 0.02).finished();

    PointCloudSourceConstPtr input_; 
    
}; 

} // namespace svn_icp

#include <svn_icp_impl.hpp>

#endif // SVN_ICP_H_