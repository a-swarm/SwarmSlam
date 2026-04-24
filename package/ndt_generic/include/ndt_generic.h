#ifndef NDT_GENERIC_H_
#define NDT_GENERIC_H_

#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <memory>
#include <atomic>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Matrix.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ndt_voxel_grid_covariance.h> 

namespace ndt_generic
{

enum class NeighborSearchMethod { KDTREE, DIRECT27, DIRECT7, DIRECT1 };
enum class OptimizationMethod { BACKTRACKING_ARMIJO, LEVENBERG_MARQUARDT };
enum class HessianForm{ FULL_NEWTON, GAUSS_NEWTON};

struct NdtResult
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    gtsam::Pose3 final_pose;
    bool converged = false;
    int iterations = 0;
    Eigen::Matrix<double, 6, 6> hessian;
    Eigen::Matrix<double, 6, 6> final_covariance;

    NdtResult()
        : final_pose(gtsam::Pose3()),
          hessian(Eigen::Matrix<double, 6, 6>::Identity()),
          final_covariance(Eigen::Matrix<double, 6, 6>::Identity()) {}
};

template <typename PointSource, typename PointTarget>
class NormalDistributionsTransform
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using PointCloudSource = pcl::PointCloud<PointSource>;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
    using PointCloudTarget = pcl::PointCloud<PointTarget>;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using TargetGrid = ndt_generic::VoxelGridCovariance<PointTarget>;
    using TargetGridLeafConstPtr = typename TargetGrid::LeafConstPtr;

    NormalDistributionsTransform(const std::string& json_path);
    virtual ~NormalDistributionsTransform() = default;

    // --- Configuration ---
    void setMinPointPerVoxel(int min_points);
    void setResolution(float resolution);
    void setStepSize(double step_size_trans, double step_size_rot) { step_size_trans_ = step_size_trans; step_size_rot_ = step_size_rot;}
    void setTransformationEpsilon(double trans_eps, double rot_eps) { trans_eps_ = trans_eps; rot_eps_ = rot_eps; }
    void setMaxIterations(int max_iter) { max_iter_ = max_iter; }
    void setOutlierRatio(double ratio);
    void setInitialCovariance(const Vector6d& initial_covariance) {initial_covariance_ = initial_covariance;}
    
    void setNumThreads(int n);
    
    void setNeighborhoodSearchMethod(NeighborSearchMethod method) { search_method_ = method; }
    void setOptimizationMethod(OptimizationMethod method) { opt_method_ = method; }
    void setHessianForm(HessianForm method) { hessian_form_ = method; }

    void setInputTarget(const PointCloudTargetConstPtr& cloud);
    NdtResult align(const PointCloudSource& source_cloud, const gtsam::Pose3& prior_pose);

    float getResolution() const { return resolution_; }

private:
    double computeDerivatives(Vector6d& score_gradient, Matrix6d& hessian,
                              const PointCloudSource& source_cloud,
                              const gtsam::Pose3& transform);

    inline double updateDerivatives(Vector6d& score_gradient, Matrix6d& hessian,
                             const Eigen::Vector3d& x_trans,
                             const Eigen::Matrix3d& c_inv,
                             const Eigen::Matrix<double, 3, 6>& point_jacobian) const;

    double calculateScore(const PointCloudSource& source_cloud, const gtsam::Pose3& transform);

    void updateNdtConstants();

    // --- State Variables ---
    TargetGrid target_cells_;
    PointCloudSourceConstPtr input_;

    float resolution_ = 1.0f;                       
    double step_size_trans_ = 0.1;
    double step_size_rot_ = 0.05;
    double outlier_ratio_ = 0.55;
    double trans_eps_ = 0.001;
    double rot_eps_ = 0.001;
    int max_iter_ = 50;
    int num_threads_ = 1; 
    
    NeighborSearchMethod search_method_ = NeighborSearchMethod::DIRECT7;
    OptimizationMethod opt_method_ = OptimizationMethod::LEVENBERG_MARQUARDT;
    HessianForm hessian_form_ = HessianForm::FULL_NEWTON;

    double gauss_d1_{}, gauss_d2_{}, gauss_d3_{};   

    // --- ADAPTIVE UNCERTAINTY VARS ---
    Matrix6d last_covariance_ = Matrix6d::Identity();
    bool has_last_covariance_ = false;

    // Initial particle spread [rot_x, rot_y, rot_z, x, y, z]
    Vector6d initial_covariance_ = (Vector6d() << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished();
};

} // namespace ndt_generic

#include "ndt_generic_impl.hpp"

#endif // NDT_GENERIC_H_