#ifndef SVN_ICP_IMPL_HPP_
#define SVN_ICP_IMPL_HPP_

#include <svn_icp.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky> 
#include <pcl/common/point_tests.h> 
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/linear/Sampler.h>
#include <gtsam/linear/NoiseModel.h>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_max_active_levels(int) {} // Stub
#endif

namespace svn_icp
{

template <typename PointSource, typename PointTarget>
SvnIterativeClosestPoint<PointSource, PointTarget>::SvnIterativeClosestPoint(const std::string& json_path)
    : kdtree_()
{
    // [Unchanged Constructor Logic]
    int K = 5;
    int max_iter = 30;
    double kernel_h = 1.0;
    double step_size_trans = 1.0;
    double step_size_rot = 1.0;
    double trans_eps = 1e-4;
    double rot_eps = 1e-4;
    double max_corr_dist = 1.0;
    int num_threads = 1;
    Vector6d initial_particle_sigmas = Vector6d::Ones();
    Vector6d inflation_sigmas = Vector6d::Ones();
    bool enable_warm_start = false;

    std::ifstream file(json_path);
    if (!file.is_open()) throw std::runtime_error("Failed to open JSON file: " + json_path);
    nlohmann::json json_data;
    file >> json_data;

    const auto& icp_param = json_data["icp"];
    max_corr_dist = icp_param.value("max_correspondence_distance", 1.0);
    K = icp_param.value("K", 5);
    max_iter = icp_param.value("max_iter", 30);
    kernel_h = icp_param.value("kernel_h", 1.0);
    step_size_trans = icp_param.value("step_size_trans", 1.0);
    step_size_rot = icp_param.value("step_size_rot", 1.0);
    trans_eps = icp_param.value("trans_eps", 1e-4);
    rot_eps = icp_param.value("rot_eps", 1e-4);
    enable_warm_start = icp_param.value("enable_warm_start", false);

    if (icp_param.contains("initial_particle_sigmas")) {
        auto arr = icp_param["initial_particle_sigmas"];
        initial_particle_sigmas << arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>(), arr[3].get<double>(), arr[4].get<double>(), arr[5].get<double>();
    }
    if (icp_param.contains("inflation_sigmas")) {
        auto arr = icp_param["inflation_sigmas"];
        inflation_sigmas << arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>(), arr[3].get<double>(), arr[4].get<double>(), arr[5].get<double>();
    }
    num_threads = json_data.value("num_threads", 16);

    setNumThreads(num_threads);
    setParticleCount(K);
    setMaxIterations(max_iter);
    setKernelBandwidth(kernel_h);
    setStepSize(step_size_trans, step_size_rot);
    setTransformationEpsilon(trans_eps, rot_eps);
    setInitialParticleSpread(initial_particle_sigmas);
    setParticleInflation(inflation_sigmas);
    setMaxCorrespondenceDistance(max_corr_dist);
    setEnableWarmStart(enable_warm_start);
}

template <typename PointSource, typename PointTarget>
void SvnIterativeClosestPoint<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
    if (!cloud || cloud->empty()) return;
    target_cloud_ = cloud;
    kdtree_.setInputCloud(target_cloud_);
}

template <typename PointSource, typename PointTarget>
void SvnIterativeClosestPoint<PointSource, PointTarget>::setMaxCorrespondenceDistance(double dist) {
    max_corr_dist_ = dist;
    max_corr_dist_sq_ = dist * dist;
}

template <typename PointSource, typename PointTarget>
double SvnIterativeClosestPoint<PointSource, PointTarget>::rbf_kernel(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    return std::exp(-diff_log.squaredNorm() / kernel_h_);
}

template <typename PointSource, typename PointTarget>
typename SvnIterativeClosestPoint<PointSource, PointTarget>::Vector6d
SvnIterativeClosestPoint<PointSource, PointTarget>::rbf_kernel_gradient(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
    if (kernel_h_ <= 1e-12) return Vector6d::Zero();
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double k_val = std::exp(-diff_log.squaredNorm() / kernel_h_);
    return k_val * (2.0 / kernel_h_) * diff_log; 
}

template <typename PointSource, typename PointTarget>
void SvnIterativeClosestPoint<PointSource, PointTarget>::computePointDerivatives(
    const Eigen::Vector3d& p_rot, 
    const Eigen::Matrix3d& R,     
    Eigen::Matrix<double, 3, 6>& point_jacobian) const 
{
    double x = p_rot[0], y = p_rot[1], z = p_rot[2];
    Eigen::Matrix3d p_cross;
    p_cross <<  0.0, -z,   y,
                z,   0.0, -x,
               -y,   x,    0.0;
    point_jacobian.block<3,3>(0,0) = -p_cross * R;
    point_jacobian.block<3,3>(0,3) = R;
}

template <typename PointSource, typename PointTarget>
void SvnIterativeClosestPoint<PointSource, PointTarget>::updateDerivatives(
    Vector6d& score_gradient, Matrix6d& hessian,        
    const Eigen::Matrix<double, 3, 6>& point_jacobian, 
    const Eigen::Vector3d& error_vector) const 
{
    score_gradient += point_jacobian.transpose() * error_vector;
    hessian += point_jacobian.transpose() * point_jacobian;
}

// --- UPDATED: computeParticleDerivatives using 2D isolated memory buffers ---
template <typename PointSource, typename PointTarget>
double SvnIterativeClosestPoint<PointSource, PointTarget>::computeParticleDerivatives(
    Vector6d& final_score_gradient, Matrix6d& final_hessian,        
    const gtsam::Pose3& pose, int particle_idx, int inner_threads) 
{
    const size_t num_points = input_->size();
    if (num_points == 0) return 0.0;

    // Isolate this particle's specific buffer row
    if (scores_buffer_[particle_idx].size() < num_points) {
        scores_buffer_[particle_idx].resize(num_points);
        score_gradients_buffer_[particle_idx].resize(num_points);
        hessians_buffer_[particle_idx].resize(num_points);
    }

    Eigen::Matrix4d p_transform = pose.matrix(); 
    Eigen::Matrix3d R = pose.rotation().matrix(); 

    // Inner loop thread execution using passed 'inner_threads' variable
    #pragma omp parallel for num_threads(inner_threads) schedule(guided)
    for (size_t idx = 0; idx < num_points; ++idx) {
        
        const PointSource& src_pt = (*input_)[idx];
        if (!pcl::isFinite(src_pt)) continue;

        double rx = R(0,0)*src_pt.x + R(0,1)*src_pt.y + R(0,2)*src_pt.z;
        double ry = R(1,0)*src_pt.x + R(1,1)*src_pt.y + R(1,2)*src_pt.z;
        double rz = R(2,0)*src_pt.x + R(2,1)*src_pt.y + R(2,2)*src_pt.z;

        double tx = rx + p_transform(0,3);
        double ty = ry + p_transform(1,3);
        double tz = rz + p_transform(2,3);

        PointTarget search_pt;
        search_pt.x = static_cast<float>(tx);
        search_pt.y = static_cast<float>(ty);
        search_pt.z = static_cast<float>(tz);

        // Local vectors for thread safety during KD-Tree query
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        
        int found = kdtree_.nearestKSearch(search_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance);

        if (found > 0 && pointNKNSquaredDistance[0] < max_corr_dist_sq_) {
            
            const PointTarget& target_pt = (*target_cloud_)[pointIdxNKNSearch[0]];

            Eigen::Vector3d error_vector;
            error_vector << (tx - target_pt.x), (ty - target_pt.y), (tz - target_pt.z);

            Eigen::Vector3d p_rot(rx, ry, rz);
            Eigen::Matrix<double, 3, 6> point_jacobian;
            this->computePointDerivatives(p_rot, R, point_jacobian);

            Vector6d pt_grad = Vector6d::Zero();
            Matrix6d pt_hess = Matrix6d::Zero();

            this->updateDerivatives(pt_grad, pt_hess, point_jacobian, error_vector);

            // Write exclusively to this particle's row
            scores_buffer_[particle_idx][idx] = 0.5 * pointNKNSquaredDistance[0]; // MSE contribution
            score_gradients_buffer_[particle_idx][idx] = pt_grad;
            hessians_buffer_[particle_idx][idx] = pt_hess;
        } else {
            scores_buffer_[particle_idx][idx] = 0.0;
            score_gradients_buffer_[particle_idx][idx].setZero();
            hessians_buffer_[particle_idx][idx].setZero();
        }
    }

    // Accumulate results
    final_score_gradient.setZero();
    final_hessian.setZero();
    double final_total_score = 0.0;
    
    for (size_t i = 0; i < num_points; ++i) {
        final_total_score += scores_buffer_[particle_idx][i];
        final_score_gradient += score_gradients_buffer_[particle_idx][i];
        final_hessian += hessians_buffer_[particle_idx][i];
    }
    
    final_hessian += 1e-6 * Matrix6d::Identity();
    return final_total_score;
}

template <typename PointSource, typename PointTarget>
SvnIcpResult SvnIterativeClosestPoint<PointSource, PointTarget>::align(
    const PointCloudSource& source_cloud, const gtsam::Pose3& prior_mean)
{
    SvnIcpResult result;
    result.ksd_score = 0.0;

    if (!target_cloud_ || target_cloud_->empty() || source_cloud.empty()) {
        result.final_pose = prior_mean;
        result.final_covariance.setIdentity();
        return result;
    }

    input_ = source_cloud.makeShared();

    // =========================================================================
    // 0. NESTED OPENMP & BUFFER SETUP
    // =========================================================================
    
    // Explicitly allow OpenMP to run teams within teams
    #ifdef _OPENMP
    omp_set_max_active_levels(2); 
    #endif

    // Resize outer particle bounds for buffers
    if (scores_buffer_.size() < K_) {
        scores_buffer_.resize(K_);
        score_gradients_buffer_.resize(K_);
        hessians_buffer_.resize(K_);
    }

    // Dynamic Thread Splitting logic based on dedicated P-Cores
    int T_avail = num_threads_; 
    int outer_threads = std::min(K_, T_avail);
    int inner_threads = std::max(1, T_avail / outer_threads);

    // =========================================================================
    // 1. INITIALIZE PARTICLES
    // =========================================================================
    std::vector<gtsam::Pose3, Eigen::aligned_allocator<gtsam::Pose3>> particles(K_);
    Matrix6d sampling_cov; 
    bool using_warm_start = false;

    if (enable_warm_start_ && has_last_covariance_) {
        Vector6d proc_noise_var = inflation_sigmas_.array().square();
        sampling_cov = last_covariance_;
        sampling_cov.diagonal() += proc_noise_var; 
        using_warm_start = true;
    } else {
        sampling_cov = initial_particle_sigmas_.array().square().matrix().asDiagonal();
    }

    Eigen::LLT<Matrix6d> llt(sampling_cov);
    Matrix6d L;

    if (llt.info() == Eigen::Success) {
        L = llt.matrixL();
    } else {
        if (using_warm_start) {
             std::cerr << "[SVN-ICP] Warning: Warm start covariance invalid. Reverting to Cold Start." << std::endl;
        }
        sampling_cov = initial_particle_sigmas_.array().square().matrix().asDiagonal();
        L = sampling_cov.llt().matrixL();
    }

    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    particles[0] = prior_mean;

    for (int k = 1; k < K_; ++k) {
        Vector6d u;
        for (int i=0; i<6; ++i) u(i) = norm_dist(rng);
        Vector6d perturbation = L * u;
        particles[k] = prior_mean.retract(perturbation);
    }

    gtsam::Pose3 mean_pose_current = prior_mean;
    gtsam::Pose3 mean_pose_last_iter;

    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> loss_gradients(K_);
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> loss_hessians(K_);
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> particle_updates(K_);
    Matrix6d I6 = Matrix6d::Identity();

    // =========================================================================
    // 3. OPTIMIZATION LOOP
    // =========================================================================
    for (int iter = 0; iter < max_iter_; ++iter)
    {
        mean_pose_last_iter = mean_pose_current;
        double iter_ksd_accumulator = 0.0;

        // A. Compute Derivatives
        // Outer loop pinned to 'outer_threads', schedule static for uniform speed
        #pragma omp parallel for num_threads(outer_threads) schedule(static)
        for (int k = 0; k < K_; ++k) {
            computeParticleDerivatives(loss_gradients[k], loss_hessians[k], particles[k], k, inner_threads);
        }

        // B. Stein Variational Update
        #pragma omp parallel for num_threads(outer_threads) schedule(static)
        for (int k = 0; k < K_; ++k) {
            Vector6d phi_k_star = Vector6d::Zero();
            Matrix6d H_k_tilde = Matrix6d::Zero();

            for (int l = 0; l < K_; ++l) {
                double k_val = rbf_kernel(particles[l], particles[k]);
                Vector6d k_grad_l = rbf_kernel_gradient(particles[l], particles[k]);
                Vector6d force_in_l = k_val * loss_gradients[l] + k_grad_l;

                gtsam::Pose3 T_kl = particles[l].between(particles[k]);
                Matrix6d Adj_kl = T_kl.AdjointMap();
                Matrix6d Adj_kl_T = Adj_kl.transpose(); 

                phi_k_star += k_val * (Adj_kl_T * force_in_l);
                H_k_tilde += (k_val * k_val) * (Adj_kl_T * loss_hessians[l] * Adj_kl);
            }

            if (K_ > 0) {
                phi_k_star /= K_;
                H_k_tilde /= K_;
            }

            double phi_norm = phi_k_star.norm();
            #pragma omp atomic
            iter_ksd_accumulator += phi_norm;

            H_k_tilde += 1e-8 * I6; 

            Eigen::LDLT<Matrix6d> solver(H_k_tilde);
            if (solver.info() == Eigen::Success) {
                particle_updates[k] = solver.solve(-phi_k_star);
            } else {
                particle_updates[k].setZero();
            }
        }

        if (input_->size() > 0 && K_ > 0) {
            result.ksd_score = (iter_ksd_accumulator / K_) / static_cast<double>(input_->size());
        }

        // C. Apply Updates
        Vector6d mean_xi_sum = Vector6d::Zero();
        for (int k = 0; k < K_; ++k) {
            if (particle_updates[k].allFinite()) {
                Vector6d delta = particle_updates[k];
                
                double norm_rot = delta.head<3>().norm();
                double norm_trans = delta.tail<3>().norm();

                double scale_rot = (norm_rot > step_size_rot_) ? step_size_rot_ / norm_rot : 1.0;
                double scale_trans = (norm_trans > step_size_trans_) ? step_size_trans_ / norm_trans : 1.0;
                double final_scale = std::min(scale_rot, scale_trans);

                if (final_scale < 1.0) delta *= final_scale;
                particles[k] = particles[k].compose(gtsam::Pose3::Expmap(delta));
            }
            mean_xi_sum += gtsam::Pose3::Logmap(prior_mean.between(particles[k]));
        }

        if (K_ > 0) mean_pose_current = prior_mean.retract(mean_xi_sum / K_);
        
        result.iterations = iter + 1;

        // E. Convergence Check
        Vector6d mean_diff = gtsam::Pose3::Logmap(mean_pose_last_iter.between(mean_pose_current));
        if (mean_diff.tail<3>().norm() < trans_eps_ && mean_diff.head<3>().norm() < rot_eps_) {
            result.converged = true;
            break;
        }
    }

    // =========================================================================
    // 4. FINALIZE & SAVE COVARIANCE
    // =========================================================================
    result.final_pose = mean_pose_current;

    if (K_ > 1) {
        result.final_covariance.setZero();
        Vector6d mean_xi = Vector6d::Zero();
        std::vector<Vector6d> tangents(K_);
        for(int k=0; k<K_; ++k) {
            tangents[k] = gtsam::Pose3::Logmap(result.final_pose.between(particles[k]));
            mean_xi += tangents[k];
        }
        mean_xi /= K_;
        for(int k=0; k<K_; ++k) {
            Vector6d diff = tangents[k] - mean_xi;
            result.final_covariance += diff * diff.transpose();
        }
        result.final_covariance /= (K_ - 1);
    } else {
        result.final_covariance = initial_particle_sigmas_.array().square().matrix().asDiagonal();
    }
    result.final_covariance = 0.5 * (result.final_covariance + result.final_covariance.transpose());
    result.final_covariance.diagonal().array() += 1e-6;

    // --- SAVE STATE ---
    last_covariance_ = result.final_covariance;
    has_last_covariance_ = true;

    input_.reset();
    return result;
}

} // namespace svn_icp

#endif // SVN_ICP_IMPL_HPP_