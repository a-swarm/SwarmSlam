#ifndef SVN_NDT_IMPL_HPP_
#define SVN_NDT_IMPL_HPP_

#include <svn_ndt.h>
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
#include <Eigen/SVD>      
#include <pcl/common/transforms.h>
#include <pcl/common/point_tests.h> 
#include <pcl/io/pcd_io.h> 
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

namespace svn_ndt
{

inline double fast_exp_neg(double x)
{
    if (x > 10.0) return 0.0; 
    return std::exp(-x);
}

template <typename PointSource, typename PointTarget>
SvnNormalDistributionsTransform<PointSource, PointTarget>::SvnNormalDistributionsTransform(const std::string& json_path)
    : target_cells_()
{
    // [Unchanged Constructor Parsing Logic]
    int min_points = 3;
    float resolution = 1.0f;
    double outlier_ratio = 0.55;
    int K = 5;
    int max_iter = 30;
    double kernel_h = 1.0;
    double step_size_trans = 1.0;
    double step_size_rot = 1.0;
    double trans_eps = 1e-4;
    double rot_eps = 1e-4;
    std::string search_method = "DIRECT7";
    bool use_gauss_newton_hessian = true;
    int num_threads = 1;
    Vector6d initial_particle_sigmas = Vector6d::Ones();
    Vector6d inflation_sigmas = Vector6d::Ones();
    bool enable_warm_start = false;

    std::ifstream file(json_path);
    if (!file.is_open()) throw std::runtime_error("Failed to open JSON file: " + json_path);
    nlohmann::json json_data;
    file >> json_data;

    const auto& ndt_param = json_data["ndt"];
    min_points = ndt_param.value("min_points", 3);
    resolution = ndt_param.value("resolution", 1.0f);
    outlier_ratio = ndt_param.value("outlier_ratio", 0.55);
    K = ndt_param.value("K", 5);
    max_iter = ndt_param.value("max_iter", 30);
    kernel_h = ndt_param.value("kernel_h", 1.0);
    step_size_trans = ndt_param.value("step_size_trans", 1.0);
    step_size_rot = ndt_param.value("step_size_rot", 1.0);
    trans_eps = ndt_param.value("trans_eps", 1e-4);
    rot_eps = ndt_param.value("rot_eps", 1e-4);
    search_method = ndt_param.value("search_method", "DIRECT7");
    use_gauss_newton_hessian = ndt_param.value("use_gauss_newton_hessian", true);
    enable_warm_start = ndt_param.value("enable_warm_start", false);
    
    if (ndt_param.contains("initial_particle_sigmas")) {
        auto arr = ndt_param["initial_particle_sigmas"];
        initial_particle_sigmas << arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>(), arr[3].get<double>(), arr[4].get<double>(), arr[5].get<double>();
    }
    if (ndt_param.contains("inflation_sigmas")) {
        auto arr = ndt_param["inflation_sigmas"];
        inflation_sigmas << arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>(), arr[3].get<double>(), arr[4].get<double>(), arr[5].get<double>();
    }
    num_threads = json_data.value("num_threads", 16);

    setNumThreads(num_threads);
    setMinPointPerVoxel(min_points);
    setResolution(resolution);
    setParticleCount(K);
    setMaxIterations(max_iter);
    setKernelBandwidth(kernel_h);
    setStepSize(step_size_trans, step_size_rot);
    setTransformationEpsilon(trans_eps, rot_eps);
    setOutlierRatio(outlier_ratio);
    setInitialParticleSpread(initial_particle_sigmas);
    setParticleInflation(inflation_sigmas);
    setUseGaussNewtonHessian(use_gauss_newton_hessian);
    setEnableWarmStart(enable_warm_start);

    if (search_method == "DIRECT1") setNeighborhoodSearchMethod(NeighborSearchMethod::DIRECT1);
    else if (search_method == "DIRECT27") setNeighborhoodSearchMethod(NeighborSearchMethod::DIRECT27);
    else if (search_method == "KDTREE") setNeighborhoodSearchMethod(NeighborSearchMethod::KDTREE);
    else setNeighborhoodSearchMethod(NeighborSearchMethod::DIRECT7);

    updateNdtConstants(); 
    j_ang_.setZero();
    h_ang_.setZero();
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::updateNdtConstants()
{
    if (resolution_ <= 1e-6f) {
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
        return;
    }
    double gauss_c1 = 10.0 * (1.0 - outlier_ratio_);
    double gauss_c2 = outlier_ratio_ / pow(static_cast<double>(resolution_), 3); 
    double epsilon = 1e-9;
    if (gauss_c1 <= epsilon) gauss_c1 = epsilon;
    if (gauss_c2 <= epsilon) gauss_c2 = epsilon;

    gauss_d3_ = -log(gauss_c2);
    gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3_; 
    gauss_d2_ = -2.0 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3_) / gauss_d1_);
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
    if (!cloud || cloud->empty()) return;
    target_cells_.setInputCloud(cloud);
    if (resolution_ > 1e-6f) {
        target_cells_.setLeafSize(resolution_, resolution_, resolution_);
        target_cells_.filter(search_method_ == NeighborSearchMethod::KDTREE); 
        updateNdtConstants();
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setResolution(float resolution) {
    if (resolution <= 1e-6f) return;
    if (std::abs(resolution_ - resolution) > 1e-6f) {
        resolution_ = resolution;
        if (target_cells_.getInputCloud()) setInputTarget(target_cells_.getInputCloud());
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setMinPointPerVoxel(int min_points) {
    target_cells_.setMinPointPerVoxel(min_points);
    if (target_cells_.getInputCloud()) setInputTarget(target_cells_.getInputCloud());
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setOutlierRatio(double ratio) {
    outlier_ratio_ = ratio;
    updateNdtConstants();
}

// --- Kernel Functions ---

template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    return std::exp(-diff_log.squaredNorm() / kernel_h_);
}

template <typename PointSource, typename PointTarget>
typename SvnNormalDistributionsTransform<PointSource, PointTarget>::Vector6d
SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel_gradient(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
     if (kernel_h_ <= 1e-12) return Vector6d::Zero();
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double k_val = std::exp(-diff_log.squaredNorm() / kernel_h_);
    return k_val * (2.0 / kernel_h_) * diff_log; 
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computePointDerivatives(
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
double SvnNormalDistributionsTransform<PointSource, PointTarget>::updateDerivatives(
    Vector6d& score_gradient, Matrix6d& hessian,        
    const Eigen::Matrix<double, 3, 6>& point_jacobian, 
    const Eigen::Vector3d& x_trans, const Eigen::Matrix3d& c_inv,   
    bool compute_hessian, bool use_gauss_newton_hessian) const 
{
    double mahal_sq = x_trans.dot(c_inv * x_trans);
    if (!std::isfinite(mahal_sq) || mahal_sq < 0.0) return 0.0;

    double exponent_arg = gauss_d2_ * mahal_sq * 0.5;
    double exp_term = fast_exp_neg(exponent_arg);
    double score_inc = -gauss_d1_ * exp_term; 
    
    double factor = gauss_d1_ * gauss_d2_ * exp_term;
    Eigen::Matrix<double, 1, 6> grad_contrib = x_trans.transpose() * c_inv * point_jacobian;
    score_gradient += factor * grad_contrib.transpose();

    if (compute_hessian) {
        Matrix6d hess_contrib = point_jacobian.transpose() * c_inv * point_jacobian;
        hessian += std::abs(factor) * hess_contrib; 
    }
    return score_inc;
}

// --- UPDATED: computeParticleDerivatives using 2D isolated memory buffers ---
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::computeParticleDerivatives(
    Vector6d& final_score_gradient, Matrix6d& final_hessian,        
    const gtsam::Pose3& pose, bool compute_hessian, int particle_idx, int inner_threads) 
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
        Eigen::Matrix<double, 3, 6> point_jacobian;
        
        // Safe inside nested loops thanks to thread_local
        static thread_local std::vector<TargetGridLeafConstPtr> neighborhood_buffer;
        neighborhood_buffer.clear(); 

        const auto& target_cells = this->target_cells_;
        const PointSource& src_pt = (*input_)[idx];
        if (!pcl::isFinite(src_pt)) continue;

        double rx = R(0,0)*src_pt.x + R(0,1)*src_pt.y + R(0,2)*src_pt.z;
        double ry = R(1,0)*src_pt.x + R(1,1)*src_pt.y + R(1,2)*src_pt.z;
        double rz = R(2,0)*src_pt.x + R(2,1)*src_pt.y + R(2,2)*src_pt.z;

        double tx = rx + p_transform(0,3);
        double ty = ry + p_transform(1,3);
        double tz = rz + p_transform(2,3);

        PointTarget x_trans_pt_search;
        x_trans_pt_search.x = static_cast<float>(tx);
        x_trans_pt_search.y = static_cast<float>(ty);
        x_trans_pt_search.z = static_cast<float>(tz);

        int neighbors_found = 0;
        if (search_method_ == NeighborSearchMethod::KDTREE) {
             std::vector<float> dists;
             neighbors_found = target_cells.radiusSearch(x_trans_pt_search, resolution_, neighborhood_buffer, dists);
        } else if (search_method_ == NeighborSearchMethod::DIRECT27) {
             neighbors_found = target_cells.getNeighborhoodAtPoint27(x_trans_pt_search, neighborhood_buffer);
        } else if (search_method_ == NeighborSearchMethod::DIRECT7) {
             neighbors_found = target_cells.getNeighborhoodAtPoint7(x_trans_pt_search, neighborhood_buffer);
        } else {
             neighbors_found = target_cells.getNeighborhoodAtPoint1(x_trans_pt_search, neighborhood_buffer);
        }
        
        if (neighbors_found == 0) {
            scores_buffer_[particle_idx][idx] = 0.0;
            score_gradients_buffer_[particle_idx][idx].setZero();
            hessians_buffer_[particle_idx][idx].setZero();
            continue; 
        }

        Eigen::Vector3d p_rot(rx, ry, rz);
        this->computePointDerivatives(p_rot, R, point_jacobian);

        Vector6d pt_grad = Vector6d::Zero();
        Matrix6d pt_hess = Matrix6d::Zero();
        double pt_score = 0;

        Eigen::Vector3d x_curr(tx, ty, tz); 

        for (const TargetGridLeafConstPtr& cell : neighborhood_buffer) {
            if (!cell) continue; 
            Eigen::Vector3d x_rel = x_curr - cell->getMean();
            pt_score += this->updateDerivatives(pt_grad, pt_hess, point_jacobian, x_rel, cell->getInverseCov(), compute_hessian, this->use_gauss_newton_hessian_);
        }
        
        // Write exclusively to this particle's row
        scores_buffer_[particle_idx][idx] = pt_score;
        score_gradients_buffer_[particle_idx][idx] = pt_grad;
        hessians_buffer_[particle_idx][idx] = pt_hess;
    }

    // Accumulate results (Serial accumulation avoids OpenMP reduction complexity for Eigen)
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
SvnNdtResult SvnNormalDistributionsTransform<PointSource, PointTarget>::align(
    const PointCloudSource& source_cloud, const gtsam::Pose3& prior_mean)
{
    SvnNdtResult result; 
    result.ksd_score = 0.0; 

    if (!target_cells_.getInputCloud() || target_cells_.getAllLeaves().empty() || source_cloud.empty()) {
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

    // Dynamic Thread Splitting logic based on 16 P-Cores
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
             std::cerr << "[SVN-NDT] Warning: Warm start covariance invalid. Reverting to Cold Start." << std::endl;
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

    // =========================================================================
    // 2. OPTIMIZATION LOOP
    // =========================================================================
    for (int iter = 0; iter < max_iter_; ++iter)
    {   
        mean_pose_last_iter = mean_pose_current;
        double iter_ksd_accumulator = 0.0; 

        // A. Compute Gradients/Hessians for all particles
        // Outer loop pinned to 'outer_threads', schedule static for uniform speed
        #pragma omp parallel for num_threads(outer_threads) schedule(static)
        for (int k = 0; k < K_; ++k) {
            computeParticleDerivatives(loss_gradients[k], loss_hessians[k], particles[k], true, k, inner_threads);
        }

        // B. Stein Variational Interaction Step
        // Use outer_threads here since we are looping over particles
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

            H_k_tilde.diagonal().array() += 0.5; 

            Eigen::LDLT<Matrix6d> solver(H_k_tilde); 
            if (solver.info() == Eigen::Success) {
                 particle_updates[k] = solver.solve(phi_k_star);
            } else { 
                 particle_updates[k] = phi_k_star * 0.01; 
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

        // D. Update Mean Pose
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
    // 3. FINALIZE & SAVE COVARIANCE
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
    
    last_covariance_ = result.final_covariance;
    has_last_covariance_ = true;
    
    input_.reset(); 
    return result;
}

} // namespace svn_ndt

#endif // SVN_NDT_IMPL_HPP_