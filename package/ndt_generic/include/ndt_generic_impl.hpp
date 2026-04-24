#ifndef NDT_GENERIC_IMPL_HPP_
#define NDT_GENERIC_IMPL_HPP_

#include <ndt_generic.h>
#include <omp.h>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <algorithm>

namespace ndt_generic
{

template <typename PointSource, typename PointTarget>
NormalDistributionsTransform<PointSource, PointTarget>::NormalDistributionsTransform(const std::string& json_path)
    : target_cells_()
{   
    // [Unchanged Constructor Parsing Logic]
    int min_points = 3;
    float resolution = 1.0f;
    double step_size_trans = 0.1;
    double step_size_rot = 0.02;
    double outlier_ratio = 0.55;
    double trans_eps = 0.01;
    double rot_eps = 0.005;
    int max_iter = 30;
    std::string search_method = "DIRECT7";
    std::string opt_method = "LEVENBERG_MARQUARDT";
    std::string hessian_form = "FULL_NEWTON";
    int num_threads = 1;
    Vector6d initial_covariance = Vector6d::Ones();

    std::ifstream file(json_path);
    if (!file.is_open()) throw std::runtime_error("Failed to open JSON file: " + json_path);
    nlohmann::json json_data;
    file >> json_data;

    const auto& ndt_param = json_data["ndt"];
    min_points = ndt_param.value("min_points", 3);
    resolution = ndt_param.value("resolution", 1.0f);
    step_size_trans = ndt_param.value("step_size_trans", 0.1);
    step_size_rot = ndt_param.value("step_size_rot", 0.02);
    outlier_ratio = ndt_param.value("outlier_ratio", 0.55);
    trans_eps = ndt_param.value("trans_eps", 0.01);
    rot_eps = ndt_param.value("rot_eps", 0.005);
    max_iter = ndt_param.value("max_iter", 30);
    search_method = ndt_param.value("search_method", "DIRECT7");
    opt_method = ndt_param.value("opt_method", "LEVENBERG_MARQUARDT");
    hessian_form = ndt_param.value("hessian_form", "FULL_NEWTON");

    if (ndt_param.contains("initial_covariance")) {
        auto arr = ndt_param["initial_covariance"];
        initial_covariance << arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>(), arr[3].get<double>(), arr[4].get<double>(), arr[5].get<double>();
    }
    num_threads = json_data.value("num_threads", 16);

    setNumThreads(num_threads);
    setMinPointPerVoxel(min_points);
    setResolution(resolution);
    setStepSize(step_size_trans, step_size_rot);
    setInitialCovariance(initial_covariance);
    setTransformationEpsilon(trans_eps, rot_eps);
    setMaxIterations(max_iter);
    setOutlierRatio(outlier_ratio);

    if (search_method == "DIRECT1") setNeighborhoodSearchMethod(NeighborSearchMethod::DIRECT1);
    else if (search_method == "DIRECT27") setNeighborhoodSearchMethod(NeighborSearchMethod::DIRECT27);
    else if (search_method == "KDTREE") setNeighborhoodSearchMethod(NeighborSearchMethod::KDTREE);
    else setNeighborhoodSearchMethod(NeighborSearchMethod::DIRECT7);

    if (opt_method == "BACKTRACKING_ARMIJO") setOptimizationMethod(OptimizationMethod::BACKTRACKING_ARMIJO);
    else setOptimizationMethod(OptimizationMethod::LEVENBERG_MARQUARDT);

    if (hessian_form == "FULL_NEWTON") setHessianForm(HessianForm::FULL_NEWTON);
    else setHessianForm(HessianForm::GAUSS_NEWTON);

    updateNdtConstants();
}

template <typename PointSource, typename PointTarget>
void NormalDistributionsTransform<PointSource, PointTarget>::updateNdtConstants()
{
    if (resolution_ <= 1e-6f) {
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
        return;
    }
    
    double gauss_c1 = 10.0 * (1 - outlier_ratio_);
    double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
    double epsilon = 1e-9;
    if (gauss_c1 <= epsilon) gauss_c1 = epsilon;
    if (gauss_c2 <= epsilon) gauss_c2 = epsilon;

    gauss_d3_ = -log(gauss_c2);
    gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3_; 
    gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3_) / gauss_d1_);
}

template <typename PointSource, typename PointTarget>
void NormalDistributionsTransform<PointSource, PointTarget>::setResolution(float resolution)
{
    if (std::abs(resolution - resolution_) > 1e-6) {
        resolution_ = resolution;
        if (target_cells_.getInputCloud()) {
            setInputTarget(target_cells_.getInputCloud());
        }
        updateNdtConstants();
    }
}

template <typename PointSource, typename PointTarget>
void NormalDistributionsTransform<PointSource, PointTarget>::setMinPointPerVoxel(int min_points) {
    target_cells_.setMinPointPerVoxel(min_points);
    if (target_cells_.getInputCloud()) setInputTarget(target_cells_.getInputCloud());
}

template <typename PointSource, typename PointTarget>
void NormalDistributionsTransform<PointSource, PointTarget>::setOutlierRatio(double ratio) {
    outlier_ratio_ = ratio;
    updateNdtConstants();
}

template <typename PointSource, typename PointTarget>
void NormalDistributionsTransform<PointSource, PointTarget>::setNumThreads(int n)
{
    num_threads_ = (n > 0) ? n : omp_get_max_threads();
    target_cells_.setNumThreads(num_threads_);
    // No more manual buffer resizing needed here!
}

template <typename PointSource, typename PointTarget>
void NormalDistributionsTransform<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud)
{
    if (!cloud || cloud->empty()) return;
    target_cells_.setLeafSize(resolution_, resolution_, resolution_);
    target_cells_.setInputCloud(cloud);
    target_cells_.setNumThreads(num_threads_);
    target_cells_.filter(true); 
    updateNdtConstants();
}

template <typename PointSource, typename PointTarget>
NdtResult NormalDistributionsTransform<PointSource, PointTarget>::align(
    const PointCloudSource& source_cloud, const gtsam::Pose3& prior_pose)
{
    NdtResult result;
    result.final_pose = prior_pose;

    if (!target_cells_.getCentroids() || target_cells_.getCentroids()->empty() || source_cloud.empty()) {
        std::cout << "[NDT] Warning: Empty target or source cloud." << std::endl;
        return result;
    }

    input_ = source_cloud.makeShared();
    gtsam::Pose3 current_pose = prior_pose;
    
    Vector6d score_gradient;
    Matrix6d hessian;
    double score = 0.0;
    
    double lambda = 1e-4; 
    int iter = 0;
    bool converged = false;

    score = calculateScore(*input_, current_pose);
    computeDerivatives(score_gradient, hessian, *input_, current_pose);

    while (iter < max_iter_) {
        Matrix6d H = hessian; 
        Vector6d g = score_gradient; 

        if (opt_method_ == OptimizationMethod::LEVENBERG_MARQUARDT) {
             H.diagonal().array() += lambda;
        } else {
             H.diagonal().array() += 1e-9; 
        }

        Eigen::LDLT<Matrix6d> solver(H);
        Vector6d delta = solver.solve(g); 
        
        if (delta.array().isNaN().any() || delta.array().isInf().any()) {
            if (opt_method_ == OptimizationMethod::LEVENBERG_MARQUARDT) {
                lambda *= 10.0; 
                if(lambda > 1e6) { converged = false; break; }
                continue; 
            } else {
                std::cout << "[NDT] Hessian inversion failed in Line Search." << std::endl;
                converged = false; break; 
            }
        }

        Eigen::Vector3d delta_rot   = delta.head<3>();
        Eigen::Vector3d delta_trans = delta.tail<3>();

        double norm_rot   = delta_rot.norm();
        double norm_trans = delta_trans.norm();

        double scale_rot = 1.0;
        double scale_trans = 1.0;

        if (norm_rot > step_size_rot_) scale_rot = step_size_rot_ / norm_rot;
        if (norm_trans > step_size_trans_) scale_trans = step_size_trans_ / norm_trans;

        double final_scale = std::min(scale_rot, scale_trans);

        if (final_scale < 1.0) {
            delta *= final_scale;
            norm_rot *= final_scale;
            norm_trans *= final_scale;
        }

        if (norm_trans < trans_eps_ && norm_rot < rot_eps_) {
            converged = true;
            break;
        }

        gtsam::Pose3 next_pose = current_pose.compose(gtsam::Pose3::Expmap(delta));
        double next_score = calculateScore(*input_, next_pose);
        bool step_accepted = false;
        
        if (opt_method_ == OptimizationMethod::LEVENBERG_MARQUARDT) {
            if (next_score > score) {
                step_accepted = true;
                lambda = std::max(lambda * 0.1, 1e-7);
            } else {
                lambda = std::min(lambda * 10.0, 1e7);
            }
        } 
        else { 
            double alpha = 1.0;
            const double c1 = 1e-4; 
            const double dot_g_delta = g.dot(delta); 
            
            int ls_iter = 0;
            while (ls_iter < 10) {
                if (next_score >= score + c1 * alpha * dot_g_delta) {
                    step_accepted = true;
                    break;
                }
                
                alpha *= 0.5;
                Vector6d scaled_delta = delta * alpha; 
                
                if (scaled_delta.head<3>().norm() < 1e-6 && scaled_delta.tail<3>().norm() < 1e-6) break; 

                next_pose = current_pose.compose(gtsam::Pose3::Expmap(scaled_delta));
                next_score = calculateScore(*input_, next_pose);
                ls_iter++;
            }
            
            if (!step_accepted && next_score > score) step_accepted = true;
        }

        if (step_accepted) {
            current_pose = next_pose;
            score = next_score;
            iter++;
            computeDerivatives(score_gradient, hessian, *input_, current_pose);
        } else {
            if (opt_method_ == OptimizationMethod::LEVENBERG_MARQUARDT) {
                if (lambda > 1e6) {
                    converged = true; 
                    break;
                }
            } else {
                std::cout << "[NDT] Failed: Line search found no improvement." << std::endl;
                converged = true; 
                break;
            }
        }
    }

    result.final_pose = current_pose;
    result.hessian = hessian;
    result.iterations = iter;
    result.converged = converged;

    if (std::abs(hessian.determinant()) > 1e-9) {
        result.final_covariance = hessian.inverse();
        result.final_covariance = 0.5 * (result.final_covariance + result.final_covariance.transpose());
        result.final_covariance.diagonal().array() += 1e-6;
        last_covariance_ = result.final_covariance;
        has_last_covariance_ = true;
    } else {
        std::cout << "[NDT] Warning: Hessian is singular." << std::endl;
        if (has_last_covariance_) {
             result.final_covariance = last_covariance_; 
        } else {
             result.final_covariance = initial_covariance_.array().square().matrix().asDiagonal();
        }
    }
    
    input_.reset();
    return result;
}

template <typename PointSource, typename PointTarget>
double NormalDistributionsTransform<PointSource, PointTarget>::computeDerivatives(
    Vector6d& score_gradient, Matrix6d& hessian,
    const PointCloudSource& source_cloud,
    const gtsam::Pose3& transform)
{
    double total_score = 0.0;
    score_gradient.setZero();
    hessian.setZero();

    Eigen::Matrix3d R = transform.rotation().matrix();
    Eigen::Vector3d t = transform.translation();

    // OPTIMIZED: Thread-Local Reduction
    #pragma omp parallel num_threads(num_threads_)
    {
        double local_score = 0.0;
        Vector6d local_gradient = Vector6d::Zero();
        Matrix6d local_hessian = Matrix6d::Zero();
        
        std::vector<TargetGridLeafConstPtr> neighborhood;
        neighborhood.reserve(27);
        std::vector<float> distances;
        
        #pragma omp for schedule(guided, 8)
        for (size_t idx = 0; idx < source_cloud.size(); idx++) {
            const auto& pt_src = source_cloud[idx];
            if (!pcl::isFinite(pt_src)) continue;

            gtsam::Point3 p_gtsam(pt_src.x, pt_src.y, pt_src.z);
            gtsam::Point3 p_trans = transform.transformFrom(p_gtsam);
            Eigen::Vector3d x_trans = p_trans; 

            neighborhood.clear(); 
            PointTarget search_pt;
            search_pt.x = x_trans.x(); search_pt.y = x_trans.y(); search_pt.z = x_trans.z();

            if (search_method_ == NeighborSearchMethod::DIRECT7) {
                 target_cells_.getNeighborhoodAtPoint7(search_pt, neighborhood);
            } else if (search_method_ == NeighborSearchMethod::DIRECT27) {
                 target_cells_.getNeighborhoodAtPoint27(search_pt, neighborhood);
            } else if (search_method_ == NeighborSearchMethod::DIRECT1) {
                 target_cells_.getNeighborhoodAtPoint1(search_pt, neighborhood);
            } else {
                 target_cells_.radiusSearch(search_pt, resolution_, neighborhood, distances);
            }

            Eigen::Vector3d p_rot = x_trans - t;
            Eigen::Matrix3d p_cross;
            p_cross << 0.0, -p_rot.z(), p_rot.y(),
                       p_rot.z(), 0.0, -p_rot.x(),
                      -p_rot.y(), p_rot.x(), 0.0;

            Eigen::Matrix<double, 3, 6> J_right;
            J_right.block<3,3>(0,0) = -p_cross * R; 
            J_right.block<3,3>(0,3) = R;

            for (const auto& cell : neighborhood) {
                if(!cell) continue;
                Eigen::Vector3d d = x_trans - cell->getMean();
                Eigen::Matrix3d Cinv = cell->getInverseCov(); 
                local_score += updateDerivatives(local_gradient, local_hessian, d, Cinv, J_right);
            }
        }

        // Merge thread-local values safely into the global accumulators
        #pragma omp critical
        {
            total_score += local_score;
            score_gradient += local_gradient;
            hessian += local_hessian;
        }
    }
    return total_score;
}

template <typename PointSource, typename PointTarget>
inline double NormalDistributionsTransform<PointSource, PointTarget>::updateDerivatives(
    Vector6d& score_gradient, Matrix6d& hessian,
    const Eigen::Vector3d& d, 
    const Eigen::Matrix3d& Cinv,
    const Eigen::Matrix<double, 3, 6>& J) const
{
    double dist_sq = d.dot(Cinv * d);
    if(dist_sq > 25.0) return 0.0; 

    double e_exp = std::exp(-gauss_d2_ * 0.5 * dist_sq);
    double s = -gauss_d1_ * e_exp;
    double factor = gauss_d1_ * gauss_d2_ * e_exp;

    Eigen::Vector3d q = factor * (Cinv * d);
    score_gradient.noalias() += J.transpose() * q;

    Eigen::Matrix3d H_pos; 
    if (hessian_form_ == HessianForm::GAUSS_NEWTON) {
            H_pos = std::abs(factor) * Cinv;
    } else {
            Eigen::Vector3d Cd = Cinv * d;
            H_pos = std::abs(factor) * (Cinv - gauss_d2_ * Cd * Cd.transpose());
    }

    hessian.noalias() += J.transpose() * H_pos * J;
    return s;
}

template <typename PointSource, typename PointTarget>
double NormalDistributionsTransform<PointSource, PointTarget>::calculateScore(
    const PointCloudSource& source_cloud, const gtsam::Pose3& transform)
{
    double total_score = 0.0;
    
    // OPTIMIZED: Thread-Local Reduction
    #pragma omp parallel num_threads(num_threads_)
    {
        double local_score = 0.0;
        std::vector<TargetGridLeafConstPtr> neighborhood;
        neighborhood.reserve(27);
        std::vector<float> dist;

        #pragma omp for schedule(guided, 8)
        for (size_t idx = 0; idx < source_cloud.size(); idx++) {
            const auto& pt = source_cloud[idx];
            if(!pcl::isFinite(pt)) continue;

            gtsam::Point3 p_gtsam(pt.x, pt.y, pt.z);
            gtsam::Point3 p_trans = transform.transformFrom(p_gtsam);
            
            neighborhood.clear();
            PointTarget search_pt;
            search_pt.x = p_trans.x(); search_pt.y = p_trans.y(); search_pt.z = p_trans.z();

            if (search_method_ == NeighborSearchMethod::DIRECT7) {
                 target_cells_.getNeighborhoodAtPoint7(search_pt, neighborhood);
            } else if (search_method_ == NeighborSearchMethod::DIRECT1) {
                 target_cells_.getNeighborhoodAtPoint1(search_pt, neighborhood);
            } else {
                 target_cells_.radiusSearch(search_pt, resolution_, neighborhood, dist);
            }

            for (const auto& cell : neighborhood) {
                if(!cell) continue;
                Eigen::Vector3d d(p_trans.x() - cell->getMean().x(),
                                  p_trans.y() - cell->getMean().y(),
                                  p_trans.z() - cell->getMean().z());
                
                double dist_sq = d.dot(cell->getInverseCov() * d);
                double val = std::exp(-gauss_d2_ * 0.5 * dist_sq);
                if(!std::isnan(val)) {
                    local_score += -gauss_d1_ * val; 
                }
            }
        }
        
        #pragma omp critical
        {
            total_score += local_score;
        }
    }
    return total_score;
}

} // namespace ndt_generic

#endif // NDT_GENERIC_IMPL_HPP_