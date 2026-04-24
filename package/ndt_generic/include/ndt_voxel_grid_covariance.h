#ifndef NDT_VOXEL_GRID_COVARIANCE_H_
#define NDT_VOXEL_GRID_COVARIANCE_H_

// --- PCL Includes ---
// pcl_macros.h must come first
// clang-format off
#include <pcl/pcl_macros.h>
// clang-format on
#include <pcl/filters/voxel_grid.h> // Base class
#include <pcl/kdtree/kdtree_flann.h> // For optional centroid search
#include <pcl/point_types.h>
#include <pcl/common/eigen.h>          // For NdCopyPointEigenFunctor if needed in impl.
#include <pcl/common/point_tests.h>    // For pcl::isFinite

// --- Standard Library Includes ---
#include <vector>
#include <map>     // May be needed if interacting with code expecting std::map
#include <cmath>   // For std::floor
#include <limits>  // For std::numeric_limits

// --- External Library Includes ---
#include <tsl/robin_map.h> // High-performance hash map
#include <Eigen/Core>      // Core Eigen types (Vector3d, Matrix3d)
#include <Eigen/Eigenvalues> // For SelfAdjointEigenSolver used in Leaf calculation
#include <Eigen/StdVector> // For containers using Eigen types (though not directly here, good practice)


namespace ndt_generic
{
/**
 * @brief A searchable voxel grid structure storing point distribution statistics (mean, covariance) for NDT.
 * @details Inherits from PCL's VoxelGrid and calculates the covariance matrix, inverse covariance,
 * and eigen decomposition for each voxel (leaf) containing a minimum number of points.
 * Uses tsl::robin_map for efficient voxel storage. The core statistical calculations
 * are based on the principles used in Normal Distributions Transform (NDT).
 * @tparam PointT The PCL point type (e.g., pcl::PointXYZ, pcl::PointXYZI).
 */
template <typename PointT>
class VoxelGridCovariance : public pcl::VoxelGrid<PointT>
{
protected:
    // --- Bring protected base class members into scope ---
    // Make parent class's protected members available for implementation
    using pcl::VoxelGrid<PointT>::filter_name_;
    using pcl::VoxelGrid<PointT>::getClassName;
    using pcl::VoxelGrid<PointT>::input_;
    using pcl::VoxelGrid<PointT>::indices_;
    using pcl::VoxelGrid<PointT>::filter_limit_negative_;
    using pcl::VoxelGrid<PointT>::filter_limit_min_;
    using pcl::VoxelGrid<PointT>::filter_limit_max_;
    using pcl::VoxelGrid<PointT>::filter_field_name_;

    using pcl::VoxelGrid<PointT>::downsample_all_data_;
    using pcl::VoxelGrid<PointT>::leaf_layout_;
    using pcl::VoxelGrid<PointT>::save_leaf_layout_;
    using pcl::VoxelGrid<PointT>::leaf_size_;
    using pcl::VoxelGrid<PointT>::min_b_;
    using pcl::VoxelGrid<PointT>::max_b_;
    using pcl::VoxelGrid<PointT>::inverse_leaf_size_;
    using pcl::VoxelGrid<PointT>::div_b_;
    using pcl::VoxelGrid<PointT>::divb_mul_;

    // --- Type aliases from base class ---
    using PointCloud = typename pcl::Filter<PointT>::PointCloud;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

public:
    // --- Public Ptr/ConstPtr Typedefs ---
#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
    using Ptr = std::shared_ptr<VoxelGridCovariance<PointT>>;
    using ConstPtr = std::shared_ptr<const VoxelGridCovariance<PointT>>;
#else
    using Ptr = boost::shared_ptr<VoxelGridCovariance<PointT>>;
    using ConstPtr = boost::shared_ptr<const VoxelGridCovariance<PointT>>;
#endif

    /**
     * @brief Structure holding statistical data for a single voxel (leaf). Essential for NDT.
     * @details Contains the count of points within the voxel, their 3D mean (centroid),
     * the 3x3 covariance matrix, the inverse covariance matrix, and the
     * eigenvectors and eigenvalues of the covariance matrix. Also includes
     * an optional Nd centroid for compatibility with base PCL VoxelGrid features.
     * @note Aligned for Eigen vectorization (16/32 byte alignment).
     */
    struct alignas(32) Leaf // Ensure alignment for Eigen members
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Required for classes/structs with fixed-size Eigen members

        /** @brief Default constructor: Initializes counts to zero, means/centroids to zero,
         * covariance/evecs to identity, and icov/evals to zero. */
        Leaf()
            : nr_points_(0),
              mean_(Eigen::Vector3d::Zero()),
              centroid_(), // Resized only if downsample_all_data_ is true
              cov_(Eigen::Matrix3d::Identity()), // Initialize to identity for safety
              icov_(Eigen::Matrix3d::Zero()),
              evecs_(Eigen::Matrix3d::Identity()),
              evals_(Eigen::Vector3d::Zero())
        {}

        // --- Const Getters for Leaf Data ---
        int getPointCount() const { return nr_points_; }
        const Eigen::Vector3d& getMean() const { return mean_; }
        const Eigen::Matrix3d& getCov() const { return cov_; }
        const Eigen::Matrix3d& getInverseCov() const { return icov_; }
        const Eigen::Matrix3d& getEvecs() const { return evecs_; }
        const Eigen::Vector3d& getEvals() const { return evals_; }

        // --- Public Members (Direct access possible, consider private if stricter encapsulation needed) ---
        int nr_points_;             //!< Number of points falling within this voxel.
        Eigen::Vector3d mean_;      //!< 3D Centroid (mean) of points in the voxel (essential for NDT).
        Eigen::VectorXf centroid_;  //!< Nd Centroid (used only if downsample_all_data_ is true, for compatibility).
        Eigen::Matrix3d cov_;       //!< 3x3 Covariance matrix of points within the voxel.
        Eigen::Matrix3d icov_;      //!< Inverse of the 3x3 covariance matrix (precomputed).
        Eigen::Matrix3d evecs_;     //!< Eigenvectors of the covariance matrix (stored as columns).
        Eigen::Vector3d evals_;     //!< Eigenvalues corresponding to the eigenvectors.
    };

    /** @brief Pointer to a Leaf structure. */
    using LeafPtr = Leaf*;
    /** @brief Const pointer to a Leaf structure (typically used when retrieving leaves). */
    using LeafConstPtr = const Leaf*;

    // --- Voxel Map Definition ---
    /** @brief Defines the key-value pair stored in the map (voxel index -> Leaf data). */
    using MapPair = std::pair<const size_t, Leaf>;
    /** @brief Hash map storing voxel leaves, keyed by their computed 1D index.
     * Uses tsl::robin_map for performance and Eigen::aligned_allocator for correctness
     * due to fixed-size Eigen members in the Leaf struct. */
    using Map = tsl::robin_map<size_t, Leaf, std::hash<size_t>, std::equal_to<size_t>,
                               Eigen::aligned_allocator<MapPair>>;

public:
    /**
     * @brief Default constructor. Initializes NDT-specific parameters.
     */
    VoxelGridCovariance()
        : searchable_(false), // KdTree is not built by default
          min_points_per_voxel_(6), // Default commonly used in NDT (>=3 required)
          min_covar_eigvalue_mult_(0.01), // Regularization factor for covariance stability
          leaves_(), // Initialize the map
          voxel_centroids_(), // Initialize pointers
          voxel_centroids_leaf_indices_(),
          kdtree_(),
          num_threads_(0)
    {
        // Configure base class settings typically used for NDT
        this->downsample_all_data_ = false; // NDT usually only needs 3D stats
        this->save_leaf_layout_ = false; // Layout saving is generally not needed
        this->leaf_size_.setZero(); // Must be set by user via setLeafSize
        this->min_b_.setZero();
        this->max_b_.setZero();
        this->filter_name_ = "SvnNdtVoxelGridCovariance"; // Specific class name
    }

    // --- Configuration Methods ---

    /** @brief Sets the minimum number of points required within a voxel to compute its covariance
     * and consider it valid for NDT calculations.
     * @details A minimum of 3 points is enforced, as covariance is ill-defined otherwise.
     * @param min_points_per_voxel Minimum number of points (will be clamped to >= 3).
     */
    inline void setMinPointPerVoxel(int min_points_per_voxel)
    {
        if (min_points_per_voxel >= 3) {
            min_points_per_voxel_ = min_points_per_voxel;
        } else {
            PCL_WARN("[%s::setMinPointPerVoxel] Covariance calculation requires at least 3 points. Setting to 3.\n", getClassName().c_str());
            min_points_per_voxel_ = 3;
        }
    }

    /** @brief Gets the minimum number of points required per voxel. */
    inline int getMinPointPerVoxel() const { return min_points_per_voxel_; }

    /** @brief Sets the regularization factor used to prevent singular covariance matrices.
     * @details Eigenvalues smaller than `min_covar_eigvalue_mult_` times the largest eigenvalue
     * will be inflated to this minimum value.
     * @param min_covar_eigvalue_mult The inflation ratio (e.g., 0.01).
     */
    inline void setCovEigValueInflationRatio(double min_covar_eigvalue_mult)
    {
        min_covar_eigvalue_mult_ = min_covar_eigvalue_mult;
    }

    /** @brief Gets the eigenvalue inflation ratio used for regularization. */
    inline double getCovEigValueInflationRatio() const { return min_covar_eigvalue_mult_; }

    // --- ADDED: Threading Configuration ---
    /** @brief Sets the number of OpenMP threads to use for filtering.
     * @param nr_threads Number of threads. If <= 0, uses max available.
     */
    inline void setNumThreads(int nr_threads)
    {
        num_threads_ = nr_threads;
    }

    /** @brief Gets the configured thread count (returns 0 if set to auto/max). */
    inline int getNumThreads() const { return num_threads_; }

    // --- Filtering and Grid Building ---

    /**
     * @brief Processes the input point cloud to build the voxel grid with covariance information.
     * @details Calculates mean, covariance, inverse covariance, and eigen decomposition for each valid voxel.
     * Optionally populates an output point cloud with the centroids of valid voxels
     * and builds a KdTree on these centroids for fast neighbor searching.
     * @param[out] output Optional: Point cloud to be filled with the centroids of valid voxels.
     * @param[in] searchable If true, builds a KdTree on the valid voxel centroids to enable
     * `nearestKSearch` and `radiusSearch` methods.
     */
    inline void filter(PointCloud& output, bool searchable = false)
    {
        searchable_ = searchable;
        applyFilter(output); // Calls the overridden virtual method in the .hpp file

        // Must build the centroid cloud *after* applyFilter populates 'leaves_'
        buildCentroidCloudAndIndexMap(); // Helper defined in .hpp

        if (searchable_ && voxel_centroids_ && !voxel_centroids_->empty()) {
            kdtree_.setInputCloud(voxel_centroids_);
        } else if (searchable_) {
            // Handle case where grid is built but might have no valid centroids
            if (!voxel_centroids_) voxel_centroids_.reset(new PointCloud()); // Ensure cloud exists
            kdtree_.setInputCloud(voxel_centroids_); // Set empty cloud in KdTree
        }
    }

    /**
     * @brief Processes the input point cloud to build the voxel grid (no centroid output cloud).
     * @details Calculates mean, covariance, etc., for each valid voxel. Optionally builds
     * a KdTree on valid voxel centroids for searching.
     * @param[in] searchable If true, builds the KdTree.
     */
    inline void filter(bool searchable = false)
    {
        searchable_ = searchable;
        PointCloud dummy_output; // Temporary storage if base applyFilter needs it
        applyFilter(dummy_output); // Calls the overridden virtual method

        // Build centroid cloud and index mapping AFTER applyFilter populates 'leaves_'
        buildCentroidCloudAndIndexMap(); // Helper defined in .hpp

        if (searchable_ && voxel_centroids_ && !voxel_centroids_->empty()) {
            kdtree_.setInputCloud(voxel_centroids_);
        } else if (searchable_) {
            if (!voxel_centroids_) voxel_centroids_.reset(new PointCloud());
            kdtree_.setInputCloud(voxel_centroids_);
        }
    }

    // --- Leaf Access and Search Methods ---

    /**
     * @brief Retrieves a const pointer to the Leaf data structure for a given 1D voxel index.
     * @details Performs a lookup in the internal hash map. Only returns a valid pointer
     * if the leaf exists *and* contains the minimum required number of points.
     * @param index The 1D voxel index (calculated based on point coordinates and grid parameters).
     * @return Const pointer to the valid Leaf, or nullptr if the index is not found or the leaf is invalid.
     */
    inline LeafConstPtr getLeaf(size_t index) const
    {
        auto it = leaves_.find(index);
        // Check point count *after* finding the leaf to ensure validity for NDT
        if (it != leaves_.end() && it->second.getPointCount() >= min_points_per_voxel_) {
            return &(it->second); // Return address of the Leaf struct in the map
        }
        return nullptr; // Leaf not found or not enough points
    }

    /**
     * @brief Retrieves a const pointer to the Leaf data structure containing the given 3D point coordinates.
     * @details Checks if the point is within the grid bounds, calculates the corresponding 1D voxel index,
     * and then calls `getLeaf(size_t index)`.
     * @param p Point coordinates as an Eigen::Vector3f.
     * @return Const pointer to the valid Leaf, or nullptr if the point is outside bounds, the voxel is empty,
     * or the voxel doesn't meet the minimum point requirement.
     */
    inline LeafConstPtr getLeaf(const Eigen::Vector3f& p) const
    {
        // Ensure grid has been initialized (min_b_, max_b_, inverse_leaf_size_ are set)
        if (this->inverse_leaf_size_[0] == 0.0f) { // Check using inverse_leaf_size_ as indicator
             PCL_WARN("[%s::getLeaf] Voxel grid not initialized (leaf size is zero or filter not run).\n", getClassName().c_str());
             return nullptr;
        }

        // // Check if point is within the bounds calculated by applyFilter
        //  Eigen::Vector4f pt4(p[0], p[1], p[2], 0.0f); // Use 4D vector for base class compatibility
        //  if (!isPointWithinBounds(pt4)) { // Implementation in .hpp
        //      // PCL_DEBUG("[%s::getLeaf] Point (%.2f, %.2f, %.2f) is outside grid bounds.\n", getClassName().c_str(), p[0], p[1], p[2]);
        //      return nullptr;
        //  }

        // Compute 1D index using base class precomputed values (min_b_, divb_mul_)
        // Ensure correct casting and use floor for consistency
        int ijk0 = static_cast<int>(std::floor(p[0] * this->inverse_leaf_size_[0]) - this->min_b_[0]);
        int ijk1 = static_cast<int>(std::floor(p[1] * this->inverse_leaf_size_[1]) - this->min_b_[1]);
        int ijk2 = static_cast<int>(std::floor(p[2] * this->inverse_leaf_size_[2]) - this->min_b_[2]);
        size_t idx = static_cast<size_t>(ijk0 * this->divb_mul_[0] + ijk1 * this->divb_mul_[1] + ijk2 * this->divb_mul_[2]);

        // Delegate to the index-based lookup (which includes the min_points check)
        return getLeaf(idx);
    }

     /**
     * @brief Convenience overload to retrieve the Leaf containing a given PCL point.
     * @param p PCL point object (e.g., pcl::PointXYZ, pcl::PointXYZI).
     * @return Const pointer to the valid Leaf, or nullptr if invalid.
     */
    inline LeafConstPtr getLeaf(const PointT& p) const
    {
        // Extract Eigen vector map and delegate
        return getLeaf(p.getVector3fMap());
    }


    /** @brief Returns a const reference to the internal map holding all computed leaves.
     * @details Note: This map may contain leaves with fewer than `min_points_per_voxel_`.
     * Use `getLeaf()` methods to access only valid leaves for NDT calculations. */
    inline const Map& getAllLeaves() const { return leaves_; }

    /** @brief Returns a const shared pointer to the point cloud containing the centroids of *valid* voxels.
     * @details Requires `filter()` to have been called with `searchable = true`. Returns nullptr if not searchable. */
    inline PointCloudConstPtr getCentroids() const
    {
       return searchable_ ? voxel_centroids_ : nullptr;
    }

    // --- KdTree Search Methods (Require searchable_ == true) ---

    /**
     * @brief Finds the K nearest *valid* voxel centroids to a query point using the internal KdTree.
     * @details Requires `filter(true)` or `filter(output, true)` to have been called beforehand.
     * @param[in] point The query point.
     * @param[in] k The number of nearest neighbors to find.
     * @param[out] k_leaves Vector to be filled with const pointers to the valid Leaf structures
     * corresponding to the found neighbors.
     * @param[out] k_sqr_distances Vector to be filled with the squared Euclidean distances
     * to the centroids of the found neighbors.
     * @return The number of neighbors actually found (can be less than k if fewer valid voxels exist). Returns 0 if not searchable.
     */
    int nearestKSearch( const PointT& point, int k,
                        std::vector<LeafConstPtr>& k_leaves,
                        std::vector<float>& k_sqr_distances) const; // Implementation in .hpp


    /**
     * @brief Finds all *valid* voxel centroids within a specified radius of a query point using the internal KdTree.
     * @details Requires `filter(true)` or `filter(output, true)` to have been called beforehand.
     * @param[in] point The query point.
     * @param[in] radius The search radius.
     * @param[out] k_leaves Vector to be filled with const pointers to the valid Leaf structures
     * found within the radius.
     * @param[out] k_sqr_distances Vector to be filled with the squared Euclidean distances
     * to the centroids of the found neighbors.
     * @param[in] max_nn Optional: Maximum number of neighbors to return (0 means unlimited within the radius).
     * @return The number of neighbors found. Returns 0 if not searchable.
     */
    int radiusSearch( const PointT& point, double radius,
                      std::vector<LeafConstPtr>& k_leaves,
                      std::vector<float>& k_sqr_distances,
                      unsigned int max_nn = 0) const; // Implementation in .hpp

    // --- Direct Neighbor Search Methods (Do not require searchable_ == true) ---
     /**
     * @brief Gets the valid neighbor leaves using the DIRECT27 method (3x3x3 voxel cube).
     * @param reference_point The query point used to determine the central voxel.
     * @param[out] neighbors Vector filled with const pointers to the valid leaves found (up to 27).
     * @return The number of *valid* neighbors found.
     */
     int getNeighborhoodAtPoint27(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const;

    /**
     * @brief Gets the valid neighbor leaves using the DIRECT7 method (voxel containing the point + 6 face-adjacent voxels).
     * @param reference_point The query point used to determine the central voxel.
     * @param[out] neighbors Vector filled with const pointers to the valid leaves found (up to 7).
     * @return The number of *valid* neighbors found (may be less than 7 if some voxels are empty or have too few points).
     */
     int getNeighborhoodAtPoint7(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const; // Implementation in .hpp

    /**
     * @brief Gets the valid leaf using the DIRECT1 method (only the voxel containing the point).
     * @param reference_point The query point.
     * @param[out] neighbors Vector filled with at most one const pointer to the valid leaf containing the point.
     * @return The number of *valid* neighbors found (either 0 or 1).
     */
     int getNeighborhoodAtPoint1(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const; // Implementation in .hpp


protected:
    /**
     * @brief Overridden PCL VoxelGrid filter method where the main computation occurs.
     * @details This method iterates through the input cloud, assigns points to voxels,
     * and calculates the statistical properties (mean, covariance, etc.) for each voxel,
     * storing the results in the `leaves_` map. It handles regularization and
     * filters out invalid voxels based on `min_points_per_voxel_`.
     * @param[out] output The downsampled cloud (used internally by base class, but final centroid cloud is built separately).
     */
    void applyFilter(PointCloud& output) override; // Added override specifier

    /**
     * @brief Helper function called internally by `filter()` to populate the `voxel_centroids_`
     * point cloud and the `voxel_centroids_leaf_indices_` mapping.
     * @details Iterates through the computed `leaves_` map and adds the centroid of each valid leaf
     * (meeting `min_points_per_voxel_`) to the `voxel_centroids_` cloud, storing the
     * original leaf map index for later lookup during search operations.
     */
    void buildCentroidCloudAndIndexMap(); // Implementation in .hpp

    /**
     * @brief Helper function to check if a given 3D point lies within the grid bounds
     * established during the `applyFilter` stage.
     * @details Uses the `min_b_` and `max_b_` index bounds (converted back to float coordinates)
     * to perform the check. Essential for preventing lookups outside the allocated grid.
     * @param p Point coordinates (uses only x, y, z components of the Vector4f).
     * @return True if the point is within the grid bounds, false otherwise or if bounds not yet computed.
     */
     bool isPointWithinBounds(const Eigen::Vector4f& p) const; // Implementation in .hpp

    // --- Member Variables ---

    bool searchable_;                       //!< Flag indicating if the KdTree for centroid searching is built.
    int min_points_per_voxel_;              //!< Minimum number of points required for a voxel to be considered valid.
    double min_covar_eigvalue_mult_;        //!< Regularization factor for covariance eigenvalues (ratio to max eigenvalue).
    Map leaves_;                            //!< The core data structure: Hash map from 1D voxel index to Leaf struct.
    int num_threads_;                       // <--- ADDED: Member variable

    // --- Members related to searchability ---
    PointCloudPtr voxel_centroids_;           //!< Point cloud storing the centroids of *valid* voxels (used by KdTree).
    std::vector<size_t> voxel_centroids_leaf_indices_; //!< Mapping: index in voxel_centroids_ -> original index (key) in leaves_ map.
    pcl::KdTreeFLANN<PointT> kdtree_;         //!< KdTree built on voxel_centroids_ for fast neighbor searches.
};

} // namespace ndt_generic

// --- Implementation Inclusion ---
// The implementation file must be included at the end for template classes.
#include <ndt_voxel_grid_covariance_impl.hpp>

#endif // NDT_VOXEL_GRID_COVARIANCE_H_