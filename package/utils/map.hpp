#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream> // Added for debug prints
#include <tsl/robin_map.h>
#include <eigenreference.hpp>
#include <pclreference.hpp>

// --- Your Custom Voxel Structures ---
struct Voxel {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    Voxel(int x, int y, int z) 
        : x(static_cast<int32_t>(x)), y(static_cast<int32_t>(y)), z(static_cast<int32_t>(z)) {}

    bool operator==(const Voxel& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    static Voxel getKey(float x, float y, float z, double voxel_size) {
        // --- CRASH DANGER ZONE ---
        // If x, y, or z is NaN or Inf, std::floor returns NaN/Inf.
        // static_cast<int32_t>(NaN) is Undefined Behavior (SIGFPE on AVX).
        return {
            static_cast<int32_t>(std::floor(x / voxel_size)),
            static_cast<int32_t>(std::floor(y / voxel_size)),
            static_cast<int32_t>(std::floor(z / voxel_size))
        };
    }
};

struct VoxelHash {
    size_t operator()(const Voxel& voxel) const {
        const uint32_t* vec = reinterpret_cast<const uint32_t*>(&voxel);
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

// --- Hash Map Implementation (KISS-ICP Style) ---
template <typename PointT>
class VoxelMap {
public:
    using PointCloud = pcl::PointCloud<PointT>;
    using VoxelBlock = std::vector<PointT>;
    using MapType = tsl::robin_map<Voxel, VoxelBlock, VoxelHash>;

    VoxelMap(double voxel_size, double max_distance, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    void clear() { map_.clear(); }
    bool empty() const { return map_.empty(); }
    size_t size() const { return map_.size(); }

    void update(const typename PointCloud::ConstPtr& cloud, const Eigen::Matrix4f& pose) {
        // std::cout << "[DEBUG Map] Update Start. Cloud Size: " << cloud->size() << std::endl;
        
        // 1. Transform Cloud to Map Frame
        PointCloud transformed_cloud;
        pcl::transformPointCloud(*cloud, transformed_cloud, pose);

        // 2. Add Points (Incremental Downsampling)
        addPoints(transformed_cloud);

        // 3. Remove Points Far from Current Location
        Eigen::Vector3f origin = pose.block<3, 1>(0, 3);
        removeFar(origin);
        
        // std::cout << "[DEBUG Map] Update End. Map Size: " << map_.size() << " voxels." << std::endl;
    }

    typename PointCloud::Ptr getPointCloud() const {
        typename PointCloud::Ptr output(new PointCloud());
        output->reserve(map_.size() * max_points_per_voxel_);
        for (const auto& [voxel, points] : map_) {
            for (const auto& pt : points) {
                output->push_back(pt);
            }
        }
        output->width = output->size();
        output->height = 1;
        output->is_dense = true;
        return output;
    }

private:
    void addPoints(const PointCloud& cloud) {
        const float min_dist_sq = static_cast<float>((voxel_size_ * voxel_size_) / (4.0 * max_points_per_voxel_));
        int nan_count = 0;

        for (size_t i = 0; i < cloud.size(); ++i) {
            const auto& pt = cloud[i];

            // --- DEBUG: Check for BAD points ---
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                if (nan_count == 0) { // Print only the first one to avoid spam
                    std::cerr << "[CRITICAL Map] Found NaN/Inf Point at index " << i 
                              << "! Values: " << pt.x << ", " << pt.y << ", " << pt.z << std::endl;
                }
                nan_count++;
                continue; // SKIP IT to prevent crash
            }
            // -----------------------------------

            // If we get here, point is safe.
            // If the code crashes *after* this print but before the next, it's the Voxel::getKey
            // std::cout << "Processing pt " << i << ": " << pt.x << " " << pt.y << " " << pt.z << std::endl; 

            auto key = Voxel::getKey(pt.x, pt.y, pt.z, voxel_size_);
            auto it = map_.find(key);

            if (it == map_.end()) {
                VoxelBlock block;
                block.reserve(max_points_per_voxel_);
                block.push_back(pt);
                map_.insert({key, std::move(block)});
            } else {
                auto& points = it.value();
                if (points.size() < max_points_per_voxel_) {
                    bool too_close = false;
                    for (const auto& existing_pt : points) {
                        float dx = pt.x - existing_pt.x;
                        float dy = pt.y - existing_pt.y;
                        float dz = pt.z - existing_pt.z;
                        if ((dx*dx + dy*dy + dz*dz) < min_dist_sq) {
                            too_close = true;
                            break;
                        }
                    }
                    if (!too_close) {
                        points.push_back(pt);
                    }
                }
            }
        }
        
        if (nan_count > 0) {
            std::cerr << "[WARNING Map] Skipped " << nan_count << " NaN/Inf points this update." << std::endl;
        }
    }

    void removeFar(const Eigen::Vector3f& origin) {
        const float max_dist_sq = static_cast<float>(max_distance_ * max_distance_);
        
        for (auto it = map_.begin(); it != map_.end();) {
            if (it->second.empty()) {
                it = map_.erase(it);
                continue;
            }
            const auto& pt = it->second.front();
            float dx = pt.x - origin.x();
            float dy = pt.y - origin.y();
            float dz = pt.z - origin.z();

            if ((dx*dx + dy*dy + dz*dz) > max_dist_sq) {
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
    }

    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    MapType map_;
};