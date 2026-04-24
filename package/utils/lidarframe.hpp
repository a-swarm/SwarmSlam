#pragma once

#include <deque>
#include <vector> 
#include <limits>
#include <algorithm>   

#include <pclreference.hpp>
#include <gtsamreference.hpp>
#include <eigenreference.hpp>
#include <geoutils.hpp>

struct LidarFrame {
    // ... [Standard Members] ...
    uint16_t frame_id = std::numeric_limits<uint16_t>::min();
    double timestamp_start = std::numeric_limits<double>::lowest(); // Current timestamp, unix timestamp (PTP sync)
    double timestamp_end = std::numeric_limits<double>::lowest(); // End timestamp of current frame
    double interframe_timedelta = std::numeric_limits<double>::lowest(); // Time difference between first point in current frame and last point in last frame
    uint32_t numberpoints = std::numeric_limits<uint32_t>::min();

    std::vector<float, Eigen::aligned_allocator<float>> x; // X coordinates
    std::vector<float, Eigen::aligned_allocator<float>> y; // Y coordinates
    std::vector<float, Eigen::aligned_allocator<float>> z; // Z coordinates
    std::vector<uint16_t> c_id; // Channel indices
    std::vector<uint16_t> m_id; // Measurement indices
    std::vector<double, Eigen::aligned_allocator<double>> timestamp_points; // Absolute timestamps
    std::vector<float, Eigen::aligned_allocator<float>> relative_timestamp; // Relative timestamps
    std::vector<uint16_t> reflectivity; // Reflectivity values
    std::vector<uint16_t> signal; // Signal strengths
    std::vector<uint16_t> nir; // NIR values

    [[nodiscard]] bool getPointCloud(pcl::PointCloud<pcl::PointXYZI>& out_cloud) const {

        out_cloud.clear();
        out_cloud.reserve(this->numberpoints);

        for (size_t i = 0; i < this->numberpoints; ++i) {
            pcl::PointXYZI point;
            point.x = this->x[i];
            point.y = this->y[i];
            point.z = this->z[i];
            point.intensity = static_cast<float>(this->reflectivity[i]); // Use reflectivity for intensity
            // Add to pointsBody (raw sensor coordinates)
            out_cloud.push_back(point);
        }
        return !out_cloud.empty();
    }

    // ... [Clear/Swap/Reserve remain standard] ...
    void clear() {
        frame_id = std::numeric_limits<uint16_t>::min();
        timestamp_start = std::numeric_limits<double>::lowest();
        timestamp_end = std::numeric_limits<double>::lowest();
        interframe_timedelta = std::numeric_limits<double>::lowest();
        numberpoints = std::numeric_limits<uint32_t>::min();
        x.clear();
        y.clear();
        z.clear();
        c_id.clear();
        m_id.clear();
        timestamp_points.clear();
        relative_timestamp.clear();
        reflectivity.clear();
        signal.clear();
        nir.clear();
    }

    void swap(LidarFrame& other) noexcept {
        std::swap(frame_id, other.frame_id);
        std::swap(timestamp_start, other.timestamp_start);
        std::swap(timestamp_end, other.timestamp_end);
        std::swap(interframe_timedelta, other.interframe_timedelta);
        std::swap(numberpoints, other.numberpoints);

        // Swap the vectors. This is the key to efficiency!
        x.swap(other.x);
        y.swap(other.y);
        z.swap(other.z);
        c_id.swap(other.c_id);
        m_id.swap(other.m_id);
        timestamp_points.swap(other.timestamp_points);
        relative_timestamp.swap(other.relative_timestamp);
        reflectivity.swap(other.reflectivity);
        signal.swap(other.signal);
        nir.swap(other.nir);
    }

    void reserve(size_t size) {
        x.reserve(size);
        y.reserve(size);
        z.reserve(size);
        c_id.reserve(size);
        m_id.reserve(size);
        timestamp_points.reserve(size);
        relative_timestamp.reserve(size);
        reflectivity.reserve(size);
        signal.reserve(size);
        nir.reserve(size);
    }
};