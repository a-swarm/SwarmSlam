/**
 * @file lo_pipeline.hpp
 * @brief Helper functions and structures for the LIO Smoother Pipeline.
 */

#ifndef LO_PIPELINE_HPP_
#define LO_PIPELINE_HPP_

#include <fstream>
#include <iomanip>
#include <atomic>             
#include <condition_variable> 
#include <thread>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <limits>
#include <vector>
#include <string>

// --- OS Thread Affinity Includes ---
#include <pthread.h>
#include <sched.h>

#include <lo_smoother.h>
#include <lidarcallback.h>

#include <eigenreference.hpp>
#include <pclreference.hpp>
#include <gtsamreference.hpp>
#include <datatransferutils.hpp>
#include <udpsocket.hpp>

#include <vtkObject.h>
#include <vtkRendererCollection.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>

/**
 * @brief Pins the calling thread to a specific range of CPU cores.
 * @param start_core The lowest core index to allow.
 * @param end_core The highest core index to allow.
 * @param thread_name Optional name for debugging output.
 */
inline void setThreadAffinity(int start_core, int end_core, const std::string& thread_name = "Thread") {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = start_core; i <= end_core; ++i) {
        CPU_SET(i, &cpuset);
    }
    
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    if (rc != 0) {
        std::cerr << "[System] Failed to set affinity for " << thread_name << " (Error code: " << rc << ")\n";
    } else {
        std::cout << "[System] " << thread_name << " pinned to cores " << start_core << "-" << end_core << "\n";
    }
}

// --- TUM Format Structure ---
struct TumPose {
    double timestamp;
    double x, y, z;        // Metric Position (NED)
    double qx, qy, qz, qw; // Orientation
};

struct PerfLog {
    double timestamp;
    uint16_t frame_id;
    double alingment_time_ms;
    std::array<double, 36> cov_values;
};

struct FrameData{
    double timestamp = std::numeric_limits<double>::lowest();
    uint16_t frame_id = std::numeric_limits<uint16_t>::min(); 
    pcl::PointCloud<pcl::PointXYZI> points;
    // std::deque<CompFrameIMU> imu_window;

    /**
     * @brief Resets the struct to a default state for object pooling.
     */
    void clear() {
        timestamp = std::numeric_limits<double>::lowest();
        frame_id = std::numeric_limits<uint16_t>::lowest();
        points.clear(); // Calls PCLPointCloud::clear()
        // imu_window.clear();    // Calls std::vector::clear()
    }
};

// Save in TUM format: timestamp x y z qx qy qz qw
inline void saveTumTrajectory(const std::vector<TumPose>& trajectory, 
                              const std::string& filename,
                              double ref_lat, double ref_lon, double ref_alt,
                              double ref_qx, double ref_qy, double ref_qz, double ref_qw) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[IO] Failed to open file: " << filename << std::endl;
        return;
    }
    
    // High precision is critical
    file << std::fixed << std::setprecision(9); 
    
    // --- NEW: Print Reference Header ---
    // Format: # Reference: Lat Lon Alt Qx Qy Qz Qw
    file << "### Reference: " << ref_lat << " " << ref_lon << " " << ref_alt << " "
         << ref_qx << " " << ref_qy << " " << ref_qz << " " << ref_qw << "\n";
    // -----------------------------------

    for (const auto& pt : trajectory) {
        file << pt.timestamp << " "
             << pt.x << " " << pt.y << " " << pt.z << " "
             << pt.qx << " " << pt.qy << " " << pt.qz << " " << pt.qw << "\n";
    }
    std::cout << "[IO] Saved " << trajectory.size() << " poses to " << filename << " (TUM format)." << std::endl;
}

// 2. Update savePerfLog function
inline void savePerfLog(const std::vector<PerfLog>& logs, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    // --- UPDATE HEADER ---
    file << "Timestamp,Frame_ID,Calc_Time_ms";
    // Add 36 columns for covariance
    for(int i=0; i<36; ++i) file << ",Cov_" << i;
    file << "\n";
    
    file << std::fixed << std::setprecision(6);
    
    for (const auto& log : logs) {
        file << log.timestamp << "," 
             << log.frame_id << "," 
             << log.alingment_time_ms;

        // --- ADD THIS LOOP ---
        for(const double& val : log.cov_values) {
            file << "," << val;
        }
        file << "\n";
    }
    std::cout << "[IO] Saved performance logs to " << filename << std::endl;
}

// TEMPLATED Helper to save LIO Map
// FIX: Using std::shared_ptr<const ...> allows the compiler to deduce PointT automatically
template <typename PointT>
inline void saveLioMap(const std::shared_ptr<const pcl::PointCloud<PointT>>& map, const std::string& filename) {
    if (!map || map->empty()) return;
    std::cout << "[IO] Saving map (" << map->size() << " pts)..." << std::endl;
    pcl::io::savePCDFileBinaryCompressed(filename, *map);
}

// TEMPLATED Helper to save individual Scans
// FIX: Accepts generic shared_ptr for deduction
template <typename PointT>
inline void saveLioScan(const std::shared_ptr<const pcl::PointCloud<PointT>>& scan, const std::string& filename) {
    if (!scan || scan->empty()) return;
    pcl::io::savePCDFileBinaryCompressed(filename, *scan);
}

#endif // LO_PIPELINE_HPP_