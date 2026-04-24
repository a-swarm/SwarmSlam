// ==============================================================================
// FILE: lo_pipeline.cpp
// DESCRIPTION: 
//   High-Performance LIO Pipeline.
//   - Asynchronous Decoding of UDP packets.
//   - Precise Lidar-IMU synchronization with interpolation.
//   - Real-time LIO Optimization (GTSAM).
//   - Ground Truth generation from INS data (LLA -> NED).
//   - Export of registered, deskewed point clouds.
// ==============================================================================

#include <lo_pipeline.hpp>
#include <cstdlib>

// Define the point type used throughout the pipeline
using PointT = pcl::PointXYZI;

// ------------------------------------------------------------------------------
// GLOBAL SIGNALS & STATE
// ------------------------------------------------------------------------------
static std::atomic<bool> running{true};
static std::mutex running_mutex;
static std::condition_variable running_cv;

// [In lio_pipeline.cpp - Global Scope]
static double global_ref_lat = 0.0, global_ref_lon = 0.0, global_ref_alt = 0.0;
static double global_ref_qx = 0.0, global_ref_qy = 0.0, global_ref_qz = 0.0, global_ref_qw = 1.0;

/**
 * @brief Signal Handler for graceful shutdown (e.g., Ctrl+C)
 */
void signal_handler(int) {
    std::cout << "\n[System] Caught signal, initiating shutdown..." << std::endl;
    {
        std::lock_guard<std::mutex> lock(running_mutex);
        running = false;
    }
    running_cv.notify_all(); // Wake up all waiting threads
}

// ------------------------------------------------------------------------------
// MAIN ENTRY POINT
// ------------------------------------------------------------------------------
int main() {

    // Pin the Main Thread to E-Cores (16-23) so it stays out of the P-Core pool
    setThreadAffinity(18, 18, "Main Thread");

    // 1. SETUP SIGNAL HANDLING
    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // 2. CONFIGURATION PATHS
    // Ensure these paths point to valid JSON configuration files
    const std::string meta_lidar = "../../../package/lidarcallback/config/lidar_meta_berlin_asv1.json";
    const std::string param_lidar = "../../../package/lidarcallback/config/lidar_config_berlin.json";
    // const std::string param_imu = "../../../extern/compcallback/config/imu_config_berlin.json";
    const std::string param_lo = "../../../package/lo_smoother/config/lo_smoother_config.json";

    const std::string param_solver = "../../../package/ndt_generic/config/ndt_generic_config.json";

    // 3. INITIALIZE DECODERS
    // These classes handle the raw byte parsing of sensor packets
    LidarCallback lidarCallback(meta_lidar, param_lidar);
    // CompCallback compCallback(param_imu);

    // 4. DATA STRUCTURES (Thread-Safe Queues & Object Pools)
    // Raw UDP packet queues
    FrameQueue<DataBuffer> packetLidarQueue, packetCompQueue;
    
    // Decoded frame queues
    FrameQueue<LidarFrame> frameLidarQueue;
    // FrameQueue<std::deque<CompFrameIMU>> frameCompQueue; // For passing windows of IMU data
    
    // Synchronized data queue (Ready for LIO)
    FrameQueue<FrameData> framedata;
    
    // Object Pools to reduce heap allocation overhead
    // ObjectPool<std::deque<CompFrameIMU>> comp_window_pool(8), comp_window_sync_pool(8);
    ObjectPool<FrameData> frame_data_pool(8);
    ObjectPool<pcl::PointCloud<pcl::PointXYZI>> pcl_pool(8);

    // 5. INITIALIZE LIO SYSTEM
    auto lo_smoother = std::make_unique<lo_smoother::LoSmoother<PointT>>(param_lo, param_solver);

    // Trajectory Storage for Evaluation (TUM Format)
    std::vector<TumPose> gt_tum_trajectory;  // Ground Truth (from INS)
    std::vector<TumPose> lo_tum_trajectory; // Estimated (from SLAM)
    std::vector<PerfLog> performance_logs;
    std::mutex trajectory_mutex;
    std::mutex log_mutex;
    std::mutex lidar_mutex;
    
    // Global Origin (Lat, Lon, Alt) for Metric Conversion
    Eigen::Vector3d origin_lla = Eigen::Vector3d::Zero();

    // --------------------------------------------------------------------------
    // NETWORK SETUP (UDP SOCKETS)
    // --------------------------------------------------------------------------
    // boost::asio::io_context comp_iocontext;
    boost::asio::io_context lidar_iocontext;
    
    // // --- COMPASS (INS) SOCKET ---
    // UdpSocketConfig compUdpConfig;
    // compUdpConfig.host = "192.168.75.10";
    // compUdpConfig.localInterfaceIp = "192.168.75.10";
    // compUdpConfig.port = 6597; 
    // compUdpConfig.bufferSize = 105;
    // compUdpConfig.reuseAddress = true; 

    // // Capture packets and push to queue
    // auto comp_socket = UdpSocket::create(comp_iocontext, compUdpConfig, 
    //     [&](std::unique_ptr<DataBuffer> p) { if(running && p) packetCompQueue.push(std::move(p)); }, 
    //     [](const boost::system::error_code& ec) { if(running) std::cerr << "[UDP Comp] Error: " << ec.message() << "\n"; });

    // --- LIDAR SOCKET ---
    UdpSocketConfig lidarUdpConfig;
    lidarUdpConfig.host = "192.168.75.11";
    lidarUdpConfig.localInterfaceIp = "192.168.75.11";
    lidarUdpConfig.port = 7502;
    lidarUdpConfig.bufferSize = 24832;
    lidarUdpConfig.reuseAddress = true; 

    auto lidar_socket = UdpSocket::create(lidar_iocontext, lidarUdpConfig, 
        [&](std::unique_ptr<DataBuffer> p) { if(running && p) packetLidarQueue.push(std::move(p)); }, 
        [](const boost::system::error_code& ec) { if(running) std::cerr << "[UDP Lidar] Error: " << ec.message() << "\n"; });

    // // Run ASIO services in background threads (Pinned to E-Cores)
    // auto comp_iothread = std::thread([&comp_iocontext]() { 
    //     setThreadAffinity(16, 19, "Comp IO Thread");
    //     comp_iocontext.run(); 
    // });
    auto lidar_iothread = std::thread([&lidar_iocontext]() { 
        setThreadAffinity(12, 13, "Lidar IO Thread");
        lidar_iocontext.run(); 
    });

    // --------------------------------------------------------------------------
    // THREAD: LIDAR PROCESSING (Decoder)
    // --------------------------------------------------------------------------
    auto lidar_processing_thread = std::thread([&]() {
        setThreadAffinity(14, 16, "Lidar Processing Thread");
        std::cout << "[Thread] Lidar Processing thread started.\n";
        uint16_t last_processed_frame_id = 0; // Local state

        while (running) {
            auto packet_ptr = packetLidarQueue.pop();
            if (!packet_ptr) break; 
            
            auto frame_ptr = lidarCallback.DecodePacket(*packet_ptr);
            if (!frame_ptr) continue; 

            // Logic to discard out-of-order/old frames
            bool is_old_frame = false;
            
            if (frame_ptr->frame_id <= last_processed_frame_id && 
               (last_processed_frame_id - frame_ptr->frame_id) < 1000) { 
                is_old_frame = true;
            } else {
                last_processed_frame_id = frame_ptr->frame_id;
            }

            if (is_old_frame) {
                lidarCallback.ReturnFrameToPool(std::move(frame_ptr)); 
                continue;
            }

            frameLidarQueue.push(std::move(frame_ptr));
        }
        std::cout << "[Thread] Lidar Processing processing stopped.\n";  
    });
    
    // --------------------------------------------------------------------------
    // THREAD: COMPASS PROCESSING (Decoder & Windowing)
    // --------------------------------------------------------------------------
    // auto comp_processing_thread = std::thread([&]() {
    //     setThreadAffinity(20, 23, "Compass Processing Thread");
    //     std::cout << "[Thread] Compass Processing thread started.\n";
    //     double comp_latest_timestamp = std::numeric_limits<double>::lowest();
    //     size_t comp_window_size = 64; 
    //     auto comp_window = std::make_unique<std::deque<CompFrameIMU>>();

    //     while (running) {
    //         auto packet_ptr = packetCompQueue.pop();
    //         if (!packet_ptr) break; 
            
    //         auto frame_ptr = compCallback.DecodePacket(*packet_ptr);
    //         if (!frame_ptr) continue; 

    //         if (frame_ptr->timestamp_20 <= comp_latest_timestamp) {
    //             compCallback.ReturnFrameToPool(std::move(frame_ptr)); 
    //             continue;
    //         }
    //         comp_latest_timestamp = frame_ptr->timestamp_20;

    //         CompFrameIMU frame_imu = frame_ptr->toCompFrameIMU();
    //         comp_window->push_back(std::move(frame_imu));
    //         compCallback.ReturnFrameToPool(std::move(frame_ptr));

    //         if (comp_window->size() > comp_window_size) {
    //             comp_window->pop_front();
    //         }

    //         if (comp_window->size() == comp_window_size) {
    //             auto comp_window_copy_ptr = comp_window_pool.Get();
    //             *comp_window_copy_ptr = *comp_window; 
    //             frameCompQueue.push(std::move(comp_window_copy_ptr));
    //         }
    //     }
    //     std::cout << "[Thread] Compass Processing processing stopped.\n";    
    // });

    // --------------------------------------------------------------------------
    // THREAD: SYNC & MAPPING
    // --------------------------------------------------------------------------
    auto sync_thread = std::thread([&]() {
        setThreadAffinity(17, 18, "Sync Thread");
        std::cout << "[Thread] sync thread started.\n";

        // --- Config ---
        // bool enable_time_filtering = true; 
        // double start_timestamp = std::fmod(static_cast<double>(1753866469.72) , 86400.0);

        // std::unique_ptr<std::deque<CompFrameIMU>> comp_window_frame_ptr = nullptr;
        
        double lidar_last_timestamp = std::numeric_limits<double>::lowest();
        bool is_system_initialized = false; 
        // CompFrameIMU imu_interpolated_start; 

        // auto getInterpolated = [&](double target_time, const std::deque<CompFrameIMU>& window) -> CompFrameIMU {
        //     if (window.empty()) return CompFrameIMU();
        //     if (target_time <= window.front().timestamp_20) return window.front();
        //     if (target_time >= window.back().timestamp_20) return window.back();
            
        //     for (size_t i = 0; i < window.size() - 1; ++i) {
        //         const auto& a = window[i];
        //         const auto& b = window[i + 1];
        //         if (a.timestamp_20 <= target_time && target_time <= b.timestamp_20) {
        //             double dt = b.timestamp_20 - a.timestamp_20;
        //             double t = (dt > 1e-9) ? (target_time - a.timestamp_20) / dt : 0.0;
        //             return a.linearInterpolate(a, b, t);
        //         }
        //     }
        //     return window.back();
        // };

        while (running) {
            auto lidar_frame_ptr = frameLidarQueue.pop();
            if (!lidar_frame_ptr) break; 
            
            if (lidar_frame_ptr->timestamp_points.size() < 2) { 
                lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
                continue;
            }

            const double t_lidar_end = lidar_frame_ptr->timestamp_end;

            // if (enable_time_filtering) {
            //     if (t_lidar_end < start_timestamp) {
            //         std::cout << "[Sync] Waiting for start... (" << start_timestamp - t_lidar_end << "s left)\n";
            //         lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
            //         continue; 
            //     }
            // }

            if (!is_system_initialized) { 
                lidar_last_timestamp = t_lidar_end; 
                is_system_initialized = true;
                lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
                std::cout << "[Sync] System Initialized at: " << std::fixed << t_lidar_end << "\n";
                continue; 
            }

            const double t_start = lidar_last_timestamp; 
            const double t_end = t_lidar_end;
            // bool valid_window_found = false;
            
            // while (running) {
            //     comp_window_frame_ptr = frameCompQueue.pop();
            //     if (!comp_window_frame_ptr || comp_window_frame_ptr->empty()) break;

            //     double win_start = comp_window_frame_ptr->front().timestamp_20;
            //     double win_end   = comp_window_frame_ptr->back().timestamp_20;

            //     if (win_end < t_start) {
            //         comp_window_pool.Return(std::move(comp_window_frame_ptr));
            //         continue; 
            //     }

            //     if (win_start > t_start) {
            //         std::cerr << "[Sync] GAP ERROR: Needed " << t_start << " but IMU starts at " << win_start << ". Resetting.\n";
            //         valid_window_found = false;
            //         comp_window_pool.Return(std::move(comp_window_frame_ptr));
            //         break; 
            //     }

            //     if (win_end >= t_end) {
            //         valid_window_found = true;
            //         break; 
            //     } 
            //     else {
            //         comp_window_pool.Return(std::move(comp_window_frame_ptr));
            //         continue; 
            //     }
            // }

            if (!running) break;
            
            // if (!valid_window_found) {
            //     lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
            //     is_system_initialized = false; 
            //     continue;
            // }

            // if (imu_interpolated_start.timestamp_20 == 0.0) {
            //     imu_interpolated_start = getInterpolated(t_start, *comp_window_frame_ptr);
            // }
            
            // CompFrameIMU imu_interpolated_end = getInterpolated(t_end, *comp_window_frame_ptr);

            // auto comp_window_sync_ptr = comp_window_sync_pool.Get();
            // comp_window_sync_ptr->push_back(imu_interpolated_start); 
            
            // for (const auto& data : *comp_window_frame_ptr) {
            //     if (data.timestamp_20 > t_start && data.timestamp_20 < t_end) { 
            //         comp_window_sync_ptr->push_back(data);
            //     }
            // }

            // comp_window_sync_ptr->push_back(imu_interpolated_end);

            auto pcl_frame_ptr = pcl_pool.Get();
            if (!lidar_frame_ptr->getPointCloud(*pcl_frame_ptr)) {
                lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
                // comp_window_sync_pool.Return(std::move(comp_window_sync_ptr));
                pcl_pool.Return(std::move(pcl_frame_ptr));
                // comp_window_pool.Return(std::move(comp_window_frame_ptr));
                is_system_initialized = false;
                continue;
            }

            auto frame_data_ptr = frame_data_pool.Get();
            frame_data_ptr->timestamp = t_end;
            frame_data_ptr->frame_id = lidar_frame_ptr->frame_id;
            frame_data_ptr->points = std::move(*pcl_frame_ptr);
            // frame_data_ptr->imu_window = std::move(*comp_window_sync_ptr);

            framedata.push(std::move(frame_data_ptr));

            lidar_last_timestamp = t_end; 
            // imu_interpolated_start = imu_interpolated_end; 

            lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
            pcl_pool.Return(std::move(pcl_frame_ptr)); 
            // comp_window_pool.Return(std::move(comp_window_frame_ptr));
        }
        std::cout << "[Thread] Sync processing stopped.\n";       
    });

    // --------------------------------------------------------------------------
    // THREAD: LIO SMOOTHER & LOGGING
    // --------------------------------------------------------------------------
    auto lo_thread = std::thread([&]() {
        // --- PIN STRICTLY TO P-CORES (0-15) ---
        std::cout << "[Thread] Smoother thread started.\n";
        setThreadAffinity(1, 7, "LO/Solver Thread");
        

        // --- Config (Moved End Logic Here) ---
        // bool enable_time_filtering = true;
        // double end_timestamp = std::fmod(static_cast<double>(1753866652.62) , 86400.0);

        bool origin_set = false;
        bool is_first_sync_packet = true;
        
        while (running) {
            auto frame_ptr = framedata.pop();
            if (!frame_ptr) break;

            // if (enable_time_filtering) {
            //     if (frame_ptr->timestamp > end_timestamp) {
            //         std::cout << "[LO] Reached End Timestamp (" << frame_ptr->timestamp << " > " << end_timestamp << "). Stopping...\n";
            //         {
            //             std::lock_guard<std::mutex> lock(running_mutex);
            //             running = false; 
            //         }
            //         running_cv.notify_all(); 
            //         frame_data_pool.Return(std::move(frame_ptr));
            //         break; 
            //     }
            // }

            // if (!frame_ptr->imu_window.empty()) {
            if (!origin_set) {
                // const auto& first_imu = frame_ptr->imu_window.back();
                origin_lla = Eigen::Vector3d(0.0, 0.0, 0.0);
                origin_set = true;
                
                global_ref_lat = origin_lla.x();
                global_ref_lon = origin_lla.y();
                global_ref_alt = origin_lla.z();
                global_ref_qw = 1.0;
                global_ref_qx = 0.0;
                global_ref_qy = 0.0;
                global_ref_qz = 0.0;
                
                lo_smoother->setOrigin(origin_lla.x(), origin_lla.y(), origin_lla.z());
            }

            // {
            //     std::lock_guard<std::mutex> lock(trajectory_mutex);
                
            //     double last_gt_time = -1.0;
            //     if (!gt_tum_trajectory.empty()) {
            //         last_gt_time = gt_tum_trajectory.back().timestamp;
            //     }
            //     if (is_first_sync_packet) {
            //         const auto& imu = frame_ptr->imu_window.back();
            //         Eigen::Vector3d ned = GeoUtils::LLAtoNED_Exact(
            //             imu.latitude_20, imu.longitude_20, imu.altitude_20,
            //             origin_lla.x(), origin_lla.y(), origin_lla.z()
            //         );
            //         gt_tum_trajectory.push_back({
            //                 imu.timestamp_20,
            //                 ned.x(), ned.y(), ned.z(),
            //                 imu.qx_20, imu.qy_20, imu.qz_20, imu.qw_20 
            //             });
            //         is_first_sync_packet = false;
            //     } else {
            //         for (const auto& imu : frame_ptr->imu_window) {
            //             if (imu.timestamp_20 <= last_gt_time) {
            //                 continue; 
            //             }

            //             Eigen::Vector3d ned = GeoUtils::LLAtoNED_Exact(
            //                 imu.latitude_20, imu.longitude_20, imu.altitude_20,
            //                 origin_lla.x(), origin_lla.y(), origin_lla.z()
            //             );
                        
            //             gt_tum_trajectory.push_back({
            //                 imu.timestamp_20,
            //                 ned.x(), ned.y(), ned.z(),
            //                 imu.qx_20, imu.qy_20, imu.qz_20, imu.qw_20 
            //             });
            //         }
            //     }
            // }
            // }

            if (!origin_set) {
                frame_data_pool.Return(std::move(frame_ptr));
                continue; 
            }

            auto cloud_ptr = frame_ptr->points.makeShared(); 
            lo_smoother->update(frame_ptr->timestamp, cloud_ptr);//, frame_ptr->imu_window);
            lo_smoother::LoState state;
            lo_smoother->get_current_state(state);

            {
                std::lock_guard<std::mutex> lock(log_mutex);
                
                std::array<double, 36> cov_flat;
                int idx = 0;
                for(int r = 0; r < 6; ++r) {
                    for(int c = 0; c < 6; ++c) {
                        cov_flat[idx++] = state.reg_covariance(r, c);
                    }
                }

                performance_logs.push_back({
                    state.timestamp, 
                    frame_ptr->frame_id, 
                    state.alingment_time_ms,
                    cov_flat 
                });
            }
            {
                std::lock_guard<std::mutex> lock(trajectory_mutex);
                lo_tum_trajectory.push_back({
                    state.timestamp,
                    state.north, state.east, state.down, 
                    state.q_nb.x(), state.q_nb.y(), state.q_nb.z(), state.q_nb.w()
                });
            }

            frame_data_pool.Return(std::move(frame_ptr));
        }
        std::cout << "\n[Thread] Smoother thread stopped.\n";
    });

    // -------------------------------------------------------------------------- //
    // THREAD: VISUALIZATION (PCL) - DYNAMIC LIO PIPELINE WITH CUSTOM STL
    // -------------------------------------------------------------------------- //
    auto viz_thread = std::thread([&]() {
        setThreadAffinity(8, 11, "Visualization Thread");
        std::cout << "[Thread] Visualization thread started.\n";

        vtkObject::GlobalWarningDisplayOff();

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("LO Pipeline"));
        viewer->setSize(1920, 1080);

        int v_main = 0;
        viewer->setBackgroundColor(0.05, 0.05, 0.05, v_main);
        viewer->addText("Initializing...", 1450, 50, 32, 1.0, 1.0, 1.0, "gps_text", v_main);

        // =================================================================
        // 1. LOAD CUSTOM STL (No pre-processing needed)
        // =================================================================
        pcl::PolygonMesh boat_mesh;
        bool has_boat_model = false;
        std::string mesh_id = "boat_mesh";

        if (pcl::io::loadPolygonFileSTL("../../../package/utils/boat.stl", boat_mesh) > 0) {
            has_boat_model = true;
            viewer->addPolygonMesh(boat_mesh, mesh_id);
            
            // Bright Silver color for high visibility (RGB: 0.9, 0.9, 0.9)
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.9, 0.9, 0.9, mesh_id); 
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1.0, mesh_id);
            std::cout << "[Viz] Successfully loaded custom NED boat.stl\n";
        } else {
            std::cerr << "[Viz] Failed to load boat.stl entirely.\n";
        }

        // --- SMOOTHING & TRAJECTORY VARIABLES ---
        Eigen::Vector3d smoothed_look_at = Eigen::Vector3d::Zero();
        Eigen::Vector3d smoothed_cam_pos = Eigen::Vector3d::Zero();
        bool camera_initialized = false;
        const double lerp_alpha = 0.05; 

        size_t last_kf_idx = 0;
        int traj_id = 0;
        Eigen::Vector3d last_pos = Eigen::Vector3d::Zero();
        bool has_last_pos = false;

        while (running && !viewer->wasStopped()) {
            lo_smoother::LoState state;
            lo_smoother->get_current_state(state);

            // Only update visualization if we have a valid initialized state
            if (lo_smoother->is_initialized()) {
                Eigen::Vector3d current_pos(state.north, state.east, state.down);

                // =================================================================
                // 2. SMOOTHED CAMERA UPDATE (Original wide-tracking view)
                // =================================================================
                Eigen::Vector3d forward_vec = state.q_nb * Eigen::Vector3d(1, 0, 0);

                // Target camera position: 180m behind, 90m above (Original zoom)
                Eigen::Vector3d target_cam_pos = current_pos - (180.0 * forward_vec) + Eigen::Vector3d(0, 0, -90.0);

                if (!camera_initialized) {
                    smoothed_look_at = current_pos;
                    smoothed_cam_pos = target_cam_pos;
                    camera_initialized = true;
                } else {
                    smoothed_look_at = (lerp_alpha * current_pos) + ((1.0 - lerp_alpha) * smoothed_look_at);
                    smoothed_cam_pos = (lerp_alpha * target_cam_pos) + ((1.0 - lerp_alpha) * smoothed_cam_pos);
                }

                viewer->setCameraPosition(
                    smoothed_cam_pos.x(), smoothed_cam_pos.y(), smoothed_cam_pos.z(),
                    smoothed_look_at.x(), smoothed_look_at.y(), smoothed_look_at.z(),
                    0.0, 0.0, -1.0, // NED Sky
                    v_main
                );

                // =================================================================
                // 3. UPDATE GPS TEXT
                // =================================================================
                // const double rad_to_deg = 180.0 / M_PI;
                // double lat_deg = state.lat * rad_to_deg;
                // double lon_deg = state.lon * rad_to_deg;

                // std::stringstream ss;
                // ss << std::fixed << std::setprecision(7)
                // << "Lat: " << lat_deg << "\u00B0\n"
                // << "Lon: " << lon_deg << "\u00B0\n"
                // << std::setprecision(2) << "Alt: " << state.alt << " m";

                // viewer->updateText(ss.str(), 1450, 50, 32, 1.0, 1.0, 1.0, "gps_text");

                // =================================================================
                // 4. DYNAMIC VEHICLE MESH POSE UPDATE (WITH YAW CORRECTION)
                // =================================================================
                Eigen::Quaternionf q_final = state.q_nb.cast<float>().normalized();

                // Apply a 4-degree counter-clockwise rotation offset.
                // In NED, Z is down. Looking from above, CCW is a negative rotation around Z.
                float yaw_correction_rad = -5.711004f * (M_PI / 180.0f); 
                Eigen::Quaternionf q_offset(Eigen::AngleAxisf(yaw_correction_rad, Eigen::Vector3f::UnitZ()));
                
                // Right-multiply to apply the offset in the vehicle's local frame
                q_final = q_final * q_offset;

                Eigen::Affine3f vehicle_pose = Eigen::Affine3f::Identity();
                vehicle_pose.translation() << current_pos.cast<float>();
                vehicle_pose.rotate(q_final); 

                if (has_boat_model) {
                    viewer->updatePointCloudPose(mesh_id, vehicle_pose);
                }

                // =================================================================
                // 5. TRAJECTORY 
                // =================================================================
                if (has_last_pos && (current_pos - last_pos).norm() > 0.5) {
                    std::string line_id = "traj_" + std::to_string(traj_id++);
                    pcl::PointXYZ p1(last_pos.x(), last_pos.y(), last_pos.z());
                    pcl::PointXYZ p2(current_pos.x(), current_pos.y(), current_pos.z());
                    viewer->addLine(p1, p2, 0.0, 1.0, 1.0, line_id);
                    
                    // Increased line width from 1.5 to 4.0 for better visibility
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4.0, line_id);
                    last_pos = current_pos;
                } else if (!has_last_pos) {
                    last_pos = current_pos;
                    has_last_pos = true;
                }

                // =================================================================
                // 6. DYNAMIC MAP LOADING
                // =================================================================
                const auto& keyframes = lo_smoother->get_keyframes();
                if (last_kf_idx < keyframes.size()) {
                    pcl::PointCloud<PointT>::Ptr kf_cloud(new pcl::PointCloud<PointT>());
                    if (pcl::io::loadPCDFile(keyframes[last_kf_idx].cloud_filename, *kf_cloud) == 0) {
                        
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
                        for (const auto& pt : kf_cloud->points) {
                            pcl::PointXYZRGB p;
                            p.x = pt.x; p.y = pt.y; p.z = pt.z;
                            float r = std::min(1.0f, std::max(0.0f, pt.intensity / 255.0f));
                            if (r < 0.5f) {
                                float t = r * 2.0f;
                                p.r = 65 + t * (57 - 65); p.g = 105 + t * (255 - 105); p.b = 225 + t * (20 - 225);
                            } else {
                                float t = (r - 0.5f) * 2.0f;
                                p.r = 57 + t * (255 - 57); p.g = 255 + t * (182 - 255); p.b = 20 + t * (193 - 20);
                            }
                            rgb_cloud->push_back(p);
                        }

                        std::string layer_id = "kf_" + std::to_string(last_kf_idx);
                        viewer->addPointCloud(rgb_cloud, layer_id);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, layer_id);
                    }
                    last_kf_idx++;
                }
            }

            viewer->spinOnce(10);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    // --------------------------------------------------------------------------
    // MAIN LOOP (Keep main thread alive)
    // --------------------------------------------------------------------------
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // --------------------------------------------------------------------------
    // SHUTDOWN & SAVE
    // --------------------------------------------------------------------------
    std::cout << "\n[System] Stopping IO Sockets...\n";
    lidar_socket->stop();
    // comp_socket->stop();
    // comp_iocontext.stop();
    lidar_iocontext.stop();

    std::cout << "[System] Stopping Queues...\n";
    packetCompQueue.stop();
    packetLidarQueue.stop();
    frameLidarQueue.stop();
    // frameCompQueue.stop();
    framedata.stop();
    
    std::cout << "[System] Joining Threads...\n";
    // if (comp_iothread.joinable()) comp_iothread.join();
    if (lidar_iothread.joinable()) lidar_iothread.join();
    if (lidar_processing_thread.joinable()) lidar_processing_thread.join();
    // if (comp_processing_thread.joinable()) comp_processing_thread.join();
    if (sync_thread.joinable()) sync_thread.join();
    if (lo_thread.joinable()) lo_thread.join();
    if (viz_thread.joinable()) viz_thread.join();

    // --------------------------------------------------------------------------
    // FINAL SAVE
    // --------------------------------------------------------------------------
    std::cout << "[System] Saving Trajectories..." << std::endl;
    
    // saveTumTrajectory(gt_tum_trajectory, "./../output/gt/gt.tum", 
    //                   global_ref_lat, global_ref_lon, global_ref_alt,
    //                   global_ref_qx, global_ref_qy, global_ref_qz, global_ref_qw);

    saveTumTrajectory(lo_tum_trajectory, "./../output/odom/ndt-generic.tum", 
                      global_ref_lat, global_ref_lon, global_ref_alt,
                      global_ref_qx, global_ref_qy, global_ref_qz, global_ref_qw);
    
    savePerfLog(performance_logs, "./../output/log/ndt-generic.csv");
    
    saveLioMap(lo_smoother->get_map(), "./../output/map/gm.pcd");

    std::cout << "[System] Shutdown complete." << std::endl;
    return 0;
}