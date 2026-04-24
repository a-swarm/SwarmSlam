#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <mutex>
#include <deque>
#include <tsl/robin_set.h>
#include <eigenreference.hpp>
#include <lidarframe.hpp>
#include <map.hpp>

class LidarCallback {
    public:

        explicit LidarCallback(const std::string& json_meta_path, const std::string& json_param_path);
        explicit LidarCallback(const nlohmann::json& json_meta,const nlohmann::json& json_param);
        std::unique_ptr<LidarFrame> DecodePacket(const std::vector<uint8_t>& packet);
        std::unique_ptr<LidarFrame> DecodePacketRng19(const std::vector<uint8_t>& packet);
        std::unique_ptr<LidarFrame> DecodePacketLegacy(const std::vector<uint8_t>& packet); 
        void ReturnFrameToPool(std::unique_ptr<LidarFrame> frame);

    private:

        void Initialize();
        void InitializePool(size_t pool_size = 4);
        std::unique_ptr<LidarFrame> GetFrameFromPool();
        void ParseMetadata(const nlohmann::json& json_data);
        void ParseParamdata(const nlohmann::json& json_data);
        void DownsampleActiveFrame();

        float resolution_ = 0.0f;

        nlohmann::json metadata_;
        nlohmann::json parameter_;

        Eigen::Matrix3d body_to_lidar_rotation_ = Eigen::Matrix3d::Zero();
        Eigen::Vector3d body_to_lidar_translation_ = Eigen::Vector3d::Zero();
        uint16_t channel_stride_ = 1; // Number of rows to skip (N), default to 1 (process all rows)
        uint16_t column_stride_ = 1; // Number of rows to skip (N), default to 1 (process all rows)
        uint16_t subset_channels_;    // Number of channels in subset tables (ceiling(pixels_per_column_ / N))

        // Original lookup tables
        std::vector<float, Eigen::aligned_allocator<float>> x_1_;
        std::vector<float, Eigen::aligned_allocator<float>> y_1_;
        std::vector<float, Eigen::aligned_allocator<float>> z_1_;
        std::vector<float> x_2_;
        std::vector<float> y_2_;
        std::vector<float> z_2_;
        std::vector<float> r_min_;
        std::vector<float> r_max_;
        std::vector<float> sin_beam_azimuths_;
        std::vector<float> cos_beam_azimuths_;
        std::vector<float> sin_beam_altitudes_;
        std::vector<float> cos_beam_altitudes_;
        std::vector<int> pixel_shifts_;
        // Subset lookup tables for channels that are multiples of N
        std::vector<float, Eigen::aligned_allocator<float>> x_1_subset_;
        std::vector<float, Eigen::aligned_allocator<float>> y_1_subset_;
        std::vector<float, Eigen::aligned_allocator<float>> z_1_subset_;
        std::vector<float> r_min_subset_;
        std::vector<float> r_max_subset_;
        std::vector<float> sin_beam_azimuths_subset_;
        std::vector<float> cos_beam_azimuths_subset_;
        std::vector<float> sin_beam_altitudes_subset_;
        std::vector<float> cos_beam_altitudes_subset_;
        std::vector<int> pixel_shifts_subset_;
        std::vector<uint16_t> subset_c_ids_; // Maps subset indices to original c_id

        Eigen::Matrix4d lidar_to_sensor_transform_;
        float lidar_origin_to_beam_origin_mm_;
        size_t block_size_;
        size_t expected_size_;
        size_t PACKET_HEADER_BYTES = 32;
        size_t PACKET_FOOTER_BYTES = 32;
        size_t COLUMN_HEADER_BYTES = 12;
        size_t CHANNEL_STRIDE_BYTES = 12;
        size_t MEASUREMENT_BLOCK_STATUS_BYTES = 0;
        std::string udp_profile_lidar_ = "UNKNOWN";
        int columns_per_frame_ = 2048;
        int pixels_per_column_ = 128;
        int columns_per_packet_ = 16;
        uint16_t frame_id_ = 0;
        uint32_t number_points_ = 0;
        double latest_timestamp_s = 0.0;

        std::unique_ptr<LidarFrame> active_frame_;
        std::deque<std::unique_ptr<LidarFrame>> frame_pool_;
        std::mutex pool_mutex_;
        
        float zfiltermax_ = 0.0f;
        float zfiltermin_ = -300.0f;
        float rfiltermax_ = 200.0f;
        float rfiltermin_ = 1.0f;
        uint8_t reflectivity_threshold_ = 0;
        Eigen::Vector3f vehicle_box_center_ = Eigen::Vector3f::Zero();
        Eigen::Vector3f vehicle_box_dimensions_ = Eigen::Vector3f::Zero();
        Eigen::Vector3f vehicle_box_min_ = Eigen::Vector3f::Zero();
        Eigen::Vector3f vehicle_box_max_ = Eigen::Vector3f::Zero();
        size_t poolsize_ = 4;
};

#include "lidarcallback_impl.hpp"