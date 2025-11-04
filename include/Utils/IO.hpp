#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <flatbuffers/flatbuffers.h>
#include <spdlog/spdlog.h>

#include <slam_common/foxglove_messages.hpp>
#include <slam_adapter/sensor_preprocess.hpp>
#include <slam_core/imu.hpp>

#include "Utils/Config.hpp"
#include "Utils/PCL.hpp"

namespace limoncello
{

struct PointFieldOffsets
{
    int32_t x{-1};
    int32_t y{-1};
    int32_t z{-1};
    int32_t intensity{-1};
    int32_t time{-1};
    int32_t ring{-1};
    int32_t reflectivity{-1};
    int32_t timestamp{-1};
};

inline PointFieldOffsets ExtractOffsets(const foxglove::PointCloud& msg)
{
    PointFieldOffsets offsets;
    const auto* fields = msg.fields();
    if (!fields) {
        return offsets;
    }

    for (const auto* field : *fields)
    {
        if (!field || !field->name()) {
            continue;
        }
        const std::string name = field->name()->str();
        if (name == "x") {
            offsets.x = static_cast<int32_t>(field->offset());
        } else if (name == "y") {
            offsets.y = static_cast<int32_t>(field->offset());
        } else if (name == "z") {
            offsets.z = static_cast<int32_t>(field->offset());
        } else if (name == "intensity") {
            offsets.intensity = static_cast<int32_t>(field->offset());
        } else if (name == "time" || name == "t") {
            offsets.time = static_cast<int32_t>(field->offset());
        } else if (name == "ring") {
            offsets.ring = static_cast<int32_t>(field->offset());
        } else if (name == "reflectivity") {
            offsets.reflectivity = static_cast<int32_t>(field->offset());
        } else if (name == "timestamp") {
            offsets.timestamp = static_cast<int32_t>(field->offset());
        }
    }
    return offsets;
}

inline bool ConvertPointCloudMessage(const foxglove::PointCloud& msg, PointCloudT& cloud)
{
    const auto* data = msg.data();
    const uint32_t stride = msg.point_stride();
    if (!data || stride == 0) {
        spdlog::warn("LIMOncello - received empty point cloud message");
        return false;
    }

    const PointFieldOffsets offsets = ExtractOffsets(msg);
    if (offsets.x < 0 || offsets.y < 0 || offsets.z < 0 || offsets.intensity < 0)
    {
        spdlog::error("LIMOncello - point cloud missing XYZ or intensity fields");
        return false;
    }

    const std::size_t point_count = data->size() / stride;
    cloud.clear();
    cloud.reserve(point_count);
    cloud.header.frame_id = msg.frame_id() ? msg.frame_id()->str() : "";
    cloud.width = static_cast<uint32_t>(point_count);
    cloud.height = 1;
    cloud.is_dense = false;

    const auto& cfg = Config::getInstance();

    for (std::size_t i = 0; i < point_count; ++i)
    {
        const std::uint8_t* base = data->Data() + i * stride;
        PointT pt{};
        std::memcpy(&pt.x, base + offsets.x, sizeof(float));
        std::memcpy(&pt.y, base + offsets.y, sizeof(float));
        std::memcpy(&pt.z, base + offsets.z, sizeof(float));
        std::memcpy(&pt.intensity, base + offsets.intensity, sizeof(float));

        if (cfg.sensors.lidar.type == 0)  // Ouster
        {
            if (offsets.time >= 0) {
                std::memcpy(&pt.t, base + offsets.time, sizeof(std::uint32_t));
            } else {
                pt.t = 0U;
            }
        }
        else if (cfg.sensors.lidar.type == 1)  // Velodyne
        {
            if (offsets.time >= 0) {
                std::memcpy(&pt.time, base + offsets.time, sizeof(float));
            } else {
                pt.time = 0.f;
            }
        }
        else if (cfg.sensors.lidar.type == 2 || cfg.sensors.lidar.type == 3)  // Hesai / Livox
        {
            if (offsets.timestamp >= 0) {
                std::memcpy(&pt.timestamp, base + offsets.timestamp, sizeof(double));
            } else {
                pt.timestamp = 0.0;
            }
        }

        cloud.push_back(pt);
    }

    cloud.width = static_cast<uint32_t>(cloud.size());
    cloud.height = 1;

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(cloud, cloud, indices);

    if (!cloud.points.empty())
    {
        auto minmax = std::minmax_element(cloud.points.begin(), cloud.points.end(), get_point_time_comp());
        if (minmax.first != cloud.points.begin()) {
            std::iter_swap(minmax.first, cloud.points.begin());
        }
        if (minmax.second != cloud.points.end() - 1) {
            std::iter_swap(minmax.second, cloud.points.end() - 1);
        }
    }

    return !cloud.points.empty();
}

inline bool ConvertImuMessage(const foxglove::Imu& message, ms_slam::slam_core::IMU& imu_out)
{
    ms_slam::slam_core::IMU temp;
    if (!ms_slam::slam_adapter::ConvertImuMessage(message, temp, true))
    {
        return false;
    }

    const auto& cfg = Config::getInstance();
    const Eigen::Vector3d gyro = temp.angular_velocity();
    const Eigen::Vector3d accel = temp.linear_acceleration() * cfg.sensors.extrinsics.gravity;
    imu_out = ms_slam::slam_core::IMU(gyro, accel, temp.timestamp());
    return true;
}

inline bool BuildPointCloudMessage(const PointCloudT& cloud,
                                   std::string_view frame_id,
                                   double timestamp,
                                   flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();

    constexpr std::size_t kStride = sizeof(float) * 4;
    std::vector<std::uint8_t> data(cloud.points.size() * kStride);

    for (std::size_t idx = 0; idx < cloud.points.size(); ++idx)
    {
        const PointT& pt = cloud.points[idx];
        std::uint8_t* base = data.data() + idx * kStride;
        std::memcpy(base + 0, &pt.x, sizeof(float));
        std::memcpy(base + 4, &pt.y, sizeof(float));
        std::memcpy(base + 8, &pt.z, sizeof(float));
        std::memcpy(base + 12, &pt.intensity, sizeof(float));
    }

    const auto field_x = foxglove::CreatePackedElementField(builder, builder.CreateString("x"), 0, foxglove::NumericType_FLOAT32);
    const auto field_y = foxglove::CreatePackedElementField(builder, builder.CreateString("y"), 4, foxglove::NumericType_FLOAT32);
    const auto field_z = foxglove::CreatePackedElementField(builder, builder.CreateString("z"), 8, foxglove::NumericType_FLOAT32);
    const auto field_intensity = foxglove::CreatePackedElementField(builder, builder.CreateString("intensity"), 12, foxglove::NumericType_FLOAT32);

    const std::array fields{field_x, field_y, field_z, field_intensity};

    const std::uint32_t sec = static_cast<std::uint32_t>(timestamp);
    const std::uint32_t nsec = static_cast<std::uint32_t>(std::clamp((timestamp - static_cast<double>(sec)) * 1e9, 0.0, 999999999.0));
    const foxglove::Time time_struct(sec, nsec);

    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());
    const auto fields_vector = builder.CreateVector(fields.data(), fields.size());
    const auto data_vector = builder.CreateVector(data);
    const auto origin_translation = foxglove::CreateVector3(builder, 0.0, 0.0, 0.0);
    const auto origin_rotation = foxglove::CreateQuaternion(builder, 0.0, 0.0, 0.0, 1.0);
    const auto pose_offset = foxglove::CreatePose(builder, origin_translation, origin_rotation);

    const auto pointcloud_offset = foxglove::CreatePointCloud(builder, &time_struct, frame_id_offset, pose_offset, kStride, fields_vector, data_vector);
    foxglove::FinishPointCloudBuffer(builder, pointcloud_offset);
    return true;
}

struct PoseSample
{
    double timestamp{0.0};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

inline bool BuildPoseMessage(const PoseSample& pose,
                             std::string_view frame_id,
                             flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();
    const std::uint32_t sec = static_cast<std::uint32_t>(pose.timestamp);
    const std::uint32_t nsec = static_cast<std::uint32_t>(std::clamp((pose.timestamp - static_cast<double>(sec)) * 1e9, 0.0, 999999999.0));
    const foxglove::Time time_struct(sec, nsec);
    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());
    const auto position_offset = foxglove::CreateVector3(builder, pose.position.x(), pose.position.y(), pose.position.z());
    const Eigen::Quaterniond q = pose.orientation.normalized();
    const auto orientation_offset = foxglove::CreateQuaternion(builder, q.x(), q.y(), q.z(), q.w());
    const auto pose_offset = foxglove::CreatePose(builder, position_offset, orientation_offset);
    const auto message_offset = foxglove::CreatePoseInFrame(builder, &time_struct, frame_id_offset, pose_offset);
    foxglove::FinishPoseInFrameBuffer(builder, message_offset);
    return true;
}

inline bool BuildPathMessage(const std::vector<PoseSample>& path,
                             double timestamp,
                             std::string_view frame_id,
                             flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();
    std::vector<flatbuffers::Offset<foxglove::Pose>> pose_offsets;
    pose_offsets.reserve(path.size());
    for (const auto& pose : path)
    {
        const auto position_offset = foxglove::CreateVector3(builder, pose.position.x(), pose.position.y(), pose.position.z());
        const Eigen::Quaterniond q = pose.orientation.normalized();
        const auto orientation_offset = foxglove::CreateQuaternion(builder, q.x(), q.y(), q.z(), q.w());
        pose_offsets.emplace_back(foxglove::CreatePose(builder, position_offset, orientation_offset));
    }

    const auto poses_vector = builder.CreateVector(pose_offsets);
    const std::uint32_t sec = static_cast<std::uint32_t>(timestamp);
    const std::uint32_t nsec = static_cast<std::uint32_t>(std::clamp((timestamp - static_cast<double>(sec)) * 1e9, 0.0, 999999999.0));
    const foxglove::Time time_struct(sec, nsec);
    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());
    const auto path_offset = foxglove::CreatePosesInFrame(builder, &time_struct, frame_id_offset, poses_vector);
    foxglove::FinishPosesInFrameBuffer(builder, path_offset);
    return true;
}

inline double TimeToSeconds(const foxglove::Time* stamp)
{
    if (!stamp) {
        return 0.0;
    }
    return static_cast<double>(stamp->sec()) + static_cast<double>(stamp->nsec()) * 1e-9;
}

}  // namespace limoncello
