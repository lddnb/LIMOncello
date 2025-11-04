#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <optional>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <iomanip> 
#include <type_traits>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>


struct Config {

	bool verbose;
	bool debug;

  struct Topics {
  	struct {
  		std::string lidar;
  		std::string imu;
			std::string stop_ioctree_udate;
  	} input;

  	struct {
  		std::string state;
  		std::string frame;
  	} output;
  	
  	std::string frame_id;
  } topics;


  struct Sensors {
  	struct { 
  		int type; 
  		bool end_of_sweep;
  	} lidar;

  	struct {
  		int hz;
  	} imu;

  	struct {
  		bool gravity;
  		bool accel;
  		bool gyro;
  		float time;
  	} calibration;

  	bool time_offset;
    float TAI_offset;

  	struct {
  		Eigen::Affine3d imu2baselink_T;
  		Eigen::Affine3d lidar2baselink_T;
  		float gravity;
  	} extrinsics;

  	struct {
  		Eigen::Vector3d accel_bias;
  		Eigen::Vector3d gyro_bias;
  		Eigen::Matrix3d sm;
  	} intrinsics;

  } sensors;

  struct Filters {
    struct {
    	Eigen::Vector4d leaf_size;
    } voxel_grid;

    struct {
    	bool active;
    	float value;
    } min_distance;

    struct {
    	bool active;
    	float value;
    } fov;

    struct {
			bool active;
			int value;
    } rate_sampling;

  } filters;

  struct IKFoM {
  	int query_iters;
  	int max_iters;
  	float tolerance;
  	float lidar_noise;

  	struct {
  		float gyro;
  		float accel;
  		float bias_gyro;
  		float bias_accel;
  	} covariance;

  	struct {
  		int points;
  		float max_sqrt_dist;
  		float plane_threshold;
  	} plane;
  } ikfom;

  struct iOctree {
    float min_extent;
    int bucket_size;
    bool downsample;
  } ioctree;

  static Config& getInstance() {
    static Config* config = new Config();
    return *config;
  }

 private:
  // Singleton pattern
  Config() = default;

  // Delete copy/move so extra instances can't be created/moved.
  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(Config&&) = delete;
};

namespace detail
{

template <typename T>
inline void AssignIfPresent(const YAML::Node& node, std::string_view key, T& target)
{
    if (!node || !node.IsMap()) {
        return;
    }

    const YAML::Node value_node = node[std::string(key)];
    if (!value_node || value_node.IsNull()) {
        return;
    }

    try {
        if constexpr (std::is_floating_point_v<T>) {
            target = static_cast<T>(value_node.as<double>());
        } else if constexpr (std::is_same_v<T, std::string>) {
            target = value_node.as<std::string>();
        } else {
            target = value_node.as<T>();
        }
    } catch (const YAML::Exception& ex) {
        spdlog::warn("LIMOncello config - failed to convert '{}' ({})", key, ex.what());
    }
}

template <typename T, std::size_t N>
inline bool AssignArrayIfPresent(const YAML::Node& node, std::string_view key, T (&target)[N])
{
    if (!node || !node.IsMap()) {
        return false;
    }
    const YAML::Node seq = node[std::string(key)];
    if (!seq || !seq.IsSequence() || seq.size() < N) {
        return false;
    }
    try {
        for (std::size_t i = 0; i < N; ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                target[i] = static_cast<T>(seq[i].as<double>());
            } else {
                target[i] = seq[i].as<T>();
            }
        }
        return true;
    } catch (const YAML::Exception& ex) {
        spdlog::warn("LIMOncello config - failed to convert array '{}' ({})", key, ex.what());
    }
    return false;
}

inline std::optional<std::vector<double>> ReadVector(const YAML::Node& node, std::string_view key)
{
    if (!node || !node.IsMap()) {
        return std::nullopt;
    }
    const YAML::Node seq = node[std::string(key)];
    if (!seq || !seq.IsSequence()) {
        return std::nullopt;
    }
    std::vector<double> result;
    result.reserve(seq.size());
    try {
        for (const auto& item : seq) {
            result.emplace_back(item.as<double>());
        }
    } catch (const YAML::Exception& ex) {
        spdlog::warn("LIMOncello config - failed to convert sequence '{}' ({})", key, ex.what());
        return std::nullopt;
    }
    return result;
}

inline Eigen::Matrix3d RpyToRotation(const std::vector<double>& rpy_deg)
{
    if (rpy_deg.size() != 3) {
        return Eigen::Matrix3d::Identity();
    }
    const double roll  = rpy_deg[0] * M_PI / 180.0;
    const double pitch = rpy_deg[1] * M_PI / 180.0;
    const double yaw   = rpy_deg[2] * M_PI / 180.0;
    return (Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ())).toRotationMatrix();
}

inline void LoadTopics(const YAML::Node& node, Config& cfg)
{
    if (!node || !node.IsMap()) {
        return;
    }
    const auto input = node["input"];
    AssignIfPresent(input, "lidar", cfg.topics.input.lidar);
    AssignIfPresent(input, "imu", cfg.topics.input.imu);
    AssignIfPresent(input, "stop_ioctree_udate", cfg.topics.input.stop_ioctree_udate);

    const auto output = node["output"];
    AssignIfPresent(output, "state", cfg.topics.output.state);
    AssignIfPresent(output, "frame", cfg.topics.output.frame);

    AssignIfPresent(node, "frame_id", cfg.topics.frame_id);
}

inline void LoadSensors(const YAML::Node& node, Config& cfg)
{
    if (!node || !node.IsMap()) {
        return;
    }

    const auto lidar = node["lidar"];
    AssignIfPresent(lidar, "type", cfg.sensors.lidar.type);
    AssignIfPresent(lidar, "end_of_sweep", cfg.sensors.lidar.end_of_sweep);

    const auto imu = node["imu"];
    AssignIfPresent(imu, "hz", cfg.sensors.imu.hz);

    const auto calibration = node["calibration"];
    AssignIfPresent(calibration, "gravity", cfg.sensors.calibration.gravity);
    AssignIfPresent(calibration, "accel", cfg.sensors.calibration.accel);
    AssignIfPresent(calibration, "gyro", cfg.sensors.calibration.gyro);
    AssignIfPresent(calibration, "time", cfg.sensors.calibration.time);

    AssignIfPresent(node, "time_offset", cfg.sensors.time_offset);
    AssignIfPresent(node, "TAI_offset", cfg.sensors.TAI_offset);

    const auto extrinsics = node["extrinsics"];
    cfg.sensors.extrinsics.imu2baselink_T.setIdentity();
    cfg.sensors.extrinsics.lidar2baselink_T.setIdentity();

    if (extrinsics && extrinsics.IsMap())
    {
        if (auto vec = ReadVector(extrinsics["imu2baselink"], "t"))
        {
            cfg.sensors.extrinsics.imu2baselink_T.translate(Eigen::Vector3d((*vec)[0], (*vec)[1], (*vec)[2]));
        }
        if (auto vec = ReadVector(extrinsics["imu2baselink"], "R"))
        {
            cfg.sensors.extrinsics.imu2baselink_T.rotate(RpyToRotation(*vec));
        }

        if (auto vec = ReadVector(extrinsics["lidar2baselink"], "t"))
        {
            cfg.sensors.extrinsics.lidar2baselink_T.translate(Eigen::Vector3d((*vec)[0], (*vec)[1], (*vec)[2]));
        }
        if (auto vec = ReadVector(extrinsics["lidar2baselink"], "R"))
        {
            cfg.sensors.extrinsics.lidar2baselink_T.rotate(RpyToRotation(*vec));
        }

        AssignIfPresent(extrinsics, "gravity", cfg.sensors.extrinsics.gravity);
    }

    const auto intrinsics = node["intrinsics"];
    if (auto vec = ReadVector(intrinsics, "accel_bias"))
    {
        cfg.sensors.intrinsics.accel_bias = Eigen::Vector3d((*vec)[0], (*vec)[1], (*vec)[2]);
    }
    if (auto vec = ReadVector(intrinsics, "gyro_bias"))
    {
        cfg.sensors.intrinsics.gyro_bias = Eigen::Vector3d((*vec)[0], (*vec)[1], (*vec)[2]);
    }
    if (auto vec = ReadVector(intrinsics, "sm"))
    {
        if (vec->size() == 9)
        {
            cfg.sensors.intrinsics.sm << (*vec)[0], (*vec)[1], (*vec)[2],
                                         (*vec)[3], (*vec)[4], (*vec)[5],
                                         (*vec)[6], (*vec)[7], (*vec)[8];
        }
    }
}

inline void LoadFilters(const YAML::Node& node, Config& cfg)
{
    if (!node || !node.IsMap()) {
        return;
    }

    if (auto vec = ReadVector(node["voxel_grid"], "leaf_size"))
    {
        if (vec->size() >= 3)
        {
            cfg.filters.voxel_grid.leaf_size = Eigen::Vector4d((*vec)[0], (*vec)[1], (*vec)[2], 1.0);
        }
    }

    const auto min_distance = node["min_distance"];
    AssignIfPresent(min_distance, "active", cfg.filters.min_distance.active);
    AssignIfPresent(min_distance, "value", cfg.filters.min_distance.value);

    const auto fov = node["fov"];
    AssignIfPresent(fov, "active", cfg.filters.fov.active);
    if (fov && fov.IsMap())
    {
        if (const YAML::Node value = fov["value"]; value && value.IsScalar())
        {
            cfg.filters.fov.value = static_cast<float>(value.as<double>() * M_PI / 360.0);
        }
    }

    const auto rate_sampling = node["rate_sampling"];
    AssignIfPresent(rate_sampling, "active", cfg.filters.rate_sampling.active);
    AssignIfPresent(rate_sampling, "value", cfg.filters.rate_sampling.value);
}

inline void LoadIKFoM(const YAML::Node& node, Config& cfg)
{
    if (!node || !node.IsMap()) {
        return;
    }

    AssignIfPresent(node, "query_iters", cfg.ikfom.query_iters);
    AssignIfPresent(node, "max_iters", cfg.ikfom.max_iters);
    AssignIfPresent(node, "tolerance", cfg.ikfom.tolerance);
    AssignIfPresent(node, "lidar_noise", cfg.ikfom.lidar_noise);

    const auto covariance = node["covariance"];
    AssignIfPresent(covariance, "gyro", cfg.ikfom.covariance.gyro);
    AssignIfPresent(covariance, "accel", cfg.ikfom.covariance.accel);
    AssignIfPresent(covariance, "bias_gyro", cfg.ikfom.covariance.bias_gyro);
    AssignIfPresent(covariance, "bias_accel", cfg.ikfom.covariance.bias_accel);

    const auto plane = node["plane"];
    AssignIfPresent(plane, "points", cfg.ikfom.plane.points);
    AssignIfPresent(plane, "max_sqrt_dist", cfg.ikfom.plane.max_sqrt_dist);
    AssignIfPresent(plane, "plane_threshold", cfg.ikfom.plane.plane_threshold);
}

inline void LoadIoctree(const YAML::Node& node, Config& cfg)
{
    if (!node || !node.IsMap()) {
        return;
    }
    AssignIfPresent(node, "min_extent", cfg.ioctree.min_extent);
    AssignIfPresent(node, "bucket_size", cfg.ioctree.bucket_size);
    AssignIfPresent(node, "downsample", cfg.ioctree.downsample);
}

}  // namespace detail

inline bool LoadConfigFromFile(Config& cfg, const std::string& path)
{
    try
    {
        YAML::Node root = YAML::LoadFile(path);

        cfg.verbose = false;
        cfg.debug = false;
        cfg.topics = Config::Topics{};
        cfg.sensors = Config::Sensors{};
        cfg.filters = Config::Filters{};
        cfg.ikfom = Config::IKFoM{};
        cfg.ioctree = Config::iOctree{};

        cfg.sensors.extrinsics.imu2baselink_T.setIdentity();
        cfg.sensors.extrinsics.lidar2baselink_T.setIdentity();
        cfg.sensors.intrinsics.accel_bias.setZero();
        cfg.sensors.intrinsics.gyro_bias.setZero();
        cfg.sensors.intrinsics.sm.setIdentity();
        cfg.sensors.extrinsics.gravity = 9.81f;
        cfg.sensors.time_offset = false;
        cfg.sensors.TAI_offset = 0.0f;
        cfg.filters.voxel_grid.leaf_size = Eigen::Vector4d(1.0, 1.0, 1.0, 1.0);

        detail::AssignIfPresent(root, "verbose", cfg.verbose);
        detail::AssignIfPresent(root, "debug", cfg.debug);

        detail::LoadTopics(root["topics"], cfg);
        detail::LoadSensors(root["sensors"], cfg);
        detail::LoadFilters(root["filters"], cfg);
        detail::LoadIKFoM(root["IKFoM"], cfg);
        detail::LoadIoctree(root["iOctree"], cfg);

        return true;
    }
    catch (const YAML::Exception& ex)
    {
        spdlog::error("Failed to load LIMOncello configuration '{}': {}", path, ex.what());
        return false;
    }
}
