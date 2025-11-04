#pragma once

#include <Eigen/Dense>

#include <slam_core/imu.hpp>

#include "Utils/Config.hpp"

namespace limoncello
{

using Imu = ms_slam::slam_core::IMU;

inline Imu imu2baselink(const Imu& imu, double dt)
{
    const auto& cfg = Config::getInstance();
    const Eigen::Matrix3d R = cfg.sensors.extrinsics.imu2baselink_T.linear();
    const Eigen::Vector3d t = cfg.sensors.extrinsics.imu2baselink_T.translation();
    return ms_slam::slam_core::imu2baselink(imu, R, t, dt);
}

}  // namespace limoncello

using limoncello::Imu;
