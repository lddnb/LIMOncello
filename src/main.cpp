#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <future>

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/console/print.h>
#include <spdlog/spdlog.h>

#include <flatbuffers/flatbuffers.h>
#include <iox2/iceoryx2.hpp>

#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>

#include "Core/Cloud.hpp"
#include "Core/Imu.hpp"
#include "Core/Octree.hpp"
#include "Core/State.hpp"
#include "Utils/Config.hpp"
#include "Utils/IO.hpp"

using namespace ms_slam::slam_common;

namespace
{

class Manager
{
  public:
    Manager(std::shared_ptr<FBSPublisher<FoxglovePoseInFrame>> state_pub,
            std::shared_ptr<FBSPublisher<FoxglovePointCloud>> frame_pub,
            std::shared_ptr<FBSPublisher<FoxglovePosesInFrame>> path_pub)
        : state_buffer_(1000),
          pub_state_(std::move(state_pub)),
          pub_frame_(std::move(frame_pub)),
          pub_path_(std::move(path_pub)),
          path_history_(),
          pose_builder_(1024),
          cloud_builder_(1024 * 1024),
          path_builder_(1024 * 32)
    {
        Config& cfg = Config::getInstance();

        imu_calibrated_ = !(cfg.sensors.calibration.gravity || cfg.sensors.calibration.accel || cfg.sensors.calibration.gyro);
        ioctree_.setBucketSize(cfg.ioctree.bucket_size);
        ioctree_.setDownsample(cfg.ioctree.downsample);
        ioctree_.setMinExtent(cfg.ioctree.min_extent);
        stop_ioctree_update_ = false;

        first_imu_stamp_ = -1.0;
        prev_imu_timestamp_ = -1.0;
    }

    void HandleImu(const Imu& imu_msg)
    {
        Config& cfg = Config::getInstance();

        if (first_imu_stamp_ < 0.0) {
            first_imu_stamp_ = imu_msg.timestamp();
        }

        if (!imu_calibrated_) {
            static int N = 0;
            static Eigen::Vector3d gyro_avg = Eigen::Vector3d::Zero();
            static Eigen::Vector3d accel_avg = Eigen::Vector3d::Zero();
            static Eigen::Vector3d grav_vec(0.0, 0.0, cfg.sensors.extrinsics.gravity);

            if ((imu_msg.timestamp() - first_imu_stamp_) < cfg.sensors.calibration.time) {
                gyro_avg += imu_msg.angular_velocity();
                accel_avg += imu_msg.linear_acceleration();
                ++N;
                return;
            }

            if (N > 0) {
                gyro_avg /= static_cast<double>(N);
                accel_avg /= static_cast<double>(N);
            }

            if (cfg.sensors.calibration.gravity) {
                grav_vec = accel_avg.normalized() * std::abs(cfg.sensors.extrinsics.gravity);
                state_.g(-grav_vec);
            }

            if (cfg.sensors.calibration.gyro) {
                state_.b_w(gyro_avg);
            }

            if (cfg.sensors.calibration.accel) {
                state_.b_a(accel_avg - grav_vec);
            }

            imu_calibrated_ = true;
            prev_imu_ = imu_msg;
            prev_imu_timestamp_ = imu_msg.timestamp();

            spdlog::info(
                "Initialize with {} IMU:g = [{:.3f}, {:.3f}, {:.3f}], b_g = [{:.3f}, {:.3f}, {:.3f}], b_a = [{:.3f}, {:.3f}, {:.3f}], timestamp = {:.3f}",
                N,
                state_.g().x(), state_.g().y(), state_.g().z(),
                state_.b_w().x(), state_.b_w().y(), state_.b_w().z(),
                state_.b_a().x(), state_.b_a().y(), state_.b_a().z(),
                imu_msg.timestamp());
            state_.t(imu_msg.timestamp());
            state_buffer_.push_front(state_);

            return;
        }

        double dt = (prev_imu_timestamp_ > 0.0) ? imu_msg.timestamp() - prev_imu_timestamp_ : 0.0;
        if (dt <= 0.0 || dt >= imu_msg.timestamp()) {
            dt = 1.0 / static_cast<double>(cfg.sensors.imu.hz);
        }

        Imu transformed = limoncello::imu2baselink(imu_msg, dt);
        const Eigen::Vector3d corrected_acc = cfg.sensors.intrinsics.sm * transformed.linear_acceleration();
        const Imu corrected(transformed.angular_velocity(), corrected_acc, transformed.timestamp());

        prev_imu_ = corrected;
        prev_imu_timestamp_ = corrected.timestamp();

        {
            std::lock_guard<std::mutex> lock(mtx_state_);
            state_.predict(corrected, dt);
        }
        // spdlog::info("[state] predict ts {}, pos: {:.3f} {:.3f} {:.3f}, quat: {:.3f} {:.3f} {:.3f} {:.3f}", state_.stamp, state_.p().x(), state_.p().y(), state_.p().z(), state_.quat().x(), state_.quat().y(), state_.quat().z(), state_.quat().w());

        {
            std::lock_guard<std::mutex> lock(mtx_buffer_);
            state_buffer_.push_front(state_);
        }

        cv_prop_stamp_.notify_one();

        PublishPose();
    }

    void HandlePointCloud(const PointCloudT::Ptr& raw, double header_stamp)
    {
        Config& cfg = Config::getInstance();

        if (!raw || raw->points.empty()) {
            spdlog::error("[LIMOncello] Raw point cloud is empty");
            return;
        }

        if (!imu_calibrated_) {
            return;
        }

        if (state_buffer_.empty()) {
            spdlog::error("[LIMOncello] No IMUs received");
            return;
        }

        PointTime point_time = point_time_func();
        double sweep_time = header_stamp + cfg.sensors.TAI_offset;

        double offset = 0.0;
        if (cfg.sensors.time_offset) {
            offset = state_.stamp - point_time(raw->points.back(), sweep_time) - 1.e-4;
            if (offset > 0.0) {
                offset = 0.0;
            }
        }

        double start_stamp = point_time(raw->points.front(), sweep_time) + offset;
        double end_stamp = point_time(raw->points.back(), sweep_time) + offset;

        if (state_buffer_.front().stamp < end_stamp) {
            std::unique_lock<std::mutex> lock(mtx_buffer_);
            cv_prop_stamp_.wait(lock, [this, end_stamp] { return state_buffer_.front().stamp >= end_stamp; });
        }

        States interpolated(250);
        {
            std::lock_guard<std::mutex> lock(mtx_buffer_);
            interpolated = filter_states(state_buffer_, start_stamp, end_stamp);
        }
        spdlog::info("[Lidar] stamp: {:.3f}, size: {}", start_stamp, raw->points.size());
        spdlog::info("[state] state_buffer_.front: {:.3f}", state_buffer_.front().stamp);
        spdlog::info("[state] interpolated.front: {:.3f}", interpolated.front().stamp);

        if (interpolated.empty() || start_stamp < interpolated.front().stamp) {
            spdlog::warn("Not enough interpolated states for deskewing point cloud");
            return;
        }

        PointCloudT::Ptr deskewed = deskew(raw, state_, interpolated, offset, sweep_time);
        PointCloudT::Ptr downsampled = voxel_grid(deskewed);
        spdlog::info("[Lidar] downsize {}", downsampled->points.size());
        PointCloudT::Ptr processed = process(downsampled);
        spdlog::info("[Lidar] processed {}", processed->points.size());
        static size_t count = 0;
        // pcl::io::savePCDFileBinary("/home/ubuntu/data/test_bag/noros_processed_" + std::to_string(count++) + ".pcd", *processed);

        if (processed->points.empty()) {
            spdlog::error("[LIMOncello] Processed & downsampled cloud is empty");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mtx_state_);
            state_.update(processed, ioctree_);
        }

        const Eigen::Affine3f T = (state_.affine3d() * state_.I2L_affine3d()).cast<float>();
        PointCloudT::Ptr global(std::make_shared<PointCloudT>());
        pcl::transformPointCloud(*deskewed, *global, T);
        pcl::transformPointCloud(*processed, *processed, T);

        if (!stop_ioctree_update_) {
            ioctree_.update(processed->points);
        }

        PublishPose();
        PublishPointCloud(*global);
        UpdateAndPublishPath();

        if (cfg.verbose) {
            PROFC_PRINT()
        }
    }

    void StopUpdates(bool stop)
    {
        stop_ioctree_update_ = stop;
        spdlog::info("LIMOncello - ioctree updates {}", stop ? "disabled" : "enabled");
    }

  private:
    void PublishPose()
    {
        if (!pub_state_) {
            return;
        }
        limoncello::PoseSample pose;
        {
            std::lock_guard<std::mutex> lock(mtx_state_);
            pose.timestamp = state_.stamp;
            pose.position = state_.p();
            pose.orientation = state_.quat();
        }

        if (limoncello::BuildPoseMessage(pose, Config::getInstance().topics.frame_id, pose_builder_)) {
            pub_state_->publish_from_builder(pose_builder_);
        }

        latest_pose_ = pose;
    }

    void PublishPointCloud(const PointCloudT& cloud)
    {
        if (!pub_frame_) {
            return;
        }
        if (limoncello::BuildPointCloudMessage(cloud, Config::getInstance().topics.frame_id, latest_pose_.timestamp, cloud_builder_)) {
            pub_frame_->publish_from_builder(cloud_builder_);
        }
    }

    void UpdateAndPublishPath()
    {
        if (!pub_path_) {
            return;
        }

        path_history_.push_back(latest_pose_);
        constexpr std::size_t kMaxPathSize = 2048;
        if (path_history_.size() > kMaxPathSize) {
            path_history_.erase(path_history_.begin(), path_history_.begin() + (path_history_.size() - kMaxPathSize));
        }

        if (limoncello::BuildPathMessage(path_history_, latest_pose_.timestamp, Config::getInstance().topics.frame_id, path_builder_)) {
            pub_path_->publish_from_builder(path_builder_);
        }
    }

    State state_;
    States state_buffer_;

    Imu prev_imu_;
    double prev_imu_timestamp_{-1.0};
    double first_imu_stamp_{-1.0};
    bool imu_calibrated_{false};

    std::mutex mtx_state_;
    std::mutex mtx_buffer_;
    std::condition_variable cv_prop_stamp_;

    charlie::Octree ioctree_;
    std::atomic<bool> stop_ioctree_update_{false};

    std::shared_ptr<FBSPublisher<FoxglovePoseInFrame>> pub_state_;
    std::shared_ptr<FBSPublisher<FoxglovePointCloud>> pub_frame_;
    std::shared_ptr<FBSPublisher<FoxglovePosesInFrame>> pub_path_;

    limoncello::PoseSample latest_pose_{};
    std::vector<limoncello::PoseSample> path_history_;
    flatbuffers::FlatBufferBuilder pose_builder_;
    flatbuffers::FlatBufferBuilder cloud_builder_;
    flatbuffers::FlatBufferBuilder path_builder_;
};

}  // namespace

int main(int argc, char** argv)
{
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    const std::string config_path = (argc > 1) ? std::string(argv[1]) : std::string("../config/mid360.yaml");
    if (!LoadConfigFromFile(Config::getInstance(), config_path))
    {
        spdlog::error("Failed to load configuration file: {}", config_path);
        return 1;
    }

    const Config& cfg = Config::getInstance();

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Failed to create iceoryx node"));

    auto pub_state = std::make_shared<FBSPublisher<FoxglovePoseInFrame>>(node, "/odom");
    auto pub_frame = std::make_shared<FBSPublisher<FoxglovePointCloud>>(node, "/cloud_registered");
    auto pub_path = std::make_shared<FBSPublisher<FoxglovePosesInFrame>>(node, "/path");

    Manager manager(pub_state, pub_frame, pub_path);

    auto lidar_callback = [&manager](const FoxglovePointCloud& wrapper) {
        const auto* msg = wrapper.get();
        if (!msg) {
            return;
        }
        PointCloudT cloud;
        if (!limoncello::ConvertPointCloudMessage(*msg, cloud)) {
            return;
        }
        PointCloudT::Ptr raw(std::make_shared<PointCloudT>(cloud));
        manager.HandlePointCloud(raw, limoncello::TimeToSeconds(msg->timestamp()));
    };

    auto imu_callback = [&manager](const FoxgloveImu& wrapper) {
        const auto* msg = wrapper.get();
        if (!msg) {
            return;
        }
        Imu imu;
        if (!limoncello::ConvertImuMessage(*msg, imu)) {
            return;
        }
        manager.HandleImu(imu);
    };

    auto lidar_sub = std::make_shared<ThreadedFBSSubscriber<FoxglovePointCloud>>(node, cfg.topics.input.lidar, lidar_callback);
    auto imu_sub = std::make_shared<ThreadedFBSSubscriber<FoxgloveImu>>(
        node,
        cfg.topics.input.imu,
        imu_callback,
        ms_slam::slam_common::PubSubConfig{.subscriber_max_buffer_size = 100});
    lidar_sub->start();
    imu_sub->start();

    spdlog::info("LIMOncello started with configuration: {}", config_path);

    std::promise<void>().get_future().wait();
    lidar_sub->stop();
    imu_sub->stop();
    return 0;
}
