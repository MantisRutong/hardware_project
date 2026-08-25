/**
* This file is part of ORB-SLAM3 (local addition, not upstream).
* See include/orb_capi.h for the usage contract.
*/

#include "orb_capi.h"

#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>

#include "System.h"
#include "ImuTypes.h"
#include "Tracking.h"

namespace {
struct OrbHandle {
  std::unique_ptr<ORB_SLAM3::System> sys;
  std::mutex mtx; // guards every call below - caller may feed IMU and stereo frames from different threads
  std::vector<ORB_SLAM3::IMU::Point> imu_buffer;
  bool shut_down = false;
};
} // namespace

void *orb_create(const char *vocab_path, const char *settings_path, int use_viewer) {
  if (vocab_path == nullptr || settings_path == nullptr)
    return nullptr;

  auto *h = new OrbHandle();
  try {
    h->sys = std::make_unique<ORB_SLAM3::System>(std::string(vocab_path), std::string(settings_path),
                                                  ORB_SLAM3::System::IMU_STEREO, use_viewer != 0);
  } catch (const std::exception &) {
    delete h;
    return nullptr;
  }
  return h;
}

void orb_feed_imu(void *handle, double timestamp, double gx, double gy, double gz, double ax, double ay, double az) {
  if (handle == nullptr)
    return;
  auto *h = static_cast<OrbHandle *>(handle);
  std::lock_guard<std::mutex> lck(h->mtx);
  // IMU::Point's ctor is (acc_x,acc_y,acc_z, gyro_x,gyro_y,gyro_z, timestamp) - see include/ImuTypes.h.
  h->imu_buffer.emplace_back(static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(az),
                              static_cast<float>(gx), static_cast<float>(gy), static_cast<float>(gz), timestamp);
}

int orb_track_stereo(void *handle, double timestamp, int width, int height, const unsigned char *left_gray,
                      const unsigned char *right_gray, double *out_xyz3, double *out_quat_xyzw4, long *out_map_epoch) {
  if (handle == nullptr || left_gray == nullptr || right_gray == nullptr || width <= 0 || height <= 0)
    return 0;
  auto *h = static_cast<OrbHandle *>(handle);
  if (h->shut_down)
    return 0;

  // MUST clone here, not just wrap the caller's buffer: Tracking::
  // GrabImageStereo does `mImGray = imRectLeft;` (a *shallow* cv::Mat
  // assignment for single-channel/grayscale input - no cvtColor branch
  // triggers, since these are already CV_8UC1) into Tracking:: member
  // variables that outlive this call (used later by FrameDrawer etc). A
  // Mat built over external memory (as below, before .clone()) carries no
  // OpenCV refcount, so once the caller's buffer is freed/reused (e.g. the
  // RealSense frame it came from gets recycled, or a later call overwrites
  // the same Python-side numpy buffer), ORB-SLAM3 is left holding a
  // dangling pointer - a delayed use-after-free that doesn't crash
  // immediately, only once the stale memory is actually read again. The
  // stock stereo_inertial_realsense_D435i.cc example avoids this the same
  // way (its `im = imCV.clone(); imRight = imRightCV.clone();` before
  // calling TrackStereo) - .clone() here makes left/right own their own
  // refcounted heap buffer, independent of the caller's memory lifetime.
  cv::Mat left = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char *>(left_gray)).clone();
  cv::Mat right = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char *>(right_gray)).clone();

  std::vector<ORB_SLAM3::IMU::Point> imu_meas;
  {
    std::lock_guard<std::mutex> lck(h->mtx);
    imu_meas.swap(h->imu_buffer);
  }

  Sophus::SE3f Tcw = h->sys->TrackStereo(left, right, timestamp, imu_meas);

  // Written regardless of tracking success below, and before the early
  // returns - see out_map_epoch's doc comment in orb_capi.h: the caller
  // needs to see an epoch change even on frames where tracking itself
  // failed, since that's exactly when a reset is most likely to have just
  // happened.
  if (out_map_epoch != nullptr)
    *out_map_epoch = static_cast<long>(h->sys->GetMapEpoch());

  if (h->sys->GetTrackingState() != ORB_SLAM3::Tracking::OK)
    return 0;
  if (out_xyz3 == nullptr || out_quat_xyzw4 == nullptr)
    return 0;

  // Tcw is world-in-camera; invert to get the (left) camera's pose in the
  // world frame, which is what callers actually want to log/plot.
  Sophus::SE3f Twc = Tcw.inverse();
  Eigen::Vector3f t = Twc.translation();
  Eigen::Quaternionf q = Twc.unit_quaternion(); // Eigen stores coeffs as (x,y,z,w) already
  out_xyz3[0] = t.x();
  out_xyz3[1] = t.y();
  out_xyz3[2] = t.z();
  out_quat_xyzw4[0] = q.x();
  out_quat_xyzw4[1] = q.y();
  out_quat_xyzw4[2] = q.z();
  out_quat_xyzw4[3] = q.w();
  return 1;
}

void orb_shutdown(void *handle) {
  if (handle == nullptr)
    return;
  auto *h = static_cast<OrbHandle *>(handle);
  std::lock_guard<std::mutex> lck(h->mtx);
  if (!h->shut_down) {
    h->sys->Shutdown();
    h->shut_down = true;
  }
}

void orb_save_trajectory_tum(void *handle, const char *path) {
  if (handle == nullptr || path == nullptr)
    return;
  auto *h = static_cast<OrbHandle *>(handle);
  h->sys->SaveTrajectoryTUM(std::string(path));
}

void orb_destroy(void *handle) {
  if (handle == nullptr)
    return;
  auto *h = static_cast<OrbHandle *>(handle);
  {
    std::lock_guard<std::mutex> lck(h->mtx);
    if (!h->shut_down) {
      h->sys->Shutdown();
      h->shut_down = true;
    }
  }
  delete h;
}
