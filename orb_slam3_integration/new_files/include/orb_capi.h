/**
* This file is part of ORB-SLAM3 (local addition, not upstream).
*
* orb_capi: a minimal C-linkage wrapper around ORB_SLAM3::System (Stereo-
* Inertial mode), so an ORB-SLAM3 tracking session can be embedded live
* inside a non-C++ process (e.g. driven via Python's ctypes) that already
* owns the camera/IMU source itself - mirrors open_vins/ov_msckf/src/ov_capi.h,
* which this project already uses the same way for OpenVINS.
*
* Every call is internally mutex-guarded, so the caller can feed IMU samples
* and stereo frames from different threads without its own locking.
*
* Usage from the caller's side, once per tracking session:
*   void *h = orb_create("/path/to/ORBvoc.txt", "/path/to/settings.yaml", 0);
*   orb_feed_imu(h, t, gx, gy, gz, ax, ay, az);         // called repeatedly, as samples arrive
*   double xyz[3], quat_xyzw[4]; long map_epoch;
*   if (orb_track_stereo(h, t, w, h, left_gray, right_gray, xyz, quat_xyzw, &map_epoch)) { ... }
*   ...
*   orb_shutdown(h);                                     // stops SLAM threads - call before saving
*   orb_save_trajectory_tum(h, "/path/to/camera_trajectory.txt");
*   orb_destroy(h);
*
* left_gray/right_gray must each be a single-channel (grayscale), row-major,
* width*height byte buffer, already rectified to match the settings file's
* calibration (Camera.type: "Rectified") - same expectation as the stock
* stereo_inertial_realsense_D435i.cc example.
*/

#ifndef ORB_CAPI_H
#define ORB_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Builds an ORB_SLAM3::System in IMU_STEREO mode from the given vocabulary
// and settings files. use_viewer: nonzero to launch the Pangolin viewer
// (useful for standalone visual sanity checks), 0 for headless (the normal
// case when embedding inside a process that already has its own live view,
// e.g. synced_capture.py's monitor window). Returns an opaque handle, or
// nullptr if construction failed (e.g. bad vocab/settings path). Each handle
// is a fresh, independent tracking session - to start over (e.g. between
// recording episodes), destroy this handle and create a new one rather than
// trying to reset one in place.
void *orb_create(const char *vocab_path, const char *settings_path, int use_viewer);

// Buffers one IMU sample (gyro gx,gy,gz in rad/s, accel ax,ay,az in m/s^2,
// timestamp in seconds - any consistent clock, must be the same clock used
// for orb_track_stereo's timestamps). Samples accumulate internally and are
// all consumed (and cleared) by the next orb_track_stereo call, same
// buffering pattern as the stock RealSense example.
void orb_feed_imu(void *handle, double timestamp, double gx, double gy, double gz, double ax, double ay, double az);

// Feeds one rectified stereo frame pair and runs tracking, using every IMU
// sample buffered via orb_feed_imu since the last call. Writes the resulting
// camera pose (position of the left camera in the world frame, and
// orientation as a Hamilton-convention xyzw quaternion) into out_xyz3 (3
// doubles) and out_quat_xyzw4 (4 doubles) if tracking succeeded. Returns 1 in
// that case, 0 if tracking is lost / not yet initialized (in which case the
// output buffers are left untouched).
//
// out_map_epoch (may be NULL if the caller doesn't care) is always written,
// tracked or not: System::GetMapEpoch(), read in the same call as the pose
// so it can't race a reset happening between two separate calls. ORB-SLAM3
// can silently reset in a way that moves the active map's pose origin --
// e.g. after "Not enough motion for initializing. Reseting..." or "IMU is
// not or recently initialized. Reseting active map..." -- without that
// being visible in the return value or the tracking state (which reports
// OK either way once tracking resumes post-reset). A change in
// *out_map_epoch from one call to the next is how a caller detects "the
// pose just written is NOT in the same coordinate frame as poses from
// before this call". NOT simply "the active map's ID": live testing showed
// the common reset path clears and re-initializes the SAME map object in
// place (its own ID never changes) -- see GetMapEpoch()'s doc comment in
// System.h for why a dedicated counter is used instead.
int orb_track_stereo(void *handle, double timestamp, int width, int height, const unsigned char *left_gray,
                      const unsigned char *right_gray, double *out_xyz3, double *out_quat_xyzw4, long *out_map_epoch);

// The id of the map tracking is currently in
// (ORB_SLAM3::System::GetCurrentMapId -- see its doc comment in System.h).
//
// What this is for: with an atlas loaded, "tracking is OK" does NOT mean
// "localized in that atlas". LoadAtlas restores the saved map (id 0) but the
// session then cold-starts its OWN map ("Creation of new map with id: 1")
// and tracks there, which is precisely the cold start the atlas exists to
// avoid, until LoopClosing recognises the region and merges -- at which
// point Atlas::ChangeMap switches the active map and this id becomes the
// atlas's. So a caller wanting to know "am I in the atlas yet" watches this,
// not the tracking state and not out_map_epoch (a merge is not a reset, so
// the epoch counter never moves for it).
//
// 0 if handle is NULL. Cheap enough to call per frame.
long orb_get_current_map_id(void *handle);

// Stops all ORB-SLAM3 threads (local mapping, loop closing). Must be called
// before orb_save_trajectory_tum. Safe to call more than once (a no-op after
// the first call) and safe to skip calling directly - orb_destroy calls it
// automatically if not already done.
void orb_shutdown(void *handle);

// Saves the tracked trajectory in TUM format (timestamp x y z qx qy qz qw).
// Call after orb_shutdown.
void orb_save_trajectory_tum(void *handle, const char *path);

// Frees the handle (shutting it down first if orb_shutdown wasn't already
// called). Safe to call once per successful orb_create; do not use the
// handle again afterward.
void orb_destroy(void *handle);

#ifdef __cplusplus
}
#endif

#endif // ORB_CAPI_H
