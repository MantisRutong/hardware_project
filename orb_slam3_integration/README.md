# ORB-SLAM3 setup

This directory captures our local additions to [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
that make it usable as a live Stereo-Inertial tracking backend embedded directly inside
this project's Python capture process (`output_script/synced_capture.py`), instead of
running as ORB-SLAM3's own standalone example binary. The full ORB-SLAM3 (and Pangolin)
source trees are NOT checked into this repo -- they're large (1.8GB, mostly upstream
code and build artifacts) and separately cloneable. This directory is just the diff.

## Why ORB-SLAM3

We first tried OpenVINS (monocular VIO). It requires periodic stationary pauses to
correct drift via zero-velocity updates -- under continuous handheld motion with no
pauses, its scale estimate diverges unboundedly (seen up to thousands of meters in
testing, for a few feet of real motion). ORB-SLAM3's Stereo-Inertial mode gets real
metric scale directly from the D435i's known IR stereo baseline instead of estimating
it online, which eliminates that failure mode -- validated over 50+ seconds of
continuous motion with zero pauses.

## Setup from scratch

```bash
# 1. Clone ORB-SLAM3 at the exact base commit these changes apply to, and Pangolin.
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
git checkout 4452a3c
git clone https://github.com/stevenlovegrove/Pangolin.git ../Pangolin   # no local changes needed

# 2. Apply our changes to upstream files, and add our new files.
git apply /path/to/hardware_project/orb_slam3_integration/orb_slam3_local_changes.patch
cp -r /path/to/hardware_project/orb_slam3_integration/new_files/include/orb_capi.h include/
cp -r /path/to/hardware_project/orb_slam3_integration/new_files/src/orb_capi.cc src/
mkdir -p config
cp /path/to/hardware_project/orb_slam3_integration/new_files/config/RealSense_D435i_ours.yaml config/

# 3. Download the vocabulary file (145MB, not checked in anywhere -- GitHub's
#    100MB hard file-size limit would reject it, and it's a stock ORB-SLAM3 asset
#    anyway, not something we modified).
cd Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd ..
# (if ORBvoc.txt.tar.gz isn't present, see the upstream repo's own instructions)

# 4. Build Pangolin, then ORB-SLAM3 (including the new orb_capi target).
#    IMPORTANT: this machine's shell auto-activates conda, which picks up the
#    wrong glog/gflags/Eigen/etc. and breaks the build -- strip conda out of
#    PATH for every cmake/make invocation:
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v anaconda3 | paste -sd:)

cd ../Pangolin && mkdir build && cd build && cmake .. && make -j$(nproc)

cd ../../ORB_SLAM3
chmod +x build.sh && ./build.sh   # builds Thirdparty/DBoW2, Thirdparty/g2o, libORB_SLAM3.so, Examples/
mkdir -p build && cd build && cmake .. && make -j$(nproc) orb_capi   # our extra target -> lib/liborb_capi.so
```

## What each piece is

- **`orb_slam3_local_changes.patch`** -- a `git diff` against upstream commit `4452a3c`,
  covering:
  - `CMakeLists.txt`: C++11 -> C++14 (compiler compatibility on this machine); adds the
    `orb_capi` shared-library target (`src/orb_capi.cc` + `include/orb_capi.h`, linked
    against `ORB_SLAM3`) that builds `lib/liborb_capi.so`.
  - `src/Settings.cc`: fixes a crash when printing stereo params for
    `Camera.type: "Rectified"` configs (a single shared pinhole model + `Stereo.b`
    baseline -- exactly the calibration format `config/RealSense_D435i_ours.yaml` uses).
  - `include/System.h` / `src/System.cc`: adds `System::GetMapEpoch()`, a counter that
    changes on every ORB-SLAM3 reset that can move the active map's pose origin.
    `GetCurrentMap()->GetId()` alone is NOT a reliable "did a reset just happen" signal:
    the common reset path (`Tracking::ResetActiveMap()`, triggered by e.g. "Not enough
    motion for initializing" or "IMU is not or recently initialized") clears and
    re-initializes the SAME `Map` object in place via `Atlas::clearMap()`, so its ID
    never changes even though the origin does. Confirmed empirically: a live recording
    showed two real resets that a naive map-ID check reported as zero.
  - `include/Atlas.h` / `src/Atlas.cc`: the actual counter (`mnResetGeneration`),
    incremented inside both `CreateNewMap()` and `clearMap()` so it catches every reset
    path, backing `System::GetMapEpoch()` above.
  - `Examples/*.cc`: three example binaries had `bUseViewer` flipped `true`->`false`;
    unrelated to our tracking path (different dataset-replay examples), left over from
    getting the initial build working headless on this machine.
- **`new_files/include/orb_capi.h`, `new_files/src/orb_capi.cc`** -- a minimal
  C-linkage wrapper around `ORB_SLAM3::System` (Stereo-Inertial mode), so a tracking
  session can be driven from Python via `ctypes` (see
  `camera/orbslam_bridge.py`'s `OrbSlamTracker`) by a process that already owns the
  camera, instead of ORB-SLAM3 opening its own RealSense connection the way the stock
  `Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc` example does. Mirrors
  `open_vins/ov_msckf/src/ov_capi.h`'s role for OpenVINS in this same project.
- **`new_files/config/RealSense_D435i_ours.yaml`** -- Stereo-Inertial calibration
  measured live from this specific D435I unit (serial 346222071398) via
  `rs-enumerate-devices -c`, not assumed/borrowed. See the file's own header comment
  for the full provenance and cross-check against a prior session's calibration for
  the same camera.

## How this plugs into the rest of the project

- `camera/orbslam_bridge.py` -- ctypes wrapper (`OrbSlamTracker`) around
  `lib/liborb_capi.so`. Default paths assume ORB-SLAM3 is built at
  `/home/hakan/Desktop/umi/ORB_SLAM3` (sibling of this repo's parent `umi/` folder) --
  override via `OrbSlamTracker(settings_path=..., vocab_path=..., lib_path=...)` or the
  matching `--orbslam-*` CLI flags on `output_script/synced_capture.py` if built
  elsewhere.
- `output_script/synced_capture.py` -- `--orbslam`/`--no-orbslam` (default: **on**, the
  primary live tracker) runs ORB-SLAM3 Stereo-Inertial tracking on its own thread
  (`OrbSlamWorker`) alongside recording, writing live pose to `camera_trajectory.csv`
  per episode (columns include `map_epoch` -- see its docstring on
  `write_camera_trajectory_csv` for what a reset means for that file, and for
  downstream training-data use). Also auto-enables stereo IR capture
  (`camera/camera_collecting.py`'s `record_ir`) since Stereo-Inertial tracking needs
  both IR streams, not just color.

Run `output_script/synced_capture.py --help` for the full flag list, or see this
directory's patch/new-file comments for anything not covered above.

## Known limitations (as of this integration)

- `min_interval` in `OrbSlamWorker` (`output_script/synced_capture.py`) is throttled to
  10Hz (0.1s). This value was reached the hard way -- see the extensive comment above
  `OrbSlamWorker.__init__` in that file before changing it; a prior attempt at 20Hz
  reproduced a crash that took several rounds of live testing to root-cause and fix.
- ORB-SLAM3 resets mid-recording are real and were observed live (not rare) --
  `map_epoch` in `camera_trajectory.csv` flags them, but nothing downstream currently
  *acts* on that (e.g. splitting an episode into per-segment trajectories for training).
  That's the next piece of work, not yet built.
