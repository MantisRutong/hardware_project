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
  - `src/Optimizer.cc`: fixes vertex-id collisions in `LocalInertialBA()` and
    `MergeInertialBA()`. Both pack three kinds of g2o vertex into one id space against
    `maxKFid` (pose `= mnId`, velocity/bias `= maxKFid + 3*mnId + {1,2,3}`, map point
    `= maxKFid*5 + mnId + 1`), but upstream sets `maxKFid` to the *current* keyframe's id
    rather than the largest one in the optimization. Any keyframe above that lands its
    pose vertex inside the IMU range, and g2o rejects the duplicate ("addVertex: FATAL, a
    vertex with ID N has already been registered") -- rejected vertices are silently left
    out of the graph and edges then resolve that id to a vertex of the wrong type. This
    only shows up once an atlas is **loaded**: cold-started ids ascend with the current
    keyframe, so upstream's value happens to be the maximum. A loaded map's ids and the
    session's interleave. Observed live as 20 rejected vertices in two clusters during a
    single merge. Computing `maxKFid` properly also retires the `if(pKFi->mnId > maxKFid)
    continue;` guard in `MergeInertialBA`'s observation loop, which existed only to paper
    over the overflow and dropped otherwise-valid observations.
  - `src/System.cc`: `Shutdown()` now actually waits for the mapping and loop-closing
    threads to stop (bounded at 20s, then says so loudly and continues). Upstream requests
    them to finish and returns immediately -- the wait is present but commented out --
    while everything after that point walks the atlas: `SaveAtlas()`, and the
    `SaveTrajectoryTUM()` / `SaveKeyFrameTrajectory*()` calls that callers are told to make
    *after* `Shutdown()`. Those were reading containers `LocalMapping` and `LoopClosing`
    were still mutating; neither thread is ever joined either. Seen three times on this
    project before the change: `free(): corrupted unsorted chunks` right after an atlas
    save, and `FATAL: exception not rethrown` + `SIGABRT` at process exit after a
    recording -- in every case the output file was already complete, which is what a
    teardown-only race looks like.
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
  Besides per-frame tracking it exposes `orb_shutdown()` and
  `orb_save_trajectory_tum()`, which is what makes the refined trajectory below
  possible -- ORB-SLAM3 will only resolve frame poses against the final map after
  its threads have stopped.
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
- `output_script/synced_capture.py` -- records. `--record-ir` (default **on**) saves the
  left/right IR streams, which is what the tracking pass consumes, live or offline.
  `--orbslam` (default **off**) additionally runs ORB-SLAM3 on its own thread
  (`OrbSlamWorker`) during the recording, writing `camera_trajectory.csv` and, at episode
  end, `camera_trajectory_refined.csv`. That is now a rig sanity check rather than the
  source of training labels -- see **Recording and tracking are separate steps** below.
- `output_script/replay_slam.py` -- re-runs ORB-SLAM3 over an ALREADY-RECORDED episode
  from its saved IR frames and IMU, instead of tracking while recording. The recordings
  keep everything needed for this on purpose (raw unmasked IR PNGs, both index files, the
  IMU CSVs), so a recording can be re-tracked against an atlas that did not exist when it
  was made, at full frame rate rather than the live 10Hz throttle, with the gripper mask
  as a clean A/B on identical input. On scan_0011 against `maps/workspace`: 1614 tracked
  poses vs the live run's 582, exported as one episode with zero speed splits instead of
  four episodes with three. Only for the data-collection branch -- teleoperation drives
  the arm from the live pose and has nothing to be offline about.
- `output_script/build_atlas_map.py` + `synced_capture.py --orbslam-map-dir` -- the
  two-stage approach: build a persistent map ONCE from a deliberate mapping pass, then
  LOCALIZE every demo against it rather than cold-starting a fresh map per episode.
  See the **Two-stage mapping** section below.
- `camera/camera_collecting.py`'s `_set_emitter_for_depth()` -- switches the D435i's IR
  projector OFF whenever the IR streams are feeding ORB-SLAM3 (ON only when the depth
  stream is actually running, which needs it). The projected dot pattern is
  *camera-fixed*, not world-fixed: the dots slide across surfaces as the camera moves,
  so they are not valid landmarks, and they're near-identical to each other so their
  descriptors mismatch easily. This matches where upstream draws the line -- ORB-SLAM3's
  own Stereo, Stereo-Inertial, Monocular and Calibration RealSense examples all switch
  the emitter off; only the RGB-D ones (which consume the depth map, not IR features)
  leave it on.

Run `output_script/synced_capture.py --help` for the full flag list, or see this
directory's patch/new-file comments for anything not covered above.

## Recording and tracking are separate steps

For data collection, `synced_capture.py` records and `replay_slam.py` tracks, afterwards.
Live tracking (`--orbslam`) defaults to OFF.

```bash
# 1. record (IR + IMU + RGB + servo). No SLAM running.
python3 output_script/synced_capture.py --episodes 5

# 2. track, at full rate, against the atlas
python3 output_script/replay_slam.py recording --map-dir maps/workspace

# 3. export
python3 output_script/export_dataset.py recording --output dataset.zarr
```

Measured on scan_0011 -- the same recording taken both ways. Replay is NOT
deterministic (local mapping and loop closing run on their own threads, so their
interleaving with tracking differs every run), so the replay row is seven runs of the
same command on the same bytes:

| | tracked poses | exported frames | episodes | speed splits | usable H=16 windows |
|---|---|---|---|---|---|
| live | 582 | 581 | 4 | 3 | 521 |
| replay, median | 1526 | 1541 | 5 | 2 | 1477 |
| replay, range | 1490-1614 | 1525-1557 | 1-6 | 0-4 | 1462-1542 |

The headline counts swing a lot between runs -- one run came out as a single episode
with zero splits, another as six episodes with three -- but the number that actually
matters barely moves: usable training windows span 1462-1542, a 5% spread, because the
splits mostly land near the ends of the trajectory. Every run, including the worst, is
about 2.8x live.

So do not treat a non-zero split count in a replay as a failed run to be repeated. The
split is the mechanism working; re-running to chase zero buys around 5% and costs a
selection process. What the count IS good for is comparing the refined trajectory
against the same run's live-equivalent (replay_slam.py prints both, as `jumps a -> b`):
b should be well below a, which is what says refinement did its job on that run.

The gain over live comes from bytes that were already on disk, and its reasons are
structural rather than incidental:

- **Rate.** Live tracking shares the machine with the capture thread and is pinned to
  10Hz (`OrbSlamWorker.min_interval` -- 20Hz reproduced a crash), against a camera
  recording at 30fps. `export_dataset.py` is pose-driven, so the other two thirds of the
  frames are simply dropped. Offline there is no capture thread to starve.
- **One attempt.** A live recording is tracked once, with whatever atlas, mask and
  settings existed that day, and the result is final. Offline it can be re-tracked when
  any of those improve.
- **Experiments.** `--gripper-mask` against `--no-gripper-mask` live means two
  recordings, which differ in how the operator moved as well as in the mask. Offline it
  is the same bytes twice.

What is given up: live tracking told you during the recording whether tracking was
working. Offline you find out afterwards. `--orbslam` is still there for exactly that --
run one episode with it before recording a batch -- but its output is a check, not a
label.

Teleoperation is unaffected and always tracks live: `umi_teleop.py` drives the arm from
the pose, and a pose that arrives after the fact is no use to it.

Two things follow from the default. Recordings must keep their IR (`--record-ir`, on by
default) or they can never be re-tracked. And an episode recorded this way has no
`camera_trajectory.csv` at all -- only the `camera_trajectory_refined.csv` that
`replay_slam.py` writes -- which is why `export_dataset.py` accepts either
(`has_trajectory`).

### Limits, measured

Replaying does not rescue a recording whose scene was untrackable. scan_0006-0009 were
recorded before the workspace texture went up, and today they export to literally
nothing ("No usable frames survived processing" -- their longest continuous tracked run
is 4 frames). Re-tracked against `maps/workspace` they reach 33-40% of frames and 139-243
resets, yielding one usable segment each, up to 68 frames. Compare scan_0011, recorded
after, in the scene the atlas actually covers: 98% and one reset. An atlas only helps
where it recognises the scene, and no amount of offline processing recovers texture that
was not in front of the camera.

## Two-stage mapping (build once, localize per demo)

Live testing showed Stereo-Inertial failing catastrophically during careful, precise
pick-and-place -- near-constant resets, ~33% of frames tracked, every tracked segment
only ~3 frames long -- **after** low texture and motion blur had both been independently
ruled out as the cause. Root cause, traced into `LocalMapping.cc`: IMU initialization
requires accumulated keyframe-to-keyframe translation >= 2cm within a ~10s window or it
resets ("Not enough motion for initializing"). Precise manipulation has small net
displacement by nature, so a map cold-started from inside a single short, careful demo
may simply never clear that bar. The official UMI pipeline
(`scripts_slam_pipeline/03_batch_slam.py`) never hits this because it never cold-starts
a map from a demo's own motion.

```bash
# 1. Build the map ONCE -- expansive, unhurried motion over the whole workspace.
python3 output_script/build_atlas_map.py --output-dir maps/kitchen_table

# 2. Every demo localizes against it instead of cold-starting.
python3 output_script/synced_capture.py --orbslam-map-dir maps/kitchen_table
```

**Why loading a map actually removes the reset**, rather than just making it less likely:
`Map.h`'s `serialize()` includes `mbImuInitialized` / `mbIsInertial` / `mbIMU_BA1` /
`mbIMU_BA2` in what gets written to the `.osa`. A map built by a proper mapping pass
saves with `GetIniertialBA2() == true`, and the reset above is gated specifically on
`!GetIniertialBA2()` -- so a loaded, already-BA2-complete map never re-triggers it, no
matter how small the demo's own motion is. The hard part (cold initialization) happens
once, offline; each demo only does the easier part.

Mechanics worth knowing before running either half:

- ORB-SLAM3 hardcodes both the save and load path as `"./" + name + ".osa"`, relative to
  the **process's cwd at the moment the `System` is constructed** -- not configurable.
  `build_atlas_map.py` `chdir`s into `--output-dir` before starting; `synced_capture.py`'s
  `construct_orb_tracker()` chdirs in, constructs, and chdirs back (see its docstring for
  why that's thread-safe where it's called).
- The atlas is checksum-verified against the **vocabulary file** used to build it -- don't
  point `--orbslam-vocab` at a different `ORBvoc.txt` between the two stages.
- Saving/loading needs no C API changes: `System::Shutdown()` calls `SaveAtlas()` when
  `System.SaveAtlasToFile` is set, and the constructor calls `LoadAtlas()` when
  `System.LoadAtlasFromFile` is. Hence the two config variants,
  `config/RealSense_D435i_ours_mapping.yaml` and `..._relocalize.yaml` -- identical
  calibration to `..._ours.yaml`, plus that one key.

### Refined trajectories

Loading an atlas introduces a failure the live trajectory cannot flag. `map_epoch` catches
every **reset**, but not a **merge**: when the session's own map is recognized as somewhere
the loaded atlas already covers, `LoopClosing` rigidly transforms every keyframe and map
point into the atlas's frame (`ApplyScaledRotation`) and switches the active map
(`Atlas::ChangeMap`). Neither touches the reset counter, because both of its increment
sites are reset paths. So the pose origin moves mid-episode with nothing marking it --
observed live as a 38cm step between two consecutive 10Hz rows, `is_lost=0` and `map_epoch`
unchanged on both sides. Every downstream sanity check calls that data healthy.

ORB-SLAM3 already solves this for offline use, and the official UMI pipeline relies on it:
`Tracking` stores each frame's pose *relative to its reference keyframe*
(`mlRelativeFramePoses`), and `System::SaveTrajectoryTUM` resolves it at save time as
`Trw * pKF->GetPose() * Two` using that keyframe's **final** pose. Merges and bundle
adjustment move keyframes; the frame-to-keyframe relation survives, so every frame comes
out in one frame -- the final map's -- with the merge discontinuity absorbed rather than
recorded. After a merge that frame is the *atlas's* own, so refined trajectories from
different episodes that both merged share one world frame.

So each episode ends with `orb_shutdown()` (the wait added to `System::Shutdown()` above is
what makes this safe), then `orb_save_trajectory_tum()` into
`camera_trajectory_orbslam.tum`, which `write_refined_camera_trajectory_csv()`
(`synced_capture.py`) joins back onto the live rows by timestamp to produce
`camera_trajectory_refined.csv`. Frames the tracker ended up considering lost are written
`is_lost=1`. `map_epoch` is carried over unchanged -- it still describes what happened
during the run, and a refined file spanning a **reset** is still discontinuous (a reset
destroys the keyframes, so there is nothing to re-resolve those frames against). Splitting
on `map_epoch` stays correct; this only removes the discontinuities `map_epoch` could never
see. The per-episode print reports the largest correction applied; anything above ~5cm
means the live trajectory did contain a coordinate-frame shift.

Note that the tracker is torn down and reconstructed per episode
(`construct_orb_tracker()`), so this costs one atlas reload per demo rather than one per
batch -- the price of getting a final map to resolve against while the batch is still
running.

### Scene texture

With the emitter off (above), the workspace has to supply its own **world-fixed** texture.
Adding it deliberately -- printed patterns stuck around the manipulation region -- is the
intended way to do that. Two constraints:

- They are used **only as ordinary ORB features**. Nothing in this project detects them as
  markers and no pose is derived from their geometry (there is no `aruco`/`apriltag` code
  anywhere here, deliberately). Deriving pose from marker geometry would tie the recorded
  trajectories to an instrumented environment, which defeats the point for policies that
  have to run in un-instrumented ones.
- Tracking happens in **near-IR**, where many colored dyes are effectively transparent.
  Use black-on-white laser-printed patterns, non-repeating (a regular grid produces
  ambiguous descriptors), at mixed scales.

Whatever texture is present during the mapping pass is baked into the atlas as map points,
so it must stay physically put between mapping and the demos that localize against it.
Rearranging it means re-running `build_atlas_map.py`.

## Known limitations (as of this integration)

- `min_interval` in `OrbSlamWorker` (`output_script/synced_capture.py`) is throttled to
  10Hz (0.1s). This value was reached the hard way -- see the extensive comment above
  `OrbSlamWorker.__init__` in that file before changing it; a prior attempt at 20Hz
  reproduced a crash that took several rounds of live testing to root-cause and fix.
- ORB-SLAM3 resets mid-recording are real and were observed live (not rare). `map_epoch`
  flags them and `output_script/export_dataset.py` now acts on it, splitting a recording
  into one training episode per contiguous `map_epoch` segment rather than training across
  a coordinate-frame teleport. The two-stage approach above is aimed squarely at removing
  the dominant cause of those resets; if it works, most recordings should come out as a
  single segment, and `map_epoch` is still the thing to check to find out.
- **None of the two-stage path has been validated on hardware yet.** The mapping script,
  the `--orbslam-map-dir` localize path, and the emitter change are all written and
  compile, but no live run has confirmed that resets actually drop. The order to test in:
  put the scene texture up, run `build_atlas_map.py` and watch its tracked/total ratio,
  then record a demo with `--orbslam-map-dir` and compare reset counts against a
  cold-start run.
- `IR_EXPOSURE_CAP_US` (`camera/camera_collecting.py`) was chosen to stop the *emitter's
  dot pattern* smearing under handheld motion. With the emitter now off for tracking, that
  basis no longer holds -- real world-fixed texture has a different blur budget, so the
  value needs re-measuring rather than assuming it still applies.

- **A grasped object becomes camera-fixed for as long as it's held, and nothing handles
  that.** The static gripper mask cannot: where a held object lands in frame depends on
  where and how it was grasped. Measured on existing recordings with a sliding 1.5s
  temporal-mean sharpness test (the same camera-fixed detector used to derive the gripper
  mask, minus the known gripper region), the signal is real and lines up with grasp
  phases -- scan_0009 shows eight consecutive elevated windows from 12s, scan_0007 shows
  19 of 26. During those windows the held object is ~2.3-2.8% of the frame but takes
  **11-13% of the features that remain after gripper masking** (104 of 786 per frame in
  scan_0009; 91 of 798 in scan_0007).

  Probably survivable as-is: `Optimizer::PoseOptimization` uses a Huber robust kernel
  (`thHuber2D = sqrt(5.99)`, `src/Optimizer.cc`) and four rounds of chi-squared outlier
  rejection (`if(e->chi2() > 5.991) ... nBad++`), which is exactly the mechanism for a
  minority of points whose motion disagrees with the rest. 11-13% is a clear minority.

  Three reasons not to treat that as settled: (1) it's a minority *in these recordings* --
  a closer grasp, a larger object, or a view pointed at the blankest part of the workspace
  could multiply it, and this scene is texture-poor to begin with; (2) the object was
  mapped in its pre-grasp position, so after it moves those map points are stale and can
  mis-associate -- that pollutes the *map*, which outlives any single frame's pose, and it
  is baked into `atlas.osa` if it happens during the mapping pass; (3) it happens
  precisely during the careful, small-displacement phase that is already closest to the
  IMU-init motion threshold.

  Deliberately not fixed in code. A static mask can't cover it, and per-frame dynamic
  rejection would be re-implementing what the robust kernel already does, with more ways
  to get it wrong. The real lever is the denominator: strong world-fixed texture *around*
  the manipulation region dilutes the object's share and gives outlier rejection a solid
  inlier majority to work against -- which is the reason for putting added texture around
  the workspace rather than in it. One cheap option held in reserve: extending the gripper
  mask upward by 40-60px would cover much of where held objects sit (they appear directly
  above the current boundary), at the cost of the manipulation region's own pixels -- which
  are pixels SLAM wants nothing from anyway. Not done blind; decide it against live
  `camera_trajectory.csv` behaviour during grasp phases once the scene texture is in place.
