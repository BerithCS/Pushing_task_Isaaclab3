#!/usr/bin/env python3

# ==============================================================================
# 1) APP LAUNCH  — must come first, before any omni/isaacsim/pxr/cuRobo import.
#    PORT NOTE: replaces `from omni.isaac.kit import SimulationApp; SimulationApp(...)`
# ==============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Doosan M0609 tactile pushing (Isaac Lab 3.0).")
# AppLauncher adds --headless, --device, --width, --height, etc.
# In Lab 3.0 it also adds backend/renderer selection args.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# PORT NOTE (Lab 3.0): the physics backend is NOT selected by a CLI string --
# it is selected by which PhysicsCfg subclass is passed to SimulationCfg.physics
# further down. See the SimulationCfg block in __main__.

# Match the standalone window size default; harmless when --headless.
if getattr(args_cli, "width", None) is None:
    args_cli.width = 1920
if getattr(args_cli, "height", None) is None:
    args_cli.height = 1080

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # <-- same name as before, so downstream code is unchanged

# PORT NOTE (IsaacLab @0b5605c2): AppLauncher no longer exposes --headless; the
# viewer is selected with --visualizer instead. Derive one render flag here so
# the sim loop doesn't touch args_cli.headless (AttributeError on this commit).
_viz = getattr(args_cli, "visualizer", None)
RENDER = (not getattr(args_cli, "headless", True)) or (_viz not in (None, "", "none", "None"))
print(f"[APP] visualizer={_viz!r} -> render={RENDER}")

# ==============================================================================
# 2) EVERYTHING ELSE IMPORTS ONLY AFTER THE APP EXISTS
# ==============================================================================
import os
import csv
import datetime
import numpy as np
import pandas as pd
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import torch

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf

# Warp is now the native buffer type for all `.data.*` properties in Lab 3.0.
import warp as wp

# PORT NOTE (Lab 2.x -> 3.0): Isaac Lab 3.0 is multi-backend. PhysX is no longer
# the implicit default everywhere; it lives in `isaaclab_physx`. Deformable
# bodies are PHYSX-ONLY in 3.0 (Newton has no deformable support), so this
# script MUST select the PhysX backend explicitly.
#
# VERIFIED against this install (isaaclab3 source tree):
#   * SimulationContext / SimulationCfg stay in isaaclab.sim
#   * PhysxCfg moved to isaaclab_physx.physics.physx_manager_cfg
#     (NOT isaaclab_physx.sim -- that subpackage is lazy_export only)
#   * SimulationCfg.physx=  is gone; the field is now  SimulationCfg.physics=
#     typed PhysicsCfg|None, defaulting to PhysxCfg(). The BACKEND IS THE
#     CONFIG OBJECT -- passing PhysxCfg() selects PhysX, passing
#     NewtonManagerCfg() would select Newton. There is no backend string.
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

from isaaclab.assets import Articulation, ArticulationCfg

# PORT NOTE (Sim 6.0 / Lab 3.0): `isaacsim.core.utils` no longer exists -- Isaac
# Sim 6.0 restructured toward isaacsim.core.experimental. Isaac Lab 3.0 now
# provides the stage-reference helper itself. `add_reference_to_stage` survives
# in isaaclab.sim.utils.legacy but logs a deprecation warning; the new name is
# `add_usd_reference`.
#
# CAUTION: the two functions do NOT share a parameter order --
#   legacy:  add_reference_to_stage(usd_path, path)
#   new:     add_usd_reference(prim_path, usd_path)   <- reversed!
# Passing positionally silently feeds the prim path in as the USD path and
# fails with "Unable to open the usd file at path: /World". So bind by NAME,
# resolved from the real signature at import time rather than assumed.
import inspect as _inspect

def _make_stage_ref_adapter():
    _fn = None
    for _modname, _attr in (("isaaclab.sim", "add_usd_reference"),
                            ("isaaclab.sim", "add_reference_to_stage")):
        try:
            _fn = getattr(__import__(_modname, fromlist=[_attr]), _attr)
            break
        except (ImportError, AttributeError):
            continue
    if _fn is None:
        raise ImportError("No stage-reference helper found in isaaclab.sim")

    _params = list(_inspect.signature(_fn).parameters)
    # the USD-file parameter is the one whose name mentions 'usd'; the prim
    # parameter is the other positional one.
    _usd_kw = next((p for p in _params if "usd" in p.lower()), None)
    _prim_kw = next((p for p in _params
                     if p != _usd_kw and ("path" in p.lower() or "prim" in p.lower())), None)
    if _usd_kw is None or _prim_kw is None:
        raise ImportError(f"Could not map args of {_fn.__name__}{tuple(_params)}")

    print(f"[PORT] stage ref -> {_fn.__name__}({_usd_kw}=<usd>, {_prim_kw}=<prim>)")

    def _adapter(usd_path: str, prim_path: str):
        return _fn(**{_usd_kw: usd_path, _prim_kw: prim_path})

    return _adapter

add_reference_to_stage = _make_stage_ref_adapter()

# ==============================================================================
# cuRobo  -- PORT NOTE (cuRobo 0.7.7 -> 0.8.0 "curobov2" rewrite, commit 8e734f3c)
# The 0.8 changelog says "Major refactor breaks most existing api." Mapping used:
#   MotionGen / MotionGenConfig / plan_single  -> MotionPlanner / MotionPlannerCfg.create / plan_pose
#   MotionGenPlanConfig                        -> keyword args of plan_pose (max_attempts, enable_graph_attempt)
#   PoseCostMetric(hold_partial_pose, hold_vec_weight)
#                                              -> ToolPoseCriteria(non_terminal_pose_axes_weight_factor)
#                                                 via planner.update_tool_pose_criteria({tool_frame: ...})
#                                                 NOTE axis order changed: 0.7 was [r,p,y,x,y,z],
#                                                 0.8 is [x,y,z,roll,pitch,yaw]
#   TensorDeviceType                           -> DeviceCfg
#   UsdHelper                                  -> UsdSceneParser (same load_stage / get_obstacles_from_stage)
#   load_yaml (curobo.util_file)               -> yaml.safe_load
#   CollisionCheckerType.MESH                  -> gone; mesh checking comes from collision_cache={"mesh": n}
#   ee_link_name=                              -> robot_cfg["kinematics"]["tool_frames"] = [EE_LINK_NAME]
#   time_dilation_factor                       -> gone; emulated with STEPS_PER_CMD (see TIMING)
#   JointState.get_ordered_joint_state         -> gone; small _reorder_js helper below
# ==============================================================================
import yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose, ToolPoseCriteria
from curobo._src.util.usd_scene_parser import UsdSceneParser

# ============================================================
# SETTINGS  (unchanged)
# ============================================================
SCRIPT_DIR      = Path(__file__).resolve().parent
PROJECT_DIR     = str(SCRIPT_DIR)
ROBOT_DIR       = str(SCRIPT_DIR / "coro_doosan_station" / "m0609")
SCENE_USD       = str(SCRIPT_DIR / "scenes" / "doosan_station_full.usd")
ROBOT_CFG_FILE  = "m0609_adap_curobo.yml"
ROBOT_PRIM_PATH = "/World/m0609/m0609"
EE_LINK_NAME    = "link_6"

INITIAL_JOINT_DEGREES = np.array(
    [-72.09, 49.03, 57.46, -0.08, 73.51, -73.04],
    dtype=np.float32,
)

# ============================================================
# CNN / INFERENCE SETTINGS  (restored, same as original standalone)
# ============================================================
NODES_FILE      = str(SCRIPT_DIR / "CNN_tactile" / "Nodes_id_filtered.csv")
ONNX_MODEL_PATH = str(SCRIPT_DIR / "CNN_tactile" / "best.onnx")
X_TRAIN_MAX_NPY = str(SCRIPT_DIR / "CNN_tactile" / "CNN_max.npy")

N_ROWS = 18
N_COLS = 12

# ============================================================
# OBJECT PUSHED  (unchanged)
# ============================================================
OBJECT_PUSHED_BOTTOM_CENTER_WORLD = np.array([0.22, -0.26, 0.96557], dtype=np.float64)
OBJECT_HEIGHT         = 0.10
OBJECT_DIAMETER       = 0.075
OBJECT_RADIUS         = OBJECT_DIAMETER / 2.0
OBJECT_MASS_KG        = 0.9
OBJECT_COLOR          = np.array([1.0, 0.0, 0.0])
OBJECT_CONTACT_OFFSET = 0.0001
OBJECT_REST_OFFSET    = 0.0
OBJECT_XFORM_PATH     = "/World/Object_pushed"
OBJECT_MESH_PATH      = "/World/Object_pushed/Cylinder"

CYLINDER_SOLVER_POS_ITERS = 240
CYLINDER_SOLVER_VEL_ITERS = 30

# ============================================================
# TARGET COMPUTATION  (unchanged)
# ============================================================
DISTANCE_TARGETS = 0.3

_cyl_center_x = float(OBJECT_PUSHED_BOTTOM_CENTER_WORLD[0])
_cyl_center_y = float(OBJECT_PUSHED_BOTTOM_CENTER_WORLD[1])
_cyl_center_z = float(OBJECT_PUSHED_BOTTOM_CENTER_WORLD[2]) + OBJECT_HEIGHT / 2.0

_target_y = _cyl_center_y - OBJECT_RADIUS - 0.012 - 0.012 - 0.001
_target_z = _cyl_center_z + 0.087715

TARGET_A_WORLD = np.array([_cyl_center_x, _target_y,                    _target_z], dtype=np.float32)
TARGET_B_WORLD = np.array([_cyl_center_x, _target_y + DISTANCE_TARGETS, _target_z], dtype=np.float32)

# ============================================================
# DEFORMABLE MESH / CSV  (unchanged)
# ============================================================
SPONGE_ROOT_PATH = "/World/CoRo_tactile/CoRo_tactile/Sponge"
SENSOR_POSE_PATH = "/World/CoRo_tactile/CoRo_tactile/Case_m"
CSV_DIR          = PROJECT_DIR
CSV_BASENAME     = "sponge_data"

# ============================================================
# TIMING  (unchanged)
# ============================================================
# PORT NOTE (Sim 6.0): the new deformable-rigid contact resolves penetration
# per physics step much less completely than 5.1 did, so the cylinder sinks into
# the pad by an amount proportional to the step. Measured non-rigid residual
# during the push: 0.92 mm @ 60 Hz, 0.25 @ 240, 0.07 @ 480, 0.024 @ 960; the
# dent shape/depth/uniform-compression converge onto the 5.1 reference at 960 Hz
# (dent-map correlation 0.89). No material / offset / collider setting moved it.
# Override with PHYSICS_HZ=...
PHYSICS_HZ         = float(os.environ.get("PHYSICS_HZ", "960"))
PHYSICS_DT         = 1.0 / PHYSICS_HZ
OUTPUT_HZ          = 60.0                                  # CSV / tactile sample rate (as before)
OUTPUT_EVERY       = max(1, int(round(PHYSICS_HZ / OUTPUT_HZ)))   # physics steps per recorded frame
INTERPOLATION_DT   = 0.008
TIME_DILATION      = 0.5
# PORT NOTE (cuRobo 0.8): MotionGenPlanConfig.time_dilation_factor no longer
# exists. 0.5 meant "play the plan at half speed"; holding each interpolated
# waypoint for 1/TIME_DILATION physics steps gives the same execution speed.
# Scaled by PHYSICS_HZ/60 so a finer physics step does not change the push speed.
STEPS_PER_CMD      = max(1, int(round((1.0 / TIME_DILATION) * (PHYSICS_HZ / 60.0))))
MAX_EFFORT         = 1500.0
KP_GAINS           = 200000.0
KD_GAINS           = 10000.0

# Phase durations are physics-step counts; scaled by OUTPUT_EVERY so they keep
# the same wall-clock length they had at 60 Hz.
SETTLE_FRAMES      = 10  * OUTPUT_EVERY
MAX_SETTLE_FRAMES  = 900 * OUTPUT_EVERY   # cap on waiting for the pad to go quiet (~15 s)
SETTLE_WINDOW      = 5   * OUTPUT_EVERY   # stability window for the settle gate (83 ms)
MIN_SETTLE_FRAMES  = 60  * OUTPUT_EVERY   # never accept a baseline in the first 1 s
BASELINE_FRAMES    = 10  * OUTPUT_EVERY
END_FRAMES         = 10  * OUTPUT_EVERY

# ============================================================
# PLANNING  (unchanged)
# ============================================================
MAX_PLAN_FAILS  = 5
MAX_ATTEMPTS         = 4   # was MotionGenPlanConfig.max_attempts
ENABLE_GRAPH_ATTEMPT = 4   # was MotionGenPlanConfig.enable_graph_attempt (== max_attempts -> graph never used)

# PORT NOTE (cuRobo 0.8): ToolPoseCriteria axis order is [x, y, z, roll, pitch, yaw].
# cuRobo 0.7's hold_vec_weight was [roll, pitch, yaw, x, y, z]. 1 = hold that
# axis along the whole path, 0 = free. Same intent as before, re-ordered:
PUSH_Y_WEIGHT   = [1, 0, 1, 1, 1, 1]   # 0.7: [1,1,1, 1,0,1] -> hold x,z + orientation, free along Y
HOLD_ORI_WEIGHT = [0, 0, 0, 1, 1, 1]   # 0.7: [1,1,1, 0,0,0] -> hold orientation only

OBSTACLE_IGNORE = [
    "/World/m0609", "/World/looks",
    "/World/target_A", "/World/target_B",
    "/World/Object_pushed", "/World/GroundPlane",
    "/World/defaultGroundPlane", "/World/SphereLight",
    "/World/physicsScene", "/World/robot_mount",
    "/World/Environment/Geometry",
]

# ============================================================
# CNN HELPERS  (restored VERBATIM from the original standalone,
#  except get_dz_live which replaces the USD-based get_dz)
# ============================================================

def load_ordered_node_ids(path):
    df  = pd.read_csv(path)
    col = "node_id" if "node_id" in df.columns else df.columns[0]
    seen, ids = set(), []
    for nid in df[col].dropna().astype(int):
        if nid not in seen:
            seen.add(nid); ids.append(nid)
    return ids


def rotation_matrix_np(w, x, y, z):
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - z*w),       2*(x*z + y*w)    ],
        [2*(x*y + z*w),        1 - 2*(x**2 + z**2),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),        2*(y*z + x*w),        1 - 2*(x**2+y**2)],
    ])


def scale_to_cube(pts, cx, cy, cz, sx, sy, sz):
    out = np.zeros_like(pts)
    out[:, 0] = (pts[:, 0] - cx) / sx + cx
    out[:, 1] = (pts[:, 1] - cy) / sy + cy
    out[:, 2] = (pts[:, 2] - cz) / sz + cz
    return out


def compute_dz_from_arrays(rest_pts, curr_pts, R_mat, T, ordered_ids, all_node_ids):
    sx = rest_pts[:, 0].max() - rest_pts[:, 0].min()
    sy = rest_pts[:, 1].max() - rest_pts[:, 1].min()
    sz = rest_pts[:, 2].max() - rest_pts[:, 2].min()

    rot_pts = (R_mat @ rest_pts.T).T + T
    cx_w = (rot_pts[:, 0].min() + rot_pts[:, 0].max()) / 2
    cy_w = (rot_pts[:, 1].min() + rot_pts[:, 1].max()) / 2
    cz_w = (rot_pts[:, 2].min() + rot_pts[:, 2].max()) / 2

    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    filter_idx  = [node_to_idx[nid] for nid in ordered_ids if nid in node_to_idx]

    rest_f = rest_pts[filter_idx]
    curr_f = curr_pts[filter_idx]
    rot_f  = (R_mat @ rest_f.T).T + T

    black  = scale_to_cube(rot_f,  cx_w, cy_w, cz_w, sx, sy, sz)
    orange = scale_to_cube(curr_f, cx_w, cy_w, cz_w, sx, sy, sz)

    return orange[:, 2] - black[:, 2]


def kabsch_fit(rest_pts, cur_pts):
    """Return (R_mat, T) such that R_mat @ rest + T best matches cur (least squares).

    PORT NOTE (Sim 6.0): this REPLACES the case_view -> parent-relative ->
    case_q_corr chain for producing compute_dz's transform.

    Why: compute_dz_from_arrays only needs the rigid transform that maps the
    mesh-local rest points onto the current nodes. The old path reconstructed
    that indirectly from the Case_m rigid-body pose, which required (a) the
    live body quaternion, (b) the parent's static world pose, and (c) a
    one-time constant offset solved at t=0. In Isaac Sim 6.0 the live body
    quaternion comes back in a different body frame than it did in 5.1, which
    left a CONSTANT ~106 deg error in the reconstructed rotation (verified
    against a 5.1 baseline CSV: dz blew up to 7-19 m vs a true range of
    ~0.013 m, and tactile output hit 400-1600).

    Kabsch measures the transform DIRECTLY from the data compute_dz consumes,
    so it cannot drift out of sync with the body-frame convention. Measured
    residual on this scene: ~0.045 mm mean. It also matches 5.1 behaviour, so
    it is not a 6.0-only workaround.

    The Case_m pose is still read and logged to the CSV for reference, but it
    no longer feeds the dz computation.
    """
    A = rest_pts - rest_pts.mean(axis=0)
    B = cur_pts - cur_pts.mean(axis=0)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R_mat = Vt.T @ D @ U.T
    T = cur_pts.mean(axis=0) - R_mat @ rest_pts.mean(axis=0)
    return R_mat, T


def get_dz_live(deform_view, case_view, rest_ref,
                parent_pos, parent_rot_inv, case_q_corr, ordered_ids):
    """Compute dz from the live deformable view using a PER-FRAME Kabsch fit.

    PORT NOTE (Sim 6.0): previously this reconstructed R_mat/T from the Case_m
    rigid-body pose (parent-local + a constant case_q_corr offset). That path
    is broken in Sim 6.0 -- see kabsch_fit() for the full diagnosis. We now fit
    the transform directly from rest -> current, which is what compute_dz
    actually needs. case_view / parent_pos / parent_rot_inv / case_q_corr are
    kept in the signature for call-site compatibility but are no longer used
    for the transform."""
    try:
        cur = deform_view.get_simulation_nodal_positions()
        if isinstance(cur, wp.array): cur = cur.numpy()
        elif hasattr(cur, "detach"): cur = cur.detach().cpu().numpy()
        cur = np.asarray(cur).reshape(-1, 3)
        # NOTE: no parent-relative conversion. Kabsch is frame-agnostic -- it
        # fits rest -> cur in whatever frame `cur` arrives in, and compute_dz
        # normalises by the rest-shape extents, so the result is invariant.
        R_mat, T = kabsch_fit(rest_ref, cur)
        n = cur.shape[0]
        return compute_dz_from_arrays(rest_ref, cur, R_mat, T,
                                      ordered_ids, list(range(n)))
    except Exception:
        return None


def infer_tactile(dz, baseline, x_train_max, session, input_name):
    dz_corrected = dz - baseline
    dz_grid      = dz_corrected.reshape(N_ROWS, N_COLS).astype(np.float32)
    X            = np.clip(dz_grid / x_train_max, -1.0, 1.0)
    X            = X.reshape(1, N_ROWS, N_COLS, 1).astype(np.float32)
    y_pred       = session.run(None, {input_name: X})[0]
    return y_pred[0].flatten()  # (28,)


# ============================================================
# DEFORMABLE MESH HELPERS  (unchanged)
# ============================================================

def _has_any_deformable_api(prim: Usd.Prim) -> bool:
    """PORT NOTE (Sim 6.0): PhysxSchema.PhysxDeformableBodyAPI was REMOVED in
    Isaac Sim 6.0 -- referencing it raises AttributeError, which previously ate
    the whole deformable-detection path via a bare `except Exception` upstream
    and left rest_ref=None (silently producing all-zero tactile output).

    Detect by applied-schema NAME instead of by class attribute, so this works
    across schema renames. Any class attributes we do try are guarded."""
    # 1) name-based: covers OmniPhysicsDeformable*, PhysxDeformable*, and the
    #    Sim 6.0 volume/surface deformable schemas, whatever they are called.
    for s in prim.GetAppliedSchemas():
        sl = s.lower()
        if "deformable" in sl or "softbody" in sl or "soft_body" in sl:
            return True

    # 2) class-based fallback, each guarded -- these may not exist on this build.
    for _api_name in ("PhysxDeformableBodyAPI",
                      "PhysxDeformableSurfaceAPI",
                      "PhysxAutoDeformableAPI",
                      "PhysxDeformableVolumeAPI"):
        _api = getattr(PhysxSchema, _api_name, None)
        if _api is not None:
            try:
                if prim.HasAPI(_api):
                    return True
            except Exception:
                pass
    return False


def find_deformable_simulation_mesh(root_path: str):
    stage = omni.usd.get_context().get_stage()
    root  = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[DEFORMABLE] WARNING: root prim not found: {root_path}")
        return None
    # PORT NOTE: dump what is actually under the root, so a schema rename shows
    # up as readable output instead of a silent None.
    print(f"[DEFORMABLE] scanning under {root_path}:")
    _cands = []
    for prim in Usd.PrimRange(root):
        _schemas = list(prim.GetAppliedSchemas())
        if _schemas:
            print(f"    {prim.GetPath()}  type={prim.GetTypeName()}  schemas={_schemas}")
        if _has_any_deformable_api(prim):
            for child in prim.GetChildren():
                _cands.append(child)
                if "simulation_mesh" in child.GetName().lower():
                    print(f"[DEFORMABLE] Found: {child.GetPath()}")
                    return child
    print(f"[DEFORMABLE] WARNING: simulation_mesh not found under {root_path}")
    if _cands:
        print(f"[DEFORMABLE]   children of deformable prims seen: "
              f"{[c.GetName() for c in _cands]}")
    return None


# ============================================================
# CSV HELPERS  (unchanged)
# ============================================================

def prepare_csv(csv_dir: str, basename: str):
    """DEBUG BUILD: creates ONLY the deformation CSV (no tactile file)."""
    os.makedirs(csv_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def_path   = os.path.join(csv_dir, f"{basename}_deformation_{ts}.csv")
    def_file   = open(def_path, "w", newline="")
    def_writer = csv.writer(def_file)
    def_writer.writerow([
        "frame", "t", "node_id",
        "s1_x",  "s1_y",  "s1_z",
        "s1_vx", "s1_vy", "s1_vz",
        "s1_Rx", "s1_Ry", "s1_Rz",
        "s1_Trans_x", "s1_Trans_y", "s1_Trans_z",
        "s1_Ori_w",   "s1_Ori_x",   "s1_Ori_y",   "s1_Ori_z",
    ])
    def_file.flush()
    print(f"[CSV] Deformation file : {def_path}")

    tactile_cols = [f"T_{i+1:02d}" for i in range(28)]
    tac_path   = os.path.join(csv_dir, f"{basename}_tactiledata_{ts}.csv")
    tac_file   = open(tac_path, "w", newline="")
    tac_writer = csv.writer(tac_file)
    tac_writer.writerow(["frame", "t"] + tactile_cols)
    tac_file.flush()
    print(f"[CSV] Tactile file     : {tac_path}")

    return def_file, def_writer, tac_file, tac_writer


def save_sponge_data(deform_view, rest_ref, case_view,
                     parent_pos, parent_rot_inv, case_q_corr,
                     def_writer, def_file,
                     tac_writer, tac_file,
                     frame: int, sim_time: float,
                     ordered_ids, baseline, x_train_max, session, input_name):
    """Writes one row per mesh node per frame + one tactile prediction row.

    Columns match the ORIGINAL standalone CSV; the dz + inference path is the
    original's, fed by the live tensor-view data in the same local frames.
    """
    def _np(x):
        # PORT NOTE (Lab 3.0 / Sim 6.0): physics tensor views can hand back
        # warp arrays now. wp.array has .numpy(); torch tensors have .detach().
        if isinstance(x, wp.array):
            return x.numpy()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    try:
        cur = _np(deform_view.get_simulation_nodal_positions()).reshape(-1, 3)
        vel = _np(deform_view.get_simulation_nodal_velocities()).reshape(-1, 3)
        # JUNE-26 CONVENTION: nodes and pose all in the PARENT-LOCAL frame.
        cur = parent_rot_inv.apply(cur - parent_pos)
        vel = parent_rot_inv.apply(vel)
    except Exception as e:
        if frame % 60 == 0:
            print(f"[CSV] frame={frame}: deformable view read failed: {e}")
        return

    # LIVE Case_m WORLD pose from the rigid-body view, then convert to LOCAL
    # (relative to the static parent), matching the original xformOp read.
    trans = (0.0, 0.0, 0.0)
    ori_w, ori_x, ori_y, ori_z = 1.0, 0.0, 0.0, 0.0
    if case_view is not None:
        try:
            T = _np(case_view.get_transforms()).reshape(-1)
            p_world = np.array([float(T[0]), float(T[1]), float(T[2])])
            q_world = R.from_quat([float(T[3]), float(T[4]), float(T[5]), float(T[6])])  # xyzw

            p_local = parent_rot_inv.apply(p_world - parent_pos)
            q_local = (parent_rot_inv * q_world) * case_q_corr

            trans = (float(p_local[0]), float(p_local[1]), float(p_local[2]))
            qx, qy, qz, qw = q_local.as_quat()  # scipy returns (x,y,z,w)
            ori_w, ori_x, ori_y, ori_z = float(qw), float(qx), float(qy), float(qz)
        except Exception as e:
            if frame % 60 == 0:
                print(f"[CSV] frame={frame}: case pose read failed: {e}")

    n    = cur.shape[0]
    n_re = rest_ref.shape[0] if rest_ref is not None else 0

    # ── dz + CNN inference ───────────────────────────────────────────────
    # PORT NOTE (Sim 6.0): the transform now comes from a PER-FRAME Kabsch fit
    # of rest -> cur, NOT from the Case_m quaternion above. The Case_m pose is
    # still written to the CSV (s1_Trans_*/s1_Ori_*) for reference and for
    # comparison against 5.1 baselines, but it no longer drives dz. See
    # kabsch_fit() for why. The Kabsch residual is logged periodically below
    # so a bad fit cannot pass silently.
    tactile_vals = np.zeros(28, dtype=np.float32)
    _kab_resid = float("nan")
    if rest_ref is not None and baseline is not None:
        try:
            R_mat, T_vec = kabsch_fit(rest_ref, cur)
            _kab_resid = float(np.linalg.norm(
                (rest_ref @ R_mat.T + T_vec) - cur, axis=1).mean())
            dz    = compute_dz_from_arrays(rest_ref, cur, R_mat, T_vec,
                                           ordered_ids, list(range(n)))
            tactile_vals = infer_tactile(dz, baseline, x_train_max, session, input_name)
        except Exception as e:
            print(f"[WARN] Inference failed at frame {frame}: {e}")

    dmax = 0.0
    for i in range(n):
        cx, cy, cz = float(cur[i, 0]), float(cur[i, 1]), float(cur[i, 2])
        vx, vy, vz = float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])
        rx = ry = rz = 0.0
        if i < n_re:
            rx, ry, rz = float(rest_ref[i, 0]), float(rest_ref[i, 1]), float(rest_ref[i, 2])
            d = ((cx - rx)**2 + (cy - ry)**2 + (cz - rz)**2) ** 0.5
            if d > dmax: dmax = d

        def_writer.writerow([
            frame, f"{sim_time:.6f}", i,
            cx, cy, cz,
            vx, vy, vz,
            rx, ry, rz,
            trans[0], trans[1], trans[2],
            ori_w, ori_x, ori_y, ori_z,   # (w,x,y,z) exactly as the original wrote
        ])

    tac_writer.writerow([frame, f"{sim_time:.6f}"] + tactile_vals.tolist())

    if frame % 60 == 0:
        print(f"[CSV] frame={frame}  t={sim_time:.3f}s  nodes={n}  "
              f"kabsch_resid={_kab_resid*1000:.3f}mm  "
              f"tactile={tactile_vals[:4].round(3)}")
        if _kab_resid == _kab_resid and _kab_resid > 0.005:  # >5 mm
            print(f"[CSV] WARNING: Kabsch residual {_kab_resid*1000:.1f}mm is large -- "
                  f"the rest/current correspondence may be wrong.")
        def_file.flush()
        tac_file.flush()


# ============================================================
# ROBOT / MOTION HELPERS
# ============================================================

def apply_physics_material(stage, prim_path, static_friction, dynamic_friction,
                            restitution=0.0):
    mat_path    = "/World/looks/" + prim_path.replace("/", "_").strip("_") + "_physicsMat"
    mat_prim    = stage.DefinePrim(mat_path, "Material")
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    physics_mat.GetStaticFrictionAttr().Set(float(static_friction))
    physics_mat.GetDynamicFrictionAttr().Set(float(dynamic_friction))
    physics_mat.GetRestitutionAttr().Set(float(restitution))
    target_prim = stage.GetPrimAtPath(prim_path)
    if target_prim.IsValid():
        UsdShade.MaterialBindingAPI(target_prim).Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.weakerThanDescendants, "physics",
        )
        print(f"  Physics mat -> {prim_path}  (s={static_friction}, d={dynamic_friction})")
    else:
        print(f"  WARNING: prim not found: {prim_path}")


def create_object_pushed(stage):
    bottom     = OBJECT_PUSHED_BOTTOM_CENTER_WORLD
    xform_prim = UsdGeom.Xform.Define(stage, OBJECT_XFORM_PATH)
    xform_prim.AddTranslateOp().Set(Gf.Vec3d(*bottom.tolist()))

    xform_p  = stage.GetPrimAtPath(OBJECT_XFORM_PATH)
    UsdPhysics.RigidBodyAPI.Apply(xform_p)
    mass_api = UsdPhysics.MassAPI.Apply(xform_p)
    mass_api.GetMassAttr().Set(float(OBJECT_MASS_KG))
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, float(OBJECT_HEIGHT / 2.0)))

    # DIAG switch: CYL_SHAPE=mesh -> build the cylinder as a tessellated Mesh with
    # a convex-hull collider instead of a UsdGeom.Cylinder shape. PhysX has no
    # native cylinder collider; the Cylinder shape is handled as "custom geometry",
    # which deformable (GPU) contacts do not resolve properly in Sim 6.0.
    if os.environ.get("CYL_SHAPE", "shape") == "mesh":
        n_seg = 48
        r, h = float(OBJECT_RADIUS), float(OBJECT_HEIGHT)
        ring = [(r * np.cos(2 * np.pi * i / n_seg), r * np.sin(2 * np.pi * i / n_seg)) for i in range(n_seg)]
        pts = [Gf.Vec3f(x, y, -h / 2) for x, y in ring] + [Gf.Vec3f(x, y, h / 2) for x, y in ring]
        counts, idx = [], []
        for i in range(n_seg):                       # side quads
            j = (i + 1) % n_seg
            counts.append(4); idx += [i, j, n_seg + j, n_seg + i]
        counts.append(n_seg); idx += list(range(n_seg))[::-1]              # bottom cap
        counts.append(n_seg); idx += [n_seg + i for i in range(n_seg)]     # top cap
        mesh = UsdGeom.Mesh.Define(stage, OBJECT_MESH_PATH)
        mesh.GetPointsAttr().Set(pts)
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(idx)
        mesh.GetSubdivisionSchemeAttr().Set("none")
        cyl_xform = UsdGeom.Xformable(mesh.GetPrim())
        cyl_xform.ClearXformOpOrder()
        cyl_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, h / 2))
        cyl_prim = stage.GetPrimAtPath(OBJECT_MESH_PATH)
        UsdPhysics.CollisionAPI.Apply(cyl_prim)
        UsdPhysics.MeshCollisionAPI.Apply(cyl_prim).GetApproximationAttr().Set("convexHull")
        print(f"  [DIAG] cylinder built as Mesh ({n_seg} segments) with convexHull collider")
    else:
        cylinder = UsdGeom.Cylinder.Define(stage, OBJECT_MESH_PATH)
        cylinder.GetHeightAttr().Set(float(OBJECT_HEIGHT))
        cylinder.GetRadiusAttr().Set(float(OBJECT_RADIUS))
        cylinder.GetAxisAttr().Set("Z")
        cyl_xform = UsdGeom.Xformable(cylinder.GetPrim())
        cyl_xform.ClearXformOpOrder()
        cyl_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(OBJECT_HEIGHT / 2.0)))
        cyl_prim = stage.GetPrimAtPath(OBJECT_MESH_PATH)
        UsdPhysics.CollisionAPI.Apply(cyl_prim)
    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(cyl_prim)
    physx_col.GetContactOffsetAttr().Set(float(OBJECT_CONTACT_OFFSET))
    physx_col.GetRestOffsetAttr().Set(float(OBJECT_REST_OFFSET))

    mat_path    = "/World/looks/ObjectPushedMat"
    mat_prim    = stage.DefinePrim(mat_path, "Material")
    shader_prim = stage.DefinePrim(mat_path + "/Shader", "Shader")
    shader_prim.CreateAttribute("info:id", Sdf.ValueTypeNames.Token).Set("UsdPreviewSurface")
    shader_prim.CreateAttribute("inputs:diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*OBJECT_COLOR.tolist()))
    shader_prim.CreateAttribute("inputs:metallic",  Sdf.ValueTypeNames.Float).Set(0.0)
    shader_prim.CreateAttribute("inputs:roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material = UsdShade.Material(mat_prim)
    shader   = UsdShade.Shader(shader_prim)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI(cyl_prim).Bind(material)
    print(f"Object_pushed created at Z={bottom[2]:.5f} m")


def apply_cylinder_solver_iters(stage):
    xform_p = stage.GetPrimAtPath(OBJECT_XFORM_PATH)
    if not xform_p.IsValid():
        print("  WARNING: Cylinder Xform not found.")
        return
    for name, val in [
        ("physxRigidBody:solverPositionIterationCount", CYLINDER_SOLVER_POS_ITERS),
        ("physxRigidBody:solverVelocityIterationCount", CYLINDER_SOLVER_VEL_ITERS),
    ]:
        attr = xform_p.GetAttribute(name)
        if attr and attr.IsValid(): attr.Set(val)
        else: xform_p.CreateAttribute(name, Sdf.ValueTypeNames.UInt).Set(val)
        print(f"  Cylinder {name} = {val}")


def create_visual_marker(stage, path, position, size=0.04, color=(0.1, 0.1, 0.1)):
    """PORT NOTE: replaces omni.isaac.core.objects.cuboid.VisualCuboid + OmniPBR.
    A purely-cosmetic cube marker built from plain USD (no physics)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(p) for p in position]))
    xf.AddScaleOp().Set(Gf.Vec3f(size, size, size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cube.GetPrim()


def set_marker_color(prim, color):
    """Recolor a visual marker prim (cosmetic only)."""
    if prim and prim.IsValid():
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color])])


def world_to_base_frame(point_world, base_pos, base_ori_wxyz):
    w, x, y, z = base_ori_wxyz
    return R.from_quat([x, y, z, w]).inv().apply(point_world - base_pos).astype(np.float32)


# ==============================================================================
# cuRobo 0.8 helpers  (PORT NOTE: replace 0.7 JointState conveniences)
# ==============================================================================

def _dev_tensor(x, device_cfg: DeviceCfg) -> torch.Tensor:
    """np/list/tensor -> float32 tensor on cuRobo's device."""
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32,
                           device=device_cfg.device)


def make_js(qp, joint_names, device_cfg: DeviceCfg) -> JointState:
    """1-D joint positions (already in `joint_names` order) -> batched [1, dof]
    JointState with zero derivatives. plan_pose requires a 2-D state."""
    q = _dev_tensor(qp, device_cfg).reshape(1, -1)
    return JointState.from_position(q, joint_names=list(joint_names))


def flatten_plan(js: JointState) -> JointState:
    """Interpolated plans come back with leading batch dims ([1, H, dof] or
    [1, 1, H, dof]). Collapse to [H, dof] so `position[-1]` / `len()` behave like
    the 0.7 plan the rest of this script was written against."""
    dof = js.position.shape[-1]
    f = lambda t: None if t is None else t.reshape(-1, dof)
    return JointState(position=f(js.position), velocity=f(js.velocity),
                      acceleration=f(js.acceleration), jerk=f(js.jerk),
                      joint_names=list(js.joint_names))


def reorder_js(js: JointState, names) -> JointState:
    """Replacement for JointState.get_ordered_joint_state: select/reorder the
    last (dof) axis by joint name."""
    idx = [js.joint_names.index(n) for n in names]
    f = lambda t: None if t is None else t[..., idx]
    return JointState(position=f(js.position), velocity=f(js.velocity),
                      acceleration=f(js.acceleration), jerk=f(js.jerk),
                      joint_names=list(names))


def set_hold_axes(planner: MotionPlanner, weights, device_cfg: DeviceCfg):
    """Replacement for PoseCostMetric(hold_partial_pose=True, hold_vec_weight=w).
    `weights` is [x,y,z,roll,pitch,yaw]; non-zero entries are held at the goal
    pose along the whole trajectory (non-terminal timesteps)."""
    planner.update_tool_pose_criteria({
        EE_LINK_NAME: ToolPoseCriteria(
            non_terminal_pose_axes_weight_factor=[float(w) for w in weights],
            device_cfg=device_cfg,
        )
    })


def make_goal(pos_xyz, quat_wxyz, device_cfg: DeviceCfg) -> GoalToolPose:
    """Replacement for the bare Pose ik_goal: 0.8 wants a GoalToolPose keyed by
    tool frame."""
    pose = Pose(position=_dev_tensor(pos_xyz, device_cfg),
                quaternion=_dev_tensor(quat_wxyz, device_cfg),
                normalize_rotation=True)
    return GoalToolPose.from_poses({EE_LINK_NAME: pose}, ordered_tool_frames=[EE_LINK_NAME])


def get_curobo_fk(planner: MotionPlanner, joint_positions, joint_names, device_cfg: DeviceCfg):
    """PORT NOTE: kinematics.get_state(q).ee_position -> kinematics.get_link_poses(q, [ee])."""
    q    = _dev_tensor(joint_positions, device_cfg).reshape(1, -1)
    pose = planner.kinematics.get_link_poses(q, [EE_LINK_NAME])
    pos  = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
    quat = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
    return pos, quat


def load_curobo_robot_cfg_v2():
    """Load m0609_adap_curobo.yml and translate the 0.7 keys to the 0.8
    (format_version 2.0) layout. Paths are re-pointed at this repo so the
    stale /home/berith/Documents/Pushing_task paths in the YAML are ignored."""
    with open(os.path.join(ROBOT_DIR, ROBOT_CFG_FILE)) as f:
        data = yaml.safe_load(f)
    kin = data["robot_cfg"]["kinematics"]
    kin["format_version"]  = 2.0
    kin["urdf_path"]       = os.path.join(ROBOT_DIR, "m0609_adap.blue.urdf")
    kin["asset_root_path"] = str(SCRIPT_DIR / "coro_doosan_station")
    kin["tool_frames"]     = [EE_LINK_NAME]           # was ee_link_name= at MotionGenConfig
    for legacy in ("ee_link", "external_asset_path", "external_robot_configs_path"):
        kin.pop(legacy, None)
    cs = kin["cspace"]
    if "retract_config" in cs:                        # renamed in 2.0
        cs["default_joint_position"] = cs.pop("retract_config")
    return data


# ==============================================================================
# ISAAC LAB ARTICULATION ADAPTERS
# PORT NOTE: these wrap isaaclab.assets.Articulation so the rest of the code can
# read/write joints in the same shape the standalone Robot API used (1-D numpy
# arrays indexed by cuRobo joint order). Isaac Lab is batched: all buffers carry
# a leading env dimension (here always 1), and joint order follows
# `robot.joint_names`, so we build a name->column map once.
# ==============================================================================

def as_torch(x):
    """PORT NOTE (Lab 3.0): every `.data.*` property on assets/sensors now returns
    a `warp.array`, not a `torch.Tensor`. `wp.to_torch` is a zero-copy view, so
    this is cheap. Kept tolerant so the script still runs on a 2.x install."""
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return x


def build_dof_index_map(robot: Articulation):
    """Return {joint_name: column_index} for the single-env articulation."""
    names = robot.joint_names
    return {n: i for i, n in enumerate(names)}


def art_get_dof_indices(dof_map, joint_names):
    return [dof_map[n] for n in joint_names]


def art_set_joint_positions(robot: Articulation, positions_1d, col_indices, device):
    """Hard-set joint positions (teleport) for the given columns, env 0."""
    q = as_torch(robot.data.joint_pos).clone()            # (1, num_dof)
    pos = torch.as_tensor(positions_1d, dtype=torch.float32, device=device)
    q[0, col_indices] = pos
    robot.write_joint_state_to_sim(
        position=q,
        velocity=torch.zeros_like(q),
    )


def art_get_joint_state(robot: Articulation, col_indices):
    """Return (positions, velocities) as 1-D numpy arrays in the given col order."""
    qp = as_torch(robot.data.joint_pos)[0, col_indices].detach().cpu().numpy()
    qv = as_torch(robot.data.joint_vel)[0, col_indices].detach().cpu().numpy()
    return qp, qv


def art_apply_position_target(robot: Articulation, positions_1d, col_indices, device):
    """PORT NOTE: replaces ctrl.apply_action(ArticulationAction(...)).
    Sends a PD position target to the given joint columns for env 0."""
    tgt = torch.as_tensor(positions_1d, dtype=torch.float32, device=device).unsqueeze(0)
    joint_ids = list(col_indices)
    robot.set_joint_position_target(tgt, joint_ids=joint_ids)
    robot.write_data_to_sim()


def get_prim_world_pose(stage, prim_path):
    """World pose (pos_xyz numpy, quat_wxyz numpy) of a prim via USD xform cache.

    PORT NOTE (Lab 3.0): Isaac Lab flipped its default quaternion convention from
    wxyz to xyzw to match PhysX/Warp/Newton. That change affects `robot.data.*`
    quaternions (e.g. root_quat_w), NOT this function -- we read the USD xform
    cache directly, and Gf/USD are untouched by the Lab convention. So this still
    returns wxyz and `world_to_base_frame` below still expects wxyz. If you ever
    swap this out for `robot.data.root_quat_w`, drop the reordering in
    world_to_base_frame, because that buffer is now already xyzw.
    """
    prim = stage.GetPrimAtPath(prim_path)
    xcache = UsdGeom.XformCache()
    m = xcache.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    imag = q.GetImaginary()
    pos  = np.array([t[0], t[1], t[2]], dtype=np.float32)
    quat = np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)
    return pos, quat


def init_targets(stage, robot, planner, init_positions, j_names,
                 device_cfg, target_a_prim, target_b_prim):
    # PORT NOTE: robot.get_world_pose() -> USD xform cache on the robot root prim.
    base_pos, base_ori           = get_prim_world_pose(stage, ROBOT_PRIM_PATH)
    curobo_ee_pos, curobo_ee_ori = get_curobo_fk(planner, init_positions, j_names, device_cfg)
    ee_pos, ee_ori               = get_prim_world_pose(stage, f"{ROBOT_PRIM_PATH}/{EE_LINK_NAME}")

    print(f"Robot base pos   : {base_pos.tolist()}")
    print(f"cuRobo FK EE pos : {curobo_ee_pos.tolist()}")
    print(f"Target A (world) : {TARGET_A_WORLD.tolist()}")
    print(f"Target B (world) : {TARGET_B_WORLD.tolist()}")

    goal_ori_base    = curobo_ee_ori.astype(np.float32)
    target_list_base = []
    target_ori_base  = []

    for pt_world, tgt_prim in zip([TARGET_A_WORLD, TARGET_B_WORLD],
                                  [target_a_prim, target_b_prim]):
        pt_base = world_to_base_frame(pt_world, base_pos, base_ori)
        target_list_base.append(pt_base)
        target_ori_base.append(goal_ori_base.copy())
        # move the cosmetic marker to the world target (optional)
        if tgt_prim and tgt_prim.IsValid():
            xf = UsdGeom.Xformable(tgt_prim)
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*[float(v) for v in pt_world]))
                    break

    return target_list_base, target_ori_base


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Scene USD      : {SCENE_USD}")
    print(f"Target A       : {TARGET_A_WORLD.tolist()}")
    print(f"Target B       : {TARGET_B_WORLD.tolist()}")

    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"

    # ── Load CNN resources (restored, same as original) ────────
    # NOTE: on this laptop (RTX A2000, cuDNN 9.7.1) the CUDA EP can fail on one
    # Conv and ONNX Runtime falls back to CPU — that fallback worked before and
    # is acceptable here. Ensure best.onnx is the cuDNN-patched export.
    print("Loading CNN resources...")
    ordered_ids = load_ordered_node_ids(NODES_FILE)
    # PORT NOTE (NumPy 2.x): Isaac Lab 3 ships NumPy 2.x, where float() on a
    # 1-element (non-0-d) array is a TypeError rather than a deprecation warning.
    # CNN_max.npy stores a 1-element array, so go through .item(), which works
    # for both 0-d and 1-element arrays.
    x_train_max = float(np.load(X_TRAIN_MAX_NPY).item())
    session     = ort.InferenceSession(ONNX_MODEL_PATH,
                                       providers=["CUDAExecutionProvider",
                                                  "CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    print(f"  Nodes       : {len(ordered_ids)}")
    print(f"  X_train_max : {x_train_max:.6f}")
    print(f"  ONNX input  : {input_name}")

    baseline        = None
    baseline_buffer = []
    settle_hist     = []      # Kabsch residual history while waiting for the pad
    settle_ok       = False   # True once the pad's deformation has gone quiet

    # ==========================================================
    # PORT NOTE: World(...) -> SimulationContext(SimulationCfg(...))
    #
    # DEFORMABLE READ (final, proven by probe on this stack):
    # Sim 5.1 / Lab 2.3.2 / Physics 107.3. The sponge is a PhysxAutoDeformable
    # VOLUME deformable. USD/Fabric `points` never stream the live mesh (they
    # return a frozen post-reset snapshot — that's why velocities were constant
    # 796mm/s every frame). The ONLY live source is the physics tensor view
    # `create_volume_deformable_body_view`, read via get_simulation_nodal_positions().
    # Fabric setting is irrelevant to that view, so we leave it at default.
    # ==========================================================
    # PORT NOTE (Lab 3.0): SimulationCfg.physx= is GONE. The field is now
    # `physics`, typed PhysicsCfg|None. Passing a PhysxCfg selects the PhysX
    # backend; passing NewtonManagerCfg would select Newton. Newton has NO
    # deformable-body support, and this scene is a PhysxAutoDeformable sponge
    # read through a volume deformable tensor view, so PhysX is mandatory here.
    # Constructing PhysxCfg explicitly IS the backend pin.
    _physx_kwargs = dict(
        enable_ccd=False,
        # GPU pipeline is required for deformables:
        gpu_max_soft_body_contacts=2 ** 20,
    )
    try:
        _physx_cfg = PhysxCfg(**_physx_kwargs)
    except TypeError as e:
        # Some gpu_* / ccd fields were reshuffled in 3.0. Fall back to defaults
        # rather than dying, but say so loudly -- gpu_max_soft_body_contacts
        # being unset can overflow the soft-body contact buffer on this scene.
        print(f"[BACKEND] WARNING: PhysxCfg rejected a kwarg ({e}).\n"
              f"  Falling back to PhysxCfg() defaults. If the sponge misbehaves or\n"
              f"  PhysX reports a contact-buffer overflow, check the current field\n"
              f"  names in isaaclab_physx/physics/physx_manager_cfg.py.")
        _physx_cfg = PhysxCfg()

    sim_cfg = SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        physics=_physx_cfg,
    )
    print(f"[BACKEND] SimulationCfg.physics = {type(_physx_cfg).__name__} "
          f"(PhysX required for deformables)")
    print("[BACKEND] PhysxCfg gpu_*/ccd fields: " + ", ".join(
        f"{k}={v}" for k, v in vars(_physx_cfg).items()
        if "gpu" in k or "ccd" in k or "solver" in k))

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    stage = omni.usd.get_context().get_stage()

    # Ensure /World exists and is the default prim, then reference the scene USD.
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # PORT NOTE: standalone appended SCENE_USD as a sublayer of the root layer.
    # In Isaac Lab we reference it under /World instead (cleaner + matches the
    # prim paths your cuRobo config expects). If your USD's default prim is
    # itself "/World", add_reference_to_stage keeps the same absolute paths.
    add_reference_to_stage(usd_path=SCENE_USD, prim_path="/World")
    simulation_app.update()

    # ── GPU dynamics (kept as a belt-and-suspenders on the scene prim) ──
    physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
    if physics_scene_prim.IsValid():
        physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        physics_scene_api.GetEnableGPUDynamicsAttr().Set(True)
        physics_scene_api.GetBroadphaseTypeAttr().Set("GPU")
        print("GPU dynamics enabled.")

    # ── Disable table_base collision (unchanged) ──────────────
    _tbc = stage.GetPrimAtPath("/World/table_base/table_base/collisions")
    if _tbc.IsValid():
        _tbc.SetActive(False)
        print("  table_base collisions deactivated.")

    # ── Deformable physics tuning (unchanged) ─────────────────
    def _set_attr(prim, name, value, type_name):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid(): attr.Set(value)
        else: prim.CreateAttribute(name, type_name).Set(value)
        print(f"  [Deformable] {name} = {value}")

    _sponge_prim = stage.GetPrimAtPath("/World/CoRo_tactile/CoRo_tactile/Sponge")
    _mat_prim    = stage.GetPrimAtPath("/World/CoRo_tactile/CoRo_tactile/Sponge/Looks/Deformable_Material")

    if _sponge_prim.IsValid():
        # omniphysics:mass = 0.09 is the value the 5.1 script sets (line 704 there)
        # and is part of the "5.1 parameters". Keep it. SPONGE_MASS=<kg> overrides
        # for A/B tests; SPONGE_MASS=density leaves it to the material density
        # (measured on 6.0 @960 Hz: 0.09 -> 0.024 mm indentation like 5.1 but
        # noisy maps; density -> 0.28 mm, 8x too deep -- mass wrongly changes
        # static indentation in this solver).
        _m = os.environ.get("SPONGE_MASS", "0.09")
        if _m.lower() == "density":
            _ma = _sponge_prim.GetAttribute("omniphysics:mass")
            if _ma and _ma.IsValid() and _ma.IsAuthored():
                _ma.Clear()
            print("  [Deformable] omniphysics:mass left to density (no override)")
        else:
            _set_attr(_sponge_prim, "omniphysics:mass", float(_m), Sdf.ValueTypeNames.Float)
        _set_attr(_sponge_prim, "physxDeformable:solverPositionIterationCount", 80,      Sdf.ValueTypeNames.UInt)
        _set_attr(_sponge_prim, "physxDeformable:sleepThreshold",               0.00001, Sdf.ValueTypeNames.Float)
        _set_attr(_sponge_prim, "physxDeformable:disableGravity",               True,    Sdf.ValueTypeNames.Bool)
    else:
        print("  WARNING: Sponge prim not found.")

    # DIAG switches (env vars; defaults leave behaviour unchanged):
    #   SPONGE_E_SCALE=10      -> multiply Young's modulus (tests whether the solver
    #                             reacts to the material at all)
    #   SPONGE_BIND_PHYSICS=1  -> also bind the material with purpose "physics"
    #                             (6.0 currently only sees the all-purpose binding)
    _E_SCALE      = float(os.environ.get("SPONGE_E_SCALE", "1.0"))
    _BIND_PHYSICS = os.environ.get("SPONGE_BIND_PHYSICS", "0") == "1"

    if _mat_prim.IsValid():
        _set_attr(_mat_prim, "omniphysics:youngsModulus",                 1289000.0 * _E_SCALE, Sdf.ValueTypeNames.Float)
        _set_attr(_mat_prim, "omniphysics:poissonsRatio",                 0.1729,    Sdf.ValueTypeNames.Float)
        _set_attr(_mat_prim, "omniphysics:density",                       1240.0,    Sdf.ValueTypeNames.Float)
        _set_attr(_mat_prim, "physxDeformableMaterial:elasticityDamping",
                  float(os.environ.get("SPONGE_DAMPING", "15.0")), Sdf.ValueTypeNames.Float)
        if _E_SCALE != 1.0:
            print(f"  [DIAG] Young's modulus scaled x{_E_SCALE}")
        if _BIND_PHYSICS and _sponge_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(_sponge_prim).Bind(
                UsdShade.Material(_mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
            print("  [DIAG] material also bound with purpose='physics'")
    else:
        print("  WARNING: Deformable_Material prim not found.")

    # DIAG switch: SPONGE_CONTACT_OFFSET=0.002 -> override collision_mesh contact
    # offset (authored value in the USD is 1e-5 m = 0.01 mm; at 60 Hz the pad moves
    # 1-2.4 mm per step during the push, so contacts are only generated after the
    # cylinder is already inside the pad -> ~2.4 mm E-independent "indentation").
    _co = os.environ.get("SPONGE_CONTACT_OFFSET")
    if _co is not None and _sponge_prim.IsValid():
        _cm = _sponge_prim.GetChild("collision_mesh")
        if _cm.IsValid():
            _set_attr(_cm, "physxCollision:contactOffset", float(_co), Sdf.ValueTypeNames.Float)
            print(f"  [DIAG] collision_mesh contactOffset overridden -> {float(_co)} m")

    # ── DIAG: what does Sim 6.0 actually see on the material? ──
    # The pad indents ~2.4 mm under the push on 6.0 vs ~0.4 mm on 5.1 with
    # identical writes above, so check whether the material prim carries the
    # 6.0 OmniPhysics deformable-material schema and is bound for physics.
    for _p, _lbl in ((_mat_prim, "material"), (_sponge_prim, "sponge")):
        if _p.IsValid():
            print(f"  [MAT-DIAG] {_lbl} {_p.GetPath()}  type={_p.GetTypeName()!r}")
            print(f"  [MAT-DIAG]   applied schemas: {list(_p.GetAppliedSchemas())}")
            for _a in _p.GetAttributes():
                _n = _a.GetName()
                if any(k in _n.lower() for k in ("physics", "physx", "young", "poisson", "damp", "density", "mass", "material")):
                    print(f"  [MAT-DIAG]   {_n} = {_a.Get()}  (authored={_a.IsAuthored()})")
    if _sponge_prim.IsValid():
        _mb = UsdShade.MaterialBindingAPI(_sponge_prim)
        for _purpose in ("physics", UsdShade.Tokens.allPurpose):
            _rel = _mb.GetDirectBindingRel(materialPurpose=_purpose)
            print(f"  [MAT-DIAG] sponge material binding purpose={_purpose!r}: {list(_rel.GetTargets()) if _rel else None}")
        for _child in Usd.PrimRange(_sponge_prim):
            if _child != _sponge_prim:
                print(f"  [MAT-DIAG]   child {_child.GetPath()} type={_child.GetTypeName()!r} schemas={list(_child.GetAppliedSchemas())}")
        # GEOM-DIAG: contact offsets + extents of visual / simulation / collision meshes
        _xc = UsdGeom.XformCache()
        for _name in ("Cube", "simulation_mesh", "collision_mesh"):
            _c = _sponge_prim.GetChild(_name)
            if not _c.IsValid():
                continue
            for _a in _c.GetAttributes():
                _n = _a.GetName()
                if "offset" in _n.lower() or "physxCollision" in _n or "physxDeformable" in _n or "collision" in _n.lower():
                    print(f"  [GEOM-DIAG] {_name}.{_n} = {_a.Get()}  (authored={_a.IsAuthored()})")
            _pts_attr = _c.GetAttribute("points")
            _pts = _pts_attr.Get() if _pts_attr else None
            if _pts:
                _P = np.array([[p[0], p[1], p[2]] for p in _pts], dtype=float)
                _m = np.array(_xc.GetLocalToWorldTransform(_c))  # 4x4 row-major (Gf)
                _Pw = _P @ _m[:3, :3] + _m[3, :3]
                print(f"  [GEOM-DIAG] {_name}: {len(_P)} pts  local min={_P.min(0).round(5).tolist()} max={_P.max(0).round(5).tolist()}"
                      f"  | world min={_Pw.min(0).round(5).tolist()} max={_Pw.max(0).round(5).tolist()}")
            _sc = UsdGeom.Xformable(_c).GetLocalTransformation() if _c.IsA(UsdGeom.Xformable) else None
            if _sc is not None:
                print(f"  [GEOM-DIAG] {_name}: local xform = {_sc}")

    # ── Object pushed + solver (unchanged) ────────────────────
    # DIAG switch: SPONGE_NO_CYLINDER=1 -> run the whole push with nothing to
    # push, to tell contact-driven warp from motion-driven warp.
    _NO_CYL = os.environ.get("SPONGE_NO_CYLINDER", "0") == "1"
    if _NO_CYL:
        print("  [DIAG] cylinder NOT created (SPONGE_NO_CYLINDER=1)")
    else:
        create_object_pushed(stage)
        apply_cylinder_solver_iters(stage)

    # ── Physics materials (unchanged) ─────────────────────────
    if not _NO_CYL:
        apply_physics_material(stage, OBJECT_MESH_PATH,                                    0.2, 0.15)
    apply_physics_material(stage, "/World/table_cover/Cube",                                0.9,  0.9)
    apply_physics_material(stage, "/World/m0609/m0609/link_6/adapter",                      0.4,  0.35)
    # PORT NOTE (Sim 6.0): binding a plain friction-only UsdPhysics material with
    # purpose "physics" onto the deformable's collision_mesh OVERRIDES the body's
    # Deformable_Material in 6.0 (it is the most specific physics binding), so the
    # solver runs with PhysX's default (soft) Young's modulus -- measured ~2.4 mm
    # indentation vs ~0.4 mm on 5.1, and E x10 on Deformable_Material had no effect.
    # Deformable_Material already carries s=0.9 / d=0.8 friction, so the binding is
    # redundant. Keep it available for A/B testing only.
    if os.environ.get("SPONGE_COLL_MAT", "0") == "1":
        apply_physics_material(stage, "/World/CoRo_tactile/CoRo_tactile/Sponge/collision_mesh", 0.9,  0.8)
    else:
        print("  Physics mat -> Sponge/collision_mesh SKIPPED (friction lives on Deformable_Material)")

    simulation_app.update()

    # ── Scene objects ─────────────────────────────────────────
    # PORT NOTE: my_world.scene.add(Robot(...)) -> Articulation(ArticulationCfg(...)).
    # We reference the robot that already exists in the stage via its prim path.
    # A single implicit PD actuator group covers all arm joints, matching the
    # KP/KD/effort the standalone script set per joint.
    from isaaclab.actuators import ImplicitActuatorCfg
    robot_cfg_lab = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=None,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=MAX_EFFORT,
                stiffness=KP_GAINS,
                damping=KD_GAINS,
            ),
        },
    )
    robot = Articulation(robot_cfg_lab)

    # Cosmetic target markers (optional). PORT NOTE: replaces VisualCuboid+OmniPBR.
    target_a_prim = create_visual_marker(stage, "/World/target_A", TARGET_A_WORLD)
    target_b_prim = create_visual_marker(stage, "/World/target_B", TARGET_B_WORLD)

    # ── cuRobo setup (PORT: 0.7 MotionGen -> 0.8 MotionPlanner) ──
    scene_parser = UsdSceneParser()
    scene_parser.load_stage(stage)
    world_cfg = scene_parser.get_obstacles_from_stage(
        reference_prim_path="/World", ignore_substring=OBSTACLE_IGNORE,
    )

    device_cfg = DeviceCfg()
    robot_yaml = load_curobo_robot_cfg_v2()
    robot_cfg  = robot_yaml["robot_cfg"]

    cspace_names   = list(robot_cfg["kinematics"]["cspace"]["joint_names"])
    init_by_name   = dict(zip(cspace_names, np.deg2rad(INITIAL_JOINT_DEGREES).tolist()))
    robot_cfg["kinematics"]["cspace"]["default_joint_position"] = [init_by_name[n] for n in cspace_names]

    _create_kwargs = dict(
        collision_cache={"mesh": 10, "cuboid": 10},   # 0.7 key "obb" -> "cuboid"; enables mesh checker
        num_trajopt_seeds=12,                         # num_graph_seeds no longer exists
        interpolation_dt=INTERPOLATION_DT,
        # push_y with x/z/orientation held needs ~2000 waypoints at 8 ms (~16 s);
        # 12 seeds x 6 dof x 4 buffers is only ~1.2 MB per 1000 waypoints here.
        interpolation_buffer_size=6000,
    )
    # create() accepts a dict robot config; the YAML-loader path normalises both
    # {"robot_cfg": {...}} and the bare inner dict, so try the full YAML first.
    try:
        planner_cfg = MotionPlannerCfg.create(robot=robot_yaml, **_create_kwargs)
    except (KeyError, TypeError, AttributeError) as e:
        print(f"[cuRobo] create(robot=<full yaml dict>) failed ({e!r}); retrying with robot_cfg sub-dict")
        planner_cfg = MotionPlannerCfg.create(robot=robot_cfg, **_create_kwargs)

    planner = MotionPlanner(planner_cfg)
    planner.update_world(world_cfg)          # obstacles parsed from the live stage

    # cuRobo's kinematics decides the canonical joint order now; everything
    # downstream (idx_cu, init_positions, FK) uses this order.
    j_names = list(planner.joint_names)
    if j_names != cspace_names:
        print(f"[cuRobo] joint order differs from YAML cspace: {j_names} vs {cspace_names}")
    init_positions = [init_by_name[n] for n in j_names]
    if list(planner.tool_frames) != [EE_LINK_NAME]:
        raise RuntimeError(f"[cuRobo] tool_frames={planner.tool_frames}, expected [{EE_LINK_NAME!r}]")

    print("Warming up cuRobo...")
    planner.warmup(enable_graph=False, num_warmup_iterations=3)
    print("cuRobo ready.")

    # ==========================================================
    # PORT NOTE: my_world.initialize_physics() + robot.initialize()
    #            -> sim.reset() (this initialises the physics view AND the
    #               Articulation buffers in one call).
    # ==========================================================
    sim.reset()
    robot.reset()

    # Build the joint name->column map once, and the cuRobo-order index list.
    dof_map  = build_dof_index_map(robot)
    idx_cu   = art_get_dof_indices(dof_map, j_names)   # columns in cuRobo order
    print(f"[JOINTS] Articulation joint order : {robot.joint_names}")
    print(f"[JOINTS] cuRobo joint order        : {j_names}")
    print(f"[JOINTS] cuRobo->column indices    : {idx_cu}")

    # PORT NOTE: MotionGenPlanConfig is gone. max_attempts / enable_graph_attempt
    # are now plan_pose() kwargs (MAX_ATTEMPTS / ENABLE_GRAPH_ATTEMPT above),
    # finetune is always on, time dilation is emulated by STEPS_PER_CMD, and the
    # partial-pose hold is set per phase with set_hold_axes().

    # ==========================================================
    # JOINT INIT — write, settle, verify (fixes "positions don't stick")
    # PORT NOTE: In the standalone loop you were re-setting joints for the first
    # 10 played frames and hoping they'd stick before cuRobo read them. In Isaac
    # Lab we do it deterministically BEFORE the phase loop: write the state to
    # sim, step a few times so PhysX latches it, then read it back and confirm.
    # ==========================================================
    art_set_joint_positions(robot, init_positions, idx_cu, device)
    robot.write_data_to_sim()
    for _ in range(SETTLE_FRAMES):
        # hold the target while physics settles
        art_apply_position_target(robot, init_positions, idx_cu, device)
        sim.step(render=RENDER)
        robot.update(PHYSICS_DT)
    qp_check, qv_check = art_get_joint_state(robot, idx_cu)
    print(f"[DIAG] init target (rad) : {np.round(init_positions, 4)}")
    print(f"[DIAG] joint pos  (rad)  : {np.round(qp_check, 4)}")
    print(f"[DIAG] joint vel  (rad/s): {np.round(qv_check, 4)}")
    print(f"[DIAG] max |pos err|     : {np.max(np.abs(qp_check - np.array(init_positions))):.5f}")

    # ── Deformable read setup (volume deformable tensor view) ──
    # PROBE-PROVEN on this stack (Sim 5.1 / Lab 2.3.2 / Physics 107.3):
    #   * Sponge is a PhysxAutoDeformable VOLUME deformable.
    #   * USD `points` & Fabric are frozen snapshots (constant velocities) — dead.
    #   * The live source is the physics tensor view created by
    #     create_volume_deformable_body_view(<body_path>), read each frame via
    #     get_simulation_nodal_positions(). count==1 for our single sponge.
    # PORT NOTE (Sim 6.0): isaacsim.* modules only resolve AFTER the app is
    # instantiated, so this import stays here rather than at module top. Sim 6.0
    # restructured isaacsim.core toward isaacsim.core.experimental, so try the
    # known locations in order.
    SimulationManager = None
    _sm_err = []
    for _mod in ("isaacsim.core.simulation_manager",
                 "isaacsim.core.experimental.simulation_manager"):
        try:
            SimulationManager = __import__(_mod, fromlist=["SimulationManager"]).SimulationManager
            print(f"[DEFORMABLE] SimulationManager from {_mod}")
            break
        except (ImportError, AttributeError) as e:
            _sm_err.append(f"{_mod}: {e}")
    if SimulationManager is None:
        # PORT NOTE (IsaacLab @0b5605c2): the isaacsim.core.simulation_manager Kit
        # extension is not enabled by this commit's app experience, so the module
        # is not importable. The only thing we use it for is
        # get_physics_sim_view(), which wraps omni.physics.tensors -- call that
        # directly. Same tensor view type, same create_*_view factories.
        try:
            import omni.physics.tensors as _tensors

            class SimulationManager:  # minimal shim
                _view = None

                @classmethod
                def get_physics_sim_view(cls):
                    if cls._view is None:
                        cls._view = _tensors.create_simulation_view("torch")
                        cls._view.set_subspace_roots("/")
                    return cls._view

            print("[DEFORMABLE] SimulationManager extension unavailable on this IsaacLab "
                  "commit; using omni.physics.tensors.create_simulation_view('torch') directly")
        except ImportError as e:
            _sm_err.append(f"omni.physics.tensors: {e}")
    if SimulationManager is None:
        raise ImportError(
            "Could not locate SimulationManager -- needed for the deformable "
            "tensor view. Tried:\n  " + "\n  ".join(_sm_err))

    def _to_np(x):
        if isinstance(x, wp.array):
            return x.numpy()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    deform_view = None
    rest_ref = None
    _pv = None
    try:
        _pv = SimulationManager.get_physics_sim_view()

        # PORT NOTE (Sim 6.0): SimulationManager can now hand back a sim view
        # from a NON-PhysX engine (Newton). Newton's tensor API covers rigid
        # bodies + articulations only -- it has no deformable views at all, so
        # create_volume_deformable_body_view() will simply be absent. If that
        # happens, the backend pin above did not take effect.
        _backend = getattr(SimulationManager, "get_physics_backend", None)
        if callable(_backend):
            try:
                print(f"[BACKEND] active physics engine = {_backend()}")
            except Exception:
                pass

        _mk = getattr(_pv, "create_volume_deformable_body_view", None)
        if _mk is None:
            # Sim 6.0 renamed some experimental view factories; try the aliases.
            for _alt in ("create_deformable_volume_view",
                         "create_soft_body_view"):
                _mk = getattr(_pv, _alt, None)
                if _mk is not None:
                    print(f"[DEFORMABLE] using fallback factory '{_alt}'")
                    break

        if _mk is not None:
            deform_view = _mk(SPONGE_ROOT_PATH)
            cnt = getattr(deform_view, "count", None) if deform_view is not None else None
            print(f"[DEFORMABLE] volume view count={cnt}")
            if deform_view is not None and cnt:
                # JUNE-26 CONVENTION: rest = the authored mesh-local
                # omniphysics:restShapePoints (small coords, static), exactly what
                # the standalone recorded in s1_Rx/Ry/Rz. Read once from USD.
                rest_ref = None
                _sm = find_deformable_simulation_mesh(SPONGE_ROOT_PATH)
                if _sm is not None:
                    _ra = _sm.GetAttribute("omniphysics:restShapePoints")
                    _rv = _ra.Get() if (_ra and _ra.IsValid()) else None
                    if _rv is not None:
                        rest_ref = np.array([[float(p[0]), float(p[1]), float(p[2])]
                                             for p in _rv], dtype=float)
                if rest_ref is None:
                    print("[DEFORMABLE] WARNING: restShapePoints unavailable.")
                else:
                    print(f"[DEFORMABLE] nodes={rest_ref.shape[0]}  "
                          f"restShapePoints first={rest_ref[0].round(5).tolist()}")
            else:
                deform_view = None
        else:
            print("[DEFORMABLE] ERROR: no deformable view factory on this sim view.\n"
                  "  This almost always means the NEWTON backend is active. Newton\n"
                  "  has no deformable/FEM support in Isaac Sim 6.0 -- deformables\n"
                  "  are PhysX-only. The backend is chosen by the config object passed\n"
                  "  to SimulationCfg(physics=...): PhysxCfg() selects PhysX. Check that\n"
                  "  the [BACKEND] line above says PhysxCfg.")
    except Exception as e:
        print(f"[DEFORMABLE] ERROR creating volume view: {e}")

    sensor_prim = stage.GetPrimAtPath(SENSOR_POSE_PATH)
    if not sensor_prim or not sensor_prim.IsValid():
        print(f"[WARNING] Sensor prim not found: {SENSOR_POSE_PATH}")
        sensor_prim = None

    # ── Live Case_m pose via rigid-body physics view ──────────
    # The sensor Case_m is a rigid body (PhysxRigidBodyAPI), attached to the
    # adapter. Its USD xform is FROZEN during sim (USD isn't the live source),
    # so we read its pose from the rigid-body tensor view instead. get_transforms()
    # returns (count, 7) = [x, y, z, qx, qy, qz, qw] in WORLD frame, live.
    #
    # FRAME MATCH WITH THE STANDALONE CSV: the original Test_pushing.py read
    # xformOp:translate / xformOp:orient on Case_m — i.e. the LOCAL pose,
    # relative to its parent /World/CoRo_tactile/CoRo_tactile. To keep the CSV
    # convention identical (compute_dz expects local), we convert the live
    # WORLD pose to LOCAL each frame using the parent's (static) world pose:
    #     T_local = R_parent^-1 · (T_world - p_parent)
    #     q_local = q_parent^-1 ⊗ q_world
    case_view = None
    parent_pos = np.zeros(3)
    parent_rot_inv = R.identity()
    try:
        _mkr = getattr(_pv, "create_rigid_body_view", None)
        if _mkr is not None:
            case_view = _mkr(SENSOR_POSE_PATH)
            ccnt = getattr(case_view, "count", None) if case_view is not None else None
            print(f"[CASE] rigid-body view count={ccnt}")
            if not ccnt:
                case_view = None
            else:
                _t0 = _to_np(case_view.get_transforms()).reshape(-1)
                print(f"[CASE] initial WORLD pose xyz={_t0[:3].round(5).tolist()} "
                      f"quat(xyzw)={_t0[3:7].round(5).tolist()}")
        else:
            print("[CASE] ERROR: create_rigid_body_view missing.")
    except Exception as e:
        print(f"[CASE] ERROR creating rigid-body view: {e}")

    # DIAG: rigid-body view on the pushed cylinder so "did it get pushed" is in
    # the log, not something to eyeball in the viewer.
    cyl_view = None
    if not _NO_CYL:
        try:
            cyl_view = _pv.create_rigid_body_view(OBJECT_XFORM_PATH)
            if getattr(cyl_view, "count", 0):
                _c0 = _to_np(cyl_view.get_transforms()).reshape(-1)
                print(f"[CYL] rigid-body view ok, initial WORLD xyz={_c0[:3].round(4).tolist()}")
            else:
                cyl_view = None
        except Exception as e:
            print(f"[CYL] view not created: {e}")
            cyl_view = None

    def cyl_world_xyz():
        if cyl_view is None:
            return None
        try:
            return _to_np(cyl_view.get_transforms()).reshape(-1)[:3]
        except Exception:
            return None

    # Parent world pose (static prim -> USD xform cache is fine, read once).
    _parent_path = SENSOR_POSE_PATH.rsplit("/", 1)[0]
    _parent_prim = stage.GetPrimAtPath(_parent_path)
    if _parent_prim and _parent_prim.IsValid():
        _m = UsdGeom.XformCache().GetLocalToWorldTransform(_parent_prim)
        _t = _m.ExtractTranslation()
        _q = _m.ExtractRotationQuat()
        _im = _q.GetImaginary()
        parent_pos = np.array([float(_t[0]), float(_t[1]), float(_t[2])])
        parent_rot = R.from_quat([float(_im[0]), float(_im[1]), float(_im[2]),
                                  float(_q.GetReal())])  # scipy = (x,y,z,w)
        parent_rot_inv = parent_rot.inv()
        print(f"[CASE] parent {_parent_path} world pos={parent_pos.round(5).tolist()} "
              f"rot(quat xyzw)={parent_rot.as_quat().round(5).tolist()}")
    else:
        print(f"[CASE] WARNING: parent prim not found at {_parent_path}; "
              f"pose will be written in WORLD frame.")

    # One-time ORIENTATION CALIBRATION (Kabsch). June 26's recorded orientation
    # was NOT the authored xformOp:orient (that reads identity) — it was the
    # live physics pose written back by the standalone World. Rather than guess
    # the body-frame convention, we solve for the rotation that compute_dz
    # actually needs: the R that maps the mesh-local rest points onto the
    # current (parent-local) nodes at t=0, i.e. R@rest + T ≈ cur. We then store
    # the constant offset from the view's local quat to that fitted R and apply
    # it every frame. The alignment residual is printed — it should be ~0 mm.
    # ORIENTATION: no longer calibrated. PORT NOTE (Sim 6.0): this used to solve
    # a constant offset (case_q_corr) between the live Case_m body quaternion and
    # the rotation compute_dz needs. In Sim 6.0 the live body quaternion is in a
    # different body frame than in 5.1, leaving a constant ~106 deg error that
    # the offset preserved rather than removed -- dz reached 7-19 m against a
    # true range of ~0.013 m. The transform is now fit per frame directly from
    # rest -> current (see kabsch_fit), so no calibration constant exists to be
    # wrong. We only VALIDATE the fit here.
    case_q_corr = R.identity()   # kept for call-site compatibility; unused
    try:
        if deform_view is not None and rest_ref is not None:
            cur_w0 = _to_np(deform_view.get_simulation_nodal_positions()).reshape(-1, 3)
            cur_l0 = parent_rot_inv.apply(cur_w0 - parent_pos)
            R_fit_mat, T_fit = kabsch_fit(rest_ref, cur_l0)
            resid = np.linalg.norm((rest_ref @ R_fit_mat.T + T_fit) - cur_l0, axis=1)
            print(f"[CASE] Kabsch fit (xyzw)        = "
                  f"{R.from_matrix(R_fit_mat).as_quat().round(5).tolist()}")
            print(f"[CASE] T_fit (local)            = {T_fit.round(5).tolist()}")
            print(f"[CASE] rest->cur alignment residual: mean={resid.mean()*1000:.3f}mm "
                  f"max={resid.max()*1000:.3f}mm  (should be ~0)")
            if resid.mean() > 0.005:
                print("[CASE] WARNING: residual >5mm -- rest/current node "
                      "correspondence is suspect; dz will be unreliable.")
        else:
            print("[CASE] validation skipped (missing view or rest points).")
    except Exception as e:
        print(f"[CASE] Kabsch validation failed: {e}")

    # ── Prepare CSV files (unchanged) ─────────────────────────
    # PORT NOTE: FAIL FAST. The Sim 6.0 schema rename made the whole deformable
    # path degrade SILENTLY -- the run completed, wrote CSVs, and every tactile
    # value was 0.0 because rest_ref was None and the baseline was all zeros.
    # A run that cannot produce valid data must not produce a plausible-looking
    # file. Set ALLOW_DEGRADED=1 to override (e.g. to test robot motion alone).
    _degraded = []
    if deform_view is None:
        _degraded.append("deformable volume view is None (no live node data)")
    if rest_ref is None:
        _degraded.append("rest_ref is None (restShapePoints unavailable)")
    if case_view is None:
        _degraded.append("Case_m rigid-body view is None (no live sensor pose)")
    if rest_ref is not None and deform_view is not None:
        try:
            _n_live = _to_np(deform_view.get_simulation_nodal_positions()).reshape(-1, 3).shape[0]
            if _n_live != rest_ref.shape[0]:
                _degraded.append(
                    f"node-count mismatch: live view has {_n_live}, "
                    f"restShapePoints has {rest_ref.shape[0]}")
            if len(ordered_ids) != N_ROWS * N_COLS:
                _degraded.append(
                    f"ordered_ids has {len(ordered_ids)} entries but the CNN "
                    f"expects {N_ROWS}x{N_COLS}={N_ROWS*N_COLS}")
            if max(ordered_ids) >= _n_live:
                _degraded.append(
                    f"ordered_ids max index {max(ordered_ids)} exceeds live node "
                    f"count {_n_live} -- node IDs do not match this mesh")
        except Exception as e:
            _degraded.append(f"could not validate node counts: {e}")

    if _degraded:
        _msg = ("\n" + "=" * 70 +
                "\n[ABORT] The deformable/tactile pipeline is not viable:\n" +
                "".join(f"  * {d}\n" for d in _degraded) +
                "\nAny CSV written now would contain zero or meaningless tactile\n"
                "data. Fix the above before trusting output. Set ALLOW_DEGRADED=1\n"
                "to run anyway (robot motion only -- do NOT use the tactile CSV).\n"
                + "=" * 70)
        if os.environ.get("ALLOW_DEGRADED") != "1":
            print(_msg)
            simulation_app.close()
            raise SystemExit(1)
        print(_msg + "\n[ALLOW_DEGRADED=1] Continuing anyway.")

    def_file, def_writer, tac_file, tac_writer = prepare_csv(CSV_DIR, CSV_BASENAME)

    # ── State machine (unchanged logic) ───────────────────────
    # PORT NOTE: we start at "baseline" because SETTLE already happened during
    # the deterministic joint-init block above. Set to "settle" if you'd rather
    # keep the original ordering.
    phase              = "baseline"
    phase_counter      = 0

    cmd_idx = cmd_step_idx = 0
    cmd_plan = idx_list    = None
    targets_initialized    = False
    target_list_base       = []
    target_ori_base        = []
    planned_phase          = None
    last_curobo_goal_pos   = None
    last_curobo_goal_names = None
    plan_fail_count        = 0
    recording_active       = False
    sim_time               = 0.0
    step                   = 0

    material_list_prims = [target_a_prim, target_b_prim]

    # ==========================================================
    # MAIN LOOP
    # PORT NOTE: no is_playing() gate — SimulationContext plays as soon as we
    # step it. `step` is our own frame counter (was current_time_step_index).
    #
    # ORDER MATTERS. We step FIRST so mesh/joint reads are fresh for this frame,
    # then within each phase we apply the current command AND advance the index
    # in the SAME place (matching the standalone apply_action placement). The
    # earlier port split these apart, which caused an off-by-one: the arm was
    # commanded from a stale index and snapped back at the end of each plan.
    # `hold_target` holds the last sent target so the PD arm doesn't sag when
    # there's no active trajectory.
    # ==========================================================
    hold_target = list(init_positions)   # last commanded joint target (cols in hold_idx order)
    hold_idx    = list(idx_cu)

    while simulation_app.is_running():

        # advance physics one frame, then refresh articulation buffers
        sim.step(render=RENDER)
        robot.update(PHYSICS_DT)

        # re-assert the current hold target every frame (PD position control)
        art_apply_position_target(robot, hold_target, hold_idx, device)

        step     += 1
        sim_time += PHYSICS_DT

        # ── PHASE: settle ─────────────────────────────────────
        if phase == "settle":
            phase_counter += 1
            if phase_counter >= SETTLE_FRAMES:
                print(f"[PHASE] Settle done ({SETTLE_FRAMES} frames). Starting baseline collection.")
                phase         = "baseline"
                phase_counter = 0
            continue

        # ── PHASE: baseline (restored: collect no-contact dz) ─
        if phase == "baseline":
            # PORT NOTE (Sim 6.0): the pad is NOT necessarily settled when this
            # phase begins. Measured on 6.0: the Kabsch residual is ~9 mm during
            # early baseline frames and relaxes to ~0.045 mm only later. Averaging
            # dz over an unsettled pad bakes that transient into the baseline, and
            # infer_tactile subtracts it from every later frame -- which showed up
            # as tactile values offset by a large constant (T_01 sitting near 200
            # instead of starting near 0, vs 5.1 which started at ~0.9).
            #
            # So gate on the MEASUREMENT: wait for the pad's non-rigid deformation
            # to go quiet before collecting any baseline samples. 5.1 settled fast
            # enough that a fixed frame count worked; 6.0 does not.
            if deform_view is not None and rest_ref is not None:
                if not settle_ok:
                    try:
                        _c = _to_np(deform_view.get_simulation_nodal_positions()).reshape(-1, 3)
                        _c = parent_rot_inv.apply(_c - parent_pos)
                        _Rm, _T = kabsch_fit(rest_ref, _c)
                        _r = float(np.linalg.norm((rest_ref @ _Rm.T + _T) - _c, axis=1).mean())
                    except Exception:
                        _r = float("nan")
                    settle_hist.append(_r)
                    # Gate is wall-clock based: the stability window is 5 frames
                    # at 60 Hz (83 ms) regardless of PHYSICS_HZ, and the pad must
                    # have had at least MIN_SETTLE_FRAMES to relax. At 480 Hz the
                    # raw 5-step window passed in 10 ms on an unsettled pad and
                    # contaminated the baseline (edge load on taxels 1-4).
                    _w = settle_hist[-SETTLE_WINDOW:]
                    if (len(settle_hist) >= max(SETTLE_WINDOW, MIN_SETTLE_FRAMES)
                            and (max(_w) - min(_w)) < 5e-5 and max(_w) < 1e-3):
                        settle_ok = True
                        print(f"[PHASE] Pad settled after {len(settle_hist)} frames "
                              f"(residual {_r*1000:.3f}mm). Collecting baseline.")
                    elif len(settle_hist) > MAX_SETTLE_FRAMES:
                        print(f"[PHASE] WARNING: pad did not settle within "
                              f"{MAX_SETTLE_FRAMES} frames (residual {_r*1000:.3f}mm). "
                              f"Proceeding -- baseline may be contaminated.")
                        settle_ok = True
                    else:
                        if len(settle_hist) % (60 * OUTPUT_EVERY) == 0:
                            print(f"[PHASE] waiting for pad to settle... "
                                  f"residual={_r*1000:.3f}mm (frame {len(settle_hist)})")
                        continue

                dz = get_dz_live(deform_view, case_view, rest_ref,
                                 parent_pos, parent_rot_inv, case_q_corr,
                                 ordered_ids)
                if dz is not None:
                    baseline_buffer.append(dz)

            phase_counter += 1
            if phase_counter >= BASELINE_FRAMES:
                if len(baseline_buffer) > 0:
                    baseline = np.mean(baseline_buffer, axis=0)
                    _bs = np.std(baseline_buffer, axis=0).mean()
                    if np.abs(baseline).mean() > 0.5 or _bs > 0.05:
                        print(f"[PHASE] WARNING: baseline looks contaminated "
                              f"(|mean|={np.abs(baseline).mean():.4f}, "
                              f"frame-to-frame std={_bs:.4f}). A clean no-contact "
                              f"baseline should be near zero and steady. Tactile "
                              f"output will be offset by this amount.")
                    print(f"[PHASE] Baseline done ({len(baseline_buffer)} frames). "
                          f"mean={baseline.mean():.6f}  Starting planning to Target A.")
                else:
                    baseline = np.zeros(N_ROWS * N_COLS, dtype=np.float32)
                    print("[PHASE] Baseline done but no data collected — using zeros.")
                phase         = "plan_A"
                phase_counter = 0

                if not targets_initialized:
                    target_list_base, target_ori_base = init_targets(
                        stage, robot, planner, init_positions, j_names,
                        device_cfg, target_a_prim, target_b_prim,
                    )
                    targets_initialized = True
            continue

        # ── PHASE: plan_A — move to Target A ─────────────────
        if phase == "plan_A":
            if cmd_plan is None and step % 10 == 0:
                set_hold_axes(planner, HOLD_ORI_WEIGHT, device_cfg)   # was PoseCostMetric(hold_partial_pose)
                print("-"*50 + "\nPHASE: plan_A -> Target A\n" + "-"*50)

                set_marker_color(target_a_prim, (0., 1., 0.))
                set_marker_color(target_b_prim, (0.1, 0.1, 0.1))

                # PORT NOTE: robot.get_joints_state() -> art_get_joint_state();
                # qp is read in cuRobo joint order (idx_cu) so no reordering needed.
                qp, qv  = art_get_joint_state(robot, idx_cu)
                cu_js   = make_js(qp, j_names, device_cfg)
                ik_goal = make_goal(target_list_base[0], target_ori_base[0], device_cfg)
                result  = planner.plan_pose(ik_goal, cu_js,
                                            max_attempts=MAX_ATTEMPTS,
                                            enable_graph_attempt=ENABLE_GRAPH_ATTEMPT)
                succ    = bool(result is not None and result.success.any())
                print(f"  Planning to Target A: {'succeeded' if succ else 'failed'}")

                if succ:
                    plan_fail_count        = 0
                    curobo_plan            = flatten_plan(result.get_interpolated_plan())   # [H, dof]
                    last_curobo_goal_pos   = curobo_plan.position[-1].clone()
                    last_curobo_goal_names = list(curobo_plan.joint_names)
                    # no lock_joints on the M0609, so the plan already carries the
                    # full joint set; get_full_js is not needed.
                    common                 = [x for x in robot.joint_names if x in curobo_plan.joint_names]
                    idx_list               = [dof_map[x] for x in common]
                    cmd_plan               = reorder_js(curobo_plan, common)
                    cmd_idx = cmd_step_idx = 0
                    planned_phase          = "plan_A"
                else:
                    plan_fail_count += 1
                    if plan_fail_count >= MAX_PLAN_FAILS:
                        print(f"  {MAX_PLAN_FAILS} failures to Target A. Aborting.")
                        break

            # Execute trajectory: apply the current command AND advance the
            # index in the same place (matches standalone apply_action).
            if cmd_plan is not None:
                cmd_pos = cmd_plan.position[cmd_idx].cpu().numpy()
                art_apply_position_target(robot, cmd_pos, idx_list, device)
                # keep the arm on this target on any frame where we don't re-plan
                hold_target, hold_idx = list(cmd_pos), list(idx_list)

                cmd_step_idx += 1
                if cmd_step_idx >= STEPS_PER_CMD:
                    cmd_idx += 1; cmd_step_idx = 0

                if cmd_idx >= len(cmd_plan.position):
                    final_pos = cmd_plan.position[-1].cpu().numpy()
                    # hold the final target with PD (do NOT teleport — that was
                    # the snap-back). The controller holds the arm at Target A.
                    hold_target, hold_idx = list(final_pos), list(idx_list)
                    art_apply_position_target(robot, final_pos, idx_list, device)

                    cmd_idx = cmd_step_idx = 0
                    cmd_plan      = None
                    planned_phase = None
                    plan_fail_count = 0

                    print("[PHASE] Arrived at Target A — starting recording and Y push.")
                    recording_active = True
                    phase            = "push_y"
                    phase_counter    = 0
            continue

        # ── PHASE: push_y — push from A to B, saving data ────
        if phase == "push_y":

            if recording_active and deform_view is not None and step % OUTPUT_EVERY == 0:
                save_sponge_data(
                    deform_view, rest_ref, case_view,
                    parent_pos, parent_rot_inv, case_q_corr,
                    def_writer, def_file,
                    tac_writer, tac_file,
                    frame=step // OUTPUT_EVERY, sim_time=sim_time,
                    ordered_ids=ordered_ids, baseline=baseline,
                    x_train_max=x_train_max, session=session,
                    input_name=input_name,
                )
                if (step // OUTPUT_EVERY) % 60 == 0:
                    _cw = cyl_world_xyz()
                    _sw = _to_np(case_view.get_transforms()).reshape(-1)[:3] if case_view is not None else None
                    print(f"[CYL] frame={step // OUTPUT_EVERY}  cylinder world xyz="
                          f"{None if _cw is None else _cw.round(4).tolist()}  "
                          f"sensor case world xyz={None if _sw is None else _sw.round(4).tolist()}")

            if cmd_plan is None and step % 10 == 0:
                set_hold_axes(planner, PUSH_Y_WEIGHT, device_cfg)   # hold x,z + orientation, free along Y
                print("-"*50 + "\nPHASE: push_y -> Target B\n" + "-"*50)

                set_marker_color(target_a_prim, (0.1, 0.1, 0.1))
                set_marker_color(target_b_prim, (0., 1., 0.))

                # start from the end of the previous plan (same as before); the
                # plan came out in cuRobo order so its names == j_names.
                cu_js   = reorder_js(make_js(last_curobo_goal_pos.cpu().numpy(),
                                             last_curobo_goal_names, device_cfg), j_names)
                ik_goal = make_goal(target_list_base[1], target_ori_base[1], device_cfg)
                result  = planner.plan_pose(ik_goal, cu_js,
                                            max_attempts=MAX_ATTEMPTS,
                                            enable_graph_attempt=ENABLE_GRAPH_ATTEMPT)
                succ    = bool(result is not None and result.success.any())
                print(f"  Planning to Target B: {'succeeded' if succ else 'failed'}")

                if succ:
                    plan_fail_count        = 0
                    curobo_plan            = flatten_plan(result.get_interpolated_plan())   # [H, dof]
                    last_curobo_goal_pos   = curobo_plan.position[-1].clone()
                    last_curobo_goal_names = list(curobo_plan.joint_names)
                    common                 = [x for x in robot.joint_names if x in curobo_plan.joint_names]
                    idx_list               = [dof_map[x] for x in common]
                    cmd_plan               = reorder_js(curobo_plan, common)
                    cmd_idx = cmd_step_idx = 0
                    planned_phase          = "push_y"
                else:
                    plan_fail_count += 1
                    if plan_fail_count >= MAX_PLAN_FAILS:
                        print(f"  {MAX_PLAN_FAILS} failures to Target B. Aborting.")
                        break

            # Execute trajectory: apply current command AND advance index together.
            if cmd_plan is not None:
                cmd_pos = cmd_plan.position[cmd_idx].cpu().numpy()
                art_apply_position_target(robot, cmd_pos, idx_list, device)
                hold_target, hold_idx = list(cmd_pos), list(idx_list)

                cmd_step_idx += 1
                if cmd_step_idx >= STEPS_PER_CMD:
                    cmd_idx += 1; cmd_step_idx = 0

                if cmd_idx >= len(cmd_plan.position):
                    final_pos = cmd_plan.position[-1].cpu().numpy()
                    # hold Target B with PD instead of teleporting back.
                    hold_target, hold_idx = list(final_pos), list(idx_list)
                    art_apply_position_target(robot, final_pos, idx_list, device)

                    cmd_idx = cmd_step_idx = 0
                    cmd_plan      = None
                    planned_phase = None

                    print(f"[PHASE] Arrived at Target B — waiting {END_FRAMES} frames then closing.")
                    phase         = "end_wait"
                    phase_counter = 0
            continue

        # ── PHASE: end_wait ───────────────────────────────────
        if phase == "end_wait":
            if recording_active and deform_view is not None and step % OUTPUT_EVERY == 0:
                save_sponge_data(
                    deform_view, rest_ref, case_view,
                    parent_pos, parent_rot_inv, case_q_corr,
                    def_writer, def_file,
                    tac_writer, tac_file,
                    frame=step // OUTPUT_EVERY, sim_time=sim_time,
                    ordered_ids=ordered_ids, baseline=baseline,
                    x_train_max=x_train_max, session=session,
                    input_name=input_name,
                )
                if (step // OUTPUT_EVERY) % 60 == 0:
                    _cw = cyl_world_xyz()
                    _sw = _to_np(case_view.get_transforms()).reshape(-1)[:3] if case_view is not None else None
                    print(f"[CYL] frame={step // OUTPUT_EVERY}  cylinder world xyz="
                          f"{None if _cw is None else _cw.round(4).tolist()}  "
                          f"sensor case world xyz={None if _sw is None else _sw.round(4).tolist()}")

            phase_counter += 1
            if phase_counter >= END_FRAMES:
                print(f"[PHASE] End wait done ({END_FRAMES} frames). Closing simulation.")
                break
            continue

    # ── Cleanup ───────────────────────────────────────────────
    def_file.flush(); def_file.close()
    tac_file.flush(); tac_file.close()
    print("[CSV] Files closed.")
    simulation_app.close()
