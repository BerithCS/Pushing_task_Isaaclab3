#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

"""
Sponge deformation → ONNX tactile prediction — animated over all frames.
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ================================================================
# CONFIG
# ================================================================
CSV_PATH        = "/home/berith/Documents/coro_doosan_station/sponge_data_009.csv"
NODES_FILE      = "/home/berith/Documents/Pushing_task/CNN_tactile/Nodes_id_filtered.csv"
ONNX_MODEL_PATH = "/home/berith/Documents/Pushing_task/CNN_tactile/best.onnx"
X_TRAIN_MAX_NPY = "/home/berith/Documents/Pushing_task/CNN_tactile/CNN_max.npy"

N_BASELINE = 5
N_ROWS     = 18
N_COLS     = 12
MAP_H      = 7
MAP_W      = 4
INTERVAL   = 500    # ms between frames

# ================================================================
# HELPERS
# ================================================================

def rotation_matrix(w, x, y, z):
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


def load_ordered_node_ids(path):
    df  = pd.read_csv(path)
    col = "node_id" if "node_id" in df.columns else df.columns[0]
    seen, ids = set(), []
    for nid in df[col].dropna().astype(int):
        if nid not in seen:
            seen.add(nid); ids.append(nid)
    return ids


def compute_dz(fd, ordered_ids):
    rest_pts = fd[["s1_Rx", "s1_Ry", "s1_Rz"]].to_numpy(dtype=float)

    row = fd.iloc[0]
    R   = rotation_matrix(row.s1_Ori_w, row.s1_Ori_x,
                          row.s1_Ori_y, row.s1_Ori_z)
    T   = np.array([row.s1_Trans_x, row.s1_Trans_y, row.s1_Trans_z])

    sx = rest_pts[:, 0].max() - rest_pts[:, 0].min()
    sy = rest_pts[:, 1].max() - rest_pts[:, 1].min()
    sz = rest_pts[:, 2].max() - rest_pts[:, 2].min()

    rot_pts = (R @ rest_pts.T).T + T
    cx_w = (rot_pts[:, 0].min() + rot_pts[:, 0].max()) / 2
    cy_w = (rot_pts[:, 1].min() + rot_pts[:, 1].max()) / 2
    cz_w = (rot_pts[:, 2].min() + rot_pts[:, 2].max()) / 2

    fd_f   = fd.set_index("node_id").reindex(ordered_ids).dropna(subset=["s1_x"])
    rest_f = fd_f[["s1_Rx", "s1_Ry", "s1_Rz"]].to_numpy(dtype=float)
    curr_f = fd_f[["s1_x",  "s1_y",  "s1_z" ]].to_numpy(dtype=float)
    rot_f  = (R @ rest_f.T).T + T

    black  = scale_to_cube(rot_f,  cx_w, cy_w, cz_w, sx, sy, sz)
    orange = scale_to_cube(curr_f, cx_w, cy_w, cz_w, sx, sy, sz)

    return orange[:, 2] - black[:, 2]


def process_frame(fd, ordered_ids, baseline, x_train_max, session, input_name):
    dz      = compute_dz(fd, ordered_ids)
    dz      = dz - baseline
    dz_grid = dz.reshape(N_ROWS, N_COLS).astype(np.float32)

    X = np.clip(dz_grid / x_train_max, -1.0, 1.0)
    X = X.reshape(1, N_ROWS, N_COLS, 1).astype(np.float32)

    y_pred   = session.run(None, {input_name: X})[0]
    pred_map = y_pred[0].reshape(MAP_H, MAP_W)

    return dz_grid, pred_map


# ================================================================
# LOAD DATA
# ================================================================
print("Loading data...")
sim = pd.read_csv(CSV_PATH, sep="\t")
if "frame" not in sim.columns:
    sim = pd.read_csv(CSV_PATH, sep=",")
if "frame" not in sim.columns:
    print(f"  [SKIP] Cannot find 'frame' column — skipping.")

frames      = sorted(sim["frame"].unique())
ordered_ids = load_ordered_node_ids(NODES_FILE)
x_train_max = float(np.load(X_TRAIN_MAX_NPY))

print(f"  Frames     : {len(frames)}  ({frames[0]} → {frames[-1]})")
print(f"  Nodes      : {len(ordered_ids)}")
print(f"  X_train_max: {x_train_max:.6f}")

# ================================================================
# LOAD ONNX MODEL
# ================================================================
print("Loading ONNX model...")
session    = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"  ONNX model ready. Input name: {input_name}")

# ================================================================
# COMPUTE BASELINE
# ================================================================
print(f"Computing baseline from first {N_BASELINE} frames...")
baseline_list = []
for fid in frames[:N_BASELINE]:
    fd_b = sim[sim["frame"] == fid].copy()
    baseline_list.append(compute_dz(fd_b, ordered_ids))

baseline = np.mean(baseline_list, axis=0)
print(f"  Baseline mean : {baseline.mean():.6f}")
print(f"  Baseline max  : {baseline.max():.6f}")
print(f"  Baseline min  : {baseline.min():.6f}")

# ================================================================
# PRE-COMPUTE ALL FRAMES
# ================================================================
print("Running inference on all frames...")
dz_grids, pred_maps = [], []
for fid in frames:
    fd = sim[sim["frame"] == fid].copy()
    dz_grid, pred_map = process_frame(
        fd, ordered_ids, baseline, x_train_max, session, input_name
    )
    dz_grids.append(dz_grid)
    pred_maps.append(pred_map)
print(f"  Done — {len(dz_grids)} frames processed.")

# ================================================================
# ANIMATED PLOT
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor("#1e1e2e")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e1e2e")


def style_cbar(cbar, label):
    cbar.set_label(label, color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")


def style_ax(ax, title, xlabel, ylabel, xticks, yticks):
    ax.set_title(title, fontsize=12, color="white", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color="white")
    ax.set_ylabel(ylabel, fontsize=10, color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xticks(np.arange(xticks))
    ax.set_yticks(np.arange(yticks))
    ax.set_xticks(np.arange(-0.5, xticks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, yticks, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)


# Initial frame
im1 = ax1.imshow(dz_grids[0],  cmap="jet", origin="upper", aspect="auto")
im2 = ax2.imshow(pred_maps[0], cmap="jet", origin="upper", aspect="auto")

cbar1 = plt.colorbar(im1, ax=ax1)
cbar2 = plt.colorbar(im2, ax=ax2)
style_cbar(cbar1, "dZ (normalized, baseline removed)")
style_cbar(cbar2, "Tactile count")

style_ax(ax1, f"Input dZ grid  |  Frame {frames[0]}",
         "Column", "Row", N_COLS, N_ROWS)
style_ax(ax2, f"Predicted tactile map  |  Frame {frames[0]}",
         "Column", "Row", MAP_W, MAP_H)

fig.suptitle("Sponge deformation → ONNX tactile prediction",
             fontsize=13, color="white")
fig.tight_layout()


def update(k):
    im1.set_data(dz_grids[k])
    im1.set_clim(dz_grids[k].min(), dz_grids[k].max())
    im2.set_data(pred_maps[k])
    im2.set_clim(pred_maps[k].min(), pred_maps[k].max())
    ax1.set_title(f"Input dZ grid  |  Frame {frames[k]}",
                  fontsize=12, color="white", pad=10)
    ax2.set_title(f"Predicted tactile map  |  Frame {frames[k]}",
                  fontsize=12, color="white", pad=10)
    return im1, im2


ani = animation.FuncAnimation(
    fig, update,
    frames=len(frames),
    interval=INTERVAL,
    blit=False,
    repeat=True,
)

plt.show()
print("Done.")