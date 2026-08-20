#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

# ================================================================
# CONFIG
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEF_CSV = str(max(SCRIPT_DIR.glob("sponge_data_deformation_*.csv"), key=lambda p: p.stat().st_mtime))
TAC_CSV = str(max(SCRIPT_DIR.glob("sponge_data_tactiledata_*.csv"), key=lambda p: p.stat().st_mtime))
print(f"[CSV] Using: {DEF_CSV}\n[CSV] Using: {TAC_CSV}")

NODES_FILE = str(SCRIPT_DIR / "CNN_tactile" / "Nodes_id_filtered.csv")

N_ROWS   = 18
N_COLS   = 12
MAP_H    = 7
MAP_W    = 4

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
    try:
        df  = pd.read_csv(path)
        col = "node_id" if "node_id" in df.columns else df.columns[0]
        seen, ids = set(), []
        for nid in df[col].dropna().astype(int):
            if nid not in seen:
                seen.add(nid); ids.append(nid)
        print(f"  Loaded {len(ids)} node ids from {path}")
        return ids
    except Exception as e:
        print(f"  [WARN] Could not load nodes file: {e}")
        print(f"  Using first {N_ROWS * N_COLS} nodes as fallback")
        return list(range(N_ROWS * N_COLS))


def compute_dz(fd, ordered_ids):
    rest_pts = fd[["s1_Rx", "s1_Ry", "s1_Rz"]].to_numpy(dtype=float)
    curr_pts = fd[["s1_x",  "s1_y",  "s1_z" ]].to_numpy(dtype=float)

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


# ================================================================
# LOAD DATA
# ================================================================
print("Loading data...")
def_df = pd.read_csv(DEF_CSV)
tac_df = pd.read_csv(TAC_CSV)

def_frames = set(def_df["frame"].unique())
tac_frames = set(tac_df["frame"].unique())
frames     = sorted(def_frames & tac_frames)

print(f"  Deformation frames : {len(def_frames)}")
print(f"  Tactile frames     : {len(tac_frames)}")
print(f"  Matched frames     : {len(frames)}")

ordered_ids = load_ordered_node_ids(NODES_FILE)
tac_cols    = [f"T_{i+1:02d}" for i in range(28)]

# ================================================================
# COMPUTE BASELINE from first 5 frames
# ================================================================
N_BASELINE = 5
print(f"Computing baseline from first {N_BASELINE} frames...")
baseline_list = []
for fid in frames[:N_BASELINE]:
    fd_b = def_df[def_df["frame"] == fid].copy()
    baseline_list.append(compute_dz(fd_b, ordered_ids))
baseline = np.mean(baseline_list, axis=0)
print(f"  Baseline mean: {baseline.mean():.6f}")

# ================================================================
# PRE-COMPUTE ALL FRAMES
# ================================================================
print("Pre-computing all frames...")
dz_grids = []
tac_maps = []

for fid in frames:
    fd      = def_df[def_df["frame"] == fid].copy()
    dz      = compute_dz(fd, ordered_ids)
    dz      = dz - baseline
    dz_grid = dz.reshape(N_ROWS, N_COLS).astype(np.float32)
    dz_grids.append(dz_grid)

    tac_row = tac_df[tac_df["frame"] == fid].iloc[0]
    tac_map = tac_row[tac_cols].to_numpy(dtype=np.float32).reshape(MAP_H, MAP_W)
    tac_maps.append(tac_map)

print(f"  Done — {len(dz_grids)} frames pre-computed.")

# ================================================================
# INTERACTIVE PLOT WITH SLIDER
# ================================================================
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("#1e1e2e")

# Layout: two heatmaps on top, slider + buttons at bottom
ax1  = fig.add_axes([0.05, 0.20, 0.42, 0.72])   # deformation
ax2  = fig.add_axes([0.53, 0.20, 0.42, 0.72])   # tactile
ax_s = fig.add_axes([0.10, 0.07, 0.65, 0.04])   # slider
ax_p = fig.add_axes([0.78, 0.06, 0.06, 0.06])   # prev button
ax_n = fig.add_axes([0.86, 0.06, 0.06, 0.06])   # next button

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
im1 = ax1.imshow(dz_grids[0], cmap="jet", origin="upper", aspect="auto")
im2 = ax2.imshow(tac_maps[0], cmap="jet", origin="upper", aspect="auto")

cbar1 = plt.colorbar(im1, ax=ax1)
cbar2 = plt.colorbar(im2, ax=ax2)
style_cbar(cbar1, "dZ (baseline removed)")
style_cbar(cbar2, "Tactile value")

style_ax(ax1, f"Deformation grid  |  Frame {frames[0]}",
         "Column", "Row", N_COLS, N_ROWS)
style_ax(ax2, f"Tactile map  |  Frame {frames[0]}",
         "Column", "Row", MAP_W, MAP_H)

fig.suptitle("Sponge deformation + Tactile prediction",
             fontsize=13, color="white")

# ── Slider ────────────────────────────────────────────────────
ax_s.set_facecolor("#313244")
slider = Slider(
    ax=ax_s,
    label="Frame",
    valmin=0,
    valmax=len(frames) - 1,
    valinit=0,
    valstep=1,
    color="#89b4fa",
)
slider.label.set_color("white")
slider.valtext.set_color("white")

# ── Prev / Next buttons ───────────────────────────────────────
btn_prev = Button(ax_p, "◀ Prev", color="#313244", hovercolor="#45475a")
btn_next = Button(ax_n, "Next ▶", color="#313244", hovercolor="#45475a")
btn_prev.label.set_color("white")
btn_next.label.set_color("white")


def update(k):
    k = int(k)
    im1.set_data(dz_grids[k])
    im1.set_clim(dz_grids[k].min(), dz_grids[k].max())
    im2.set_data(tac_maps[k])
    im2.set_clim(tac_maps[k].min(), tac_maps[k].max())
    ax1.set_title(f"Deformation grid  |  Frame {frames[k]}",
                  fontsize=12, color="white", pad=10)
    ax2.set_title(f"Tactile map  |  Frame {frames[k]}",
                  fontsize=12, color="white", pad=10)
    # Update slider label to show frame index and actual frame number
    slider.valtext.set_text(f"{k}  (frame {frames[k]})")
    fig.canvas.draw_idle()


def on_prev(event):
    current = int(slider.val)
    if current > 0:
        slider.set_val(current - 1)


def on_next(event):
    current = int(slider.val)
    if current < len(frames) - 1:
        slider.set_val(current + 1)


slider.on_changed(update)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)

# ── Keyboard navigation ───────────────────────────────────────
def on_key(event):
    if event.key == "left":
        on_prev(event)
    elif event.key == "right":
        on_next(event)

fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
print("Done.")