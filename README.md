# Pushing Task — Isaac Lab 3.0 / Isaac Sim 6.0

Doosan M0609 tactile pushing task with cuRobo motion planning and the CoRo_tactile
deformable sponge sensor. This is the Isaac Lab 3.0 port of the original
Isaac Sim 5.1 / Isaac Lab 2.3.2 standalone script.

## Requirements

### System (tested configuration)

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX (tested on RTX A2000 8 GB Laptop) |
| NVIDIA driver | 580.x (CUDA 13 driver — backward compatible) |
| CUDA toolkit | 12.8 |

### Software stack

| Component | Version | Notes |
|---|---|---|
| Isaac Sim | 6.0 | |
| Isaac Lab | 3.0 (develop, beta) | Provides the Python env (Python 3.12) |
| PyTorch | 2.11.0+cu128 | Installed by Isaac Lab env |
| cuRobo | v0.7.7 (commit `d64c4b0`) | **Must be built against this env** — see below |
| onnxruntime-gpu | **1.22.0 (pinned)** | Newer builds link CUDA 13 (`libcudart.so.13`) and fail on a CUDA 12.8 stack |
| numpy | 2.x | Script already handles the NumPy 2 `.item()` requirement |
| pandas, scipy | any recent | |

### Python packages (into the Isaac Lab env)

```bash
cd ~/isaac-lab-3
./isaaclab.sh -p -m pip install onnxruntime-gpu==1.22.0 pandas scipy
```

### cuRobo build

cuRobo ships compiled extensions, so a clone built for another Python/CUDA
version will not load. Build it inside the Isaac Lab 3 env:

```bash
git clone https://github.com/NVlabs/curobo.git curobo-lab3
cd curobo-lab3
git checkout d64c4b0   # v0.7.7
export CUDA_HOME=/usr/local/cuda-12.8
~/isaac-lab-3/isaaclab.sh -p -m pip install -e .[isaacsim] --no-build-isolation
```

Keep this as a separate clone if you also maintain an Isaac Sim 5.1 setup —
an editable install points at the source directory, so rebuilding one clone
in place would break the other environment.

## Repository layout

The script resolves all assets relative to its own location. The repo must
contain:

```
Pushing_task_Isaaclab3/
├── Test_pushing_isaaclab3.py
├── scenes/
│   └── doosan_station_full.usd      # + any USDs/meshes/textures it references
├── coro_doosan_station/
│   └── m0609/
│       ├── m0609_adap_curobo.yml    # cuRobo robot config
│       └── ...                      # URDF / collision meshes referenced by the yml
└── CNN_tactile/
    ├── Nodes_id_filtered.csv        # ordered node IDs (18 x 12 taxel grid)
    ├── best.onnx                    # tactile CNN (cuDNN-patched export)
    └── CNN_max.npy                  # normalization constant (1-element array)
```

> Check `m0609_adap_curobo.yml` for absolute paths after moving the repo —
> cuRobo configs often reference the URDF and collision meshes by path.

## Running

Always launch through the Isaac Lab wrapper, with an **absolute** path to the
script (`isaaclab.sh -p` does not change the working directory, so relative
paths resolve against the Isaac Lab root):

```bash
cd ~/isaac-lab-3
./isaaclab.sh -p /media/berith/DataDrive/Documents/Pushing_task_Isaaclab3/Test_pushing_isaaclab3.py
```

Headless:

```bash
./isaaclab.sh -p /media/.../Test_pushing_isaaclab3.py --headless
```

The script only uses the standard `AppLauncher` flags (`--headless`,
`--device`, `--width`, `--height`, ...); it has no custom CLI arguments. The
physics backend is selected in code (PhysX via `SimulationCfg.physics=PhysxCfg(...)`),
not by a CLI flag — deformables require PhysX (Newton has no FEM support in
Isaac Sim 6.0).

First launch from a new location can take several minutes (extension loading,
shader compilation) before the window appears. The startup warnings about
protobuf, MaterialX, the omniverse proxy, and "USD stage is not available"
are all benign.

## Outputs

Two timestamped CSVs are written next to the script per run:

- `*_deformation_<timestamp>.csv` — per-node deformation and velocities
- `*_tactiledata_<timestamp>.csv` — inferred taxel values from the CNN

## Sanity checks for a good run

- Baseline mean near zero (≈ ±0.01), not a large constant.
- Tactile values start near zero and grow monotonically with contact.
- Kabsch residual (logged every 60 frames) settles to ~0.05 mm; a warning is
  printed above 5 mm. The pad takes ~130 frames to settle before baseline
  collection starts (gated on residual stability, capped at 900 frames).
- The script fails fast (missing deformable view, node-count mismatch, etc.)
  rather than writing zero-filled CSVs; `ALLOW_DEGRADED=1` overrides this.

## Known issues

- `/World/m0609/m0609/link_6/adapter` prim is not found in Sim 6.0, so its
  friction material is never applied (sensor mounting surface).
- TGS solver warning about noisy velocities with
  `enable_external_forces_every_iteration=False` — relevant to the nodal
  velocities written to the deformation CSV.
