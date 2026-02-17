"""
Run it

Basic (neutral, default betas)

python3 smplx_walk_in_place.py \
  --smpl_model_folder ./smpl_models \
  --gender neutral \
  --device cpu

  
  
With your tuned betas (example from your log) 

python3 smplx_walk_in_place.py \
  --smpl_model_folder ./smpl_models \
  --gender neutral \
  --device cpu \
  --betas "-0.7236,0.0494,0.2609,0.0801,0.0152,-0.0322,0.0494,-0.0465,-0.0298,-0.0286"
  
  
Export a frame sequence as OBJ (optional)

python3 smplx_walk_in_place.py \
  --smpl_model_folder ./smpl_models \
  --gender neutral \
  --device cpu \
  --seconds 6 --fps 30 \
  --export_dir out/walk_frames



"""


import os
import time
import argparse
import numpy as np
import torch
import open3d as o3d
import smplx
import numpy as np
import torch

# ------------------------- OBJ export -------------------------
def export_obj(path: str, v: np.ndarray, f: np.ndarray):
    with open(path, "w") as fp:
        for x, y, z in v:
            fp.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in (f.astype(np.int64) + 1):
            fp.write(f"f {a} {b} {c}\n")


# ------------------------- Joint mapping (SMPL body 21 joints) -------------------------
# smplx body_pose is 21 joints * 3 axis-angle = 63
# This is the common SMPL body joint order used by many SMPL-X implementations.
JOINT = {
    "L_HIP": 0,
    "R_HIP": 1,
    "SPINE1": 2,
    "L_KNEE": 3,
    "R_KNEE": 4,
    "SPINE2": 5,
    "L_ANKLE": 6,
    "R_ANKLE": 7,
    "SPINE3": 8,
    "L_FOOT": 9,
    "R_FOOT": 10,
    "NECK": 11,
    "L_COLLAR": 12,
    "R_COLLAR": 13,
    "HEAD": 14,
    "L_SHOULDER": 15,
    "R_SHOULDER": 16,
    "L_ELBOW": 17,
    "R_ELBOW": 18,
    "L_WRIST": 19,
    "R_WRIST": 20,
}


def set_joint_x(body_pose, j, angle_rad):
    """Set axis-angle rotation around X axis for joint j."""
    body_pose[0, 3 * j + 0] = angle_rad
    body_pose[0, 3 * j + 1] = 0.0
    body_pose[0, 3 * j + 2] = 0.0


def add_joint_x(body_pose, j, angle_rad):
    """Add axis-angle around X axis."""
    body_pose[0, 3 * j + 0] += angle_rad


def set_joint_y(body_pose, j, angle_rad):
    body_pose[0, 3 * j + 0] = 0.0
    body_pose[0, 3 * j + 1] = angle_rad
    body_pose[0, 3 * j + 2] = 0.0


def set_joint_z(body_pose, j, angle_rad):
    body_pose[0, 3 * j + 0] = 0.0
    body_pose[0, 3 * j + 1] = 0.0
    body_pose[0, 3 * j + 2] = angle_rad


# ------------------------- Procedural walk cycle -------------------------

# --- helpers: add axis-angle around single axis (safe to layer) ---
def add_joint_x(body_pose, j, a):
    body_pose[0, 3*j + 0] += a

def add_joint_y(body_pose, j, a):
    body_pose[0, 3*j + 1] += a

def add_joint_z(body_pose, j, a):
    body_pose[0, 3*j + 2] += a


def walk_pose(
    t: float,
    # slower cadence => walking (not running)
    freq_hz: float = 0.85,         # 0.7~1.0 looks like walk; 1.2+ looks like jog/run
    # leg amplitudes (smaller => slower/softer walk)
    hip_swing: float = 0.35,       # rad
    knee_lift: float = 0.55,       # rad
    ankle_comp: float = 0.35,      # rad
    # arm swing (smaller than legs)
    #arm_swing: float = 0.35,       # rad
    arm_swing: float = 0.45,       # rad
    #elbow_bend: float = 0.30,      # rad
    elbow_bend: float = 0.40,      # rad
    # "arms down" offsets (key to avoid T-pose look)
    shoulder_adduct: float = 0.95, # rad (brings arms down at sides)
    shoulder_twist: float = 0.10,  # rad (tiny forward roll)
    # body bob
    torso_bob: float = 0.012,      # meters
    device: str = "cpu",
):
    """
    Walk-in-place pose generator.
    - Adds a constant shoulder adduction so arms are down.
    - Adds arm swing opposite to legs.
    - Slower cadence + smaller amplitudes => walking, not running.
    """
    w = 2.0 * np.pi * freq_hz
    phase = w * t

    # left and right legs out of phase by pi
    sL = np.sin(phase)
    sR = np.sin(phase + np.pi)

    # shape for stance/swing
    # swing gate: 0..1 (more bend when leg is in forward swing)
    swingL = np.clip((sL + 1.0) * 0.5, 0.0, 1.0)
    swingR = np.clip((sR + 1.0) * 0.5, 0.0, 1.0)

    body_pose = torch.zeros((1, 63), dtype=torch.float32, device=device)

    # ---------------- legs ----------------
    # hips: forward/back
    add_joint_x(body_pose, JOINT["L_HIP"], hip_swing * sL)
    add_joint_x(body_pose, JOINT["R_HIP"], hip_swing * sR)

    # knees: bend mainly during swing
    kneeL = knee_lift * (swingL ** 1.8)
    kneeR = knee_lift * (swingR ** 1.8)
    add_joint_x(body_pose, JOINT["L_KNEE"], kneeL)
    add_joint_x(body_pose, JOINT["R_KNEE"], kneeR)

    # ankles: compensate
    add_joint_x(body_pose, JOINT["L_ANKLE"], -ankle_comp * kneeL)
    add_joint_x(body_pose, JOINT["R_ANKLE"], -ankle_comp * kneeR)

    # slight toe-off near end of stance (cos-shaped)
    cL = np.cos(phase)
    cR = np.cos(phase + np.pi)
    toeL = 0.18 * np.clip((cL + 1.0) * 0.5, 0.0, 1.0) ** 2.2
    toeR = 0.18 * np.clip((cR + 1.0) * 0.5, 0.0, 1.0) ** 2.2
    add_joint_x(body_pose, JOINT["L_FOOT"], toeL)
    add_joint_x(body_pose, JOINT["R_FOOT"], toeR)

    # ---------------- arms ----------------
    # Goal:
    # - arms rest down (not T-pose)
    # - swing around the vertical line by about +/-10 degrees (Z axis)
    # - avoid strong forward/back swing (X axis)

    deg = np.pi / 180.0

    # constant arms-down offset (adduction): adjust if arms still too lifted
    # (bigger -> arms closer to body)
    #base_adduct = 55.0 * deg   # ~55 deg
    base_adduct = 65.0 * deg   # ~55 deg
    
    add_joint_z(body_pose, JOINT["L_SHOULDER"], -base_adduct)
    add_joint_z(body_pose, JOINT["R_SHOULDER"], +base_adduct)

    # small swing around vertical line: +/-10 deg
    # use opposite phase to legs: left arm ~ right leg
    swing_amp = 10.0 * deg     # +/-10 deg
    add_joint_z(body_pose, JOINT["L_SHOULDER"], +(swing_amp * np.sin(phase + np.pi)))
    add_joint_z(body_pose, JOINT["R_SHOULDER"], -(swing_amp * np.sin(phase)))

    # tiny forward roll so it doesn't look like a rigid hinge (optional)
    add_joint_x(body_pose, JOINT["L_SHOULDER"], 0.06)
    add_joint_x(body_pose, JOINT["R_SHOULDER"], 0.06)

    # elbows: keep slight bend, tiny modulation
    add_joint_x(body_pose, JOINT["L_ELBOW"], elbow_bend + 0.06 * (swingR ** 1.5))
    add_joint_x(body_pose, JOINT["R_ELBOW"], elbow_bend + 0.06 * (swingL ** 1.5))

    # wrists: very small counter swing (optional)
    add_joint_z(body_pose, JOINT["L_WRIST"], -0.04 * np.sin(phase + np.pi))
    add_joint_z(body_pose, JOINT["R_WRIST"], +0.04 * np.sin(phase)
)

    # ---------------- torso / head ----------------
    # subtle torso counter motion (small!)
    add_joint_x(body_pose, JOINT["SPINE1"], 0.05 * np.sin(phase + np.pi/2))
    add_joint_x(body_pose, JOINT["SPINE2"], 0.03 * np.sin(phase + np.pi/2))
    add_joint_y(body_pose, JOINT["SPINE1"], 0.03 * np.sin(phase))  # tiny yaw

    # global: small forward lean
    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
    global_orient[0, 0] = 0.08

    # translation: bob up/down only (walk in place)
    transl = torch.zeros((1, 3), dtype=torch.float32, device=device)
    transl[0, 1] = torso_bob * (0.5 + 0.5 * np.sin(2.0 * phase))

    return global_orient, body_pose, transl


# ------------------------- Open3D animation -------------------------
def animate_walk(
    model,
    betas,
    seconds=6.0,
    fps=30,
    freq_hz=1.2,
    export_dir="",
    device="cpu",
):
    os.makedirs(export_dir, exist_ok=True) if export_dir else None

    faces = model.faces.astype(np.int32)

    # Initial frame
    g0, p0, tr0 = walk_pose(0.0, freq_hz=freq_hz, device=device)
    out0 = model(
        betas=betas,
        global_orient=g0,
        body_pose=p0,
        transl=tr0,
        return_verts=True,
    )
    verts0 = out0.vertices[0].detach().cpu().numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts0.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SMPL-X walk-in-place", width=1200, height=900)
    vis.add_geometry(mesh)

    # Make the view nicer
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)

    dt = 1.0 / fps
    total_frames = int(seconds * fps)

    for i in range(total_frames):
        t = i * dt
        g, p, tr = walk_pose(t, freq_hz=freq_hz, device=device)

        out = model(
            betas=betas,
            global_orient=g,
            body_pose=p,
            transl=tr,
            return_verts=True,
        )
        verts = out.vertices[0].detach().cpu().numpy()

        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.compute_vertex_normals()

        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        if export_dir:
            export_obj(os.path.join(export_dir, f"frame_{i:05d}.obj"), verts, faces)

        time.sleep(dt)

    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl_model_folder", type=str, default="./smpl_models")
    ap.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--num_betas", type=int, default=10)

    # You can paste your tuned betas here (10 numbers). If empty -> zeros.
    ap.add_argument("--betas", type=str, default="", help='Comma list, e.g. "-0.72,0.05,0.26,..."')

    # animation controls
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--freq_hz", type=float, default=1.2, help="steps cadence ~1.0-1.8 looks ok")
    ap.add_argument("--export_dir", type=str, default="", help="optional: export OBJ frames into this folder")

    args = ap.parse_args()
    device = args.device

    model = smplx.create(
        model_path=args.smpl_model_folder,
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        num_betas=args.num_betas,
        num_expression_coeffs=10,
        create_transl=True,
    ).to(device)

    # betas
    betas = torch.zeros((1, model.num_betas), dtype=torch.float32, device=device)
    if args.betas.strip():
        parts = [p.strip() for p in args.betas.split(",") if p.strip()]
        vals = np.array([float(x) for x in parts], dtype=np.float32)
        if vals.size < model.num_betas:
            vals = np.pad(vals, (0, model.num_betas - vals.size))
        vals = vals[: model.num_betas]
        betas[:] = torch.tensor(vals, dtype=torch.float32, device=device).reshape(1, -1)

    animate_walk(
        model=model,
        betas=betas,
        seconds=args.seconds,
        fps=args.fps,
        freq_hz=args.freq_hz,
        export_dir=args.export_dir,
        device=device,
    )


if __name__ == "__main__":
    main()