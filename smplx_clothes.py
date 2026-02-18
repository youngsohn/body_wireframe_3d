"""
how to run:

source ~/codes/venv-smplx311/bin/activate

# 1) FAST: export body frames WITHOUT opening Open3D viewer
# 2) run Blender cloth sim
# 3) visualize body + cloth

python3 smplx_clothes.py \
  --smpl_model_folder ./smpl_models \
  --gender neutral \
  --device cpu \
  --seconds 6 --fps 30 --freq_hz 1.0 \
  --export_dir out/walk_frames \
  --cloth_obj assets/shirt.obj \
  --cloth_out out/cloth_frames \
  --blender "$(which blender)" \
  --run_cloth_sim \
  --no_view_export


If you only want to visualize body+cloth and cloth is already baked:
python3 smplx_clothes.py \
  --smpl_model_folder ./smpl_models \
  --gender neutral \
  --device cpu \
  --seconds 6 --fps 30 --freq_hz 1.0 \
  --cloth_out out/cloth_frames \
  --view_cloth_only
"""

import os
import time
import json
import argparse
import subprocess
import numpy as np
import torch
import open3d as o3d
import smplx


def log(msg: str):
    print(msg, flush=True)

# ------------------------- OBJ export -------------------------
def export_obj(path: str, v: np.ndarray, f: np.ndarray):
    with open(path, "w") as fp:
        for x, y, z in v:
            fp.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in (f.astype(np.int64) + 1):
            fp.write(f"f {a} {b} {c}\n")


# ------------------------- Joint mapping (SMPL body 21 joints) -------------------------
JOINT = {
    "L_HIP": 0, "R_HIP": 1, "SPINE1": 2, "L_KNEE": 3, "R_KNEE": 4, "SPINE2": 5,
    "L_ANKLE": 6, "R_ANKLE": 7, "SPINE3": 8, "L_FOOT": 9, "R_FOOT": 10,
    "NECK": 11, "L_COLLAR": 12, "R_COLLAR": 13, "HEAD": 14,
    "L_SHOULDER": 15, "R_SHOULDER": 16, "L_ELBOW": 17, "R_ELBOW": 18,
    "L_WRIST": 19, "R_WRIST": 20,
}


# --- helpers: add axis-angle around single axis (safe to layer) ---
def add_joint_x(body_pose, j, a): body_pose[0, 3*j + 0] += a
def add_joint_y(body_pose, j, a): body_pose[0, 3*j + 1] += a
def add_joint_z(body_pose, j, a): body_pose[0, 3*j + 2] += a


# ------------------------- Procedural walk cycle -------------------------
def walk_pose(
    t: float,
    freq_hz: float = 0.85,
    hip_swing: float = 0.35,
    knee_lift: float = 0.55,
    ankle_comp: float = 0.35,
    arm_swing: float = 0.45,
    elbow_bend: float = 0.40,
    torso_bob: float = 0.012,
    device: str = "cpu",
):
    w = 2.0 * np.pi * freq_hz
    phase = w * t

    sL = np.sin(phase)
    sR = np.sin(phase + np.pi)

    swingL = np.clip((sL + 1.0) * 0.5, 0.0, 1.0)
    swingR = np.clip((sR + 1.0) * 0.5, 0.0, 1.0)

    body_pose = torch.zeros((1, 63), dtype=torch.float32, device=device)

    # legs
    add_joint_x(body_pose, JOINT["L_HIP"], hip_swing * sL)
    add_joint_x(body_pose, JOINT["R_HIP"], hip_swing * sR)

    kneeL = knee_lift * (swingL ** 1.8)
    kneeR = knee_lift * (swingR ** 1.8)
    add_joint_x(body_pose, JOINT["L_KNEE"], kneeL)
    add_joint_x(body_pose, JOINT["R_KNEE"], kneeR)

    add_joint_x(body_pose, JOINT["L_ANKLE"], -ankle_comp * kneeL)
    add_joint_x(body_pose, JOINT["R_ANKLE"], -ankle_comp * kneeR)

    # toe-off
    cL = np.cos(phase)
    cR = np.cos(phase + np.pi)
    toeL = 0.18 * np.clip((cL + 1.0) * 0.5, 0.0, 1.0) ** 2.2
    toeR = 0.18 * np.clip((cR + 1.0) * 0.5, 0.0, 1.0) ** 2.2
    add_joint_x(body_pose, JOINT["L_FOOT"], toeL)
    add_joint_x(body_pose, JOINT["R_FOOT"], toeR)

    # arms
    deg = np.pi / 180.0
    base_adduct = 65.0 * deg

    add_joint_z(body_pose, JOINT["L_SHOULDER"], -base_adduct)
    add_joint_z(body_pose, JOINT["R_SHOULDER"], +base_adduct)

    swing_amp = 10.0 * deg
    add_joint_z(body_pose, JOINT["L_SHOULDER"], +(swing_amp * np.sin(phase + np.pi)))
    add_joint_z(body_pose, JOINT["R_SHOULDER"], -(swing_amp * np.sin(phase)))

    add_joint_x(body_pose, JOINT["L_SHOULDER"], 0.06)
    add_joint_x(body_pose, JOINT["R_SHOULDER"], 0.06)

    add_joint_x(body_pose, JOINT["L_ELBOW"], elbow_bend + 0.06 * (swingR ** 1.5))
    add_joint_x(body_pose, JOINT["R_ELBOW"], elbow_bend + 0.06 * (swingL ** 1.5))

    add_joint_z(body_pose, JOINT["L_WRIST"], -0.04 * np.sin(phase + np.pi))
    add_joint_z(body_pose, JOINT["R_WRIST"], +0.04 * np.sin(phase))

    # torso/head tiny motion
    add_joint_x(body_pose, JOINT["SPINE1"], 0.05 * np.sin(phase + np.pi/2))
    add_joint_x(body_pose, JOINT["SPINE2"], 0.03 * np.sin(phase + np.pi/2))
    add_joint_y(body_pose, JOINT["SPINE1"], 0.03 * np.sin(phase))

    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
    global_orient[0, 0] = 0.08

    transl = torch.zeros((1, 3), dtype=torch.float32, device=device)
    transl[0, 1] = torso_bob * (0.5 + 0.5 * np.sin(2.0 * phase))

    return global_orient, body_pose, transl


# ------------------------- Blender cloth sim script generator -------------------------
def write_blender_script(path: str):
    """
    Writes a Blender Python script.
    IMPORTANT FIX:
      Blender's sys.argv includes its own flags. We must parse only args after '--'.
    """
    script = r'''
import bpy
import os
import sys

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # purge meshes to avoid memory build-up
    for block in list(bpy.data.meshes):
        try:
            bpy.data.meshes.remove(block)
        except:
            pass

def import_obj(filepath):
    # Resolve relative paths relative to this .py script location
    if not os.path.isabs(filepath):
        base = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base, filepath)
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise RuntimeError(f"OBJ Import: file does not exist: {filepath}")

    bpy.ops.wm.obj_import(filepath=filepath)
    return bpy.context.selected_objects[0]


def export_obj(obj, filepath):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True)

def set_scene_fps(fps):
    bpy.context.scene.render.fps = int(fps)

def set_frame_range(start, end):
    bpy.context.scene.frame_start = int(start)
    bpy.context.scene.frame_end = int(end)

def add_collision(obj, thickness=0.003):
    if obj is None or obj.type != 'MESH':
        return

    # ensure editable
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_render = False

    # ensure modifier exists
    if not any(m.type == 'COLLISION' for m in obj.modifiers):
        obj.modifiers.new(name="Collision", type='COLLISION')

    col = obj.collision
    if col is None:
        return

    col.thickness_outer = float(thickness)
    col.thickness_inner = 0.0
    col.damping = 0.1
    col.friction_factor = 0.5
        

def add_cloth(obj, quality=10):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type='CLOTH')
    m = obj.modifiers[-1]
    st = m.settings

    st.quality = int(quality)

    # collisions
    st.use_collision = True
    st.collision_quality = 4
    st.distance_min = 0.003
    st.use_self_collision = True
    st.self_distance_min = 0.002
    return m

def make_body_sequence(body_dir, n_frames):
    bodies = []
    for f in range(n_frames):
        path = os.path.join(body_dir, f"frame_{f:05d}.obj")
        obj = import_obj(path)
        obj.name = f"Body_{f:05d}"
        bodies.append(obj)

    # add collision first (before any hide)
    for obj in bodies:
        add_collision(obj, thickness=0.003)

    # then animate visibility
    for f in range(n_frames):
        bpy.context.scene.frame_set(f + 1)
        for k, obj in enumerate(bodies):
            visible = (k == f)
            obj.hide_viewport = not visible
            obj.hide_render = not visible
            obj.keyframe_insert(data_path="hide_viewport", frame=f + 1)
            obj.keyframe_insert(data_path="hide_render", frame=f + 1)

    return bodies



def bake_and_export(cloth_obj, out_dir, n_frames):
    os.makedirs(out_dir, exist_ok=True)

    cloth_mod = None
    for mod in cloth_obj.modifiers:
        if mod.type == 'CLOTH':
            cloth_mod = mod
            break
    if cloth_mod is None:
        raise RuntimeError("No cloth modifier found")

    cache = cloth_mod.point_cache
    cache.frame_start = 1
    cache.frame_end = n_frames

    bpy.ops.ptcache.free_bake_all()
    bpy.ops.ptcache.bake_all(bake=True)

    for f in range(n_frames):
        bpy.context.scene.frame_set(f + 1)
        export_obj(cloth_obj, os.path.join(out_dir, f"cloth_{f:05d}.obj"))

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--body_dir", required=True)
    ap.add_argument("--cloth_obj", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n_frames", type=int, required=True)

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = ap.parse_args(argv)

    clear_scene()
    set_scene_fps(args.fps)
    set_frame_range(1, args.n_frames)

    bodies = make_body_sequence(args.body_dir, args.n_frames)

    cloth = import_obj(args.cloth_obj)
    cloth.name = "Cloth"
    add_cloth(cloth, quality=10)

    # small offset to reduce initial intersections
    cloth.location.z += 0.002

    bake_and_export(cloth, args.out_dir, args.n_frames)

if __name__ == "__main__":
    main()
'''
    with open(path, "w") as f:
        f.write(script)


def run_blender_cloth(blender_bin: str, blender_script: str, body_dir: str, cloth_obj: str,
                      out_dir: str, fps: int, n_frames: int):

    blender_bin = os.path.abspath(blender_bin)
    blender_script = os.path.abspath(blender_script)
    body_dir = os.path.abspath(body_dir)
    cloth_obj = os.path.abspath(cloth_obj)
    out_dir = os.path.abspath(out_dir)

    cmd = [
        blender_bin,
        "--background",
        "--python", blender_script,
        "--",
        "--body_dir", body_dir,
        "--cloth_obj", cloth_obj,
        "--out_dir", out_dir,
        "--fps", str(fps),
        "--n_frames", str(n_frames),
    ]
    print("Running Blender cloth sim:\n", " ".join(cmd))
    subprocess.check_call(cmd)



def parse_betas(s: str, n: int, device: str):
    betas = torch.zeros((1, n), dtype=torch.float32, device=device)
    if s.strip():
        parts = [p.strip() for p in s.split(",") if p.strip()]
        vals = np.array([float(x) for x in parts], dtype=np.float32)
        if vals.size < n:
            vals = np.pad(vals, (0, n - vals.size))
        vals = vals[:n]
        betas[:] = torch.tensor(vals, dtype=torch.float32, device=device).reshape(1, -1)
    return betas


# ------------------------- FAST export body frames (no Open3D window) -------------------------
@torch.no_grad()
def export_body_sequence(model, betas, seconds, fps, freq_hz, export_dir, device):
    os.makedirs(export_dir, exist_ok=True)
    faces = model.faces.astype(np.int32)

    dt = 1.0 / fps
    total_frames = int(seconds * fps)

    for i in range(total_frames):
        t = i * dt
        g, p, tr = walk_pose(t, freq_hz=freq_hz, device=device)
        out = model(betas=betas, global_orient=g, body_pose=p, transl=tr, return_verts=True)
        verts = out.vertices[0].detach().cpu().numpy()
        export_obj(os.path.join(export_dir, f"frame_{i:05d}.obj"), verts, faces)

    print(f"[OK] Exported body OBJ frames: {export_dir} (frames={total_frames})")


# ------------------------- Open3D visualize body + optional cloth -------------------------
def visualize_open3d(model, betas, seconds, fps, freq_hz, device, cloth_dir=""):
    faces = model.faces.astype(np.int32)

    # initial body mesh
    g0, p0, tr0 = walk_pose(0.0, freq_hz=freq_hz, device=device)
    out0 = model(betas=betas, global_orient=g0, body_pose=p0, transl=tr0, return_verts=True)
    verts0 = out0.vertices[0].detach().cpu().numpy()

    body_mesh = o3d.geometry.TriangleMesh()
    body_mesh.vertices = o3d.utility.Vector3dVector(verts0.astype(np.float64))
    body_mesh.triangles = o3d.utility.Vector3iVector(faces)
    body_mesh.compute_vertex_normals()

    cloth_mesh = None
    if cloth_dir and os.path.isdir(cloth_dir):
        c0 = os.path.join(cloth_dir, "cloth_00000.obj")
        if os.path.exists(c0):
            cloth_mesh = o3d.io.read_triangle_mesh(c0)
            cloth_mesh.compute_vertex_normals()
        else:
            print(f"[WARN] cloth_dir set but missing: {c0}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SMPL-X (+cloth)", width=1200, height=900)
    vis.add_geometry(body_mesh)
    if cloth_mesh is not None:
        vis.add_geometry(cloth_mesh)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)

    dt = 1.0 / fps
    total_frames = int(seconds * fps)

    for i in range(total_frames):
        t = i * dt
        g, p, tr = walk_pose(t, freq_hz=freq_hz, device=device)
        out = model(betas=betas, global_orient=g, body_pose=p, transl=tr, return_verts=True)
        verts = out.vertices[0].detach().cpu().numpy()

        body_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        body_mesh.compute_vertex_normals()

        if cloth_mesh is not None:
            cpath = os.path.join(cloth_dir, f"cloth_{i:05d}.obj")
            if os.path.exists(cpath):
                new_cloth = o3d.io.read_triangle_mesh(cpath)
                new_cloth.compute_vertex_normals()
                cloth_mesh.vertices = new_cloth.vertices
                cloth_mesh.triangles = new_cloth.triangles
                cloth_mesh.vertex_normals = new_cloth.vertex_normals

        vis.update_geometry(body_mesh)
        if cloth_mesh is not None:
            vis.update_geometry(cloth_mesh)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(dt)

    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--smpl_model_folder", type=str, default="./smpl_models")
    ap.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--betas", type=str, default="", help='Comma list, e.g. "-0.72,0.05,0.26,..."')

    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--freq_hz", type=float, default=1.2)

    ap.add_argument("--export_dir", type=str, default="", help="export body OBJs to this folder (needed for cloth sim)")

    # cloth sim via Blender
    ap.add_argument("--cloth_obj", type=str, default="", help="cloth asset OBJ (rest pose)")
    ap.add_argument("--cloth_out", type=str, default="out/cloth_frames", help="output cloth OBJ sequence folder")
    ap.add_argument("--run_cloth_sim", action="store_true", help="run Blender cloth simulation")
    ap.add_argument("--blender", type=str, default="", help="Blender binary path (macOS app path)")
    ap.add_argument("--blender_script", type=str, default="out/blender_cloth_sim.py", help="auto-generated bpy script path")

    # NEW flags
    ap.add_argument("--no_view_export", action="store_true", help="during export pass, do not open Open3D window (recommended)")
    ap.add_argument("--view_cloth_only", action="store_true", help="skip export/sim and only visualize using existing --cloth_out")

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

    betas = parse_betas(args.betas, model.num_betas, device)
    total_frames = int(args.seconds * args.fps)

    # If user only wants to view cloth results already baked
    if args.view_cloth_only:
        visualize_open3d(model, betas, args.seconds, args.fps, args.freq_hz, device, cloth_dir=args.cloth_out)
        return

    # If cloth sim is requested, we MUST export body frames (headless recommended)
    if args.run_cloth_sim:
        if not args.export_dir:
            raise ValueError("--run_cloth_sim requires --export_dir to export body OBJ sequence first.")
        if not args.cloth_obj:
            raise ValueError("--run_cloth_sim requires --cloth_obj (your clothing asset OBJ).")
        if not args.blender:
            raise ValueError("--run_cloth_sim requires --blender (path to Blender binary).")

        log("[1/3] Exporting body OBJ sequence (headless)...")
        export_body_sequence(
            model=model,
            betas=betas,
            seconds=args.seconds,
            fps=args.fps,
            freq_hz=args.freq_hz,
            export_dir=args.export_dir,
            device=device,
        )

        log("[2/3] Writing Blender script...")
        os.makedirs(os.path.dirname(args.blender_script), exist_ok=True) if os.path.dirname(args.blender_script) else None
        write_blender_script(args.blender_script)

        log("[2/3] Running Blender cloth simulation (this can take a while)...")
        os.makedirs(args.cloth_out, exist_ok=True)
        run_blender_cloth(
            blender_bin=args.blender,
            blender_script=args.blender_script,
            body_dir=args.export_dir,
            cloth_obj=args.cloth_obj,
            out_dir=args.cloth_out,
            fps=args.fps,
            n_frames=total_frames,
        )

        # quick sanity check
        c0 = os.path.join(args.cloth_out, "cloth_00000.obj")
        if not os.path.exists(c0):
            raise RuntimeError(f"Blender finished but no cloth output found: {c0}")

        log("[3/3] Visualizing body + cloth in Open3D...")
        visualize_open3d(model, betas, args.seconds, args.fps, args.freq_hz, device, cloth_dir=args.cloth_out)
        return

    # If no cloth sim requested: just view body
    visualize_open3d(model, betas, args.seconds, args.fps, args.freq_hz, device, cloth_dir="")


if __name__ == "__main__":
    main()