import os
import time
import json
import argparse
import subprocess
import numpy as np
import torch
import open3d as o3d
import smplx

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
    shoulder_adduct: float = 0.95,
    shoulder_twist: float = 0.10,
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

    # torso/head
    add_joint_x(body_pose, JOINT["SPINE1"], 0.05 * np.sin(phase + np.pi/2))
    add_joint_x(body_pose, JOINT["SPINE2"], 0.03 * np.sin(phase + np.pi/2))
    add_joint_y(body_pose, JOINT["SPINE1"], 0.03 * np.sin(phase))

    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
    global_orient[0, 0] = 0.08

    transl = torch.zeros((1, 3), dtype=torch.float32, device=device)
    transl[0, 1] = torso_bob * (0.5 + 0.5 * np.sin(2.0 * phase))

    return global_orient, body_pose, transl

# ------------------------- Attachment export (head transform) -------------------------
def make_head_transform(joints_np: np.ndarray):
    """
    Build a simple rigid transform for head-attached objects.
    We approximate head orientation using (NECK->HEAD) as 'up' and (L_COLLAR->R_COLLAR) as 'right'.
    Returns 4x4 transform matrix in world coordinates.
    """
    neck = joints_np[JOINT["NECK"]]
    head = joints_np[JOINT["HEAD"]]
    lcol = joints_np[JOINT["L_COLLAR"]]
    rcol = joints_np[JOINT["R_COLLAR"]]

    up = head - neck
    up_norm = np.linalg.norm(up) + 1e-8
    up = up / up_norm

    right = rcol - lcol
    right_norm = np.linalg.norm(right) + 1e-8
    right = right / right_norm

    forward = np.cross(right, up)
    f_norm = np.linalg.norm(forward) + 1e-8
    forward = forward / f_norm

    # re-orthogonalize right
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)

    R = np.stack([right, up, forward], axis=1)  # columns
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = head.astype(np.float32)          # origin at HEAD joint
    return T

# ------------------------- Blender cloth sim script generator -------------------------
def write_blender_script(path: str):
    """
    Writes a Blender Python script which:
    - Imports body OBJ sequence (frame_00000.obj...)
    - Imports a single cloth OBJ (rest mesh)
    - Adds cloth sim + collision
    - Bakes and exports cloth OBJ sequence

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
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)

def import_obj(filepath):
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
    bpy.context.view_layer.objects.active = obj
    if obj.collision is None:
        bpy.ops.object.modifier_add(type='COLLISION')
    obj.collision.thickness_outer = thickness

def add_cloth(obj, quality=8, time_scale=1.0, pressure=0.0):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type='CLOTH')
    m = obj.modifiers[-1]
    st = m.settings
    st.quality = int(quality)
    st.time_scale = float(time_scale)
    st.use_pressure = (pressure != 0.0)
    st.uniform_pressure_force = float(pressure)

    # collisions
    st.use_collision = True
    st.collision_quality = 4
    st.distance_min = 0.003
    st.use_self_collision = True
    st.self_distance_min = 0.002

    return m

def make_body_sequence(body_dir, n_frames):
    # import first body
    body0 = import_obj(os.path.join(body_dir, "frame_00000.obj"))
    body0.name = "Body"

    # animate by swapping mesh each frame (fast and simple)
    for f in range(n_frames):
        bpy.context.scene.frame_set(f + 1)
        path = os.path.join(body_dir, f"frame_{f:05d}.obj")

        # import temp
        tmp = import_obj(path)
        tmp.data.name = f"BodyMesh_{f:05d}"

        # keyframe body mesh datablock switch
        body0.data = tmp.data
        body0.keyframe_insert(data_path="data", frame=f + 1)

        # delete tmp object but keep mesh datablock (linked)
        bpy.data.objects.remove(tmp, do_unlink=True)

    return body0

def bake_and_export(cloth_obj, out_dir, n_frames):
    os.makedirs(out_dir, exist_ok=True)

    # ensure cloth modifier exists
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

    # export per frame
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

    # -------- FIX: parse only arguments after '--' --------
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = ap.parse_args(argv)
    # -----------------------------------------------------

    clear_scene()
    set_scene_fps(args.fps)
    set_frame_range(1, args.n_frames)

    body = make_body_sequence(args.body_dir, args.n_frames)
    add_collision(body, thickness=0.003)

    cloth = import_obj(args.cloth_obj)
    cloth.name = "Cloth"
    add_cloth(cloth, quality=10, time_scale=1.0, pressure=0.0)

    # Put cloth slightly away from body to reduce initial intersections
    cloth.location.z += 0.002

    bake_and_export(cloth, args.out_dir, args.n_frames)

if __name__ == "__main__":
    main()
'''
    with open(path, "w") as f:
        f.write(script)

# ------------------------- Open3D animation (body only OR body+cloth) -------------------------
def animate_walk_open3d(
    model,
    betas,
    seconds=6.0,
    fps=30,
    freq_hz=1.2,
    export_dir="",
    device="cpu",
    cloth_dir="",
    hair_obj="",
    attach_json="",
):
    os.makedirs(export_dir, exist_ok=True) if export_dir else None

    faces = model.faces.astype(np.int32)

    # initial body
    g0, p0, tr0 = walk_pose(0.0, freq_hz=freq_hz, device=device)
    out0 = model(betas=betas, global_orient=g0, body_pose=p0, transl=tr0, return_verts=True)
    verts0 = out0.vertices[0].detach().cpu().numpy()

    body_mesh = o3d.geometry.TriangleMesh()
    body_mesh.vertices = o3d.utility.Vector3dVector(verts0.astype(np.float64))
    body_mesh.triangles = o3d.utility.Vector3iVector(faces)
    body_mesh.compute_vertex_normals()

    # optional cloth display (load cloth objs per frame)
    cloth_mesh = None
    if cloth_dir and os.path.isdir(cloth_dir):
        # load first cloth
        cpath0 = os.path.join(cloth_dir, "cloth_00000.obj")
        if os.path.exists(cpath0):
            cloth_mesh = o3d.io.read_triangle_mesh(cpath0)
            cloth_mesh.compute_vertex_normals()

    # optional hair rigid attachment
    hair_mesh = None
    if hair_obj and os.path.exists(hair_obj):
        hair_mesh = o3d.io.read_triangle_mesh(hair_obj)
        hair_mesh.compute_vertex_normals()

    # Open3D window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SMPL-X walk (+cloth)", width=1200, height=900)
    vis.add_geometry(body_mesh)
    if cloth_mesh is not None:
        vis.add_geometry(cloth_mesh)
    if hair_mesh is not None:
        vis.add_geometry(hair_mesh)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)

    dt = 1.0 / fps
    total_frames = int(seconds * fps)

    # attachment records (head transform per frame)
    attach = {"head_T_4x4": []} if attach_json else None

    for i in range(total_frames):
        t = i * dt
        g, p, tr = walk_pose(t, freq_hz=freq_hz, device=device)

        out = model(betas=betas, global_orient=g, body_pose=p, transl=tr, return_verts=True)
        verts = out.vertices[0].detach().cpu().numpy()

        body_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        body_mesh.compute_vertex_normals()

        # joints for head attachment
        joints = out.joints[0].detach().cpu().numpy()
        head_T = make_head_transform(joints)

        if attach is not None:
            attach["head_T_4x4"].append(head_T.tolist())

        # update cloth mesh if present
        if cloth_mesh is not None:
            cpath = os.path.join(cloth_dir, f"cloth_{i:05d}.obj")
            if os.path.exists(cpath):
                new_cloth = o3d.io.read_triangle_mesh(cpath)
                new_cloth.compute_vertex_normals()
                # replace geometry data (Open3D doesn't like swapping object references)
                cloth_mesh.vertices = new_cloth.vertices
                cloth_mesh.triangles = new_cloth.triangles
                cloth_mesh.vertex_normals = new_cloth.vertex_normals

        # update hair rigid transform
        if hair_mesh is not None:
            # reset each frame: easiest is to store original hair and retransform
            # Here we assume hair_obj is modeled near origin (head-local). If it's already world-positioned, skip.
            # We'll apply head_T to hair vertices.
            # NOTE: For performance, pre-load vertices once and transform in-place. Kept simple here.
            base_hair = o3d.io.read_triangle_mesh(hair_obj)
            V = np.asarray(base_hair.vertices)
            Vh = (head_T[:3, :3] @ V.T).T + head_T[:3, 3]
            hair_mesh.vertices = o3d.utility.Vector3dVector(Vh.astype(np.float64))
            hair_mesh.triangles = base_hair.triangles
            hair_mesh.compute_vertex_normals()

        vis.update_geometry(body_mesh)
        if cloth_mesh is not None:
            vis.update_geometry(cloth_mesh)
        if hair_mesh is not None:
            vis.update_geometry(hair_mesh)

        vis.poll_events()
        vis.update_renderer()

        if export_dir:
            export_obj(os.path.join(export_dir, f"frame_{i:05d}.obj"), verts, faces)

        time.sleep(dt)

    vis.destroy_window()

    if attach is not None:
        with open(attach_json, "w") as f:
            json.dump(attach, f, indent=2)

def run_blender_cloth(blender_bin: str, blender_script: str, body_dir: str, cloth_obj: str, out_dir: str, fps: int, n_frames: int):
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
    ap.add_argument("--export_dir", type=str, default="", help="export body OBJs to this folder")

    # cloth sim via Blender
    ap.add_argument("--cloth_obj", type=str, default="", help="cloth asset OBJ (rest pose)")
    ap.add_argument("--cloth_out", type=str, default="out/cloth_frames", help="output cloth OBJ sequence folder")
    ap.add_argument("--run_cloth_sim", action="store_true", help="run Blender cloth simulation")
    ap.add_argument("--blender", type=str, default="", help="Blender binary path (macOS app path)")
    ap.add_argument("--blender_script", type=str, default="out/blender_cloth_sim.py", help="auto-generated bpy script path")

    # rigid attachments
    ap.add_argument("--hair_obj", type=str, default="", help="hair OBJ in head-local coordinates (optional)")
    ap.add_argument("--export_attach", type=str, default="", help="write head transforms per frame json (optional)")

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

    # 1) export body sequence if requested (needed for Blender cloth sim)
    if args.run_cloth_sim and not args.export_dir:
        raise ValueError("--run_cloth_sim requires --export_dir to export body OBJ sequence first.")
    if args.run_cloth_sim and not args.cloth_obj:
        raise ValueError("--run_cloth_sim requires --cloth_obj (your clothing asset OBJ).")
    if args.run_cloth_sim and not args.blender:
        raise ValueError("--run_cloth_sim requires --blender (path to Blender binary).")

    # Export body OBJs (and optionally attach json) by running the Open3D loop once,
    # but you can also do "headless export" without visualizer if you want.
    animate_walk_open3d(
        model=model,
        betas=betas,
        seconds=args.seconds,
        fps=args.fps,
        freq_hz=args.freq_hz,
        export_dir=args.export_dir,
        device=device,
        cloth_dir="",                   # cloth not yet
        hair_obj=args.hair_obj,
        attach_json=args.export_attach,
    )

    # 2) run Blender cloth sim (creates cloth_00000.obj ... cloth_N.obj)
    if args.run_cloth_sim:
        os.makedirs(os.path.dirname(args.blender_script), exist_ok=True) if os.path.dirname(args.blender_script) else None
        write_blender_script(args.blender_script)

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

        # 3) visualize again with cloth
        animate_walk_open3d(
            model=model,
            betas=betas,
            seconds=args.seconds,
            fps=args.fps,
            freq_hz=args.freq_hz,
            export_dir="",                # no need to re-export body
            device=device,
            cloth_dir=args.cloth_out,
            hair_obj=args.hair_obj,
            attach_json="",               # already exported if you wanted
        )

if __name__ == "__main__":
    main()