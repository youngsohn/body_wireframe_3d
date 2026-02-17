"""
howto run:
    
    
source venv-body/bin/activate

python smplx_visualize_betas.py \
  --smpl_model_folder ./smpl_models \
  --gender neutral \
  --num_betas 10 \
  --betas "1.5,-0.5,0.7,0,0,0,0,0,0,0" \
  --wireframe \
  --show_joints \
  --export_obj out/smplx_neutral_custom.obj

"""

import os
import argparse
import numpy as np
import torch
import open3d as o3d
import smplx


def export_obj(path: str, v: np.ndarray, f: np.ndarray):
    """Export mesh to OBJ."""
    with open(path, "w") as fp:
        for x, y, z in v:
            fp.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        # OBJ faces are 1-indexed
        for a, b, c in (f.astype(np.int64) + 1):
            fp.write(f"f {a} {b} {c}\n")


def make_wireframe_lineset(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.LineSet:
    """
    Build a wireframe LineSet from triangle faces (unique edges).
    """
    # Collect edges
    edges = set()
    f = faces.astype(np.int64)
    for tri in f:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        e1 = (min(i, j), max(i, j))
        e2 = (min(j, k), max(j, k))
        e3 = (min(k, i), max(k, i))
        edges.add(e1)
        edges.add(e2)
        edges.add(e3)

    edges = np.array(list(edges), dtype=np.int64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(edges)
    return ls


def build_smplx_mesh(
    smpl_model_folder: str,
    gender: str,
    num_betas: int,
    betas_values: np.ndarray,
    device: str,
):
    """
    Returns:
      verts (N,3) np.float32
      faces (F,3) np.int32
      joints (J,3) np.float32
    """
    model = smplx.create(
        model_path=smpl_model_folder,
        model_type="smplx",
        gender=gender,
        use_pca=False,            # easier: axis-angle hand poses
        num_betas=num_betas,
        num_expression_coeffs=10,
        create_body_pose=True,
        create_global_orient=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
    ).to(device)

    B = 1
    betas = torch.tensor(betas_values, dtype=torch.float32, device=device).reshape(B, -1)
    if betas.shape[1] != model.num_betas:
        raise ValueError(f"betas length mismatch: got {betas.shape[1]}, model expects {model.num_betas}")

    # Zero pose (model default)
    global_orient = torch.zeros([B, 3], device=device)  # axis-angle
    body_pose = torch.zeros([B, model.NUM_BODY_JOINTS * 3], device=device)

    left_hand_pose = torch.zeros([B, 45], device=device)
    right_hand_pose = torch.zeros([B, 45], device=device)

    expression = torch.zeros([B, model.num_expression_coeffs], device=device)
    jaw_pose = torch.zeros([B, 3], device=device)
    leye_pose = torch.zeros([B, 3], device=device)
    reye_pose = torch.zeros([B, 3], device=device)

    transl = torch.zeros([B, 3], device=device)

    out = model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        transl=transl,
        return_verts=True,
        return_joints=True,
    )

    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)
    faces = model.faces.astype(np.int32)

    return verts, faces, joints


def visualize_open3d(
    verts: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    show_wireframe: bool,
    show_joints: bool,
):
    # Surface mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    geoms = [mesh]

    # Optional wireframe
    if show_wireframe:
        ls = make_wireframe_lineset(verts, faces)
        geoms.append(ls)

    # Optional joints as spheres + lines (simple skeleton visual)
    if show_joints:
        # spheres for joints
        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(joints.astype(np.float64))
        geoms.append(pts)

        # also add tiny spheres (more visible than points sometimes)
        # (kept small to avoid clutter)
        for p in joints:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sph.translate(p.astype(np.float64))
            sph.compute_vertex_normals()
            geoms.append(sph)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="SMPL-X betas visualization (Open3D)",
        width=1200,
        height=900,
        mesh_show_back_face=True,
    )


def parse_betas_list(s: str) -> np.ndarray:
    """
    Parse betas from:
      --betas "1.5,-0.5,0.7,0,0,0,0,0,0,0"
    """
    parts = [p.strip() for p in s.replace(" ", "").split(",") if p.strip() != ""]
    if len(parts) == 0:
        return np.zeros((10,), dtype=np.float32)
    return np.array([float(x) for x in parts], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl_model_folder", type=str, default="./smpl_models")
    ap.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument(
        "--betas",
        type=str,
        default="1.5,-0.5,0.7,0,0,0,0,0,0,0",
        help='Comma list, e.g. "1.5,-0.5,0.7,0,0,0,0,0,0,0"',
    )
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--wireframe", action="store_true")
    ap.add_argument("--show_joints", action="store_true")
    ap.add_argument("--export_obj", type=str, default="", help="e.g. out/custom.obj")
    args = ap.parse_args()

    betas_values = parse_betas_list(args.betas)

    # If user provided fewer betas than num_betas, pad with zeros.
    if betas_values.shape[0] < args.num_betas:
        betas_values = np.pad(betas_values, (0, args.num_betas - betas_values.shape[0]))
    # If user provided more, truncate.
    if betas_values.shape[0] > args.num_betas:
        betas_values = betas_values[: args.num_betas]

    # Build mesh
    verts, faces, joints = build_smplx_mesh(
        smpl_model_folder=args.smpl_model_folder,
        gender=args.gender,
        num_betas=args.num_betas,
        betas_values=betas_values,
        device=args.device,
    )

    # Export if requested
    if args.export_obj:
        os.makedirs(os.path.dirname(args.export_obj) or ".", exist_ok=True)
        export_obj(args.export_obj, verts, faces)
        print("[OK] Exported OBJ:", args.export_obj)

    # Visualize
    visualize_open3d(
        verts=verts,
        faces=faces,
        joints=joints,
        show_wireframe=args.wireframe,
        show_joints=args.show_joints,
    )


if __name__ == "__main__":
    main()