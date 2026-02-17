"""
how to run:
    python smplx_tune_regions.py \
  --model_path ./smpl_models \
  --gender neutral \
  --device cpu \
  --target_shoulder 0.44 \
  --target_hip 0.32 \
  --target_chest_proxy 0.205 \
  --steps 350 \
  --export_obj out/tuned_all.obj

"""

import os
import argparse
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


# ------------------------- Open3D view -------------------------
def visualize_mesh(verts: np.ndarray, faces: np.ndarray, title="SMPL-X"):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [mesh],
        window_name=title,
        width=1200,
        height=900,
        mesh_show_back_face=True,
    )


# ------------------------- Create model -------------------------
def create_smplx(model_path: str, gender: str, num_betas: int, device: str):
    model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        create_transl=True,
    ).to(device)
    return model


# ------------------------- Measurements (proxies) -------------------------
def get_named_joint_indices():
    """
    SMPL-X joint ordering can vary by implementation.
    In SMPL-X from smplx library, the first joints usually include:
      pelvis, left_hip, right_hip, spine1, left_knee, right_knee, ...
    Shoulder joints exist later.
    We avoid guessing by providing a fallback manual index section below.

    If you want *guaranteed* indices, print joint count and visually inspect.
    """
    return None


def shoulder_width_from_joints(joints: torch.Tensor, idx_L: int, idx_R: int):
    return torch.norm(joints[idx_L] - joints[idx_R])


def hip_width_from_joints(joints: torch.Tensor, idx_L: int, idx_R: int):
    return torch.norm(joints[idx_L] - joints[idx_R])


def chest_girth_proxy(verts: torch.Tensor, y_low: float, y_high: float):
    """
    Proxy: take vertices in a chest 'band' of Y, compute mean radius in XZ.
    Bigger chest/upper torso -> larger average radius.
    """
    y = verts[:, 1]
    mask = (y > y_low) & (y < y_high)
    band = verts[mask]
    # If band selection fails (rare), return 0 safely
    if band.shape[0] < 50:
        return torch.tensor(0.0, device=verts.device)

    # radius around vertical axis (approx)
    r = torch.sqrt(band[:, 0] ** 2 + band[:, 2] ** 2)
    return r.mean()


# ------------------------- Optimization -------------------------
def optimize_betas(
    model,
    targets,
    joint_ids,
    device="cpu",
    steps=250,
    lr=0.05,
    beta_reg=0.01,
):
    """
    targets:
      shoulder_w: float (meters-ish)
      hip_w: float
      chest_r: float  (unit: average radius proxy)
    joint_ids:
      dict with "L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP"
    """
    B = 1
    betas = torch.zeros((B, model.num_betas), device=device, requires_grad=True)

    # keep pose neutral to isolate shape effects
    global_orient = torch.zeros((B, 3), device=device)
    body_pose = torch.zeros((B, model.NUM_BODY_JOINTS * 3), device=device)
    left_hand_pose = torch.zeros((B, 45), device=device)
    right_hand_pose = torch.zeros((B, 45), device=device)
    expression = torch.zeros((B, model.num_expression_coeffs), device=device)
    jaw_pose = torch.zeros((B, 3), device=device)
    leye_pose = torch.zeros((B, 3), device=device)
    reye_pose = torch.zeros((B, 3), device=device)
    transl = torch.zeros((B, 3), device=device)

    opt = torch.optim.Adam([betas], lr=lr)

    # Chest band in Y: we choose relative band based on current mesh height each iter (robust)
    for it in range(steps):
        opt.zero_grad()

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

        verts = out.vertices[0]
        joints = out.joints[0]

        # compute a rough height to place chest band
        y = verts[:, 1]
        y_min, y_max = y.min(), y.max()
        height = y_max - y_min

        # chest band ~ 65% to 78% of height from bottom (tunable)
        y_low = y_min + 0.65 * height
        y_high = y_min + 0.78 * height

        sW = shoulder_width_from_joints(joints, joint_ids["L_SHOULDER"], joint_ids["R_SHOULDER"])
        hW = hip_width_from_joints(joints, joint_ids["L_HIP"], joint_ids["R_HIP"])
        cR = chest_girth_proxy(verts, y_low, y_high)

        loss = 0.0
        if "shoulder_w" in targets and targets["shoulder_w"] is not None:
            loss = loss + (sW - targets["shoulder_w"]) ** 2
        if "hip_w" in targets and targets["hip_w"] is not None:
            loss = loss + (hW - targets["hip_w"]) ** 2
        if "chest_r" in targets and targets["chest_r"] is not None:
            loss = loss + (cR - targets["chest_r"]) ** 2

        # regularize betas to keep realistic
        loss = loss + beta_reg * torch.sum(betas ** 2)

        loss.backward()
        opt.step()

        # keep betas in a sane range to avoid weird artifacts
        with torch.no_grad():
            betas.clamp_(-3.0, 3.0)

        if (it + 1) % 25 == 0:
            print(
                f"[{it+1:04d}/{steps}] loss={loss.item():.6f} "
                f"shoulder={sW.item():.4f} hip={hW.item():.4f} chest_proxy={cR.item():.4f} "
                f"||betas||={betas.norm().item():.3f}"
            )

    return betas.detach()


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./smpl_models")
    ap.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    # Targets (meters-ish). You can set only the ones you care about.
    ap.add_argument("--target_shoulder", type=float, default=None, help="e.g. 0.42")
    ap.add_argument("--target_hip", type=float, default=None, help="e.g. 0.30")
    ap.add_argument("--target_chest_proxy", type=float, default=None, help="e.g. 0.20 (proxy)")

    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.05)

    ap.add_argument("--export_obj", type=str, default="out/tuned.obj")
    args = ap.parse_args()

    os.makedirs("out", exist_ok=True)
    device = args.device

    model = create_smplx(args.model_path, args.gender, args.num_betas, device)

    # ---- IMPORTANT: joint indices ----
    # These indices can differ depending on your SMPL-X setup/version.
    # If these are wrong, shoulder/hip optimization will behave incorrectly.
    #
    # Typical smplx joint indices often:
    #   L_HIP=1, R_HIP=2  (commonly correct)
    #   L_SHOULDER and R_SHOULDER often around ~16-17 or later depending on extended joints.
    #
    # If you want, I can provide a tiny helper to print joints to confirm.
    joint_ids = {
        "L_HIP": 1,
        "R_HIP": 2,
        # These two are the only "iffy" ones:
        "L_SHOULDER": 16,
        "R_SHOULDER": 17,
    }

    targets = {
        "shoulder_w": args.target_shoulder,
        "hip_w": args.target_hip,
        "chest_r": args.target_chest_proxy,
    }

    print("Optimizing betas for targets:", targets)
    betas = optimize_betas(
        model=model,
        targets=targets,
        joint_ids=joint_ids,
        device=device,
        steps=args.steps,
        lr=args.lr,
        beta_reg=0.01,
    )

    # Generate final mesh
    B = 1
    out = model(
        betas=betas,
        global_orient=torch.zeros((B, 3), device=device),
        body_pose=torch.zeros((B, model.NUM_BODY_JOINTS * 3), device=device),
        return_verts=True,
        return_joints=True,
    )
    verts = out.vertices[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int32)

    # Export + visualize
    export_obj(args.export_obj, verts, faces)
    print("[OK] saved:", args.export_obj)
    print("[OK] betas:", betas.detach().cpu().numpy().round(4).tolist())

    visualize_mesh(verts, faces, title="SMPL-X tuned (betas)")

if __name__ == "__main__":
    main()