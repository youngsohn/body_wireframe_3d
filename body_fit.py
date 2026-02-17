#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import torch
import open3d as o3d
import smplx

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================
DEFAULT_POSE_TASK = "pose_landmarker_heavy.task"

MP_ID = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

SWAP_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]


# ============================================================
# Camera
# ============================================================
@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float


def estimate_camera_from_image(w: int, h: int) -> Camera:
    f = 1.2 * max(w, h)
    return Camera(fx=f, fy=f, cx=w / 2.0, cy=h / 2.0)


def project_points(points_3d: torch.Tensor, cam: Camera) -> torch.Tensor:
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = torch.clamp(points_3d[:, 2], min=1e-6)
    u = cam.fx * (x / z) + cam.cx
    v = cam.fy * (y / z) + cam.cy
    return torch.stack([u, v], dim=-1)


# ============================================================
# MediaPipe Pose Landmarker (Tasks API)
# ============================================================
def create_pose_landmarker(task_path: str):
    if not os.path.exists(task_path):
        raise FileNotFoundError(
            f"Pose task model not found: {task_path}\n"
            f"Download it with:\n"
            f"curl -L -o pose_landmarker_heavy.task "
            f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        )

    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.PoseLandmarker.create_from_options(options)


def mp_pose_keypoints_tasks(
    landmarker,
    image_bgr: np.ndarray
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Returns:
      kp2d: dict name->(x,y) pixels
      conf: (J,) visibility
      kp3d: dict name->(x,y,z) pose_world_landmarks if available else None
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return {}, np.zeros((len(JOINT_NAMES),), dtype=np.float32), None

    lm2d = result.pose_landmarks[0]
    lm3d = None
    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
        lm3d = result.pose_world_landmarks[0]

    kp2d: Dict[str, np.ndarray] = {}
    conf_list: List[float] = []
    for name in JOINT_NAMES:
        idx = MP_ID[name]
        x = lm2d[idx].x * w
        y = lm2d[idx].y * h
        v = getattr(lm2d[idx], "visibility", 1.0)
        kp2d[name] = np.array([x, y], dtype=np.float32)
        conf_list.append(float(v))
    conf = np.array(conf_list, dtype=np.float32)

    kp3d = None
    if lm3d is not None:
        kp3d = {}
        for name in JOINT_NAMES:
            idx = MP_ID[name]
            kp3d[name] = np.array([lm3d[idx].x, lm3d[idx].y, lm3d[idx].z], dtype=np.float32)

    return kp2d, conf, kp3d


# ============================================================
# LR SANITIZATION (fix local LR mistakes)
# ============================================================
def _midline_x(kp2d: Dict[str, np.ndarray]) -> float:
    xs = []
    for a, b in [("left_hip", "right_hip"), ("left_shoulder", "right_shoulder")]:
        if a in kp2d and b in kp2d:
            xs.append(0.5 * (kp2d[a][0] + kp2d[b][0]))
    if xs:
        return float(np.mean(xs))
    return float(np.mean([kp2d[n][0] for n in kp2d]))


def _swap_pair(kp: Dict[str, np.ndarray], a: str, b: str):
    kp[a], kp[b] = kp[b], kp[a]


def sanitize_lr_per_limb(kp2d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Fixes local LR mistakes (e.g., only one elbow/wrist swapped).
    1) Midline test for each LR pair
    2) For elbows/wrists: ensure each is closer to its own-side shoulder
    """
    kp = dict(kp2d)
    mx = _midline_x(kp)

    # Step 1: swap if both endpoints are on wrong side of midline
    for a, b in SWAP_PAIRS:
        if a not in kp or b not in kp:
            continue
        xa = kp[a][0]
        xb = kp[b][0]
        if (xa > mx) and (xb < mx):
            _swap_pair(kp, a, b)

    # Step 2: arms closer-to-shoulder check (fix single-arm issues)
    if ("left_shoulder" in kp) and ("right_shoulder" in kp):
        LS = kp["left_shoulder"]
        RS = kp["right_shoulder"]

        def closer_to_left(pt):
            return np.linalg.norm(pt - LS) <= np.linalg.norm(pt - RS)

        if ("left_elbow" in kp) and ("right_elbow" in kp):
            LE = kp["left_elbow"]
            RE = kp["right_elbow"]
            if (not closer_to_left(LE)) and closer_to_left(RE):
                _swap_pair(kp, "left_elbow", "right_elbow")

        if ("left_wrist" in kp) and ("right_wrist" in kp):
            LW = kp["left_wrist"]
            RW = kp["right_wrist"]
            if (not closer_to_left(LW)) and closer_to_left(RW):
                _swap_pair(kp, "left_wrist", "right_wrist")

    return kp


def swap_lr_keypoints_2d(kp2d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    kp = dict(kp2d)
    for a, b in SWAP_PAIRS:
        if a in kp and b in kp:
            kp[a], kp[b] = kp[b], kp[a]
    return kp


def swap_lr_keypoints_3d(kp3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    kp = dict(kp3d)
    for a, b in SWAP_PAIRS:
        if a in kp and b in kp:
            kp[a], kp[b] = kp[b], kp[a]
    return kp


# ============================================================
# SMPL-X helpers
# ============================================================
def smplx_extras_zeros(device: torch.device):
    return dict(
        left_hand_pose=torch.zeros((1, 45), device=device),
        right_hand_pose=torch.zeros((1, 45), device=device),
        jaw_pose=torch.zeros((1, 3), device=device),
        leye_pose=torch.zeros((1, 3), device=device),
        reye_pose=torch.zeros((1, 3), device=device),
        expression=torch.zeros((1, 10), device=device),
    )


def load_smplx_model(model_folder: str, device: torch.device, gender: str = "neutral"):
    """
    Supports BOTH:
      <model_folder>/smplx/SMPLX_*.npz
      <model_folder>/smpl/SMPLX_*.npz
    """
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}\ncwd: {os.getcwd()}")

    smplx_dir = os.path.join(model_folder, "smplx")
    smpl_dir = os.path.join(model_folder, "smpl")

    if os.path.isdir(smplx_dir):
        model_path = model_folder
    elif os.path.isdir(smpl_dir):
        if not os.path.exists(smplx_dir):
            try:
                os.symlink("smpl", smplx_dir)
            except Exception:
                pass
        model_path = model_folder
    else:
        raise FileNotFoundError(
            f"No subfolder 'smplx' or 'smpl' under: {model_folder}\n"
            f"Expected: {model_folder}/smplx/SMPLX_NEUTRAL.npz (or in smpl/)"
        )

    model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        gender=gender,
        num_betas=10,
        batch_size=1,
        use_pca=False,
    ).to(device)
    return model


# ============================================================
# Joint mapping selection
# ============================================================
def _idx_map_normal():
    return {
        "left_hip": 1,
        "right_hip": 2,
        "left_knee": 4,
        "right_knee": 5,
        "left_ankle": 7,
        "right_ankle": 8,
        "left_shoulder": 16,
        "right_shoulder": 17,
        "left_elbow": 18,
        "right_elbow": 19,
        "left_wrist": 20,
        "right_wrist": 21,
    }


def _idx_map_swap_lr(idx: Dict[str, int]):
    idx2 = dict(idx)
    for a, b in SWAP_PAIRS:
        idx2[a], idx2[b] = idx2[b], idx2[a]
    return idx2


def smpl_joints_for_names(out_joints: torch.Tensor, idx_map: Dict[str, int]) -> torch.Tensor:
    J = out_joints[0]
    pts = []
    for name in JOINT_NAMES:
        pts.append(J[idx_map[name]])
    return torch.stack(pts, dim=0)


# ============================================================
# Loss / init / direction constraint
# ============================================================
def robust_reproj_loss(diff_xy: torch.Tensor, delta: float = 10.0) -> torch.Tensor:
    d = torch.sqrt((diff_xy ** 2).sum(dim=-1) + 1e-6)
    delta_t = torch.tensor(delta, device=diff_xy.device, dtype=diff_xy.dtype)
    return torch.where(d < delta_t, 0.5 * d * d, delta_t * (d - 0.5 * delta_t))


def estimate_init_depth_from_2d(kp2d: Dict[str, np.ndarray], cam: Camera) -> float:
    try:
        ls = kp2d["left_shoulder"]
        rs = kp2d["right_shoulder"]
        lh = kp2d["left_hip"]
        rh = kp2d["right_hip"]
    except KeyError:
        return 2.5

    mid_sh = 0.5 * (ls + rs)
    mid_hp = 0.5 * (lh + rh)
    L_px = float(np.linalg.norm(mid_sh - mid_hp))
    if L_px < 10.0:
        return 2.5

    L_m = 0.55
    z = float(cam.fx * (L_m / L_px))
    return float(np.clip(z, 0.8, 6.0))


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def limb_direction_loss(
    smpl_joints: torch.Tensor,   # (J,3) torch, subset order = JOINT_NAMES
    mp3d: torch.Tensor,          # (J,3) torch, same subset order
    w_joint: torch.Tensor,       # (J,) weights
) -> torch.Tensor:
    """
    Compare normalized limb direction vectors (scale/translation invariant).
    Prevents "reversed" forearm solutions.
    """
    LS, RS = 0, 1
    LE, RE = 2, 3
    LW, RW = 4, 5

    suL = _unit(smpl_joints[LE] - smpl_joints[LS])
    sfL = _unit(smpl_joints[LW] - smpl_joints[LE])
    suR = _unit(smpl_joints[RE] - smpl_joints[RS])
    sfR = _unit(smpl_joints[RW] - smpl_joints[RE])

    muL = _unit(mp3d[LE] - mp3d[LS])
    mfL = _unit(mp3d[LW] - mp3d[LE])
    muR = _unit(mp3d[RE] - mp3d[RS])
    mfR = _unit(mp3d[RW] - mp3d[RE])

    def cos_dist(a, b):
        return 1.0 - torch.clamp((a * b).sum(), -1.0, 1.0)

    wL = (w_joint[LS] + w_joint[LE] + w_joint[LW]) / 3.0
    wR = (w_joint[RS] + w_joint[RE] + w_joint[RW]) / 3.0

    loss = (
        wL * (cos_dist(suL, muL) + cos_dist(sfL, mfL)) +
        wR * (cos_dist(suR, muR) + cos_dist(sfR, mfR))
    )
    return loss


# ============================================================
# Fitting (staged)
#   - per-limb LR sanitize (2D)
#   - 4-way global selection (idx + kp swapLR)
#   - optional world3d direction constraint (fix reversed arm)
# ============================================================
def fit_smplx_to_keypoints(
    smpl_model,
    kp2d_in: Dict[str, np.ndarray],
    conf: np.ndarray,
    cam: Camera,
    device: torch.device,
    kp3d_in: Optional[Dict[str, np.ndarray]] = None,
    iters: int = 420,
    lr: float = 0.03,
    use_world3d: bool = False,
    w3d: float = 10.0,
    w_dir: float = 50.0,
    w_pose_prior: float = 0.01,
    w_shape_prior: float = 0.01,
    sanitize_lr: bool = True,
) -> Dict[str, torch.Tensor]:

    kp2d_base = sanitize_lr_per_limb(kp2d_in) if sanitize_lr else dict(kp2d_in)
    kp3d_base = dict(kp3d_in) if kp3d_in is not None else None

    z0 = estimate_init_depth_from_2d(kp2d_base, cam)
    transl = torch.tensor([[0.0, 0.0, z0]], device=device, requires_grad=True)
    extras = smplx_extras_zeros(device)

    idxA = _idx_map_normal()
    idxB = _idx_map_swap_lr(idxA)

    kp2dA = kp2d_base
    kp2dB = swap_lr_keypoints_2d(kp2d_base)
    kp3dA = kp3d_base
    kp3dB = swap_lr_keypoints_3d(kp3d_base) if kp3d_base is not None else None

    w = torch.tensor(conf, dtype=torch.float32, device=device).clamp(0.0, 1.0)
    w = torch.where(w > 0.2, w, torch.zeros_like(w))

    def make_target2(kp2d: Dict[str, np.ndarray]) -> torch.Tensor:
        return torch.tensor(
            np.stack([kp2d[n] for n in JOINT_NAMES], axis=0),
            dtype=torch.float32,
            device=device,
        )

    def score(idx_map: Dict[str, int], kp2d: Dict[str, np.ndarray]) -> float:
        with torch.no_grad():
            out0 = smpl_model(
                global_orient=torch.zeros((1, 3), device=device),
                body_pose=torch.zeros((1, 63), device=device),
                betas=torch.zeros((1, 10), device=device),
                transl=transl.detach(),
                **extras,
            )
            j = smpl_joints_for_names(out0.joints, idx_map)
            p = project_points(j, cam)
            tgt = make_target2(kp2d)
            e = (w * ((p - tgt) ** 2).sum(dim=-1)).sum() / (w.sum() + 1e-6)
            return float(e.item())

    candidates = [
        ("idx=normal, kp=normal", idxA, kp2dA, kp3dA),
        ("idx=swap,   kp=normal", idxB, kp2dA, kp3dA),
        ("idx=normal, kp=swapLR", idxA, kp2dB, kp3dB),
        ("idx=swap,   kp=swapLR", idxB, kp2dB, kp3dB),
    ]

    scored = []
    for name, idx_map, kp2d, kp3d in candidates:
        s = score(idx_map, kp2d)
        scored.append((s, name, idx_map, kp2d, kp3d))
    scored.sort(key=lambda x: x[0])

    best_s, best_name, chosen_idx, kp2d, kp3d = scored[0]
    print("[INFO] best mapping:", best_name, "score=", best_s)

    target2 = make_target2(kp2d)

    target3 = None
    if use_world3d and kp3d is not None:
        target3 = torch.tensor(
            np.stack([kp3d[n] for n in JOINT_NAMES], axis=0),
            dtype=torch.float32,
            device=device,
        )

    # Params (SMPL-X)
    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((1, 63), device=device, requires_grad=True)
    betas = torch.zeros((1, 10), device=device, requires_grad=True)

    def pose_prior(p): return (p ** 2).mean()
    def shape_prior(b): return (b ** 2).mean()

    itA = max(60, int(0.20 * iters))
    itB = max(220, int(0.60 * iters))
    itC = max(60, iters - itA - itB)

    # Stage A
    body_pose.requires_grad_(False)
    betas.requires_grad_(False)
    optimA = torch.optim.Adam([global_orient, transl], lr=lr)

    for _ in range(itA):
        optimA.zero_grad()
        out = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            **extras,
        )
        joints3d = smpl_joints_for_names(out.joints, chosen_idx)
        proj2d = project_points(joints3d, cam)

        loss2 = (w * robust_reproj_loss(proj2d - target2)).sum() / (w.sum() + 1e-6)
        loss = loss2

        if target3 is not None:
            loss_dir = limb_direction_loss(joints3d, target3, w)
            loss = loss + w_dir * loss_dir

            j3c = joints3d - joints3d.mean(dim=0, keepdim=True)
            t3c = target3 - target3.mean(dim=0, keepdim=True)
            loss3 = (w * ((j3c - t3c) ** 2).sum(dim=-1)).sum() / (w.sum() + 1e-6)
            loss = loss + (w3d * 0.2) * loss3

        loss.backward()
        optimA.step()

    # Stage B
    body_pose.requires_grad_(True)
    betas.requires_grad_(False)
    optimB = torch.optim.Adam([global_orient, transl, body_pose], lr=lr * 0.7)

    for _ in range(itB):
        optimB.zero_grad()
        out = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            **extras,
        )
        joints3d = smpl_joints_for_names(out.joints, chosen_idx)
        proj2d = project_points(joints3d, cam)

        loss2 = (w * robust_reproj_loss(proj2d - target2)).sum() / (w.sum() + 1e-6)
        loss = loss2 + w_pose_prior * pose_prior(body_pose)

        if target3 is not None:
            loss_dir = limb_direction_loss(joints3d, target3, w)
            loss = loss + w_dir * loss_dir

            j3c = joints3d - joints3d.mean(dim=0, keepdim=True)
            t3c = target3 - target3.mean(dim=0, keepdim=True)
            loss3 = (w * ((j3c - t3c) ** 2).sum(dim=-1)).sum() / (w.sum() + 1e-6)
            loss = loss + (w3d * 0.2) * loss3

        loss.backward()
        optimB.step()

    # Stage C
    body_pose.requires_grad_(True)
    betas.requires_grad_(True)
    optimC = torch.optim.Adam([global_orient, transl, body_pose, betas], lr=lr * 0.3)

    for _ in range(itC):
        optimC.zero_grad()
        out = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            **extras,
        )
        joints3d = smpl_joints_for_names(out.joints, chosen_idx)
        proj2d = project_points(joints3d, cam)

        loss2 = (w * robust_reproj_loss(proj2d - target2)).sum() / (w.sum() + 1e-6)
        loss = loss2 + w_pose_prior * pose_prior(body_pose) + w_shape_prior * shape_prior(betas)

        if target3 is not None:
            loss_dir = limb_direction_loss(joints3d, target3, w)
            loss = loss + w_dir * loss_dir

            j3c = joints3d - joints3d.mean(dim=0, keepdim=True)
            t3c = target3 - target3.mean(dim=0, keepdim=True)
            loss3 = (w * ((j3c - t3c) ** 2).sum(dim=-1)).sum() / (w.sum() + 1e-6)
            loss = loss + (w3d * 0.2) * loss3

        loss.backward()
        optimC.step()

    return {
        "global_orient": global_orient.detach(),
        "body_pose": body_pose.detach(),
        "betas": betas.detach(),
        "transl": transl.detach(),
    }


def smplx_mesh_from_params(smpl_model, params: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    device = params["global_orient"].device
    extras = smplx_extras_zeros(device)
    out = smpl_model(
        global_orient=params["global_orient"],
        body_pose=params["body_pose"],
        betas=params["betas"],
        transl=params["transl"],
        **extras,
    )
    verts = out.vertices[0].detach().cpu().numpy()
    faces = smpl_model.faces.astype(np.int32)
    return verts, faces


# ============================================================
# Open3D helpers
# ============================================================
def visualize_mesh_open3d(verts: np.ndarray, faces: np.ndarray, title: str):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name=title)


def export_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
    print(f"[OK] Exported OBJ: {path}")


# ============================================================
# Modes
# ============================================================
def mode_fit_photo(args):
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    h, w = img.shape[:2]
    cam = estimate_camera_from_image(w, h)

    landmarker = create_pose_landmarker(args.pose_task)
    kp2d, conf, kp3d = mp_pose_keypoints_tasks(landmarker, img)
    if not kp2d:
        print("[ERR] No pose detected in image.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = load_smplx_model(args.smpl_model_folder, device, gender=args.gender)

    params = fit_smplx_to_keypoints(
        smpl_model=smpl_model,
        kp2d_in=kp2d,
        conf=conf,
        cam=cam,
        device=device,
        kp3d_in=kp3d,
        iters=args.iters,
        lr=args.lr,
        use_world3d=args.use_world3d,
        w3d=args.w3d,
        w_dir=args.w_dir,
        w_pose_prior=args.pose_prior,
        w_shape_prior=args.shape_prior,
        sanitize_lr=not args.no_sanitize_lr,
    )

    verts, faces = smplx_mesh_from_params(smpl_model, params)

    if args.export_obj:
        export_obj(args.export_obj, verts, faces)
    if args.view:
        visualize_mesh_open3d(verts, faces, "Fit photo -> SMPL-X mesh")


def mode_image(args):
    mode_fit_photo(args)


def mode_animate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = load_smplx_model(args.smpl_model_folder, device, gender=args.gender)

    params = {
        "global_orient": torch.zeros((1, 3), device=device),
        "body_pose": torch.zeros((1, 63), device=device),
        "betas": torch.zeros((1, 10), device=device),
        "transl": torch.tensor([[0.0, 0.0, 2.5]], device=device),
    }

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Animate -> SMPL-X mesh", width=960, height=720)

    verts, faces = smplx_mesh_from_params(smpl_model, params)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)

    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            pose = params["body_pose"].clone()
            amp = 0.6
            pose[:, 3:6] = torch.tensor([[amp * np.sin(2.0 * t), 0.0, 0.0]], device=device)
            pose[:, 9:12] = torch.tensor([[-amp * np.sin(2.0 * t), 0.0, 0.0]], device=device)
            params["body_pose"] = pose

            verts, _ = smplx_mesh_from_params(smpl_model, params)
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()


def mode_webcam(args):
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print("[ERR] Cannot open webcam.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = load_smplx_model(args.smpl_model_folder, device, gender=args.gender)
    landmarker = create_pose_landmarker(args.pose_task)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Webcam -> SMPL-X mesh (baseline)", width=960, height=720)

    params0 = {
        "global_orient": torch.zeros((1, 3), device=device),
        "body_pose": torch.zeros((1, 63), device=device),
        "betas": torch.zeros((1, 10), device=device),
        "transl": torch.tensor([[0.0, 0.0, 2.5]], device=device),
    }

    verts, faces = smplx_mesh_from_params(smpl_model, params0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            cam = estimate_camera_from_image(w, h)

            kp2d, conf, kp3d = mp_pose_keypoints_tasks(landmarker, frame)
            if kp2d:
                params_fit = fit_smplx_to_keypoints(
                    smpl_model=smpl_model,
                    kp2d_in=kp2d,
                    conf=conf,
                    cam=cam,
                    device=device,
                    kp3d_in=kp3d,
                    iters=args.webcam_steps,
                    lr=args.lr,
                    use_world3d=args.use_world3d,
                    w3d=args.w3d,
                    w_dir=args.w_dir,
                    w_pose_prior=args.pose_prior,
                    w_shape_prior=max(args.shape_prior, 0.1),
                    sanitize_lr=not args.no_sanitize_lr,
                )
                verts, _ = smplx_mesh_from_params(smpl_model, params_fit)
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.compute_vertex_normals()
                vis.update_geometry(mesh)

            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("webcam", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()


# ============================================================
# Entry
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["image", "fit_photo", "animate", "webcam"])
    ap.add_argument("--smpl_model_folder", required=True)
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])

    ap.add_argument("--image", default=None)
    ap.add_argument("--pose_task", default=DEFAULT_POSE_TASK)

    ap.add_argument("--iters", type=int, default=420)
    ap.add_argument("--lr", type=float, default=0.03)

    ap.add_argument("--use_world3d", action="store_true")
    ap.add_argument("--w3d", type=float, default=10.0)
    ap.add_argument("--w_dir", type=float, default=50.0)

    ap.add_argument("--pose_prior", type=float, default=0.01)
    ap.add_argument("--shape_prior", type=float, default=0.01)

    ap.add_argument("--no_sanitize_lr", action="store_true",
                    help="Disable per-limb LR sanitization (not recommended).")

    ap.add_argument("--view", action="store_true")
    ap.add_argument("--export_obj", default=None)

    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--webcam_steps", type=int, default=25)

    args = ap.parse_args()

    if args.mode in ("image", "fit_photo") and not args.image:
        raise ValueError("--image is required for mode=image or fit_photo")

    if args.mode == "fit_photo":
        mode_fit_photo(args)
    elif args.mode == "image":
        mode_image(args)
    elif args.mode == "animate":
        mode_animate(args)
    elif args.mode == "webcam":
        mode_webcam(args)


if __name__ == "__main__":
    main()