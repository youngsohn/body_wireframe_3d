import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import open3d as o3d
import smplx

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG: MediaPipe Pose Landmarker task file
# You already downloaded it as:
#   pose_landmarker_heavy.task
# ============================================================
DEFAULT_POSE_TASK = "pose_landmarker_heavy.task"


# ============================================================
# 2D joint subset mapping (MediaPipe 33 landmarks -> our subset)
# ============================================================
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


# ============================================================
# Camera model (simple pinhole)
# ============================================================
@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float


def smplx_extras_zeros(device: torch.device):
    return dict(
        left_hand_pose=torch.zeros((1, 45), device=device),
        right_hand_pose=torch.zeros((1, 45), device=device),
        jaw_pose=torch.zeros((1, 3), device=device),
        leye_pose=torch.zeros((1, 3), device=device),
        reye_pose=torch.zeros((1, 3), device=device),
        expression=torch.zeros((1, 10), device=device),
    )


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
# MediaPipe Tasks API (NO mp.solutions usage)
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


def mp_pose_keypoints_2d_tasks(landmarker, image_bgr: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return {}, np.zeros((len(JOINT_NAMES),), dtype=np.float32)

    lm = result.pose_landmarks[0]  # 33 landmarks

    kp = {}
    conf = []
    for name in JOINT_NAMES:
        idx = MP_ID[name]
        x = lm[idx].x * w
        y = lm[idx].y * h
        v = getattr(lm[idx], "visibility", 1.0)
        kp[name] = np.array([x, y], dtype=np.float32)
        conf.append(float(v))

    return kp, np.array(conf, dtype=np.float32)


# ============================================================
# SMPL model load + joint extractionbody_pose = torch.zeros
# ============================================================
def load_smpl_model(model_folder: str, device: torch.device):
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"SMPL-X model folder not found: {model_folder}")

    # NOTE: For SMPL-X, smplx expects:
    #   <model_folder>/smplx/SMPLX_NEUTRAL.npz
    model = smplx.create(
        model_path=model_folder,
        model_type="smplx",     # <-- changed
        #gender="neutral",
        gender="female",
        num_betas=10,
        batch_size=1,
        use_pca=False,
    ).to(device)
    return model


def smpl_joints_for_names(smpl_output_joints: torch.Tensor) -> torch.Tensor:
    """
    IMPORTANT:
    SMPL joint indices can vary by model file.
    These indices are common in many SMPL setups.

    If your fit looks wrong, print out joint names/index mapping
    from your model or adjust these indices.
    """
    J = smpl_output_joints[0]  # (J,3)

    idx = {
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

    pts = []
    for name in JOINT_NAMES:
        pts.append(J[idx[name]])
    return torch.stack(pts, dim=0)


# ============================================================
# Fitting: minimize 2D reprojection error
# ============================================================
def fit_smpl_to_2d_keypoints(
    smpl_model,
    keypoints_2d: Dict[str, np.ndarray],
    conf: np.ndarray,
    cam: Camera,
    iters: int,
    lr: float,
    device: torch.device,
    init: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:

    target = []
    for name in JOINT_NAMES:
        target.append(keypoints_2d[name])
    target = torch.tensor(np.stack(target, axis=0), dtype=torch.float32, device=device)

    w = torch.tensor(conf, dtype=torch.float32, device=device).clamp(0.0, 1.0)
    w = torch.where(w > 0.2, w, torch.zeros_like(w))

    if init is None:
        global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
        body_pose = torch.zeros((1, 63), device=device, requires_grad=True)  # SMPL-X: 21*3
        betas = torch.zeros((1, 10), device=device, requires_grad=True)
        transl = torch.tensor([[0.0, 0.0, 2.5]], device=device, requires_grad=True)
    else:
        global_orient = init["global_orient"].detach().clone().requires_grad_(True)
        body_pose = init["body_pose"].detach().clone().requires_grad_(True)
        betas = init["betas"].detach().clone().requires_grad_(True)
        transl = init["transl"].detach().clone().requires_grad_(True)

    optim = torch.optim.Adam([global_orient, body_pose, betas, transl], lr=lr)

    def prior_pose(p): return (p ** 2).mean()
    def prior_betas(b): return (b ** 2).mean()

    for _ in range(iters):
        optim.zero_grad()

        extras = smplx_extras_zeros(device)
        out = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            **extras,
        )

        joints3d = smpl_joints_for_names(out.joints)
        proj2d = project_points(joints3d, cam)

        diff = proj2d - target
        repro = (diff ** 2).sum(dim=-1)
        loss_repro = (w * repro).sum() / (w.sum() + 1e-6)

        loss = loss_repro + 0.002 * prior_pose(body_pose) + 0.001 * prior_betas(betas)
        loss.backward()
        optim.step()

    return {
        "global_orient": global_orient.detach(),
        "body_pose": body_pose.detach(),
        "betas": betas.detach(),
        "transl": transl.detach(),
    }


def smpl_mesh_from_params(smpl_model, params: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
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
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
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
    kp, conf = mp_pose_keypoints_2d_tasks(landmarker, img)

    if not kp:
        print("[ERR] No pose detected in image.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = load_smpl_model(args.smpl_model_folder, device)

    params = fit_smpl_to_2d_keypoints(
        smpl_model=smpl_model,
        keypoints_2d=kp,
        conf=conf,
        cam=cam,
        iters=max(args.iters, 400),
        lr=args.lr,
        device=device,
        init=None,
    )

    verts, faces = smpl_mesh_from_params(smpl_model, params)

    if args.export_obj:
        export_obj(args.export_obj, verts, faces)
    if args.view:
        visualize_mesh_open3d(verts, faces, "Fit photo -> SMPL mesh")


def mode_image(args):
    # same as fit_photo but fewer iters
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    h, w = img.shape[:2]
    cam = estimate_camera_from_image(w, h)

    landmarker = create_pose_landmarker(args.pose_task)
    kp, conf = mp_pose_keypoints_2d_tasks(landmarker, img)
    if not kp:
        print("[ERR] No pose detected.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = load_smpl_model(args.smpl_model_folder, device)

    params = fit_smpl_to_2d_keypoints(
        smpl_model=smpl_model,
        keypoints_2d=kp,
        conf=conf,
        cam=cam,
        iters=args.iters,
        lr=args.lr,
        device=device,
        init=None,
    )

    verts, faces = smpl_mesh_from_params(smpl_model, params)
    if args.export_obj:
        export_obj(args.export_obj, verts, faces)
    if args.view:
        visualize_mesh_open3d(verts, faces, "Static image -> SMPL mesh")


def mode_animate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = load_smpl_model(args.smpl_model_folder, device)

    params = {
        "global_orient": torch.zeros((1, 3), device=device),
        "body_pose": torch.zeros((1, 63), device=device),
        "betas": torch.zeros((1, 10), device=device),
        "transl": torch.tensor([[0.0, 0.0, 2.5]], device=device),
    }

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Animate -> SMPL mesh", width=960, height=720)

    verts, faces = smpl_mesh_from_params(smpl_model, params)
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

            # crude motion demo
            amp = 0.6
            pose[:, 3:6] = torch.tensor([[amp * np.sin(2.0 * t), 0.0, 0.0]], device=device)
            pose[:, 9:12] = torch.tensor([[-amp * np.sin(2.0 * t), 0.0, 0.0]], device=device)

            params["body_pose"] = pose
            verts, _ = smpl_mesh_from_params(smpl_model, params)

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
    smpl_model = load_smpl_model(args.smpl_model_folder, device)

    landmarker = create_pose_landmarker(args.pose_task)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Webcam -> SMPL mesh (baseline)", width=960, height=720)

    prev_params = {
        "global_orient": torch.zeros((1, 3), device=device),
        "body_pose": torch.zeros((1, 63), device=device),
        "betas": torch.zeros((1, 10), device=device),
        "transl": torch.tensor([[0.0, 0.0, 2.5]], device=device),
    }

    verts, faces = smpl_mesh_from_params(smpl_model, prev_params)
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

            kp, conf = mp_pose_keypoints_2d_tasks(landmarker, frame)
            if kp:
                params = fit_smpl_to_2d_keypoints(
                    smpl_model=smpl_model,
                    keypoints_2d=kp,
                    conf=conf,
                    cam=cam,
                    iters=args.webcam_steps,
                    lr=args.lr,
                    device=device,
                    init=prev_params,
                )
                prev_params = params
                verts, _ = smpl_mesh_from_params(smpl_model, params)
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
    ap.add_argument("--image", default=None)
    ap.add_argument("--pose_task", default=DEFAULT_POSE_TASK, help="Path to pose_landmarker_*.task file")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--export_obj", default=None)
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--webcam_steps", type=int, default=15)
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