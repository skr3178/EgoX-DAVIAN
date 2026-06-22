"""
Section 3 test: GGA geometry preprocessing (CPU only, no model).
Lifts the exact block from infer.py:94-236 and runs it on one example take,
printing the resulting attn_maps / attn_masks / point_vecs shapes.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
import cv2
from core.finetune.datasets.utils import load_from_json_file, iproj_disp

META = sys.argv[1] if len(sys.argv) > 1 else "./example/egoexo4D/meta.json"
IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0

meta_data = load_from_json_file(META)["test_datasets"]
meta = meta_data[IDX]

# --- path derivation (infer.py:45-51) ---
exo_path = meta["exo_path"]
take_name = exo_path.split("/")[-2]
depth_root = "/".join(exo_path.split("/")[:3])
depth_map_path = Path(os.path.join(depth_root, "depth_maps", take_name))
print(f"take={take_name}  depth_dir={depth_map_path}  exists={depth_map_path.exists()}")

camera_intrinsic = meta["camera_intrinsics"]
camera_extrinsic = meta["camera_extrinsics"]
ego_extrinsic = meta["ego_extrinsics"]
ego_intrinsic = meta["ego_intrinsics"]

# ===== verbatim from infer.py:101-231 =====
device = "cpu"
C, F, H, W = 16, 13, 56, 154  #! Hard coding
exo_H, exo_W = H, W - H
W = H

depth_maps = []
for depth_map_file in sorted(depth_map_path.glob("*.npy")):
    depth_map = np.load(depth_map_file)
    depth_maps.append(torch.from_numpy(depth_map).unsqueeze(0))
depth_maps = torch.cat(depth_maps, dim=0)
print(f"depth_maps: {tuple(depth_maps.shape)}  (#npy={len(list(depth_map_path.glob('*.npy')))})")

ego_intrinsic = torch.tensor(ego_intrinsic)
ego_extrinsic = torch.tensor(ego_extrinsic)
camera_extrinsic = torch.tensor(camera_extrinsic)
camera_intrinsic = torch.tensor(camera_intrinsic)
print(f"ego_intrinsic {tuple(ego_intrinsic.shape)}  ego_extrinsic {tuple(ego_extrinsic.shape)}  "
      f"cam_extrinsic {tuple(camera_extrinsic.shape)}  cam_intrinsic {tuple(camera_intrinsic.shape)}")

if ego_extrinsic.shape[1] == 3 and ego_extrinsic.shape[2] == 4:
    ego_extrinsic = torch.cat([ego_extrinsic, torch.tensor([[[0, 0, 0, 1]]], dtype=ego_extrinsic.dtype).expand(ego_extrinsic.shape[0], -1, -1)], dim=1)
if camera_extrinsic.shape == (3, 4):
    camera_extrinsic = torch.cat([torch.tensor(camera_extrinsic, dtype=ego_extrinsic.dtype), torch.tensor([[0, 0, 0, 1]], dtype=ego_extrinsic.dtype)], dim=0)

scale = 1 / 8
scaled_intrinsic = ego_intrinsic.clone()
scaled_intrinsic[0, 0] *= scale
scaled_intrinsic[1, 1] *= scale
scaled_intrinsic[0, 2] *= scale
scaled_intrinsic[1, 2] *= scale

ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
ones = torch.ones_like(xs)
pixel_coords = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).to(dtype=ego_intrinsic.dtype)

pixel_coords_cv = pixel_coords[..., :2].cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
K = scaled_intrinsic.cpu().numpy().astype(np.float32)

distortion_coeffs = np.array([[-0.02340373583137989,0.09388021379709244,-0.06088035926222801,0.0053304750472307205,0.003342868760228157,-0.0006356257363222539,0.0005087381578050554,-0.0004747129278257489,-0.0011330085108056664,-0.00025734835071489215,0.00009328465239377692,0.00009424977179151028]])
D = distortion_coeffs.astype(np.float32)
normalized_points = cv2.undistortPoints(pixel_coords_cv, K, D, R=np.eye(3), P=np.eye(3))
normalized_points = torch.from_numpy(normalized_points).squeeze(1).to(device)

ones = torch.ones_like(normalized_points[..., :1])
cam_rays_fish = torch.cat([normalized_points, ones], dim=-1)
cam_rays = cam_rays_fish / torch.norm(cam_rays_fish, dim=-1, keepdim=True)
cam_rays = cam_rays @ ego_extrinsic[::4, :3, :3]
cam_rays = cam_rays.view(F, H, W, 3)

height, width = depth_maps.shape[1], depth_maps.shape[2]
cx = width / 2.0
cy = height / 2.0
camera_intrinsic_scale_y = cy / camera_intrinsic[1, 2]
camera_intrinsic_scale_x = cx / camera_intrinsic[0, 2]
camera_intrinsic[0, 0] = camera_intrinsic[0, 0] * camera_intrinsic_scale_x
camera_intrinsic[1, 1] = camera_intrinsic[1, 1] * camera_intrinsic_scale_y
camera_intrinsic[0, 2] = cx
camera_intrinsic[1, 2] = cy
camera_intrinsic = np.array([camera_intrinsic[0, 0], camera_intrinsic[1, 1], cx, cy])

disp_v, disp_u = torch.meshgrid(
    torch.arange(depth_maps.shape[1], device=device).float(),
    torch.arange(depth_maps.shape[2], device=device).float(),
    indexing="ij",
)
disp = torch.ones_like(disp_v)
pts, _, _ = iproj_disp(torch.from_numpy(camera_intrinsic), disp.cpu(), disp_u.cpu(), disp_v.cpu())
pts = pts.to(device) if isinstance(pts, torch.Tensor) else torch.from_numpy(pts).to(device).float()

rays = pts[..., :3]
rays = rays / rays[..., 2:3]
rays = rays.unsqueeze(0).expand(depth_maps.size(0), -1, -1, -1)
camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)

pcd_camera = rays * depth_maps.unsqueeze(-1)
point_map = pcd_camera.to(dtype=camera_extrinsics_c2w.dtype)
point_map = torch.tensor(point_map)

p_f, p_h, p_w, p_p = point_map.shape
point_map_world = point_map.reshape(-1, 3)
camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)
ones_point = torch.ones(point_map_world.shape[0], 1, device=point_map_world.device)
point_map_world = torch.cat([point_map_world, ones_point], dim=-1)
point_map_world = (camera_extrinsics_c2w @ point_map_world.T).T[..., :3]
point_map = point_map_world.reshape(p_f, p_h, p_w, 3).permute(0, 3, 1, 2)

point_map = point_map[:, :, (point_map.shape[2] - 448)//2:(point_map.shape[2] + 448)//2, (point_map.shape[3] - 784)//2:(point_map.shape[3] + 784)//2]
point_map = torch.nn.functional.interpolate(point_map, size=(exo_H, exo_W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

ego_extrinsic_c2w = torch.linalg.inv(ego_extrinsic)
cam_origins = ego_extrinsic_c2w[::4, :3, 3].unsqueeze(1).expand(-1, exo_H * exo_W, -1)
cam_origins = cam_origins.view(F, exo_H, exo_W, 3)

if point_map.size(0) != ego_extrinsic_c2w.size(0):
    min_size = min(point_map.size(0), ego_extrinsic_c2w.size(0))
    point_map = point_map[:min_size]

point_vecs_per_frame = []
for j in range(cam_origins.size(0)):
    point_vec = point_map[::4] - cam_origins[j].unsqueeze(0)
    point_vec = point_vec / torch.norm(point_vec, dim=-1, keepdim=True)
    point_vecs_per_frame.append(point_vec)
point_vecs_per_frame = torch.stack(point_vecs_per_frame, dim=0)

point_vecs = point_map[::4] - cam_origins
point_vecs = point_vecs / torch.norm(point_vecs, dim=-1, keepdim=True)
cam_rays = torch.rot90(cam_rays, k=-1, dims=[1, 2])

attn_maps = torch.cat((point_vecs, cam_rays), dim=2)
attn_masks = torch.cat((torch.ones_like(point_vecs), torch.zeros_like(cam_rays)), dim=2)

# ===== report =====
def stat(name, t):
    print(f"  {name:22s} {str(tuple(t.shape)):22s} dtype={t.dtype} "
          f"nan={torch.isnan(t).any().item()} finite={torch.isfinite(t).all().item()} "
          f"range=[{t.min():.3f},{t.max():.3f}]")

print("\n=== GGA outputs (fed to generate_video, unsqueezed to add batch) ===")
stat("attn_maps", attn_maps)
stat("attn_masks", attn_masks)
stat("point_vecs_per_frame", point_vecs_per_frame)
stat("cam_rays", cam_rays)
print("\nSection 3 OK")
