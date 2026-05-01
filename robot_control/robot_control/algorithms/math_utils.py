import numpy as np
import cv2

def pixel_to_world_on_plane(pixel, mtx, rvec, tvec, z_world_offset):
    """
    将2D像素坐标反投影到指定Z高度的世界坐标平面上。
    """
    if rvec is None or tvec is None:
        return None

    # 1. 确保 rvec/tvec 格式正确 (3x1)
    rvec = np.array(rvec, dtype=np.float32).reshape(3, 1)
    tvec = np.array(tvec, dtype=np.float32).reshape(3, 1)

    # 2. 获取旋转矩阵 R (World -> Camera)
    R, _ = cv2.Rodrigues(rvec)
    
    # 3. 计算相机光心在世界坐标系的位置 C_w
    # Camera Center: C = -R^T * t
    cam_center_world = -np.dot(R.T, tvec)
    
    # 4. 将像素转换为归一化相机坐标系下的方向向量
    # pixel 可能是 (u, v) 元组或 numpy 数组
    u, v = pixel[0], pixel[1]
    uv_homo = np.array([u, v, 1.0]).reshape(3, 1)
    
    # P_norm = K^(-1) * P_pix
    ray_cam = np.dot(np.linalg.inv(mtx), uv_homo)
    
    # 5. 将射线方向从相机坐标系转到世界坐标系
    # Ray_world = R^T * Ray_cam
    ray_world = np.dot(R.T, ray_cam)
    
    # 6. 射线平面求交
    # 射线方程: P = C_w + lambda * Ray_world
    # 我们已知 P.z = z_world_offset
    ray_z = ray_world[2][0]
    cam_z = cam_center_world[2][0]

    if abs(ray_z) < 1e-6:
        return None  # 射线与Z轴垂直，无法求交

    lam = (z_world_offset - cam_z) / ray_z
    
    # 计算最终的世界坐标
    p_world = cam_center_world + lam * ray_world
    
    # 确保返回的是平铺的 (3,) 数组
    return p_world.flatten()

def get_projected_circle_pts(center_world, radius_mm, rvec, tvec, mtx, samples=30):
    """
    在世界坐标系平面的 center_world 处生成一个水平圆 (Z不变)，
    并将其投影回 2D 像素坐标。
    """
    if rvec is None or tvec is None: return None
    
    # [关键修复] 强壮的坐标提取：不管你是 (3,1) 还是 (3,) 还是 list，统统展平取前3个
    center_flat = np.array(center_world).flatten()
    if len(center_flat) < 3: return None
    cx, cy, cz = center_flat[:3]
    
    # 1. 在 3D 空间生成圆周上的点
    thetas = np.linspace(0, 2*np.pi, samples)
    
    # 假设圆躺在 Z = cz 的平面上 (平行于 XY 平面)
    circle_pts_3d = []
    for theta in thetas:
        x = cx + radius_mm * np.cos(theta)
        y = cy + radius_mm * np.sin(theta)
        z = cz 
        circle_pts_3d.append([x, y, z])
    
    circle_pts_3d = np.array(circle_pts_3d, dtype=np.float32)
    
    # 2. 3D -> 2D 投影
    # distCoeffs 设为 None，因为我们在 rectify 过的图像上画图
    pts_2d, _ = cv2.projectPoints(circle_pts_3d, rvec, tvec, mtx, None)
    
    # 转换为 int32 并 reshape 为 polylines 需要的格式 (N, 1, 2)
    return np.int32(pts_2d).reshape(-1, 1, 2)