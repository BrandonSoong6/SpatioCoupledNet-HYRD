import cv2
import numpy as np
import itertools
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import deque # <--- 新增：用于历史数据记录
import robot_control.config as cfg

# ===========================================
# Helper Functions (完全保持原样)
# ===========================================
def extract_global_ellipses(hsv_frame):
    yellow_mask = cv2.inRange(hsv_frame, cfg.YELLOW_LOWER, cfg.YELLOW_UPPER)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_ellipses = []
    for cnt in contours:
        if len(cnt) < 5: continue
        area = cv2.contourArea(cnt)
        if cfg.ELLIPSE_MIN_AREA < area < cfg.ELLIPSE_MAX_AREA:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area 
            if solidity > 0.5:
                ellipse = cv2.fitEllipse(cnt)
                (xc, yc), (ew, eh), angle = ellipse
                ellipse_area = (np.pi * ew * eh) / 4.0
                if ellipse_area == 0: continue
                fit_quality = float(area) / ellipse_area
                if fit_quality > 0.5:
                    if min(ew, eh) > 0:
                        ratio = min(ew, eh) / max(ew, eh)
                        if ratio > cfg.ELLIPSE_MIN_RATIO:
                            valid_ellipses.append({'params': ellipse, 'center': (int(xc), int(yc))})
    return yellow_mask, valid_ellipses

def get_contained_ellipses(roi_box_pts, all_ellipses):
    contained = []
    roi_cnt = np.int32(roi_box_pts)
    for item in all_ellipses:
        pt = item['center']
        if cv2.pointPolygonTest(roi_cnt, pt, False) >= 0:
            contained.append(item['params'])
    return contained

def get_contour_min_dist(cnt1, cnt2):
    poly1 = cv2.approxPolyDP(cnt1, 3.0, True).reshape(-1, 2)
    poly2 = cv2.approxPolyDP(cnt2, 3.0, True).reshape(-1, 2)
    dists = cdist(poly1, poly2)
    min_dist = np.min(dists)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    return min_dist, tuple(poly1[min_idx[0]]), tuple(poly2[min_idx[1]])

def check_circle_intersection(center, radius, dots):
    cx, cy = center
    r_sq = radius * radius
    for d in dots:
        dx = d['center'][0]
        dy = d['center'][1]
        dist_sq = (cx - dx)**2 + (cy - dy)**2
        if dist_sq <= r_sq:
            return True
    return False

# ===========================================
# Marker Tracker Logic
# ===========================================
class MarkerTracker:
    def __init__(self):
        self.is_initialized = False
        self.markers = {} 
        self.kf_dict = {}       # {id: KalmanFilter}
        self.lost_counters = {} # {id: int} 看门狗计数
        self.lost_frames_counter = 0 # 全局丢失计数
        
        # 鲁棒性参数
        self.max_gating_dist = 200  # 匹配距离阈值
        self.max_lost_watchdog = 5  # 单个ID允许连续预测的最大帧数
        
        # --- 新增：几何稳定性历史记录 ---
        self.geo_history = {} # {id: {'areas': deque, 'ratios': deque}}

    def reset(self):
        self.is_initialized = False
        self.markers = {}
        self.kf_dict = {}
        self.lost_counters = {}
        self.lost_frames_counter = 0
        self.geo_history = {} # 清空历史
        print("⚠️ Tracker Reset: Re-initializing geometry...")

    # --- 新增：KF 初始化辅助 ---
    def _init_kf(self, x, y):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # 过程噪声 (允许速度变化)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1 
        # 观测噪声 (极小 -> 有真值时完全信任真值)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01 
        kf.statePost = np.array([x, y, 0, 0], np.float32)
        return kf

    # --- 新增：几何偏移记录辅助 ---
    def _update_geometry_offset(self, rid, center, dots):
        if len(dots) == 3:
            c_arr = np.array(center)
            raw_pts = [np.array(d[0]) for d in dots] 
            offsets = [pt - c_arr for pt in raw_pts]
            if rid not in self.markers: self.markers[rid] = {}
            self.markers[rid]['last_offsets'] = offsets

    # --- 新增：几何突变检测 ---
    def _check_geo_stability(self, rid, current_area, current_ratio):
        if rid not in self.geo_history:
            self.geo_history[rid] = {
                'areas': deque(maxlen=10),
                'ratios': deque(maxlen=10)
            }
        
        hist = self.geo_history[rid]
        
        # 历史数据不足（启动阶段），直接信任
        if len(hist['areas']) < 3:
            hist['areas'].append(current_area)
            hist['ratios'].append(current_ratio)
            return True
        
        mean_area = np.mean(hist['areas'])
        mean_ratio = np.mean(hist['ratios'])
        
        if mean_area == 0: mean_area = 1e-5
        if mean_ratio == 0: mean_ratio = 1e-5
        
        dev_area = abs(current_area - mean_area) / mean_area
        dev_ratio = abs(current_ratio - mean_ratio) / mean_ratio
        
        # 突变判断：单项>20% 或 双项>10%
        is_unstable = (dev_area > 0.20) or \
                      (dev_ratio > 0.20) or \
                      (dev_area > 0.10 and dev_ratio > 0.10)
        
        if is_unstable:
            return False
        else:
            hist['areas'].append(current_area)
            hist['ratios'].append(current_ratio)
            return True

    # --- 原有辅助函数 (保持不变) ---
    def _sort_points_ccw(self, pts):
        pts = np.array(pts)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        return pts[sorted_idx]

    def _calc_triangle_area(self, pts):
        (x1, y1), (x2, y2), (x3, y3) = pts
        return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    def _calc_signed_geometry_signature(self, dots):
        raw_pts = [d[0] for d in dots]
        if len(raw_pts) != 3: return 0.5, None, None, None 
        
        pts = self._sort_points_ccw(raw_pts)
        dists = []
        for i in range(3):
            p_start = pts[i]
            p_end = pts[(i+1)%3]
            d = np.linalg.norm(p_end - p_start)
            mid = (p_start + p_end) / 2.0
            dists.append((d, i, (i+1)%3, mid))
            
        dists.sort(key=lambda x: x[0], reverse=True)
        longest_edge = dists[0] 
        idx_A = longest_edge[1]; idx_B = longest_edge[2]
        idx_C = ({0, 1, 2} - {idx_A, idx_B}).pop()
        A = pts[idx_A]; B = pts[idx_B]; C = pts[idx_C]
        midpoint = longest_edge[3]
        vec_AB = B - A; vec_AC = C - A
        denom = np.dot(vec_AB, vec_AB)
        if denom == 0: return 0.5, None, None, None
        t = np.dot(vec_AC, vec_AB) / denom
        projection_point = A + t * vec_AB
        return t, midpoint, projection_point, C

    def _vis_blend(self, bg, mask, color):
        colored = np.zeros_like(bg)
        colored[mask > 0] = color
        return cv2.addWeighted(bg, 0.7, colored, 0.3, 0)

    # ==========================================================
    # Process 函数 (保持原样，只字未改)
    # ==========================================================
    def process(self, hsv_frame, bgr_frame=None, return_debug=False):
        # 0. 基础特征提取
        global_yellow_mask, global_ellipses = extract_global_ellipses(hsv_frame)
        binary_roi_mask = cv2.inRange(hsv_frame, cfg.CYAN_LOWER, cfg.CYAN_UPPER)
        
        # === Step 1: 精英候选筛选 ===
        tmp_contours, _ = cv2.findContours(binary_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_candidates = [] 
        
        for i, cnt in enumerate(tmp_contours):
            # [规则1] 面积范围
            area = cv2.contourArea(cnt)
            if not (cfg.ROI_MIN_AREA <= area <= cfg.ROI_MAX_AREA): 
                continue 
            
            # 计算几何属性
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if min(w, h) == 0: continue
            
            # [规则2] 长宽比
            ratio = max(w, h) / min(w, h)
            if not (cfg.ROI_MIN_RATIO <= ratio <= cfg.ROI_MAX_RATIO):
                continue

            # [规则3] 实心率
            box = np.int0(cv2.boxPoints(rect))
            rect_area = w * h
            solidity = float(area) / rect_area
            if solidity < cfg.ROI_MIN_SOLIDITY: 
                continue
                
            # [规则4] 双黄点验证
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            search_radius = max(radius * 2.0, 40.0) 
            
            dots_nearby = 0
            search_sq = search_radius ** 2
            for el in global_ellipses:
                dist_sq = (cx - el['center'][0])**2 + (cy - el['center'][1])**2
                if dist_sq <= search_sq:
                    dots_nearby += 1
            
            if dots_nearby < 2:
                continue
                
            valid_candidates.append({'cnt': cnt, 'idx': i})

        # === Step 2: 循环焊接 ===
        current_count = len(valid_candidates)
        target_count = 5
        merge_ops = [] 

        if current_count > target_count:
            all_pairs = []
            n = current_count
            for i in range(n):
                for j in range(i + 1, n):
                    c1 = valid_candidates[i]['cnt']
                    c2 = valid_candidates[j]['cnt']
                    dist, pt1, pt2 = get_contour_min_dist(c1, c2)
                    if dist < cfg.MAX_MERGE_DIST:
                        all_pairs.append((dist, i, j, pt1, pt2))
            
            all_pairs.sort(key=lambda x: x[0])
            used_indices = set()
            
            for dist, idx_a, idx_b, pt1, pt2 in all_pairs:
                if current_count <= target_count: break
                if idx_a in used_indices or idx_b in used_indices: continue
                
                cv2.line(binary_roi_mask, pt1, pt2, 255, thickness=10)
                used_indices.add(idx_a); used_indices.add(idx_b)
                merge_ops.append((pt1, pt2))
                current_count -= 1

        if len(merge_ops) > 0:
            print(f">>> [WELDING] Active! Merged {len(merge_ops)} pairs. Count: {len(valid_candidates)} -> {current_count}")

        # === Step 3: 最终提取 (数据记录) ===
        final_contours, _ = cv2.findContours(binary_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        all_debug_boxes = [] 
        
        for cnt in final_contours:
            area = cv2.contourArea(cnt)
            if area < cfg.ROI_MIN_AREA: continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            rect_area = w * h
            solidity = 0.0
            if rect_area > 0:
                solidity = area / rect_area

            # Final Check
            search_radius = max(radius * 2.0, 40.0)
            search_sq = search_radius ** 2
            dots_nearby_count = 0
            for el in global_ellipses:
                dx = cx - el['center'][0]; dy = cy - el['center'][1]
                if (dx*dx + dy*dy) <= search_sq:
                    dots_nearby_count += 1
            
            if dots_nearby_count < 2: continue

            if w > h: 
                new_w, new_h = w * cfg.EXPAND_LONG_RATIO + cfg.ROI_EXPAND_FIXED, h * cfg.EXPAND_SHORT_RATIO + cfg.ROI_EXPAND_FIXED 
            else: 
                new_w, new_h = w * cfg.EXPAND_SHORT_RATIO + cfg.ROI_EXPAND_FIXED, h * cfg.EXPAND_LONG_RATIO + cfg.ROI_EXPAND_FIXED
            
            center, _, angle = rect
            rect_expanded = (center, (new_w, new_h), angle)
            box_expanded = np.int32(cv2.boxPoints(rect_expanded))
            
            contained_dots = get_contained_ellipses(box_expanded, global_ellipses)
            
            # 记录数据
            debug_info = {
                'box': box_expanded,
                'area': area,
                'solidity': solidity,
                'color': (0, 0, 255) # 默认为红
            }

            if len(contained_dots) >= 1:
                debug_info['color'] = (255, 0, 0) # 蓝
                all_debug_boxes.append(debug_info)
                
                candidates.append({
                    'box': box_expanded,
                    'center_x': center[0], 'center_y': center[1],
                    'dots_count': len(contained_dots), 'dots': contained_dots
                })
            else:
                all_debug_boxes.append(debug_info)

        status = self.update(candidates)

        result_output = {}
        for roi in candidates:
            if 'final_id' in roi and 'midpoint' in roi:
                result_output[roi['final_id']] = {
                    'center': (roi['center_x'], roi['center_y']),
                    'midpoint_uv': roi['midpoint'],
                    'proj_uv': roi.get('proj_point'),
                    'apex_uv': roi.get('apex_point'),
                    'dots': roi.get('final_dots', [])
                }
        
        # === Debug Image Generation (极值标注版) ===
        debug_images = None
        if return_debug and bgr_frame is not None:
            vis_raw = bgr_frame.copy()
            vis_yellow = self._vis_blend(bgr_frame, global_yellow_mask, (0, 255, 255))
            vis_mask = self._vis_blend(bgr_frame, binary_roi_mask, (255, 255, 0))

            if len(merge_ops) > 0:
                 for (p1, p2) in merge_ops:
                    cv2.line(vis_mask, p1, p2, (0, 0, 255), 3)

            # --- 4. Final Candidates (只标最大/最小) ---
            vis_candidates = bgr_frame.copy()
            
            # 找出极值索引
            idx_max_area = -1
            idx_min_area = -1
            idx_min_solidity = -1
            
            if len(all_debug_boxes) > 0:
                # 按面积排序找最大最小
                sorted_by_area = sorted(range(len(all_debug_boxes)), key=lambda i: all_debug_boxes[i]['area'])
                idx_min_area = sorted_by_area[0]
                idx_max_area = sorted_by_area[-1]
                
                # 按实心率找最小
                idx_min_solidity = min(range(len(all_debug_boxes)), key=lambda i: all_debug_boxes[i]['solidity'])

            for i, item in enumerate(all_debug_boxes):
                # 画框
                cv2.polylines(vis_candidates, [item['box']], True, item['color'], 2)
                
                # 只在极值项旁边写字
                labels = []
                if i == idx_max_area: labels.append(f"MAX_A:{int(item['area'])}")
                if i == idx_min_area: labels.append(f"MIN_A:{int(item['area'])}")
                if i == idx_min_solidity: labels.append(f"MIN_S:{item['solidity']:.2f}")
                
                if labels:
                    text = " | ".join(labels)
                    # 字体加大到 1.0，加粗
                    font_scale = 1.0
                    thickness = 2
                    
                    # 找绘制点 (Max/Min 位置错开)
                    pt = item['box'][1] # 默认右上角
                    if i == idx_min_area: 
                        pt = item['box'][3] # 最小的放左下角，防遮挡
                    
                    # 黑色描边 + 白色内芯
                    cv2.putText(vis_candidates, text, (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2)
                    cv2.putText(vis_candidates, text, (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

            # --- 5. Valid Ellipses (只标最大/最小) ---
            vis_ellipses = bgr_frame.copy()
            
            # 计算所有椭圆面积
            ellipse_data = []
            for el in global_ellipses:
                (ew, eh) = el['params'][1]
                e_area = (np.pi * ew * eh) / 4.0
                ellipse_data.append({'el': el, 'area': e_area})
            
            idx_max_e = -1
            idx_min_e = -1
            if len(ellipse_data) > 0:
                sorted_e = sorted(range(len(ellipse_data)), key=lambda i: ellipse_data[i]['area'])
                idx_min_e = sorted_e[0]
                idx_max_e = sorted_e[-1]

            for i, data in enumerate(ellipse_data):
                el = data['el']
                # 画椭圆
                cv2.ellipse(vis_ellipses, el['params'], (0, 0, 255), 2)
                
                # 只标最大最小
                if i == idx_max_e or i == idx_min_e:
                    prefix = "MAX" if i == idx_max_e else "MIN"
                    text_e = f"{prefix}:{int(data['area'])}"
                    center_pt = (int(el['center'][0]), int(el['center'][1]))
                    
                    # 大字体
                    cv2.putText(vis_ellipses, text_e, center_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
                    cv2.putText(vis_ellipses, text_e, center_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
            # 6. Tracking Result
            vis_result = bgr_frame.copy()
            for roi in candidates:
                if 'final_id' in roi and 'final_dots' in roi:
                    for pt in roi['final_dots']:
                        cv2.circle(vis_result, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                    pts = np.array(roi['final_dots'], np.int32).reshape((-1, 1, 2))
                    color = cfg.ID_COLORS.get(roi['final_id'], (255, 255, 255))
                    cv2.polylines(vis_result, [pts], True, color, 2)
                    
                    # 显示文字 (如果被预测/突变拦截，显示特殊标记)
                    display_text = f"ID{roi['final_id']}"
                    if roi.get('is_predicted', False):
                        display_text += " [P]"
                        
                    cv2.putText(vis_result, display_text, (int(roi['center_x']), int(roi['center_y'])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            debug_images = {
                "1. Raw Input": vis_raw,
                "2. Global Yellow": vis_yellow,
                "3. Merged Mask": vis_mask,
                "4. Final Candidates": vis_candidates,
                "5. Valid Ellipses": vis_ellipses,
                "6. Tracking Result": vis_result
            }

        return status, result_output, debug_images
 
    # =========================================================================
    # 核心修改：Update 函数 (混合状态机：同步/预测 + 突变检测)
    # =========================================================================
    def update(self, candidates):
        # [Step 0] 全局预处理：确保每个 ROI 只取最大的 3 个点
        for roi in candidates:
            if len(roi['dots']) > 3:
                roi['dots'].sort(key=lambda d: d[1][0] * d[1][1], reverse=True)
                roi['dots'] = roi['dots'][:3]

        # 筛选出几何信息完整的 ROI (3个点) 用于可能的几何排序
        # 这里的 candidates 包含当前帧所有有效的检测框
        geo_ready_cands = [c for c in candidates if len(c['dots']) == 3]
        
        # 筛选有效的用于匹配的 (至少1个点)
        valid_candidates = [c for c in candidates if len(c['dots']) >= 1]
        
        # 记录本帧发生突变的ID
        unstable_ids_detected = []

        # =====================================================================
        # 模式 A：看到 5 个点 -> 强制同步 (Sort & Sync)
        # =====================================================================
        if len(geo_ready_cands) == 5:
            signatures = []
            for i, roi in enumerate(geo_ready_cands):
                t, mid, proj, apex = self._calc_signed_geometry_signature(roi['dots'])
                raw_pts = [d[0] for d in roi['dots']]
                area = self._calc_triangle_area(raw_pts)
                roi['midpoint'], roi['proj_point'], roi['apex_point'] = mid, proj, apex
                signatures.append((t, i, area))
            
            signatures.sort(key=lambda x: x[0])
            sorted_indices = [x[1] for x in signatures]
            sorted_data = signatures
            
            # ID 映射 (t值从小到大对应 1,3,5,4,2)
            id_map = {
                sorted_indices[0]: 1, sorted_indices[1]: 3, 
                sorted_indices[2]: 5, sorted_indices[3]: 4, sorted_indices[4]: 2
            }
            
            self.geo_history = {} # Sync时重置历史
            
            matched_ids = set()
            for idx, real_id in id_map.items():
                roi = geo_ready_cands[idx]
                roi['final_id'] = real_id
                roi['is_predicted'] = False
                roi['final_dots'] = [d[0] for d in roi['dots']]
                
                # 获取 t 和 area
                curr_t = next(item[0] for item in sorted_data if item[1] == idx)
                curr_area = next(item[2] for item in sorted_data if item[1] == idx)
                
                # A1. 强行同步 KF
                if real_id not in self.kf_dict:
                    self.kf_dict[real_id] = self._init_kf(roi['center_x'], roi['center_y'])
                else:
                    # 重置状态到当前观测值 (消除累积误差)
                    self.kf_dict[real_id].statePost = np.array([roi['center_x'], roi['center_y'], 0, 0], np.float32)
                    self.kf_dict[real_id].errorCovPost = np.eye(4, dtype=np.float32) * 0.1 
                
                # A2. 更新几何偏移记忆 & 历史初始化
                self._update_geometry_offset(real_id, (roi['center_x'], roi['center_y']), roi['dots'])
                
                self.geo_history[real_id] = {
                    'areas': deque([curr_area], maxlen=10),
                    'ratios': deque([curr_t], maxlen=10)
                }
                
                # A3. 清零看门狗
                self.lost_counters[real_id] = 0
                matched_ids.add(real_id)
            
            self.is_initialized = True
            self.lost_frames_counter = 0
            return "Sync: 5 Markers"

        # =====================================================================
        # 模式 B：少于/多于 5 个点 -> 记忆追踪 (KF Predict + Hungarian)
        # =====================================================================
        elif self.is_initialized:
            # B1. KF 预测
            row_ids = list(self.kf_dict.keys())
            shadows = {}
            for rid in row_ids:
                pred = self.kf_dict[rid].predict()
                shadows[rid] = (pred[0, 0], pred[1, 0])

            # B2. 代价矩阵
            if len(valid_candidates) > 0:
                cost_matrix = cdist([shadows[rid] for rid in row_ids], 
                                    [(c['center_x'], c['center_y']) for c in valid_candidates])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                row_ind, col_ind = [], []

            matched_ids = set()
            
            # B3. 匹配成功的 -> 观测更新 (加入几何检测)
            for r_idx, c_idx in zip(row_ind, col_ind):
                if cost_matrix[r_idx, c_idx] < self.max_gating_dist:
                    rid = row_ids[r_idx]
                    roi = valid_candidates[c_idx]
                    
                    # [新增] 几何突变检测
                    if len(roi['dots']) == 3:
                        t_val, _, _, _ = self._calc_signed_geometry_signature(roi['dots'])
                        raw_pts = [d[0] for d in roi['dots']]
                        area_val = self._calc_triangle_area(raw_pts)
                        
                        is_stable = self._check_geo_stability(rid, area_val, t_val)
                        if not is_stable:
                            unstable_ids_detected.append(rid)
                            continue # 跳过更新，强制进入下面的预测分支
                    
                    meas = np.array([[np.float32(roi['center_x'])], [np.float32(roi['center_y'])]])
                    self.kf_dict[rid].correct(meas)
                    
                    roi['final_id'] = rid
                    roi['is_predicted'] = False
                    roi['final_dots'] = [d[0] for d in roi['dots']]
                    
                    if len(roi['final_dots']) == 3:
                        fake_dots = [(pt, (0,0), 0) for pt in roi['final_dots']]
                        _, mid, proj, apex = self._calc_signed_geometry_signature(fake_dots)
                        roi['midpoint'], roi['proj_point'], roi['apex_point'] = mid, proj, apex
                        self._update_geometry_offset(rid, (roi['center_x'], roi['center_y']), roi['dots'])
                    
                    self.lost_counters[rid] = 0
                    matched_ids.add(rid)

            # B4. 未匹配的 -> 预测补盲 (关键步骤：构造虚拟 ROI 塞回 candidates)
            for rid in row_ids:
                if rid not in matched_ids:
                    self.lost_counters[rid] = self.lost_counters.get(rid, 0) + 1
                    
                    if self.lost_counters[rid] <= self.max_lost_watchdog:
                        pred_x, pred_y = shadows[rid]
                        
                        # 几何重构
                        re_dots = []
                        mid, proj, apex = None, None, None
                        if rid in self.markers and 'last_offsets' in self.markers[rid]:
                            re_dots = [tuple(np.array([pred_x, pred_y]) + off) for off in self.markers[rid]['last_offsets']]
                            fake_dots = [(pt, (0,0), 0) for pt in re_dots]
                            _, mid, proj, apex = self._calc_signed_geometry_signature(fake_dots)
                        
                        # 【关键】构造虚拟 ROI 并 append 到 candidates
                        # 这样 Process 函数最后的 result_output 循环和 debug 循环就能看到它并画出来
                        virtual_roi = {
                            'center_x': pred_x, 'center_y': pred_y,
                            'final_id': rid, 
                            'is_predicted': True,
                            'final_dots': re_dots,
                            'midpoint': mid, 'proj_point': proj, 'apex_point': apex,
                            'dots': [], # 原始 dots 为空
                            'box': np.array([[int(pred_x), int(pred_y)]], dtype=np.int32) # Dummy box 避免绘图报错
                        }
                        candidates.append(virtual_roi)
            
            base_status = ""
            if len(matched_ids) >= 3:
                self.lost_frames_counter = 0
                base_status = f"Tracking: {len(matched_ids)} match"
            else:
                self.lost_frames_counter += 1
                if self.lost_frames_counter > cfg.MAX_LOST_FRAMES:
                    self.reset()
                    return "Resetting..."
                base_status = "Unstable"
            
            if unstable_ids_detected:
                base_status += f" | GEO_ERR: {unstable_ids_detected}"
                
            return base_status

        else:
            return "Init: Waiting..."