"""
空飘物检测器 - 整合所有检测算法
"""
import cv2
import numpy as np
import time
from collections import deque
from scipy import stats
from datetime import datetime
import json
import os

# 在 AirborneDetector 类之前添加

class SimpleImageQualityEvaluator:
    """简化的图片质量评估器，主要检测天黑"""
    
    def __init__(self):
        # 质量阈值
        self.thresholds = {
            'darkness': 0.6,       # 黑暗阈值（像素值<50的比例）
            'brightness': 0.15,    # 最低亮度要求
            'contrast': 0.08,      # 对比度阈值
            'min_valid_pixels': 0.5,  # 最小有效像素比例
        }
        
        self.last_evaluation = None
        
    def evaluate(self, frame):
        """评估图片质量，主要检测是否天黑"""
        if frame is None or frame.size == 0:
            return {'is_night': True, 'overall': 0.0, 'message': '无效图像'}
        
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # 1. 计算暗像素比例（像素值<50）
        dark_pixels = np.sum(gray < 50)
        total_pixels = gray.size
        darkness_ratio = dark_pixels / total_pixels
        
        # 2. 计算平均亮度
        brightness = np.mean(gray) / 255.0
        
        # 3. 计算对比度
        contrast = np.std(gray) / 255.0
        
        # 4. 计算有效像素比例（像素值在20-235之间）
        valid_pixels = np.sum((gray > 20) & (gray < 235))
        valid_ratio = valid_pixels / total_pixels
        
        # 判断是否天黑（主要条件）
        is_night = (
            darkness_ratio > self.thresholds['darkness'] or
            brightness < self.thresholds['brightness']
        )
        
        # 判断是否可用（附加条件）
        is_usable = (
            not is_night and
            contrast > self.thresholds['contrast'] and
            valid_ratio > self.thresholds['min_valid_pixels']
        )
        
        # 计算综合分数
        overall_score = 0.0
        if not is_night:
            # 基于亮度、对比度、有效像素计算分数
            brightness_score = min(1.0, brightness / 0.3)  # 亮度越高越好
            contrast_score = min(1.0, contrast / 0.15)     # 对比度越高越好
            valid_score = valid_ratio                      # 有效像素比例
            
            # 加权平均
            overall_score = (brightness_score * 0.4 + 
                           contrast_score * 0.3 + 
                           valid_score * 0.3)
        
        # 生成建议
        if is_night:
            message = "环境过暗（天黑或光照不足）"
        elif contrast < self.thresholds['contrast']:
            message = "对比度不足"
        elif valid_ratio < self.thresholds['min_valid_pixels']:
            message = "有效像素不足"
        else:
            message = f"图像质量良好 ({overall_score:.2f})"
        
        result = {
            'is_night': is_night,
            'is_usable': is_usable,
            'overall': overall_score,
            'message': message,
            'metrics': {
                'darkness': darkness_ratio,
                'brightness': brightness,
                'contrast': contrast,
                'valid_pixels': valid_ratio,
            }
        }
        
        self.last_evaluation = result
        return result
    
class AirborneDetector:
    def __init__(self):
        # ========== 检测参数（可在GUI中调整） ==========

        # 图片质量评估参数（新增）
        self.enable_quality_check = True  # 启用质量检查
        self.skip_night_frames = True     # 跳过天黑帧
        
        # 预处理参数
        self.downsample_ratio = 0.5          # 下采样比例
        self.use_sky_detection = True        # 启用天空检测
        self.enable_roi = True               # 启用ROI
        
        # 运动检测参数
        self.motion_method = 'frame_diff'    # 运动检测方法: frame_diff, mog2, knn, combine
        self.frame_diff_threshold = 5       # 帧差阈值
        self.min_motion_area = 5            # 最小运动区域面积
        self.max_motion_area = 200000          # 最大运动区域面积
        
        # 背景减法器参数
        self.bg_history = 100                # 背景历史帧数
        self.bg_var_threshold = 16           # MOG2方差阈值
        self.bg_dist_threshold = 400         # KNN距离阈值
        self.detect_shadows = True           # 检测阴影

        # 质量评估（新增）
        self.quality_evaluator = SimpleImageQualityEvaluator()
        self.night_frame_count = 0        # 天黑帧计数
        self.consecutive_night_frames = 0 # 连续天黑帧
        
        # 轨迹跟踪参数
        self.track_history_length = 10       # 轨迹历史长度 30
        self.min_track_duration = 3          # 最小轨迹持续时间 8
        self.max_track_speed = 100           # 最大跟踪速度 100
        self.tracker_type = 'simple'         # 跟踪器类型: simple, kalman
        
        # 空飘物判定阈值
        self.thresholds = {
            'min_duration': 3,               # 增加最小持续时间，减少误检
            'min_speed_consistency': 0.1,    # 提高速度一致性要求
            'min_direction_consistency': 0.1, # 提高方向一致性要求
            'min_area_stability': 0.1,       # 提高面积稳定性要求
            'min_linearity': 0.1,            # 提高轨迹线性度要求
            'speed_range': (2, 60),          # 调整合理速度范围
            'min_confidence': 0.1            # 提高最小置信度，减少误报
        }
        
        # ========== 算法状态变量 ==========
        self.prev_frame = None
        self.background_model = None
        self.sky_mask = None
        self.roi_mask = None
        
        # 轨迹跟踪
        self.tracks = {}
        self.next_track_id = 0
        self.finished_tracks = []
        self.active_tracks = 0
        
        # 检测统计
        self.detection_count = 0
        self.last_detection_time = None
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_times = deque(maxlen=30)
        
        # 性能监控
        self.current_fps = 0
        self.processing_time = 0
        self.detection_history = deque(maxlen=100)
        
        # 初始化背景减法器
        self._init_background_subtractors()
        
        # 输出结果
        self.last_detections = []
        self.last_motion_mask = None
        self.last_debug_info = {}
    
    def _init_background_subtractors(self):
        """初始化背景减法器"""
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history,
            varThreshold=self.bg_var_threshold,
            detectShadows=self.detect_shadows
        )
        
        self.knn = cv2.createBackgroundSubtractorKNN(
            history=self.bg_history,
            dist2Threshold=self.bg_dist_threshold,
            detectShadows=self.detect_shadows
        )
    
    def check_image_quality(self, frame):
        """检查图片质量，主要检测是否天黑"""
        if not self.enable_quality_check:
            return True, "质量检查已禁用"
        
        result = self.quality_evaluator.evaluate(frame)
        
        if result['is_night'] and self.skip_night_frames:
            self.night_frame_count += 1
            self.consecutive_night_frames += 1
            return False, result['message']
        else:
            self.consecutive_night_frames = 0
            return True, result['message']
    
    def get_quality_info(self):
        """获取质量信息"""
        if self.quality_evaluator.last_evaluation is None:
            return {
                'is_night': False,
                'overall': 0.0,
                'message': '未评估',
                'night_frames': self.night_frame_count
            }
        
        result = self.quality_evaluator.last_evaluation.copy()
        result['night_frames'] = self.night_frame_count
        result['consecutive_night_frames'] = self.consecutive_night_frames
        
        return result
    
    def update_parameters(self, **kwargs):
        """更新检测参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.thresholds:
                if key == 'speed_range':
                    # 确保速度范围有效
                    if isinstance(value, tuple) and len(value) == 2:
                        if value[0] < value[1]:
                            self.thresholds[key] = value
                else:
                    self.thresholds[key] = value
        
        # 重新初始化背景减法器
        self._init_background_subtractors()
    
    def reset(self):
        """重置检测器状态"""
        self.prev_frame = None
        self.background_model = None
        self.sky_mask = None
        self.roi_mask = None
        self.tracks = {}
        self.next_track_id = 0
        self.finished_tracks = []
        self.active_tracks = 0
        self.detection_count = 0
        self.last_detection_time = None
        self.last_detections = []
        self.last_motion_mask = None
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_times.clear()
        self.detection_history.clear()
        
        # 重新初始化背景减法器
        self._init_background_subtractors()
    
    def detect_sky_region(self, frame):
        """检测天空区域"""
        if not self.use_sky_detection:
            h, w = frame.shape[:2]
            self.sky_mask = np.ones((h, w), dtype=np.uint8) * 255
            return self.sky_mask
        
        h, w = frame.shape[:2]
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 天空颜色范围（蓝色和白色）
        lower_blue = np.array([90, 40, 100])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # 合并天空区域
        sky_mask = cv2.bitwise_or(mask_blue, mask_white)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
        sky_mask = cv2.dilate(sky_mask, kernel, iterations=1)
        
        # 确保天空区域连通
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sky_mask, 8, cv2.CV_32S)
        
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                max_area_idx = np.argmax(areas) + 1
                sky_mask = np.zeros_like(sky_mask)
                sky_mask[labels == max_area_idx] = 255
        
        self.sky_mask = sky_mask
        return sky_mask
    
    def create_roi_mask(self, frame_shape, roi_points):
        """创建ROI掩码"""
        if roi_points is None or len(roi_points) < 3:
            return None
        
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        points = np.array(roi_points, np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        self.roi_mask = mask
        return mask
    
    def fast_frame_difference(self, current_frame):
        """快速帧差法"""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return np.zeros(current_frame.shape[:2], dtype=np.uint8)
        
        # 下采样加速
        if self.downsample_ratio < 1.0:
            h, w = current_frame.shape[:2]
            new_w = int(w * self.downsample_ratio)
            new_h = int(h * self.downsample_ratio)
            small_current = cv2.resize(current_frame, (new_w, new_h))
            small_prev = cv2.resize(self.prev_frame, (new_w, new_h))
        else:
            small_current = current_frame.copy()
            small_prev = self.prev_frame.copy()
        
        # 转换为灰度
        gray_current = cv2.cvtColor(small_current, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        gray_current = cv2.GaussianBlur(gray_current, (3, 3), 0)
        gray_prev = cv2.GaussianBlur(gray_prev, (3, 3), 0)
        
        # 计算帧差
        diff = cv2.absdiff(gray_current, gray_prev)
        
        # 阈值处理
        _, motion_mask = cv2.threshold(diff, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
        
        # 上采样回原尺寸
        if self.downsample_ratio < 1.0:
            motion_mask = cv2.resize(motion_mask, (w, h))
        
        self.prev_frame = current_frame.copy()
        return motion_mask
    
    def apply_background_subtraction(self, frame, method='mog2'):
        """应用背景减法"""
        if method == 'mog2':
            fg_mask = self.mog2.apply(frame)
        elif method == 'knn':
            fg_mask = self.knn.apply(frame)
        else:
            # 默认使用MOG2
            fg_mask = self.mog2.apply(frame)
        
        # 去除阴影（如果检测阴影）
        if self.detect_shadows:
            fg_mask[fg_mask == 127] = 0
        
        return fg_mask
    
    def detect_motion_regions(self, motion_mask, roi_mask=None):
        """从运动掩码中提取运动区域"""
        # 应用ROI掩码
        if roi_mask is not None and self.enable_roi:
            motion_mask = cv2.bitwise_and(motion_mask, roi_mask)
        
        # 查找轮廓
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_motion_area < area < self.max_motion_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算中心点
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # 检查中心点是否在ROI内
                in_roi = True
                if roi_mask is not None and self.enable_roi:
                    # 确保坐标在图像范围内
                    if 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]:
                        in_roi = roi_mask[cy, cx] > 0
                    else:
                        in_roi = False
                
                # 过滤过于细长的区域
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0 and in_roi:
                    regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (cx, cy),
                        'contour': contour,
                        'timestamp': time.time()
                    })
        return regions
    
    def update_tracks(self, regions, frame_timestamp):
        """更新轨迹跟踪"""
        # 只处理在ROI内的区域
        filtered_regions = []
        # TODO:应该将区域中心、颜色相近的区域合并为同一个
        for region in regions:
            # 检查区域中心是否在ROI内
            if self.roi_mask is not None and self.enable_roi:
                cx, cy = region['center']
                if 0 <= cy < self.roi_mask.shape[0] and 0 <= cx < self.roi_mask.shape[1]:
                    if self.roi_mask[cy, cx] > 0:
                        filtered_regions.append(region)
            else:
                filtered_regions.append(region)
        
        regions = filtered_regions
    
        if not regions:
            # 清理丢失的轨迹
            self._cleanup_lost_tracks(frame_timestamp)
            return []
        
        # 数据关联
        updated_tracks = {}
        used_regions = set()
        
        # 首先尝试匹配现有轨迹
        for track_id, track in self.tracks.items():
            if not track['centers']:
                continue
                
            last_center = track['centers'][-1]
            min_distance = float('inf')
            best_region_idx = -1
            
            for i, region in enumerate(regions):
                if i in used_regions:
                    continue
                    
                center = region['center']
                distance = np.sqrt((center[0] - last_center[0])**2 + 
                                 (center[1] - last_center[1])**2)
                
                # 距离和速度约束
                if distance < min_distance and distance < self.max_track_speed * 2:
                    min_distance = distance
                    best_region_idx = i
            
            if best_region_idx != -1:
                region = regions[best_region_idx]
                used_regions.add(best_region_idx)
                
                # 更新轨迹
                track['centers'].append(region['center'])
                track['bboxes'].append(region['bbox'])
                track['areas'].append(region['area'])
                track['timestamps'].append(frame_timestamp)
                
                # 限制轨迹长度
                if len(track['centers']) > self.track_history_length:
                    track['centers'] = track['centers'][-self.track_history_length:]
                    track['bboxes'] = track['bboxes'][-self.track_history_length:]
                    track['areas'] = track['areas'][-self.track_history_length:]
                
                # 计算运动特征
                if len(track['centers']) >= 2:
                    self._calculate_motion_features(track)
                
                updated_tracks[track_id] = track
            else:
                # 轨迹暂时丢失，保留一段时间
                if frame_timestamp - track['timestamps'][-1] < 1.0:  # 1秒内
                    updated_tracks[track_id] = track
        
        # 为未匹配的区域创建新轨迹
        for i, region in enumerate(regions):
            if i not in used_regions:
                new_track = {
                    'centers': [region['center']],
                    'bboxes': [region['bbox']],
                    'areas': [region['area']],
                    'timestamps': [frame_timestamp],
                    'start_time': frame_timestamp,
                    'speed_history': [],
                    'direction_history': [],
                    'status': 'new'
                }
                updated_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        self.tracks = updated_tracks
        self.active_tracks = len(self.tracks)
        
        # 分析有效轨迹
        valid_objects = self._analyze_valid_tracks()
        
        return valid_objects
    
    def _calculate_motion_features(self, track):
        """计算运动特征"""
        centers = track['centers']
        if len(centers) < 2:
            return
        
        # 计算速度和方向
        dx = centers[-1][0] - centers[-2][0]
        dy = centers[-1][1] - centers[-2][1]
        speed = np.sqrt(dx*dx + dy*dy)
        direction = np.arctan2(dy, dx) if dx != 0 else 0
        
        track['speed_history'].append(speed)
        track['direction_history'].append(direction)
        
        # 限制历史长度
        if len(track['speed_history']) > 20:
            track['speed_history'] = track['speed_history'][-20:]
            track['direction_history'] = track['direction_history'][-20:]
    
    def _analyze_trajectory_features(self, track):
        """分析轨迹特征"""
        if len(track['centers']) < 3:
            return None
        
        centers = np.array(track['centers'])
        areas = np.array(track['areas'])
        
        # 1. 速度一致性
        if track['speed_history']:
            speeds = np.array(track['speed_history'])
            speed_mean = np.mean(speeds)
            speed_std = np.std(speeds)
            speed_consistency = 1 - (speed_std / (speed_mean + 1e-6)) if speed_mean > 0 else 0
            speed_consistency = max(0, min(1, speed_consistency))
        else:
            speed_consistency = 0
        
        # 2. 方向一致性
        if track['direction_history']:
            directions = np.array(track['direction_history'])
            direction_variance = self._circular_variance(directions)
            direction_consistency = 1 - direction_variance
        else:
            direction_consistency = 0
        
        # 3. 面积稳定性
        area_mean = np.mean(areas)
        area_std = np.std(areas)
        area_stability = 1 - (area_std / (area_mean + 1e-6)) if area_mean > 0 else 0
        area_stability = max(0, min(1, area_stability))
        
        # 4. 轨迹线性度
        if len(centers) >= 5:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    centers[:, 0], centers[:, 1]
                )
                linearity = r_value**2
            except:
                linearity = 0
        else:
            linearity = 0
        
        # 5. 持续时间
        duration = len(track['centers'])
        
        # 6. 平均速度
        mean_speed = np.mean(track['speed_history']) if track['speed_history'] else 0
        
        # 7. 计算置信度（综合评分）
        confidence = (speed_consistency + direction_consistency + area_stability + linearity) / 4
        
        return {
            'speed_consistency': speed_consistency,
            'direction_consistency': direction_consistency,
            'area_stability': area_stability,
            'linearity': linearity,
            'duration': duration,
            'mean_speed': mean_speed,
            'confidence': confidence,
            'centers': centers,
            'areas': areas
        }
    
    def _is_airborne_object(self, features):
        """判断是否为空飘物"""
        thresholds = self.thresholds
        
        # 基本条件检查
        if features['duration'] < thresholds['min_duration']:
            return False
        
        # 速度范围检查
        if not (thresholds['speed_range'][0] <= features['mean_speed'] <= thresholds['speed_range'][1]):
            return False
        
        # 置信度检查
        if features['confidence'] < thresholds['min_confidence']:
            return False
        
        # 综合条件检查
        conditions = [
            features['speed_consistency'] >= thresholds['min_speed_consistency'],
            features['direction_consistency'] >= thresholds['min_direction_consistency'],
            features['area_stability'] >= thresholds['min_area_stability'],
            features['linearity'] >= thresholds['min_linearity'],
        ]
        
        # 至少满足3个条件
        return sum(conditions) >= 3
    
    def _analyze_valid_tracks(self):
        """分析有效轨迹"""
        valid_objects = []
        
        for track_id, track in self.tracks.items():
            features = self._analyze_trajectory_features(track)
            if features is None:
                continue
            
            # 判断是否为空飘物
            if self._is_airborne_object(features):
                valid_objects.append({
                    'id': track_id,
                    'bbox': track['bboxes'][-1],
                    'center': track['centers'][-1],
                    'features': features,
                    'track_length': len(track['centers']),
                    'status': 'confirmed'
                })
                track['status'] = 'confirmed'
            elif features['duration'] > self.thresholds['min_duration']:
                # 不满足条件但持续时间足够长，标记为可疑
                track['status'] = 'suspicious'
        
        return valid_objects
    
    def _cleanup_lost_tracks(self, current_time):
        """清理丢失的轨迹"""
        lost_tracks = []
        
        for track_id, track in self.tracks.items():
            if track['timestamps']:
                last_time = track['timestamps'][-1]
                if current_time - last_time > 1.0:  # 1秒未更新
                    lost_tracks.append(track_id)
        
        for track_id in lost_tracks:
            self.finished_tracks.append(self.tracks[track_id])
            del self.tracks[track_id]
        
        self.active_tracks = len(self.tracks)
    
    def _circular_variance(self, angles):
        """计算角度方差"""
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
        return 1 - R
    
    def fixed_size_motion_mask(self, motion_mask, max_growth_ratio=2.5):
        """限制运动掩码的最大尺寸"""
        if motion_mask is None or self.last_motion_mask is None:
            return motion_mask
        
        # 计算当前掩码和上一帧掩码的面积
        current_area = np.sum(motion_mask > 0)
        last_area = np.sum(self.last_motion_mask > 0)
        
        # 如果面积增长超过阈值，进行限制
        if last_area > 0 and current_area / last_area > max_growth_ratio:
            # 使用上一帧的掩码作为基础
            base_mask = self.last_motion_mask.copy()
            
            # 对当前掩码进行腐蚀，缩小区域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            eroded = cv2.erode(motion_mask, kernel, iterations=2)
            
            # 与基础掩码结合
            combined = cv2.bitwise_or(base_mask, eroded)
            
            # 形态学平滑
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
            return combined
        
        return motion_mask
    
    def detect(self, frame, roi_points=None):
        """主检测函数"""
        frame_start_time = time.time()
        self.frame_count += 1

        # 0. 图片质量检查（新增）
        if self.enable_quality_check:
            quality_ok, quality_message = self.check_image_quality(frame)
            if not quality_ok and self.skip_night_frames:
                # 如果是天黑，返回空结果，不进行检测
                self.last_debug_info = {
                    'frame_count': self.frame_count,
                    'detection_count': 0,
                    'fps': self.current_fps,
                    'processing_time_ms': self.processing_time,
                    'active_tracks': self.active_tracks,
                    'total_detections': self.detection_count,
                    'motion_regions': 0,
                    'quality_message': quality_message,
                    'is_night': True
                }
                
                # 性能计算
                frame_time = time.time() - frame_start_time
                self.frame_times.append(frame_time)
                self.processing_time = frame_time * 1000
                
                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                return [], None
        
        # 1. 创建ROI掩码
        roi_mask = None
        if roi_points is not None and len(roi_points) >= 3:
            roi_mask = self.create_roi_mask(frame.shape, roi_points)
        
        # 2. 天空检测
        if self.use_sky_detection:
            sky_mask = self.detect_sky_region(frame)
            if roi_mask is not None and self.enable_roi:
                roi_mask = cv2.bitwise_and(roi_mask, sky_mask)  # 必须掩码在天空范围内才可以
            elif self.enable_roi:
                roi_mask = sky_mask
        
        # 3. 运动检测
        motion_mask = None
        
        # 3. 运动检测
        if self.motion_method == 'frame_diff':
            motion_mask = self.fast_frame_difference(frame)
        elif self.motion_method == 'mog2':
            motion_mask = self.apply_background_subtraction(frame, 'mog2')
            # 限制背景减法器的学习
            motion_mask = self.limit_background_growth(motion_mask)
        elif self.motion_method == 'knn':
            motion_mask = self.apply_background_subtraction(frame, 'knn')
            motion_mask = self.limit_background_growth(motion_mask)
        elif self.motion_method == 'combine':
            frame_diff = self.fast_frame_difference(frame)
            bg_mask = self.apply_background_subtraction(frame, 'mog2')
            bg_mask = self.limit_background_growth(bg_mask)
            motion_mask = cv2.bitwise_and(frame_diff, bg_mask)
        
        # 4. 限制掩码尺寸
        motion_mask = self.fixed_size_motion_mask(motion_mask)
        
        # # 4. 应用ROI掩码
        # if roi_mask is not None and self.enable_roi:
        #     motion_mask = cv2.bitwise_and(motion_mask, roi_mask)
        
        # 5. 提取运动区域
        regions = self.detect_motion_regions(motion_mask)
        
        # 6. 轨迹跟踪与分析
        timestamp = time.time()
        detections = self.update_tracks(regions, timestamp)
        
        # 7. 更新检测统计
        if detections:
            self.detection_count += len(detections)
            self.last_detection_time = datetime.now()
            self.detection_history.append(len(detections))
        
        # 8. 性能计算
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        self.processing_time = frame_time * 1000  # 转换为毫秒
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # 9. 保存结果
        self.last_detections = detections
        self.last_motion_mask = motion_mask
        
        # 10. 收集调试信息
        self.last_debug_info = {
            'frame_count': self.frame_count,
            'detection_count': len(detections),
            'fps': self.current_fps,
            'processing_time_ms': self.processing_time,
            'active_tracks': self.active_tracks,
            'total_detections': self.detection_count,
            'motion_regions': len(regions),
            'motion_method': self.motion_method,
            'roi_enabled': self.enable_roi and roi_points is not None,
            'sky_detection': self.use_sky_detection,
            'quality_message': quality_message if 'quality_message' in locals() else '质量检查通过',
            'is_night': False
        }
        
        return detections, motion_mask

    def limit_background_growth(self, bg_mask, max_change_ratio=0.3):
        """限制背景减法器的输出变化"""
        if self.last_motion_mask is None:
            return bg_mask
        
        # 计算变化部分
        change_mask = cv2.absdiff(bg_mask, self.last_motion_mask)
        change_area = np.sum(change_mask > 0)
        last_area = np.sum(self.last_motion_mask > 0)
        
        # 如果变化过大，抑制新区域
        if last_area > 0 and change_area / last_area > max_change_ratio:
            # 保留大部分原有区域，只添加小部分新区域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_regions = cv2.erode(change_mask, kernel, iterations=1)
            
            # 结合原有掩码和有限的新区域
            result = cv2.bitwise_or(self.last_motion_mask, new_regions)
            return result
        
        return bg_mask
    
    def draw_detections(self, frame, detections):
        """绘制检测结果"""
        result_frame = frame.copy()
        
        # 定义颜色方案
        colors = {
            'confirmed': (0, 255, 0),      # 绿色 - 确认的目标
            'suspicious': (0, 255, 255),   # 黄色 - 可疑目标
            'new': (255, 0, 0),           # 蓝色 - 新目标
            'track': (255, 255, 0),       # 青色 - 轨迹线
            'text': (255, 255, 255)       # 白色 - 文本
        }
        
        for detection in detections:
            status = detection.get('status', 'confirmed')
            color = colors.get(status, colors['confirmed'])
            
            # 绘制边界框
            x, y, w, h = detection['bbox']
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
            
            # 绘制中心点
            center = detection['center']
            cv2.circle(result_frame, center, 4, color, -1)
            
            # 绘制ID和置信度
            features = detection['features']
            confidence = features.get('confidence', 0) * 100
            cv2.putText(result_frame, f"ID:{detection['id']}", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(result_frame, f"{confidence:.0f}%", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制轨迹
            if 'centers' in features:
                centers = features['centers']
                for j in range(1, len(centers)):
                    cv2.line(result_frame, 
                            tuple(centers[j-1].astype(int)), 
                            tuple(centers[j].astype(int)), 
                            colors['track'], 1)
        
        # 绘制统计信息
        self._draw_statistics(result_frame)
        
        return result_frame
    
    def _draw_statistics(self, frame):
        """绘制统计信息到帧上"""
        h, w = frame.shape[:2]
        
        # 创建半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文本信息
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Detections: {len(self.last_detections)}",  # 英文
            f"Total: {self.detection_count}",           # 英文
            f"Active Tracks: {self.active_tracks}",     # 英文
            f"Process Time: {self.processing_time:.1f} ms",  # 英文
            f"Method: {self.motion_method}"              # 英文
        ]
        
        y_pos = 35
        for line in info_lines:
            cv2.putText(frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_pos += 20
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'total_frames': self.frame_count,
            'total_detections': self.detection_count,
            'current_fps': self.current_fps,
            'processing_time_ms': self.processing_time,
            'active_tracks': self.active_tracks,
            'finished_tracks': len(self.finished_tracks),
            'current_detections': len(self.last_detections),
            'motion_method': self.motion_method,
            'last_detection_time': self.last_detection_time,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def get_detection_details(self):
        """获取检测详情"""
        details = []
        for detection in self.last_detections:
            features = detection['features']
            details.append({
                'id': detection['id'],
                'bbox': detection['bbox'],
                'center': detection['center'],
                'track_length': detection['track_length'],
                'speed_consistency': features['speed_consistency'],
                'direction_consistency': features['direction_consistency'],
                'area_stability': features['area_stability'],
                'linearity': features['linearity'],
                'mean_speed': features['mean_speed'],
                'confidence': features['confidence']
            })
        return details
    
    def save_configuration(self, filepath):
        """保存配置到文件"""
        config = {
            'downsample_ratio': self.downsample_ratio,
            'use_sky_detection': self.use_sky_detection,
            'enable_roi': self.enable_roi,
            'motion_method': self.motion_method,
            'frame_diff_threshold': self.frame_diff_threshold,
            'min_motion_area': self.min_motion_area,
            'max_motion_area': self.max_motion_area,
            'bg_history': self.bg_history,
            'bg_var_threshold': self.bg_var_threshold,
            'bg_dist_threshold': self.bg_dist_threshold,
            'detect_shadows': self.detect_shadows,
            'track_history_length': self.track_history_length,
            'min_track_duration': self.min_track_duration,
            'max_track_speed': self.max_track_speed,
            'tracker_type': self.tracker_type,
            'thresholds': self.thresholds
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return True
    
    def load_configuration(self, filepath):
        """从文件加载配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 应用配置
            self.update_parameters(**config)
            
            # 重新初始化背景减法器
            self._init_background_subtractors()
            
            return True
        except Exception as e:
            print(f"加载配置失败: {e}")
            return False