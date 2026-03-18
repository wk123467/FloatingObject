"""
空飘物检测系统 GUI界面 - 精简版本
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import cv2
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import json
import os
import time
from airborne_detector import AirborneDetector

class AirborneDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("北京北站空飘物智能检测系统")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        self.setup_styles()
        self.detector = AirborneDetector()
        self.detector.use_sky_detection = False
        
        # 视频相关
        self.cap = None
        self.is_running = False
        self.update_job = None
        
        # 区域相关
        self.selected_region = None
        self.region_points = []
        
        # 相机相关
        self.camera_list = []  # 完全清空，不保留任何默认相机
        self.camera_names = []
        self.camera_alert_status = {}
        self.current_camera_name = ""
        
        # 截图管理
        self.screenshots = []
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # 区域编辑
        self._editor_window = None
        self.editor_frame = None
        self.region_points = []
        self.preview_point = None
        self.point_history = []
        self.redo_stack = []
        
        # 事件管理
        self.events = []
        self.last_event_time = None
        self.event_interval = 30
        self.current_event_id = 0
        self.alert_enabled = True
        self.alert_windows = {}
        self.unread_alerts = {}
        
        # 警报目录
        self.real_alerts_dir = "alerts/real"
        self.false_alerts_dir = "alerts/false"
        for d in [self.real_alerts_dir, self.false_alerts_dir]:
            os.makedirs(d, exist_ok=True)
        
        # 相似度配置
        self.similarity_threshold = 0.6
        self.recent_time_threshold = 120
        
        # 地图相关
        self.map_lines = []
        self.is_edit_mode = False
        self.is_line_mode = False
        self.line_start = None
        self.camera_markers = {}
        
        self.create_widgets()
        self.update_all_displays()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(1000, self.update_alert_indicator)

    # ========== 工具函数 ==========
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        self.colors = {'bg': '#2c3e50', 'fg': '#ecf0f1', 'accent': '#3498db',
                      'success': '#2ecc71', 'warning': '#f39c12', 'danger': '#e74c3c'}
        style.configure('Title.TLabel', font=('Microsoft YaHei', 16, 'bold'))
        style.configure('Normal.TLabel', font=('Microsoft YaHei', 10))
        self.root.configure(bg=self.colors['bg'])

    def log_message(self, msg, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"error": "[错误]", "warning": "[警告]"}.get(level, "[信息]")
        self.log_text.insert(tk.END, f"[{timestamp}] {prefix} {msg}\n")
        self.log_text.see(tk.END)
        if int(self.log_text.index('end-1c').split('.')[0]) > 200:
            self.log_text.delete(1.0, "2.0")

    def get_safe_filename(self, name):
        return re.sub(r'[<>:"/\\|?*]', '_', name) if 're' in globals() else name

    # ========== 相机标记统一绘制 ==========
    def _draw_camera_markers(self, style="parallel"):
        for tag in ["camera", "camera_icon", "camera_label"]:
            self.map_canvas.delete(tag)
        
        self.camera_markers.clear()
        for cam in self.camera_list:
            name, x, y = cam["name"], *cam.get("position", (150, 80))
            is_alert = self.camera_alert_status.get(name, False)
            is_selected = (self.camera_var.get() == name)
            
            if is_alert:
                color, outline, tcolor = "#ff0000", "#990000", "white"
            elif is_selected and style == "parallel":
                color, outline, tcolor = "#ffff00", "#cccc00", "black"
            else:
                color, outline, tcolor = "#808080", "#404040", "#e0e0e0"
            
            marker = self.map_canvas.create_oval(x-12, y-12, x+12, y+12, fill=color,
                       outline=outline, width=2, tags=("camera", f"camera_{name}"))
            icon = self.map_canvas.create_text(x, y-2, text="📷", fill=tcolor,
                      font=("Arial", 10), tags=("camera_icon", f"icon_{name}"))
            label = self.map_canvas.create_text(x, y+18, text=name[:6], fill=tcolor,
                      font=("Courier", 7, "bold"), tags=("camera_label", f"label_{name}"))
            self.camera_markers[name] = {'marker': marker, 'icon': icon, 'label': label}

    def draw_parallel_circuit_camera_markers(self):
        self._draw_camera_markers("parallel")

    def draw_camera_markers(self):
        self._draw_camera_markers("normal")

    # ========== 闪烁统一 ==========
    def _flash_camera(self, name, count=4):
        if name not in self.camera_markers:
            return
        marker = self.camera_markers[name]['marker']
        def toggle(step=0):
            if step >= count:
                color = "#ff0000" if self.camera_alert_status.get(name) else "#808080"
                self.map_canvas.itemconfig(marker, fill=color)
                return
            self.map_canvas.itemconfig(marker, fill=["#ff9900", "#ff0000"][step % 2])
            self.root.after(200, lambda: toggle(step + 1))
        toggle()

    def flash_compact_circuit_camera_marker(self, name):
        self._flash_camera(name, 4)

    def flash_parallel_circuit_camera_marker(self, name):
        self._flash_camera(name, 4)

    def flash_camera_marker(self, name):
        self._flash_camera(name, 6)

    # ========== 特征提取与相似度 ==========
    def _extract_features(self, img):
        if img is None or img.size == 0:
            return None
        feat = {}
        if len(img.shape) == 3:
            color = []
            for i in range(3):
                h = cv2.calcHist([img], [i], None, [16], [0, 256])
                color.extend(cv2.normalize(h, h).flatten())
            feat['color'] = np.array(color)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hists = []
            for i, bins in enumerate([12, 8, 8]):
                h = cv2.calcHist([hsv], [i], None, [bins], [0, 180 if i==0 else 256])
                hists.append(cv2.normalize(h, h).flatten())
            feat['hsv'] = np.hstack(hists)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        if mag.max() > 0:
            mag = (mag / mag.max() * 255).astype(np.uint8)
            h = cv2.calcHist([mag], [0], None, [8], [0, 256])
            feat['texture'] = cv2.normalize(h, h).flatten()
        else:
            feat['texture'] = np.zeros(8)
        return feat

    def _calc_similarity(self, f1, f2):
        if not f1 or not f2:
            return 0.0
        weights = {'color': 0.4, 'hsv': 0.3, 'texture': 0.3}
        total = wsum = 0.0
        for k, w in weights.items():
            if k in f1 and k in f2:
                a = f1[k][:min(len(f1[k]), len(f2[k]))]
                b = f2[k][:min(len(f1[k]), len(f2[k]))]
                if len(a) == 0:
                    continue
                sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                total += max(0.0, min(1.0, sim)) * w
                wsum += w
        return total / wsum if wsum > 0 else 0.0

    def _extract_image_features(self, img):
        return self._extract_features(img)

    def _calculate_image_similarity(self, f1, f2):
        return self._calc_similarity(f1, f2)

    # ========== 区域编辑器显示统一 ==========
    def _update_display_base(self, frame, points, preview=None):
        display = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        for i, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 6, (255, 0, 0), -1)
            cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(display, str(i+1), (x+12, y-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], False, (0, 255, 0), 2)
        if len(points) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0, 128))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        if preview and points:
            cv2.line(display, points[-1], preview, (255, 255, 0), 2)
        return display

    def update_editor_display(self):
        if not hasattr(self, 'editor_frame') or self.editor_frame is None:
            return
        display = self._update_display_base(self.editor_frame, self.region_points, self.preview_point)
        img = Image.fromarray(display)
        photo = ImageTk.PhotoImage(img)
        if self.region_image_label:
            self.region_image_label.config(image=photo)
            self.region_image_label.image = photo
        self._update_editor_info()

    # ========== 界面创建 ==========
    def create_widgets(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(main, text="北京北站空飘物智能检测系统", style='Title.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="v1.0", style='Normal.TLabel').grid(row=0, column=1, sticky="w", padx=10)

        content = ttk.Frame(main)
        content.grid(row=1, column=0, columnspan=3, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(content)
        right.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self._create_left_panel(left)
        self._create_right_panel(right)
        self._create_status_bar(main)

    def _create_left_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="视频显示", padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(frame, text="等待启动检测...", anchor="center",
                                     relief="solid", background="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=2)

        mask_frame = ttk.LabelFrame(frame, text="运动检测掩码", padding="5")
        mask_frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        self.mask_label = ttk.Label(mask_frame, text="运动掩码", anchor="center",
                                    relief="solid", background="black")
        self.mask_label.grid(row=0, column=0, sticky="nsew")
        mask_frame.columnconfigure(0, weight=1)
        mask_frame.rowconfigure(0, weight=1)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=2, column=0, pady=(10, 0))
        self.start_btn = ttk.Button(ctrl, text="▶ 开始检测", command=self.start_selected_camera)
        self.start_btn.grid(row=0, column=0, padx=(0, 5))
        self.stop_btn = ttk.Button(ctrl, text="⏸ 停止检测", command=self.stop_detection, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        self.snapshot_btn = ttk.Button(ctrl, text="📸 保存快照", command=self.save_snapshot, state="disabled")
        self.snapshot_btn.grid(row=0, column=2, padx=(5, 0))

    def _create_right_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self._create_input_tab(nb)
        self._create_detection_tab(nb)
        self._create_tracking_tab(nb)
        self._create_threshold_tab(nb)
        self._create_screenshot_tab(nb)
        self._create_info_tab(nb)

    def _create_input_tab(self, nb):
        tab = ttk.Frame(nb, padding="10")
        nb.add(tab, text="输入源")

        # 摄像头选择
        cf = ttk.LabelFrame(tab, text="摄像头选择", padding="8")
        cf.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(cf, text="选择摄像头:").grid(row=0, column=0, sticky="w")
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(cf, textvariable=self.camera_var,
                                        values=self.camera_names, state="readonly", width=20)
        self.camera_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        cf.columnconfigure(1, weight=1)
        self.camera_info_label = ttk.Label(cf, text="请选择相机", foreground="gray")
        self.camera_info_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        self.camera_status_label = ttk.Label(cf, text="状态: 未连接", foreground="red")
        self.camera_status_label.grid(row=2, column=0, columnspan=2, sticky="w")
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # 视频文件
        ff = ttk.LabelFrame(tab, text="视频文件", padding="8")
        ff.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.file_path_var = tk.StringVar()
        ttk.Entry(ff, textvariable=self.file_path_var, state="readonly", width=20).grid(row=0, column=0, sticky="ew")
        ff.columnconfigure(0, weight=1)
        ttk.Button(ff, text="浏览...", command=self.browse_video_file).grid(row=0, column=1, padx=(3, 0))
        self.video_file_btn = ttk.Button(ff, text="打开视频文件", command=self.start_video_file)
        self.video_file_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))

        # 区域设置
        rf = ttk.LabelFrame(tab, text="检测区域设置", padding="8")
        rf.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ctrl = ttk.Frame(rf)
        ctrl.grid(row=0, column=0, sticky="ew")
        self.enable_roi_var = tk.BooleanVar(value=self.detector.enable_roi)
        ttk.Checkbutton(ctrl, text="启用区域检测", variable=self.enable_roi_var,
                       command=self.update_roi_enabled).pack(side=tk.LEFT, padx=(0, 10))
        self.select_region_btn = ttk.Button(ctrl, text="🖱️ 绘制检测区域",
                                           command=self.select_region, width=15)
        self.select_region_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.clear_region_btn = ttk.Button(ctrl, text="🗑️ 清除区域",
                                          command=self.clear_region, state="disabled", width=12)
        self.clear_region_btn.pack(side=tk.LEFT)
        self.region_info_label = ttk.Label(rf, text="未设置检测区域", foreground="gray", wraplength=350)
        self.region_info_label.grid(row=1, column=0, sticky="w", pady=(5, 0))

        # 电子地图
        mc = ttk.Frame(tab)
        mc.grid(row=3, column=0, sticky="nsew", pady=(5, 0))
        tab.rowconfigure(3, weight=1)
        tab.columnconfigure(0, weight=1)
        mf = ttk.LabelFrame(mc, text="电子地图", padding="5")
        mf.pack(fill=tk.BOTH, expand=True)
        self.map_canvas = tk.Canvas(mf, bg="#1a1a1a", height=200,
                                    highlightthickness=1, highlightbackground="#00ff00")
        self.map_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        tb = ttk.Frame(mf)
        tb.pack(fill=tk.X)
        self.add_camera_btn = ttk.Button(tb, text="➕ 添加相机", command=self.start_add_camera_mode, width=12)
        self.add_camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.edit_mode_label = ttk.Label(tb, text="", foreground="#00ff00")
        self.edit_mode_label.pack(side=tk.LEFT, padx=(5, 0))
        self.add_line_btn = ttk.Button(tb, text="〰️ 添加线段", command=self.start_add_line_mode, width=12)
        self.add_line_btn.pack(side=tk.LEFT, padx=(5, 0))

        # 图例
        leg = ttk.Frame(mf)
        leg.pack(fill=tk.X)
        for color, text, fg in [("#808080", "正常", "#00ff00"), ("#ff0000", "报警", "#ff0000")]:
            c = tk.Canvas(leg, width=20, height=20, bg="#1a1a1a", highlightthickness=0)
            c.create_oval(2, 2, 18, 18, fill=color, outline="#404040", width=2)
            c.create_text(10, 8, text="📷", fill="white", font=("Arial", 8))
            c.pack(side=tk.LEFT, padx=(0, 2))
            ttk.Label(leg, text=text, foreground=fg, background="#1a1a1a",
                     font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=(0, 8))
        c = tk.Canvas(leg, width=20, height=20, bg="#1a1a1a", highlightthickness=0)
        c.create_line(2, 10, 18, 10, fill="#00ff00", width=2)
        c.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(leg, text="线段", foreground="#00ff00", background="#1a1a1a",
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)

        self.draw_parallel_circuit_background()
        self.draw_parallel_circuit_camera_markers()
        self.map_canvas.bind("<Button-1>", self.on_map_click)
        self.map_canvas.bind("<Motion>", self.on_map_motion)
        self.map_canvas.bind("<Button-3>", self.on_map_right_click)
        self.on_camera_selected(None)

    def _create_detection_tab(self, nb):
        tab = ttk.Frame(nb, padding="15")
        nb.add(tab, text="检测参数")
        # 预处理
        pf = ttk.LabelFrame(tab, text="预处理参数", padding="10")
        pf.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        ttk.Label(pf, text="下采样比例:").grid(row=0, column=0, sticky="w")
        self.downsample_var = tk.DoubleVar(value=self.detector.downsample_ratio)
        ttk.Scale(pf, from_=0.1, to=1.0, variable=self.downsample_var,
                 orient=tk.HORIZONTAL, command=lambda x: self.update_downsample_ratio()).grid(row=0, column=1, sticky="ew", padx=10)
        pf.columnconfigure(1, weight=1)
        self.downsample_label = ttk.Label(pf, text=f"{self.downsample_var.get():.2f}")
        self.downsample_label.grid(row=0, column=2)

        # 运动检测方法
        mf = ttk.LabelFrame(tab, text="运动检测方法", padding="10")
        mf.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        self.motion_method_var = tk.StringVar(value=self.detector.motion_method)
        for i, (text, val) in enumerate([("帧差法", "frame_diff"), ("MOG2背景减法", "mog2"),
                                         ("KNN背景减法", "knn"), ("结合方法", "combine")]):
            ttk.Radiobutton(mf, text=text, variable=self.motion_method_var,
                           value=val, command=self.update_motion_method).grid(row=i, column=0, sticky="w")

        # 帧差
        df = ttk.LabelFrame(tab, text="帧差参数", padding="10")
        df.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        ttk.Label(df, text="帧差阈值:").grid(row=0, column=0, sticky="w")
        self.diff_threshold_var = tk.IntVar(value=self.detector.frame_diff_threshold)
        ttk.Scale(df, from_=5, to=50, variable=self.diff_threshold_var,
                 orient=tk.HORIZONTAL, command=lambda x: self.update_diff_threshold()).grid(row=0, column=1, sticky="ew", padx=10)
        df.columnconfigure(1, weight=1)
        self.diff_threshold_label = ttk.Label(df, text=str(self.diff_threshold_var.get()))
        self.diff_threshold_label.grid(row=0, column=2)

        # 面积
        af = ttk.LabelFrame(tab, text="区域面积参数", padding="10")
        af.grid(row=3, column=0, sticky="ew")
        ttk.Label(af, text="最小面积:").grid(row=0, column=0, sticky="w")
        self.min_area_var = tk.IntVar(value=self.detector.min_motion_area)
        ttk.Scale(af, from_=10, to=500, variable=self.min_area_var,
                 orient=tk.HORIZONTAL, command=lambda x: self.update_min_area()).grid(row=0, column=1, sticky="ew", padx=10)
        af.columnconfigure(1, weight=1)
        self.min_area_label = ttk.Label(af, text=str(self.min_area_var.get()))
        self.min_area_label.grid(row=0, column=2)
        ttk.Label(af, text="最大面积:").grid(row=1, column=0, sticky="w")
        self.max_area_var = tk.IntVar(value=self.detector.max_motion_area)
        ttk.Scale(af, from_=1000, to=20000, variable=self.max_area_var,
                 orient=tk.HORIZONTAL, command=lambda x: self.update_max_area()).grid(row=1, column=1, sticky="ew", padx=10)
        self.max_area_label = ttk.Label(af, text=str(self.max_area_var.get()))
        self.max_area_label.grid(row=1, column=2)

    def _create_tracking_tab(self, nb):
        tab = ttk.Frame(nb, padding="15")
        nb.add(tab, text="跟踪参数")
        tf = ttk.LabelFrame(tab, text="轨迹跟踪参数", padding="10")
        tf.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        for i, (label, attr, fr, to) in enumerate([
            ("轨迹历史长度:", "track_history_length", 10, 100),
            ("最小跟踪持续帧数:", "min_track_duration", 3, 30),
            ("最大跟踪速度:", "max_track_speed", 10, 200)
        ]):
            ttk.Label(tf, text=label).grid(row=i, column=0, sticky="w")
            var = tk.IntVar(value=getattr(self.detector, attr))
            setattr(self, f"{attr}_var", var)
            ttk.Scale(tf, from_=fr, to=to, variable=var, orient=tk.HORIZONTAL,
                     command=lambda x, a=attr: self._update_int_param(a)).grid(row=i, column=1, sticky="ew", padx=10)
            tf.columnconfigure(1, weight=1)
            lbl = ttk.Label(tf, text=str(var.get()))
            setattr(self, f"{attr}_label", lbl)
            lbl.grid(row=i, column=2)

        bf = ttk.LabelFrame(tab, text="背景减法器参数", padding="10")
        bf.grid(row=1, column=0, sticky="ew")
        ttk.Label(bf, text="历史帧数:").grid(row=0, column=0, sticky="w")
        self.bg_history_var = tk.IntVar(value=self.detector.bg_history)
        ttk.Scale(bf, from_=10, to=500, variable=self.bg_history_var,
                 orient=tk.HORIZONTAL, command=lambda x: self.update_bg_history()).grid(row=0, column=1, sticky="ew", padx=10)
        bf.columnconfigure(1, weight=1)
        self.bg_history_label = ttk.Label(bf, text=str(self.bg_history_var.get()))
        self.bg_history_label.grid(row=0, column=2)
        self.detect_shadows_var = tk.BooleanVar(value=self.detector.detect_shadows)
        ttk.Checkbutton(bf, text="检测阴影", variable=self.detect_shadows_var,
                       command=self.update_detect_shadows).grid(row=1, column=0, sticky="w")

    def _update_int_param(self, attr):
        val = getattr(self, f"{attr}_var").get()
        setattr(self.detector, attr, val)
        getattr(self, f"{attr}_label").config(text=str(val))

    def _create_threshold_tab(self, nb):
        tab = ttk.Frame(nb, padding="15")
        nb.add(tab, text="判定阈值")
        tf = ttk.LabelFrame(tab, text="空飘物判定阈值", padding="10")
        tf.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        self.threshold_vars = {}
        self.threshold_labels = {}
        for i, (key, label, minv, maxv) in enumerate([
            ("min_duration", "最小持续时间(帧):", 3, 30),
            ("min_speed_consistency", "速度一致性:", 0.1, 1.0),
            ("min_direction_consistency", "方向一致性:", 0.1, 1.0),
            ("min_area_stability", "面积稳定性:", 0.1, 1.0),
            ("min_linearity", "轨迹线性度:", 0.1, 1.0),
            ("min_confidence", "最小置信度:", 0.1, 1.0)
        ]):
            ttk.Label(tf, text=label).grid(row=i, column=0, sticky="w")
            val = self.detector.thresholds[key]
            if isinstance(val, float):
                var = tk.IntVar(value=int(val * 10))
                scale = ttk.Scale(tf, from_=int(minv*10), to=int(maxv*10), variable=var, orient=tk.HORIZONTAL)
                scale.bind("<ButtonRelease-1>", lambda e, k=key: self._update_threshold(k, var.get()/10.0))
            else:
                var = tk.IntVar(value=val)
                scale = ttk.Scale(tf, from_=minv, to=maxv, variable=var, orient=tk.HORIZONTAL)
                scale.bind("<ButtonRelease-1>", lambda e, k=key: self._update_threshold(k, var.get()))
            self.threshold_vars[key] = var
            scale.grid(row=i, column=1, sticky="ew", padx=10)
            tf.columnconfigure(1, weight=1)
            lbl = ttk.Label(tf, text=f"{val:.2f}" if isinstance(val, float) else str(val))
            self.threshold_labels[key] = lbl
            lbl.grid(row=i, column=2)

        sf = ttk.LabelFrame(tab, text="速度范围参数", padding="10")
        sf.grid(row=1, column=0, sticky="ew")
        ttk.Label(sf, text="最小速度:").grid(row=0, column=0, sticky="w")
        self.min_speed_var = tk.IntVar(value=self.detector.thresholds['speed_range'][0])
        ttk.Scale(sf, from_=1, to=100, variable=self.min_speed_var,
                 orient=tk.HORIZONTAL, command=lambda x: self.update_speed_range()).grid(row=0, column=1, sticky="ew", padx=10)
        sf.columnconfigure(1, weight=1)
        self.min_speed_label = ttk.Label(sf, text=str(self.min_speed_var.get()))
        self.min_speed_label.grid(row=0, column=2)
        ttk.Label(sf, text="最大速度:").grid(row=1, column=0, sticky="w")
        self.max_speed_var = tk.IntVar(value=self.detector.thresholds['speed_range'][1])
        ttk.Scale(sf, from_=10, to=200, variable=self.max_speed_var,
                 orient=tk.HORIZONTAL, command=lambda x: self.update_speed_range()).grid(row=1, column=1, sticky="ew", padx=10)
        self.max_speed_label = ttk.Label(sf, text=str(self.max_speed_var.get()))
        self.max_speed_label.grid(row=1, column=2)

    def _update_threshold(self, key, val):
        self.detector.thresholds[key] = val
        self.threshold_labels[key].config(text=f"{val:.2f}" if isinstance(val, float) else str(val))

    def _create_screenshot_tab(self, nb):
        tab = ttk.Frame(nb, padding="10")
        nb.add(tab, text="检测截图")
        main = ttk.Frame(tab)
        main.grid(row=0, column=0, sticky="nsew")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        tb = ttk.Frame(main)
        tb.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.batch_process_btn = ttk.Button(tb, text="批量处理未读警报",
                                           command=self.batch_process_unread_alerts, state="disabled")
        self.batch_process_btn.grid(row=0, column=0, padx=(0, 5))
        self.clear_screenshots_btn = ttk.Button(tb, text="清空截图", command=self.clear_screenshots, state="disabled")
        self.clear_screenshots_btn.grid(row=0, column=1, padx=5)
        self.open_screenshot_dir_btn = ttk.Button(tb, text="打开截图文件夹", command=self.open_screenshot_dir)
        self.open_screenshot_dir_btn.grid(row=0, column=2, padx=5)
        self.auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tb, text="自动保存截图", variable=self.auto_save_var).grid(row=0, column=3, padx=5)

        tf = ttk.LabelFrame(main, text="截图目录结构", padding="5")
        tf.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self.screenshot_tree = ttk.Treeview(tf, columns=("type","count","time"), show="tree headings", height=15)
        for col, w in [("#0", 250), ("type", 80), ("count", 60), ("time", 120)]:
            self.screenshot_tree.column(col, width=w, minwidth=w)
        self.screenshot_tree.heading("#0", text="名称")
        self.screenshot_tree.heading("type", text="类型")
        self.screenshot_tree.heading("count", text="数量")
        self.screenshot_tree.heading("time", text="时间")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.screenshot_tree.yview)
        self.screenshot_tree.configure(yscrollcommand=sb.set)
        self.screenshot_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        pf = ttk.LabelFrame(main, text="截图预览", padding="5")
        pf.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        main.rowconfigure(2, weight=2)
        self.preview_canvas = tk.Canvas(pf, bg="white", highlightthickness=0)
        psb = ttk.Scrollbar(pf, orient="vertical", command=self.preview_canvas.yview)
        self.preview_inner = ttk.Frame(self.preview_canvas)
        self.preview_canvas.configure(yscrollcommand=psb.set)
        self.preview_canvas.create_window((0,0), window=self.preview_inner, anchor="nw")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        psb.grid(row=0, column=1, sticky="ns")
        pf.columnconfigure(0, weight=1)
        pf.rowconfigure(0, weight=1)

        self.screenshot_tree.bind("<<TreeviewSelect>>", self.on_tree_item_selected)
        self.screenshot_tree.bind("<Double-Button-1>", self.on_tree_item_double_click)
        self.setup_tree_context_menu()

        info = ttk.Frame(main)
        info.grid(row=3, column=0, sticky="ew")
        self.tree_info_var = tk.StringVar(value="相机: 0 | 事件: 0 | 截图: 0 | 未读警报: 0")
        ttk.Label(info, textvariable=self.tree_info_var).pack(side=tk.LEFT)

        ef = ttk.LabelFrame(main, text="事件管理设置", padding="10")
        ef.grid(row=4, column=0, sticky="ew")
        ttk.Label(ef, text="事件间隔(秒):").grid(row=0, column=0, sticky="w")
        self.event_interval_var = tk.IntVar(value=30)
        ttk.Spinbox(ef, from_=10, to=300, textvariable=self.event_interval_var, width=10).grid(row=0, column=1, sticky="w", padx=10)
        self.alert_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ef, text="启用弹窗预警", variable=self.alert_enabled_var,
                       command=self.update_alert_enabled).grid(row=1, column=0, columnspan=2, sticky="w")

    def _create_info_tab(self, nb):
        tab = ttk.Frame(nb, padding="15")
        nb.add(tab, text="信息显示")
        qf = ttk.LabelFrame(tab, text="图片质量状态", padding="10")
        qf.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        self.quality_status_var = tk.StringVar(value="状态: 等待评估")
        self.quality_status_label = ttk.Label(qf, textvariable=self.quality_status_var,
                                              font=("Microsoft YaHei", 10, "bold"))
        self.quality_status_label.grid(row=0, column=0, sticky="w")
        self.quality_detail_var = tk.StringVar()
        ttk.Label(qf, textvariable=self.quality_detail_var, foreground="gray").grid(row=1, column=0, sticky="w")

        qc = ttk.Frame(qf)
        qc.grid(row=2, column=0, sticky="ew", pady=(5,0))
        self.enable_quality_check_var = tk.BooleanVar(value=self.detector.enable_quality_check)
        ttk.Checkbutton(qc, text="启用质量检查", variable=self.enable_quality_check_var,
                       command=self.update_quality_check).pack(side=tk.LEFT, padx=(0,10))
        self.skip_night_frames_var = tk.BooleanVar(value=self.detector.skip_night_frames)
        ttk.Checkbutton(qc, text="天黑时暂停检测", variable=self.skip_night_frames_var,
                       command=self.update_skip_night_frames).pack(side=tk.LEFT)

        sf = ttk.LabelFrame(tab, text="实时统计", padding="10")
        sf.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        self.stats_canvas = tk.Canvas(sf, height=150, bg="white")
        self.stats_canvas.grid(row=0, column=0, sticky="ew")
        sf.columnconfigure(0, weight=1)

        df = ttk.LabelFrame(tab, text="检测详情", padding="10")
        df.grid(row=2, column=0, sticky="nsew", pady=(0, 15))
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        self.detection_tree = ttk.Treeview(df, columns=("ID","位置","速度","置信度","轨迹长度","状态"),
                                           show="headings", height=8)
        for col in ("ID","位置","速度","置信度","轨迹长度","状态"):
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=80)
        sb = ttk.Scrollbar(df, orient="vertical", command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=sb.set)
        self.detection_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        df.columnconfigure(0, weight=1)
        df.rowconfigure(0, weight=1)

        lf = ttk.LabelFrame(tab, text="系统日志", padding="10")
        lf.grid(row=3, column=0, sticky="nsew")
        tab.rowconfigure(3, weight=1)
        self.log_text = scrolledtext.ScrolledText(lf, height=10, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)

        cf = ttk.Frame(tab)
        cf.grid(row=4, column=0, sticky="ew", pady=(10,0))
        ttk.Button(cf, text="保存配置", command=self.save_config).grid(row=0, column=0, padx=(0,5))
        ttk.Button(cf, text="加载配置", command=self.load_config).grid(row=0, column=1, padx=5)
        ttk.Button(cf, text="重置参数", command=self.reset_parameters).grid(row=0, column=2, padx=5)
        ttk.Button(cf, text="清空日志", command=self.clear_log).grid(row=0, column=3, padx=(5,0))

    def _create_status_bar(self, parent):
        sf = ttk.Frame(parent)
        sf.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10,0))
        self.status_var = tk.StringVar(value="系统就绪")
        ttk.Label(sf, textvariable=self.status_var, relief="sunken", anchor="w").grid(row=0, column=0, sticky="ew")
        sf.columnconfigure(0, weight=1)
        self.time_var = tk.StringVar()
        ttk.Label(sf, textvariable=self.time_var, relief="sunken", anchor="e", width=20).grid(row=0, column=1, sticky="e")
        self.update_time_display()

    # ========== 地图事件 ==========
    def draw_parallel_circuit_background(self):
        for item in self.map_canvas.find_all():
            if not self.map_canvas.gettags(item) or "background" in self.map_canvas.gettags(item):
                self.map_canvas.delete(item)

    def start_add_line_mode(self):
        self.is_line_mode = True
        self.is_edit_mode = False
        self.edit_mode_label.config(text="点击起点开始绘制线段")
        self.map_canvas.config(cursor="crosshair")
        self.add_line_btn.config(text="✖️ 取消线段", command=self.cancel_add_line_mode)
        self.add_camera_btn.config(state="disabled")
        self.line_start = None
        self.map_canvas.delete("temp_point", "temp_line", "temp_camera")

    def cancel_add_line_mode(self):
        self.is_line_mode = False
        self.edit_mode_label.config(text="")
        self.map_canvas.config(cursor="")
        self.add_line_btn.config(text="〰️ 添加线段", command=self.start_add_line_mode)
        self.add_camera_btn.config(state="normal")
        self.map_canvas.delete("temp_point", "temp_line", "temp_camera")
        self.line_start = None

    def on_map_click(self, e):
        if self.is_line_mode:
            self._handle_line_click(e)
        elif self.is_edit_mode:
            self.add_new_camera(e.x, e.y)
            self.cancel_add_camera_mode()
        else:
            self._check_camera_selection(e)

    def _handle_line_click(self, e):
        x, y = max(0, min(e.x, self.map_canvas.winfo_width())), max(0, min(e.y, self.map_canvas.winfo_height()))
        if self.line_start is None:
            self.line_start = (x, y)
            self.map_canvas.delete("temp_point", "temp_line")
            self.map_canvas.create_oval(x-4, y-4, x+4, y+4, fill="#00ff00", outline="#00aa00", width=2, tags="temp_point")
            self.edit_mode_label.config(text="点击终点完成线段")
        else:
            x1, y1 = self.line_start
            if abs(x1-x) < 5 and abs(y1-y) < 5:
                self.edit_mode_label.config(text="线段太短，请重新选择起点")
                self.line_start = None
                self.map_canvas.delete("temp_point", "temp_line")
                return
            idx = len(self.map_lines)
            tag = f"line_{idx}"
            line_id = self.map_canvas.create_line(x1, y1, x, y, fill="#00ff00", width=2, tags=("line", tag))
            self.map_lines.append({'id': line_id, 'tag': tag, 'coords': (x1, y1, x, y)})
            self.map_canvas.delete("temp_point", "temp_line")
            self.line_start = None
            self.edit_mode_label.config(text="点击起点开始新线段")
            self.log_message(f"已添加线段 {idx+1}")

    def on_map_motion(self, e):
        x, y = max(0, min(e.x, self.map_canvas.winfo_width())), max(0, min(e.y, self.map_canvas.winfo_height()))
        if self.is_line_mode and self.line_start:
            self.map_canvas.delete("temp_line")
            self.map_canvas.create_line(*self.line_start, x, y, fill="#00ff00", width=2, dash=(4,4), tags="temp_line")
        elif self.is_edit_mode:
            self.map_canvas.delete("temp_camera")
            self.map_canvas.create_oval(x-12, y-12, x+12, y+12, fill="#00ff00", outline="#00aa00",
                                        width=2, stipple="gray50", tags="temp_camera")
            self.map_canvas.create_text(x, y-2, text="📷", fill="black", font=("Arial", 10), tags="temp_camera")
            self.map_canvas.create_text(x, y+18, text="新相机", fill="#00ff00", font=("Courier", 7), tags="temp_camera")
        elif hasattr(self, 'dragging_camera'):
            for cam in self.camera_list:
                if cam["name"] == self.dragging_camera:
                    cam["position"] = (x, y)
                    break
            self.draw_parallel_circuit_camera_markers()

    def on_map_right_click(self, e):
        items = self.map_canvas.find_overlapping(e.x-5, e.y-5, e.x+5, e.y+5)
        for item in items:
            tags = self.map_canvas.gettags(item)
            for tag in tags:
                if tag.startswith("camera_"):
                    self.delete_camera(tag[7:])
                    return
                elif tag.startswith("line_") or tag == "line":
                    line_tag = next((t for t in tags if t.startswith("line_")), None)
                    self.delete_line(item, line_tag)
                    return

    def delete_line(self, line_id, line_tag=None):
        if messagebox.askyesno("确认", "确定要删除这条线段吗？"):
            self.map_canvas.delete(line_id)
            self.map_lines = [l for l in self.map_lines if l['id'] != line_id and (not line_tag or l['tag'] != line_tag)]
            self.log_message("线段已删除")

    def _check_camera_selection(self, e):
        for item in self.map_canvas.find_overlapping(e.x-5, e.y-5, e.x+5, e.y+5):
            for tag in self.map_canvas.gettags(item):
                if tag.startswith("camera_"):
                    self._start_drag_camera(tag[7:], e)
                    return

    def _start_drag_camera(self, name, e):
        self.dragging_camera = name
        for cam in self.camera_list:
            if cam["name"] == name:
                self.original_position = cam["position"]
                break
        self.map_canvas.config(cursor="fleur")
        self.map_canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.map_canvas.bind("<ButtonRelease-1>", self._on_drag_release)

    def _on_drag_motion(self, e):
        if hasattr(self, 'dragging_camera'):
            x, y = max(0, min(e.x, self.map_canvas.winfo_width())), max(0, min(e.y, self.map_canvas.winfo_height()))
            for cam in self.camera_list:
                if cam["name"] == self.dragging_camera:
                    cam["position"] = (x, y)
                    break
            self.draw_parallel_circuit_camera_markers()

    def _on_drag_release(self, e):
        if hasattr(self, 'dragging_camera'):
            self.map_canvas.config(cursor="")
            self.map_canvas.unbind("<B1-Motion>")
            self.map_canvas.unbind("<ButtonRelease-1>")
            delattr(self, 'dragging_camera')
            if hasattr(self, 'original_position'):
                delattr(self, 'original_position')
            self.draw_parallel_circuit_camera_markers()

    # ========== 相机操作 ==========
    def start_add_camera_mode(self):
        self.show_camera_config_dialog()

    def cancel_add_camera_mode(self):
        self.is_edit_mode = False
        self.edit_mode_label.config(text="")
        self.map_canvas.config(cursor="")
        self.add_camera_btn.config(text="➕ 添加相机", command=self.start_add_camera_mode)
        self.add_line_btn.config(state="normal")
        self.map_canvas.delete("temp_camera", "temp_point", "temp_line")

    def show_camera_config_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("配置新相机")
        dlg.geometry("500x450")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.update_idletasks()
        dlg.geometry(f"+{self.root.winfo_x() + (self.root.winfo_width()-500)//2}+{self.root.winfo_y() + (self.root.winfo_height()-450)//2}")

        main = ttk.Frame(dlg, padding="20")
        main.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main, text="配置新相机参数", font=("Microsoft YaHei", 14, "bold")).pack(pady=(0,20))

        # 名称
        nf = ttk.Frame(main)
        nf.pack(fill=tk.X, pady=(0,15))
        ttk.Label(nf, text="相机名称:", width=12).pack(side=tk.LEFT)
        name_var = tk.StringVar(value=f"相机{len(self.camera_list)+1}")
        name_entry = ttk.Entry(nf, textvariable=name_var, width=30)
        name_entry.pack(side=tk.LEFT, padx=(10,0))

        # 类型
        tf = ttk.Frame(main)
        tf.pack(fill=tk.X, pady=(0,15))
        ttk.Label(tf, text="相机类型:", width=12).pack(side=tk.LEFT)
        type_var = tk.StringVar(value="network")
        trf = ttk.Frame(tf)
        trf.pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(trf, text="网络摄像头", variable=type_var, value="network").pack(side=tk.LEFT, padx=(0,15))
        ttk.Radiobutton(trf, text="本地摄像头", variable=type_var, value="local").pack(side=tk.LEFT)

        # RTSP
        self.url_frame = ttk.Frame(main)
        self.url_frame.pack(fill=tk.X, pady=(0,15))
        ttk.Label(self.url_frame, text="RTSP地址:", width=12).pack(side=tk.LEFT)
        url_var = tk.StringVar(value="rtsp://username:password@192.168.1.100:554/stream")
        self.url_entry = ttk.Entry(self.url_frame, textvariable=url_var, width=40)
        self.url_entry.pack(side=tk.LEFT, padx=(10,0))
        ttk.Label(main, text="格式: rtsp://用户名:密码@IP地址:端口/路径",
                 foreground="gray", font=("Microsoft YaHei", 8)).pack(anchor="w", padx=(120,0), pady=(0,15))

        # 索引
        self.index_frame = ttk.Frame(main)
        self.index_frame.pack(fill=tk.X, pady=(0,15))
        ttk.Label(self.index_frame, text="摄像头索引:", width=12).pack(side=tk.LEFT)
        index_var = tk.IntVar(value=0)
        self.index_spinbox = ttk.Spinbox(self.index_frame, from_=0, to=10, textvariable=index_var, width=10)
        self.index_spinbox.pack(side=tk.LEFT, padx=(10,0))

        ttk.Button(main, text="测试连接", command=lambda: self._test_connection(type_var.get(), url_var.get(), index_var.get())).pack(pady=(10,20))

        bf = ttk.Frame(main)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="确认", width=12, command=lambda: self._confirm_add_camera(dlg, name_var.get().strip(),
                  type_var.get(), url_var.get().strip() if type_var.get()=="network" else None,
                  index_var.get() if type_var.get()=="local" else None)).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(bf, text="取消", width=12, command=dlg.destroy).pack(side=tk.LEFT)

        self._update_dialog_fields(type_var)
        type_var.trace('w', lambda *a: self._update_dialog_fields(type_var))
        name_entry.focus_set()

    def _update_dialog_fields(self, type_var):
        if type_var.get() == "network":
            for w in self.url_frame.winfo_children():
                if isinstance(w, ttk.Entry):
                    w.config(state="normal")
            for w in self.index_frame.winfo_children():
                if isinstance(w, ttk.Spinbox):
                    w.config(state="disabled")
        else:
            for w in self.url_frame.winfo_children():
                if isinstance(w, ttk.Entry):
                    w.config(state="disabled")
            for w in self.index_frame.winfo_children():
                if isinstance(w, ttk.Spinbox):
                    w.config(state="normal")

    def _test_connection(self, ctype, url, idx):
        try:
            cap = cv2.VideoCapture(url if ctype=="network" else idx)
            if cap.isOpened():
                ret, _ = cap.read()
                msg = "摄像头连接成功！\n已成功读取到视频帧。" if ret else "摄像头已连接但无法读取帧"
                messagebox.showinfo("测试成功" if ret else "警告", msg)
                cap.release()
            else:
                messagebox.showerror("测试失败", "无法连接摄像头")
        except Exception as e:
            messagebox.showerror("错误", f"测试连接失败: {str(e)}")

    def _confirm_add_camera(self, dlg, name, ctype, url, idx):
        self.pending_camera_config = {'name': name, 'type': ctype, 'url': url, 'index': idx}
        dlg.destroy()
        self.enter_add_camera_mode()

    def enter_add_camera_mode(self):
        self.is_edit_mode = True
        self.is_line_mode = False
        self.edit_mode_label.config(text="点击地图添加新相机")
        self.map_canvas.config(cursor="crosshair")
        self.add_camera_btn.config(text="✖️ 取消添加", command=self.cancel_add_camera_mode)
        self.add_line_btn.config(state="disabled")
        self.map_canvas.delete("temp_point", "temp_line")

    def add_new_camera(self, x, y):
        if not hasattr(self, 'pending_camera_config'):
            self.show_camera_config_dialog()
            return
        cfg = self.pending_camera_config
        name = cfg['name']
        if any(cam["name"] == name for cam in self.camera_list):
            nums = [int(cam["name"][2:]) for cam in self.camera_list if cam["name"].startswith("相机")]
            name = f"相机{max(nums)+1 if nums else 1}"
        self.camera_list.append({"name": name, "type": cfg['type'], "url": cfg['url'],
                                 "index": cfg['index'], "position": (x, y)})
        self.camera_names = [cam["name"] for cam in self.camera_list]
        self.camera_combo["values"] = self.camera_names
        self.camera_alert_status[name] = False
        self.draw_parallel_circuit_camera_markers()
        self.log_message(f"已添加新相机: {name}")
        delattr(self, 'pending_camera_config')
        self.cancel_add_camera_mode()

    def delete_camera(self, name):
        if len(self.camera_list) <= 1:
            messagebox.showwarning("警告", "至少保留一个相机")
            return
        if messagebox.askyesno("确认", f"确定要删除相机 '{name}' 吗？"):
            self.camera_list = [cam for cam in self.camera_list if cam["name"] != name]
            self.camera_names = [cam["name"] for cam in self.camera_list]
            self.camera_combo["values"] = self.camera_names
            self.camera_alert_status.pop(name, None)
            self.draw_parallel_circuit_camera_markers()
            self.log_message(f"已删除相机: {name}")

    # ========== 视频处理 ==========
    def on_camera_selected(self, e):
        idx = self.camera_combo.current()
        if 0 <= idx < len(self.camera_list):
            cam = self.camera_list[idx]
            if cam["type"] == "network":
                self.camera_info_label.config(text=f"类型: 网络摄像头 | RTSP: {cam['url'][:50]}...", foreground="blue")
            else:
                self.camera_info_label.config(text=f"类型: 本地摄像头 | 索引: {cam['index']}", foreground="green")
        else:
            self.camera_info_label.config(text="请选择相机", foreground="gray")

    def start_selected_camera(self):
        if self.is_running:
            messagebox.showwarning("警告", "请先停止当前检测")
            return
        idx = self.camera_combo.current()
        if idx < 0 or idx >= len(self.camera_list):
            messagebox.showwarning("警告", "请选择摄像头")
            return
        cam = self.camera_list[idx]
        self.current_camera_name = cam["name"]
        try:
            if cam["type"] == "network":
                self.cap = cv2.VideoCapture(cam["url"])
                if not self.cap.isOpened():
                    for p in [cam["url"], cam["url"].replace("rtsp://", "rtsp://@"),
                              cam["url"] + "?transport=tcp", cam["url"] + "?transport=udp"]:
                        self.cap = cv2.VideoCapture(p)
                        if self.cap.isOpened():
                            break
                    if not self.cap.isOpened():
                        messagebox.showerror("错误", f"无法连接网络摄像头: {cam['name']}")
                        return
            else:
                self.cap = cv2.VideoCapture(cam["index"])
                if not self.cap.isOpened():
                    for i in range(5):
                        self.cap = cv2.VideoCapture(i)
                        if self.cap.isOpened():
                            cam["index"] = i
                            break
                    if not self.cap.isOpened():
                        messagebox.showerror("错误", "无法打开本地摄像头")
                        return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.snapshot_btn.config(state="normal")
            self.camera_status_label.config(text="状态: 已连接", foreground="green")
            self.status_var.set(f"{cam['name']} 已启动")
            self.log_message(f"{cam['name']} 已启动")
            self.process_frame()
        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {str(e)}")

    def browse_video_file(self):
        fp = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")])
        if fp:
            self.file_path_var.set(fp)

    def start_video_file(self):
        if self.is_running:
            messagebox.showwarning("警告", "请先停止当前检测")
            return
        fp = self.file_path_var.get()
        if not fp or not os.path.exists(fp):
            messagebox.showwarning("警告", "请选择有效的视频文件")
            return
        try:
            self.cap = cv2.VideoCapture(fp)
            if not self.cap.isOpened():
                messagebox.showerror("错误", f"无法打开视频文件: {fp}")
                return
            if not any(cam["name"] == "相机1" for cam in self.camera_list):
                self.camera_list.append({"name": "相机1", "type": "network",
                                        "url": "rtsp://admin:admin123@192.168.1.101:554/Streaming/Channels/1",
                                        "index": None, "position": (150, 80)})
                self.camera_names = [cam["name"] for cam in self.camera_list]
                self.camera_combo["values"] = self.camera_names
                self.camera_alert_status["相机1"] = False
                self.draw_parallel_circuit_camera_markers()
            self.current_camera_name = "相机1"
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fc = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.is_running = True
            self.video_file_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.snapshot_btn.config(state="normal")
            self.status_var.set("视频文件已加载")
            self.log_message(f"视频文件已加载: {fp}")
            self.log_message(f"视频信息: {fps:.1f} FPS, {fc}帧, {fc/fps:.1f}秒")
            self.process_frame()
        except Exception as e:
            messagebox.showerror("错误", f"打开视频文件失败: {str(e)}")

    def stop_detection(self):
        self.is_running = False
        if self.update_job:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_file_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.snapshot_btn.config(state="disabled")
        self.video_label.config(image='', text="等待启动检测...")
        self.mask_label.config(image='', text="运动掩码")
        self.status_var.set("检测已停止")
        self.log_message("检测已停止")
        stats = self.detector.get_statistics()
        self.log_message(f"总处理帧数: {stats['total_frames']} | 总检测次数: {stats['total_detections']} | 平均FPS: {stats['current_fps']:.1f}")
        self.detector.reset()

    def process_frame(self):
        if not self.is_running or self.cap is None:
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.stop_detection()
                        return
                else:
                    self.stop_detection()
                    return

            if self.selected_region and len(self.selected_region) >= 3:
                pts = np.array(self.selected_region, np.int32)
                cv2.polylines(frame, [pts], True, (0,255,0), 2)
                for i, (x, y) in enumerate(self.selected_region):
                    cv2.circle(frame, (x, y), 4, (0,255,0), -1)
                    cv2.putText(frame, str(i+1), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            qi = self.detector.get_quality_info()
            is_night = qi.get('is_night', False)

            if is_night and self.detector.skip_night_frames:
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,100), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                frame = self.put_chinese_text_cv(frame, "  天黑/过暗 - 检测暂停", ((w-400)//2, h//2), 30, (0,0,255), (255,255,255))
                frame = self.put_chinese_text_cv(frame, qi.get('message',''), ((w-400)//2, h//2+50), 30, (0,0,255), (255,255,255))
                detections, motion_mask = [], None
                result = frame
            else:
                detections, motion_mask = self.detector.detect(frame, self.selected_region)
                if detections and self.auto_save_var.get():
                    for d in detections:
                        if d['features']['confidence'] > 0.7:
                            self.save_and_display_screenshot(frame.copy(), d)
                result = self.detector.draw_detections(frame, detections)

            h, w = result.shape[:2]
            result = self.put_chinese_text_cv(result, "图片过暗" if is_night else "图片正常",
                                             (w-140, 30), 30, (0,0,255), (255,255,255))

            self.display_frame(result, self.video_label)
            if motion_mask is not None:
                mm = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR) if len(motion_mask.shape)==2 else motion_mask
                self.display_frame(mm, self.mask_label)

            self.update_info_displays(detections)
            if self.is_running:
                self.update_job = self.root.after(30, self.process_frame)
        except Exception as e:
            self.log_message(f"处理帧时出错: {str(e)}", level="error")
            if self.is_running:
                self.update_job = self.root.after(100, self.process_frame)

    def put_chinese_text_cv(self, img, text, pos, size=30, color=(0,0,255), bg=None, pad=5):
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("simhei.ttf", size)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        if bg:
            draw.rectangle((pos[0]-pad, pos[1]-pad, pos[0]+tw+pad, pos[1]+th+pad), fill=bg)
        draw.text(pos, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def display_frame(self, frame, label):
        if frame is None or frame.size == 0:
            return
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        w, h = label.winfo_width(), label.winfo_height()
        if w > 10 and h > 10:
            ar = frame.shape[1] / frame.shape[0]
            if w / h > ar:
                frame = cv2.resize(frame, (int(h * ar), h))
            else:
                frame = cv2.resize(frame, (w, int(w / ar)))
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

    def update_info_displays(self, dets):
        stats = self.detector.get_statistics()
        self.update_quality_display()
        qi = self.detector.get_quality_info()
        self.status_var.set(f"检测中 | FPS: {stats['current_fps']:.1f} | 检测数: {len(dets)} | 光照: {'🌙天黑' if qi.get('is_night') else '☀正常'}")
        self._update_detection_tree(dets)
        self._update_stats_chart(stats)

    def _update_detection_tree(self, dets):
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
        for d in dets:
            f = d['features']
            self.detection_tree.insert("", "end", values=(
                d['id'], f"({d['center'][0]},{d['center'][1]})", f"{f['mean_speed']:.1f}",
                f"{f['confidence']*100:.1f}%", d['track_length'], d.get('status','confirmed')
            ))

    def _update_stats_chart(self, stats):
        self.stats_canvas.delete("all")
        w, h = self.stats_canvas.winfo_width(), self.stats_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        self.stats_canvas.create_text(w//2, 15, text="实时统计", font=("Microsoft YaHei", 12, "bold"))
        for i, line in enumerate([
            f"FPS: {stats['current_fps']:.1f}", f"检测数: {stats['current_detections']}",
            f"总检测: {stats['total_detections']}", f"活动轨迹: {stats['active_tracks']}",
            f"处理时间: {stats['processing_time_ms']:.1f}ms"
        ]):
            self.stats_canvas.create_text(20, 40 + i*20, text=line, font=("Microsoft YaHei", 10), anchor="w")

    def save_snapshot(self):
        if not self.is_running or self.cap is None:
            return
        try:
            ret, frame = self.cap.read()
            if ret:
                dets, _ = self.detector.detect(frame, self.selected_region)
                res = self.detector.draw_detections(frame, dets)
                fn = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fn, res)
                pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 1)
                self.status_var.set(f"快照已保存: {fn}")
                self.log_message(f"快照已保存: {fn}")
        except Exception as e:
            messagebox.showerror("错误", f"保存快照失败: {str(e)}")

    # ========== 截图保存 ==========
    def save_and_display_screenshot(self, frame, detection):
        try:
            now = time.time()
            cam = self.current_camera_name or "Unknown"
            x,y,w,h = detection['bbox']
            roi = frame[max(0,y-20):min(frame.shape[0],y+h+20), max(0,x-20):min(frame.shape[1],x+w+20)]
            if roi.size == 0:
                return
            feat = self._extract_image_features(roi)

            # 查找相似事件
            event_id = None
            best_sim = 0
            best_event = None
            for ev in self.events:
                if ev.get('camera') != cam or not ev.get('latest_features'):
                    continue
                td = now - ev.get('last_detection_time', 0)
                if td > self.recent_time_threshold:
                    continue
                sim = self._calculate_image_similarity(feat, ev['latest_features'])
                if sim >= self.similarity_threshold and sim > best_sim:
                    best_sim = sim
                    event_id = ev['event_id']
                    best_event = ev

            if event_id is None:
                self.current_event_id += 1
                event_id = self.current_event_id
                self.events.append({
                    'event_id': event_id, 'camera': cam,
                    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'detection_count': 1, 'screenshot_count': 0,
                    'last_detection_time': now, 'latest_features': feat,
                    'has_alerted': False, 'similarity_score': 1.0, 'created_time': now
                })
                if self.alert_enabled:
                    self.show_alert_dialog(event_id, cam)
                    if cam == "相机1":
                        self.update_map_camera_status(cam, True)
                        self.log_message("相机1触发报警，地图变红")
            else:
                best_event['detection_count'] += 1
                best_event['last_detection_time'] = now
                best_event['latest_features'] = feat
                best_event['similarity_score'] = best_sim
                if not best_event.get('has_alerted') and self.alert_enabled:
                    self.show_alert_dialog(event_id, cam)
                    best_event['has_alerted'] = True
                    if cam == "相机1":
                        self.update_map_camera_status(cam, True)
                        self.log_message("相机1触发报警，地图变红")

            # 保存截图（最多3张）
            existing = [s for s in self.screenshots if s['event_id'] == event_id]
            if len(existing) >= 3:
                return

            cam_dir = re.sub(r'[<>:"/\\|?*]', '_', cam)
            ev_dir = f"event_{event_id:03d}"
            full_dir = os.path.join(self.screenshot_dir, cam_dir, ev_dir)
            os.makedirs(full_dir, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fn = f"{event_id:03d}_{detection['id']}_{len(existing)+1}_{ts}.jpg"
            fp = os.path.join(full_dir, fn)
            cv2.imwrite(fp, roi)

            thumb = cv2.resize(roi, (120,90))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

            self.screenshots.append({
                'event_id': event_id, 'id': detection['id'], 'filepath': fp,
                'filename': fn, 'timestamp': ts, 'camera': cam,
                'camera_dir': cam_dir, 'event_dir': ev_dir,
                'confidence': detection['features']['confidence'],
                'thumbnail': thumb, 'original_size': (roi.shape[1], roi.shape[0]),
                'image_index': len(existing) + 1
            })

            for ev in self.events:
                if ev['event_id'] == event_id:
                    ev['screenshot_count'] = len(existing) + 1
                    break

            self.update_screenshot_tree()
            self.update_screenshot_statistics()
        except Exception as e:
            self.log_message(f"保存截图失败: {str(e)}", level="error")

    # ========== 截图树形显示 ==========
    def update_screenshot_tree(self):
        for item in self.screenshot_tree.get_children():
            self.screenshot_tree.delete(item)
        self.screenshot_tree.tag_configure("unread_camera", foreground="red")
        self.screenshot_tree.tag_configure("unread_event", foreground="red")

        data = {}
        for s in self.screenshots:
            data.setdefault(s['camera'], {}).setdefault(s['event_id'], []).append(s)

        total_cam = len(data)
        total_ev = sum(len(ev) for ev in data.values())
        total_unread = sum(len(a) for a in self.unread_alerts.values())

        for cam, events in data.items():
            unread_cnt = self.get_unread_alert_count(cam)
            cam_text = f"📷 {cam}" + (f" 🔴({unread_cnt})" if unread_cnt else "")
            cam_tags = ("camera", "unread_camera") if unread_cnt else ("camera",)
            cam_id = self.screenshot_tree.insert("", "end", text=cam_text,
                                                values=("相机", f"{len(events)}事件", ""), tags=cam_tags)
            for ev_id, ss in events.items():
                ev_info = next((e for e in self.events if e['event_id'] == ev_id), None)
                ev_time = ev_info['start_time'] if ev_info else "未知"
                is_unread = cam in self.unread_alerts and ev_id in self.unread_alerts[cam]
                ev_text = f"📁 事件 {ev_id}" + (" 🔴" if is_unread else "")
                ev_tags = ("event", "unread_event") if is_unread else ("event",)
                ev_node = self.screenshot_tree.insert(cam_id, "end", text=ev_text,
                                                      values=("事件", f"{len(ss)}张", ev_time), tags=ev_tags)
                if len(ss) <= 10:
                    for i, s in enumerate(ss[:10]):
                        self.screenshot_tree.insert(ev_node, "end",
                            text=f"🖼️ {s['filename'][:30]}...",
                            values=("截图", f"{i+1}/{len(ss)}", s['timestamp'][9:17]),
                            tags=("screenshot", s['filepath']))
                else:
                    self.screenshot_tree.insert(ev_node, "end", text=f"📊 {len(ss)}张截图",
                                                values=("截图集", f"{len(ss)}张", ""),
                                                tags=("screenshot_summary",))

        self.tree_info_var.set(f"相机: {total_cam} | 事件: {total_ev} | 截图: {len(self.screenshots)} | 未读警报: {total_unread}")
        if not data:
            self.screenshot_tree.insert("", "end", text="暂无检测截图", values=("","",""))
        for cid in self.screenshot_tree.get_children():
            self.screenshot_tree.item(cid, open=True)

    def on_tree_item_selected(self, e):
        sel = self.screenshot_tree.selection()
        if not sel:
            return
        item = sel[0]
        tags = self.screenshot_tree.item(item, "tags")
        if "screenshot" in tags:
            for t in tags:
                if t.startswith('/') or '\\' in t:
                    self._display_selected_screenshot(t)
                    break
        elif "event" in tags:
            self._display_event_screenshots(item)
        elif "camera" in tags:
            self._display_camera_screenshots(item)

    def on_tree_item_double_click(self, e):
        sel = self.screenshot_tree.selection()
        if not sel:
            return
        item = sel[0]
        tags = self.screenshot_tree.item(item, "tags")
        text = self.screenshot_tree.item(item, "text")
        if "unread_event" in tags:
            import re
            m = re.search(r'事件\s*(\d+)', text)
            if m:
                ev = int(m.group(1))
                parent = self.screenshot_tree.parent(item)
                if parent:
                    cam = re.sub(r'📷\s*|🔴.*', '', self.screenshot_tree.item(parent, "text")).strip()
                    if cam in self.unread_alerts and ev in self.unread_alerts[cam]:
                        self.show_alert_processing_dialog(ev, cam)
                        return
        if "screenshot" in tags:
            for t in tags:
                if t.startswith('/') or '\\' in t:
                    self.show_screenshot_detail_by_path(t)
                    break
        elif "event" in tags or "unread_event" in tags:
            self.screenshot_tree.item(item, open=not self.screenshot_tree.item(item, "open"))
        elif "camera" in tags or "unread_camera" in tags:
            self.screenshot_tree.item(item, open=not self.screenshot_tree.item(item, "open"))

    def _display_selected_screenshot(self, fp):
        for w in self.preview_inner.winfo_children():
            w.destroy()
        if os.path.exists(fp):
            img = cv2.imread(fp)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img)
                pil.thumbnail((300,200))
                photo = ImageTk.PhotoImage(pil)
                lbl = tk.Label(self.preview_inner, image=photo)
                lbl.image = photo
                lbl.pack(pady=10)
                ttk.Label(self.preview_inner, text=os.path.basename(fp), font=("Microsoft YaHei",9)).pack()
                self._update_preview_canvas()

    def _display_event_screenshots(self, ev_item):
        for w in self.preview_inner.winfo_children():
            w.destroy()
        import re
        m = re.search(r'事件 (\d+)', self.screenshot_tree.item(ev_item, "text"))
        if m:
            ev = int(m.group(1))
            ss = [s for s in self.screenshots if s['event_id'] == ev]
            if ss:
                ev_info = next((e for e in self.events if e['event_id'] == ev), None)
                if ev_info:
                    ttk.Label(self.preview_inner,
                             text=f"事件 {ev} - {ev_info['camera']}\n开始时间: {ev_info['start_time']}\n检测次数: {ev_info['detection_count']}\n截图数量: {ev_info['screenshot_count']}",
                             font=("Microsoft YaHei",10,"bold"), justify="left").pack(pady=(10,20))
                self._display_screenshot_grid(ss[:20])

    def _display_camera_screenshots(self, cam_item):
        for w in self.preview_inner.winfo_children():
            w.destroy()
        cam = self.screenshot_tree.item(cam_item, "text").replace("📷 ", "")
        ss = [s for s in self.screenshots if s['camera'] == cam]
        if ss:
            evs = len(set(s['event_id'] for s in ss))
            ttk.Label(self.preview_inner,
                     text=f"📷 {cam}\n总截图数: {len(ss)}\n事件数量: {evs}",
                     font=("Microsoft YaHei",10,"bold"), justify="left").pack(pady=(10,20))
            recent = sorted(ss, key=lambda x: x['timestamp'], reverse=True)[:20]
            self._display_screenshot_grid(recent)

    def _display_screenshot_grid(self, ss):
        grid = ttk.Frame(self.preview_inner)
        grid.pack(fill="both", expand=True)
        for i, s in enumerate(ss):
            r, c = i//3, i%3
            f = ttk.Frame(grid, relief="solid", borderwidth=1, padding=2)
            f.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            grid.columnconfigure(c, weight=1)
            try:
                thumb = s['thumbnail']
                pil = Image.fromarray(thumb)
                photo = ImageTk.PhotoImage(pil)
                lbl = tk.Label(f, image=photo, cursor="hand2")
                lbl.image = photo
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, p=s['filepath']: self.show_screenshot_detail_by_path(p))
                ttk.Label(f, text=f"事件 {s['event_id']}\n检测 {s['id']}\n{s['timestamp'][9:17]}",
                         font=("Microsoft YaHei",7), justify="center").pack()
            except:
                pass
        self._update_preview_canvas()

    def _update_preview_canvas(self):
        self.preview_inner.update_idletasks()
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

    def show_screenshot_detail_by_path(self, fp):
        for s in self.screenshots:
            if s['filepath'] == fp:
                self.show_screenshot_detail(s)
                break

    def show_screenshot_detail(self, info):
        win = tk.Toplevel(self.root)
        win.title(f"空飘物检测详情 - {info['filename']}")
        win.geometry("800x600")
        win.update_idletasks()
        win.geometry(f"+{self.root.winfo_x()+(self.root.winfo_width()-800)//2}+{self.root.winfo_y()+(self.root.winfo_height()-600)//2}")
        main = ttk.Frame(win, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        img_frame = ttk.Frame(main)
        img_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0,10))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        img = cv2.imread(info['filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        scale = min(780/pil.width, 450/pil.height, 1.0)
        pil = pil.resize((int(pil.width*scale), int(pil.height*scale)))
        photo = ImageTk.PhotoImage(pil)
        lbl = ttk.Label(img_frame, image=photo)
        lbl.image = photo
        lbl.pack(expand=True)

        infof = ttk.LabelFrame(main, text="检测信息", padding="10")
        infof.grid(row=1, column=0, sticky="ew", padx=(0,5))
        txt = f"文件名: {info['filename']}\n相机: {info['camera']}\n事件ID: {info['event_id']}\n检测ID: {info['id']}\n图片序号: {info['image_index']}\n时间: {info['timestamp'][:8]} {info['timestamp'][9:15]}\n置信度: {info['confidence']*100:.1f}%\n原始尺寸: {info['original_size'][0]}x{info['original_size'][1]}"
        ttk.Label(infof, text=txt, justify="left").pack(anchor="w")

        btnf = ttk.Frame(main)
        btnf.grid(row=1, column=1, sticky="ew", padx=(5,0))
        for txt, cmd in [("打开原图", lambda: self.open_original_image(info['filepath'])),
                        ("删除截图", lambda: self.delete_screenshot(info, win))]:
            ttk.Button(btnf, text=txt, command=cmd, width=15).pack(pady=(0,5))
        ttk.Button(btnf, text="关闭", command=win.destroy, width=15).pack()

    def open_original_image(self, fp):
        try:
            os.startfile(fp)
        except:
            try:
                import subprocess
                subprocess.run(['xdg-open', fp])
            except:
                messagebox.showwarning("提示", f"无法打开文件: {fp}")

    def delete_screenshot(self, info, win=None):
        if messagebox.askyesno("确认", f"确定要删除截图 '{info['filename']}' 吗？"):
            try:
                if os.path.exists(info['filepath']):
                    os.remove(info['filepath'])
                self.screenshots = [s for s in self.screenshots if s['filepath'] != info['filepath']]
                self.update_screenshot_tree()
                self.update_screenshot_statistics()
                self.log_message(f"截图已删除: {info['filename']}")
                if win and win.winfo_exists():
                    win.destroy()
            except Exception as e:
                messagebox.showerror("错误", f"删除截图失败: {str(e)}")

    # ========== 警报相关 ==========
    def show_alert_dialog(self, ev_id, cam):
        self.update_map_camera_status(cam, True)
        if ev_id in self.alert_windows:
            w = self.alert_windows[ev_id]
            if w and w.winfo_exists():
                w.lift()
                w.focus_set()
                return
        self._create_alert_window(ev_id, cam)

    def _create_alert_window(self, ev_id, cam):
        win = tk.Toplevel(self.root)
        win.title(f"⚠️ 空飘物检测预警 - 事件 {ev_id}")
        win.geometry("400x400")
        win.resizable(False, False)
        win.transient(self.root)
        self.alert_windows[ev_id] = win

        idx = len(self.alert_windows) - 1
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        x = max(0, min(sw//2 - 200 - 50 + idx*25, sw - 400))
        y = max(0, min(sh//2 - 200 - 50 + idx*25, sh - 400))
        win.geometry(f"+{int(x)}+{int(y)}")

        main = ttk.Frame(win, padding="20")
        main.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        tf = ttk.Frame(main)
        tf.grid(row=0, column=0, columnspan=2, pady=(0,15))
        ttk.Label(tf, text="⚠️", font=("Microsoft YaHei",36), foreground="orange").grid(row=0, column=0, padx=(0,10))
        ttk.Label(tf, text="空飘物检测预警", font=("Microsoft YaHei",16,"bold"), foreground="red").grid(row=0, column=1)

        infof = ttk.Frame(main)
        infof.grid(row=1, column=0, columnspan=2, pady=(0,15))
        now = datetime.now().strftime("%H:%M:%S")
        for i, line in enumerate([f"预警时间: {now}", f"事件编号: {ev_id}", f"来源相机: {cam}",
                                  "检测类型: 空飘物入侵", "状态: 等待处理"]):
            ttk.Label(infof, text=line, font=("Microsoft YaHei",10)).grid(row=i, column=0, sticky="w", pady=2)

        af = ttk.LabelFrame(main, text="警报信息", padding="10")
        af.grid(row=2, column=0, columnspan=2, pady=(10,15), sticky="ew")
        ttk.Label(af, text="请在事件详情页面处理此警报", font=("Microsoft YaHei",10), justify="center").pack()

        bf = ttk.Frame(main)
        bf.grid(row=3, column=0, columnspan=2, pady=(10,0))
        ttk.Button(bf, text="查看详情", command=lambda: self.show_event_details(ev_id, win), width=12).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(bf, text="稍后处理", command=lambda: self.close_alert_window(ev_id, False), width=12).pack(side=tk.LEFT)

        win.protocol("WM_DELETE_WINDOW", lambda: None)
        self.play_alert_sound()
        self.mark_alert_as_unread(ev_id, cam)
        self.log_message(f"事件 {ev_id} - {cam} 触发预警", level="warning")

    def close_alert_window(self, ev_id, mark_read=True):
        if ev_id in self.alert_windows:
            win = self.alert_windows[ev_id]
            cam = next((c for c, a in self.unread_alerts.items() if ev_id in a), None)
            if win and win.winfo_exists():
                if mark_read:
                    self.mark_alert_as_read(ev_id)
                win.destroy()
            del self.alert_windows[ev_id]
            if cam == "相机1" and not any(ev_id in a for a in self.unread_alerts.values() if a):
                self.update_map_camera_status(cam, False)
                self.log_message("相机1所有警报已处理，地图状态恢复为灰色")

    def mark_alert_as_unread(self, ev_id, cam):
        self.unread_alerts.setdefault(cam, []).append(ev_id)
        self.update_screenshot_tree()

    def mark_alert_as_read(self, ev_id):
        for cam, alerts in list(self.unread_alerts.items()):
            if ev_id in alerts:
                alerts.remove(ev_id)
                if not alerts:
                    del self.unread_alerts[cam]
        self.update_screenshot_tree()

    def get_unread_alert_count(self, cam=None):
        if cam:
            return len(self.unread_alerts.get(cam, []))
        return sum(len(a) for a in self.unread_alerts.values())

    def show_alert_processing_dialog(self, ev_id, cam):
        if ev_id in self.alert_windows:
            w = self.alert_windows[ev_id]
            if w and w.winfo_exists():
                w.lift()
                return
        dlg = tk.Toplevel(self.root)
        dlg.title(f"处理未读警报 - 事件 {ev_id}")
        dlg.geometry("500x400")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        self.alert_windows[ev_id] = dlg
        dlg.update_idletasks()
        dlg.geometry(f"+{self.root.winfo_x()+(self.root.winfo_width()-500)//2}+{self.root.winfo_y()+(self.root.winfo_height()-400)//2}")

        main = ttk.Frame(dlg, padding="20")
        main.grid(row=0, column=0, sticky="nsew")
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(0, weight=1)

        ttk.Label(main, text="🔴 处理未读警报", font=("Microsoft YaHei",14,"bold"), foreground="red").grid(row=0, column=0, columnspan=2, pady=(0,20))

        info = ttk.LabelFrame(main, text="警报信息", padding="10")
        info.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,20))
        now = datetime.now().strftime("%H:%M:%S")
        for i, line in enumerate([f"事件编号: {ev_id}", f"来源相机: {cam}", f"处理时间: {now}", "状态: 未处理"]):
            ttk.Label(info, text=line, font=("Microsoft YaHei",10)).grid(row=i, column=0, sticky="w", pady=2)

        opt = ttk.LabelFrame(main, text="请选择处理方式", padding="15")
        opt.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0,20))
        ttk.Button(opt, text="✅ 标记为实警", command=lambda: self.handle_alert_type(ev_id, cam, "real", dlg), width=20).grid(row=0, column=0, padx=(0,10), pady=10)
        ttk.Button(opt, text="❌ 标记为虚警", command=lambda: self.handle_alert_type(ev_id, cam, "false", dlg), width=20).grid(row=0, column=1, pady=10)

        bottom = ttk.Frame(main)
        bottom.grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Button(bottom, text="查看事件详情", command=lambda: self.show_event_details(ev_id, dlg)).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(bottom, text="关闭", command=lambda: self.close_alert_window(ev_id)).pack(side=tk.LEFT)

        opt.columnconfigure(0, weight=1)
        opt.columnconfigure(1, weight=1)
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

    def handle_alert_type(self, ev_id, cam, atype, parent=None):
        self.mark_alert_as_read(ev_id)
        atext = "实警" if atype == "real" else "虚警"
        self.log_message(f"事件 {ev_id} - {cam} 标记为{atext}")
        target = self.real_alerts_dir if atype == "real" else self.false_alerts_dir
        cam_dir = self.get_safe_filename(cam)
        ev_dir = os.path.join(target, cam_dir, f"event_{ev_id:03d}")
        os.makedirs(ev_dir, exist_ok=True)

        ss = [s for s in self.screenshots if s['event_id'] == ev_id]
        copied = []
        for s in ss:
            if os.path.exists(s['filepath']):
                dest = os.path.join(ev_dir, f"{atext}_{ev_id:03d}_{os.path.basename(s['filepath'])}")
                try:
                    import shutil
                    shutil.copy2(s['filepath'], dest)
                    copied.append(dest)
                except:
                    pass

        for ev in self.events:
            if ev['event_id'] == ev_id:
                ev['alert_type'] = atype
                ev['processed_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break

        if parent and parent.winfo_exists():
            messagebox.showinfo("处理完成", f"事件 {ev_id} 已标记为{atext}\n已保存到: {ev_dir}\n共处理 {len(copied)} 张截图", parent=parent)
            self.close_alert_window(ev_id)
        else:
            messagebox.showinfo("处理完成", f"事件 {ev_id} 已标记为{atext}\n已保存到: {ev_dir}\n共处理 {len(copied)} 张截图")

        self.update_screenshot_tree()
        if parent and parent.winfo_exists():
            parent.destroy()
        if ev_id in self.alert_windows:
            del self.alert_windows[ev_id]

        if cam == "相机1" and not any(ev_id in a for a in self.unread_alerts.values() if a):
            self.update_map_camera_status(cam, False)
            self.log_message("相机1所有警报已处理，地图状态恢复为灰色")

    def update_map_camera_status(self, cam, alerting):
        self.camera_alert_status[cam] = alerting
        self.draw_parallel_circuit_camera_markers()
        if alerting:
            self.flash_parallel_circuit_camera_marker(cam)

    def play_alert_sound(self):
        try:
            import winsound
            winsound.Beep(1000, 500)
        except:
            pass

    def show_event_details(self, ev_id, alert_win=None):
        if alert_win and alert_win.winfo_exists():
            alert_win.destroy()
        ev = next((e for e in self.events if e['event_id'] == ev_id), None)
        if not ev:
            messagebox.showinfo("信息", f"未找到事件 {ev_id} 的详细信息")
            return
        win = tk.Toplevel(self.root)
        win.title(f"事件详情 - 事件 {ev_id}")
        win.geometry("800x600")
        main = ttk.Frame(win, padding="15")
        main.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        info = ttk.LabelFrame(main, text="事件基本信息", padding="10")
        info.grid(row=0, column=0, sticky="ew", pady=(0,10))
        atype = ev.get('alert_type', '未处理')
        ptime = ev.get('processed_time', '未处理')
        txt = f"事件ID: {ev['event_id']}\n相机: {ev['camera']}\n开始时间: {ev['start_time']}\n处理状态: {atype}\n"
        if atype != '未处理':
            txt += f"处理时间: {ptime}\n"
        txt += f"检测次数: {ev['detection_count']}\n截图数量: {ev['screenshot_count']}"
        ttk.Label(info, text=txt, justify="left", font=("Microsoft YaHei",10)).pack(anchor="w")

        shots = ttk.LabelFrame(main, text="事件截图", padding="10")
        shots.grid(row=1, column=0, sticky="nsew", pady=(0,10))
        main.rowconfigure(1, weight=1)

        ss = [s for s in self.screenshots if s['event_id'] == ev_id]
        ss = sorted(ss, key=lambda x: x['timestamp'], reverse=True)[:3]
        if not ss:
            ttk.Label(shots, text="该事件没有截图", foreground="gray").pack(expand=True)
        else:
            cf = ttk.Frame(shots)
            cf.pack(fill=tk.BOTH, expand=True)
            cv = tk.Canvas(cf, bg="white")
            sb = ttk.Scrollbar(cf, orient="vertical", command=cv.yview)
            inner = ttk.Frame(cv)
            cv.configure(yscrollcommand=sb.set)
            cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            cv.create_window((0,0), window=inner, anchor="nw")
            for i, s in enumerate(ss):
                self._create_event_thumb(inner, s, i)
            inner.update_idletasks()
            cv.configure(scrollregion=cv.bbox("all"))
            inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

        bf = ttk.Frame(main)
        bf.grid(row=2, column=0, sticky="ew", pady=(10,0))
        if ss:
            ttk.Button(bf, text="导出事件截图", command=lambda: self.export_event_screenshots(ev_id)).pack(side=tk.LEFT, padx=(0,10))
        if atype == '未处理':
            ttk.Button(bf, text="处理此警报", command=lambda: self.show_alert_processing_dialog(ev_id, ev['camera'])).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(bf, text="关闭", command=win.destroy).pack(side=tk.LEFT)

    def _create_event_thumb(self, parent, s, idx):
        f = ttk.Frame(parent, relief="solid", borderwidth=1, padding=5)
        f.pack(pady=5, padx=5, fill=tk.X)
        try:
            if 'thumbnail' in s:
                thumb = s['thumbnail']
            else:
                img = cv2.imread(s['filepath'])
                thumb = cv2.resize(img, (120,90))
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(thumb)
            photo = ImageTk.PhotoImage(pil)
            lbl = tk.Label(f, image=photo, cursor="hand2")
            lbl.image = photo
            lbl.pack(side=tk.LEFT, padx=(0,10))
            lbl.bind("<Button-1>", lambda e, p=s['filepath']: self.show_screenshot_detail_by_path(p))
        except:
            pass
        infof = ttk.Frame(f)
        infof.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(infof, text=f"文件: {os.path.basename(s['filepath'])[:40]}...", font=("Microsoft YaHei",9), anchor="w").pack(anchor="w")
        ttk.Label(infof, text=f"时间: {s['timestamp'][:8]} {s['timestamp'][9:15]}", font=("Microsoft YaHei",9), anchor="w").pack(anchor="w")
        bf = ttk.Frame(infof)
        bf.pack(anchor="w", pady=(5,0))
        ttk.Button(bf, text="打开", command=lambda p=s['filepath']: self.open_original_image(p), width=8).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(bf, text="详情", command=lambda p=s['filepath']: self.show_screenshot_detail_by_path(p), width=8).pack(side=tk.LEFT)

    def export_event_screenshots(self, ev_id):
        ss = [s for s in self.screenshots if s['event_id'] == ev_id]
        if not ss:
            messagebox.showwarning("警告", "该事件没有截图")
            return
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
        ev_dir = os.path.join(save_dir, f"event_{ev_id:03d}")
        os.makedirs(ev_dir, exist_ok=True)
        cnt = 0
        for s in ss:
            if os.path.exists(s['filepath']):
                try:
                    import shutil
                    shutil.copy2(s['filepath'], os.path.join(ev_dir, os.path.basename(s['filepath'])))
                    cnt += 1
                except:
                    pass
        with open(os.path.join(ev_dir, f"event_{ev_id}_info.txt"), 'w', encoding='utf-8') as f:
            f.write(f"事件ID: {ev_id}\n截图数量: {len(ss)}\n导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n导出数量: {cnt}\n\n")
            for s in ss:
                f.write(f"文件: {os.path.basename(s['filepath'])}\n时间: {s['timestamp']}\n相机: {s['camera']}\n" + "-"*40 + "\n")
        messagebox.showinfo("成功", f"已导出 {cnt}/{len(ss)} 张截图到:\n{ev_dir}")

    def clear_screenshots(self):
        if not self.screenshots:
            return
        if messagebox.askyesno("确认", f"确定要清空所有{len(self.screenshots)}个截图吗？"):
            for s in self.screenshots:
                if os.path.exists(s['filepath']):
                    os.remove(s['filepath'])
            import shutil
            if os.path.exists(self.screenshot_dir):
                for item in os.listdir(self.screenshot_dir):
                    p = os.path.join(self.screenshot_dir, item)
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
            self.screenshots.clear()
            self.events.clear()
            self.current_event_id = 0
            self.last_event_time = None
            self.update_screenshot_tree()
            self.update_screenshot_statistics()
            self.log_message("所有截图和事件记录已清空")

    def open_screenshot_dir(self):
        try:
            os.startfile(self.screenshot_dir)
        except:
            try:
                import subprocess
                subprocess.run(['xdg-open', self.screenshot_dir])
            except:
                messagebox.showwarning("提示", f"无法打开文件夹: {self.screenshot_dir}")

    def setup_tree_context_menu(self):
        self.tree_menu = tk.Menu(self.root, tearoff=0)
        self.tree_menu.add_command(label="处理未读警报", command=self.process_selected_unread_alert)
        self.tree_menu.add_command(label="打开文件夹", command=self.open_selected_folder)
        self.tree_menu.add_command(label="刷新显示", command=self.refresh_screenshot_tree)
        self.tree_menu.add_separator()
        self.tree_menu.add_command(label="删除选中项", command=self.delete_selected_tree_item)
        self.screenshot_tree.bind("<Button-3>", self._show_tree_menu)

    def _show_tree_menu(self, e):
        item = self.screenshot_tree.identify_row(e.y)
        if item:
            self.screenshot_tree.selection_set(item)
            self.tree_menu.post(e.x_root, e.y_root)

    def process_selected_unread_alert(self):
        sel = self.screenshot_tree.selection()
        if not sel:
            return
        item = sel[0]
        tags = self.screenshot_tree.item(item, "tags")
        text = self.screenshot_tree.item(item, "text")
        import re
        if "unread_event" in tags:
            m = re.search(r'事件\s*(\d+)', text)
            if m:
                ev = int(m.group(1))
                parent = self.screenshot_tree.parent(item)
                if parent:
                    cam = re.sub(r'📷\s*|🔴.*', '', self.screenshot_tree.item(parent, "text")).strip()
                    if cam in self.unread_alerts and ev in self.unread_alerts[cam]:
                        self.show_alert_processing_dialog(ev, cam)
                        return
        messagebox.showinfo("提示", "请选择未读警报事件")

    def refresh_screenshot_tree(self):
        self.update_screenshot_tree()

    def open_selected_folder(self):
        sel = self.screenshot_tree.selection()
        if not sel:
            return
        item = sel[0]
        tags = self.screenshot_tree.item(item, "tags")
        try:
            if "camera" in tags:
                cam = self.screenshot_tree.item(item, "text").replace("📷 ", "")
                p = os.path.join(self.screenshot_dir, self.get_safe_filename(cam))
                if os.path.exists(p):
                    os.startfile(p)
            elif "event" in tags:
                parent = self.screenshot_tree.parent(item)
                if parent:
                    cam = self.screenshot_tree.item(parent, "text").replace("📷 ", "")
                    import re
                    m = re.search(r'事件 (\d+)', self.screenshot_tree.item(item, "text"))
                    if m:
                        p = os.path.join(self.screenshot_dir, self.get_safe_filename(cam), f"event_{int(m.group(1)):03d}")
                        if os.path.exists(p):
                            os.startfile(p)
            elif "screenshot" in tags:
                for t in tags:
                    if t.startswith('/') or '\\' in t:
                        p = os.path.dirname(t)
                        if os.path.exists(p):
                            os.startfile(p)
                        break
        except Exception as e:
            messagebox.showerror("错误", f"打开文件夹失败: {str(e)}")

    def delete_selected_tree_item(self):
        sel = self.screenshot_tree.selection()
        if not sel:
            return
        item = sel[0]
        tags = self.screenshot_tree.item(item, "tags")
        text = self.screenshot_tree.item(item, "text")
        if "camera" in tags:
            cam = text.replace("📷 ", "")
            if messagebox.askyesno("确认", f"确定要删除相机 '{cam}' 的所有截图吗？"):
                self._delete_camera_screenshots(cam)
        elif "event" in tags:
            parent = self.screenshot_tree.parent(item)
            if parent:
                cam = self.screenshot_tree.item(parent, "text").replace("📷 ", "")
                import re
                m = re.search(r'事件 (\d+)', text)
                if m:
                    ev = int(m.group(1))
                    if messagebox.askyesno("确认", f"确定要删除事件 {ev} 的所有截图吗？"):
                        self._delete_event_screenshots(cam, ev)
        elif "screenshot" in tags:
            for t in tags:
                if t.startswith('/') or '\\' in t:
                    if messagebox.askyesno("确认", f"确定要删除截图 '{os.path.basename(t)}' 吗？"):
                        self._delete_single_screenshot(t)
                    break

    def _delete_camera_screenshots(self, cam):
        ss = [s for s in self.screenshots if s['camera'] == cam]
        for s in ss:
            if os.path.exists(s['filepath']):
                os.remove(s['filepath'])
        self.screenshots = [s for s in self.screenshots if s['camera'] != cam]
        self.update_screenshot_tree()
        self.update_screenshot_statistics()
        self.log_message(f"相机 '{cam}' 的所有截图已删除")

    def _delete_event_screenshots(self, cam, ev):
        ss = [s for s in self.screenshots if s['camera'] == cam and s['event_id'] == ev]
        for s in ss:
            if os.path.exists(s['filepath']):
                os.remove(s['filepath'])
        self.screenshots = [s for s in self.screenshots if not (s['camera'] == cam and s['event_id'] == ev)]
        self.update_screenshot_tree()
        self.update_screenshot_statistics()
        self.log_message(f"事件 {ev} 的所有截图已删除")

    def _delete_single_screenshot(self, fp):
        if os.path.exists(fp):
            os.remove(fp)
        self.screenshots = [s for s in self.screenshots if s['filepath'] != fp]
        self.update_screenshot_tree()
        self.update_screenshot_statistics()
        self.log_message(f"截图已删除: {os.path.basename(fp)}")

    def batch_process_unread_alerts(self):
        if not self.unread_alerts:
            messagebox.showinfo("信息", "没有未读警报需要处理")
            return
        win = tk.Toplevel(self.root)
        win.title("批量处理未读警报")
        win.geometry("600x400")
        main = ttk.Frame(win, padding="15")
        main.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        ttk.Label(main, text="未读警报批量处理", font=("Microsoft YaHei",14,"bold")).grid(row=0, column=0, columnspan=3, pady=(0,15))

        tree = ttk.Treeview(main, columns=("相机","事件ID","状态","操作"), show="headings", height=10)
        for col,w in [("相机",100),("事件ID",100),("状态",100),("操作",100)]:
            tree.heading(col, text=col)
            tree.column(col, width=w)
        sb = ttk.Scrollbar(main, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.grid(row=1, column=0, columnspan=3, sticky="nsew")
        sb.grid(row=1, column=3, sticky="ns")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        items = []
        for cam, evs in self.unread_alerts.items():
            for ev in evs:
                items.append((tree.insert("", "end", values=(cam, ev, "未处理", "")), cam, ev))

        bf = ttk.Frame(main)
        bf.grid(row=2, column=0, columnspan=4, pady=(15,0))

        def mark_real():
            for item in tree.selection():
                v = tree.item(item, "values")
                self._handle_batch_alert(int(v[1]), v[0], "real")
                tree.item(item, values=(v[0], v[1], "已标记为实警", "✅"))

        def mark_false():
            for item in tree.selection():
                v = tree.item(item, "values")
                self._handle_batch_alert(int(v[1]), v[0], "false")
                tree.item(item, values=(v[0], v[1], "已标记为虚警", "❌"))

        def mark_read():
            if messagebox.askyesno("确认", "确定将所有未读警报标记为已读吗？"):
                for _, cam, ev in items:
                    self.mark_alert_as_read(ev)
                win.destroy()
                messagebox.showinfo("完成", "所有未读警报已标记为已读")

        ttk.Button(bf, text="标记选中为实警", command=mark_real).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(bf, text="标记选中为虚警", command=mark_false).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="全部标记为已读", command=mark_read).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="关闭", command=win.destroy).pack(side=tk.LEFT, padx=(5,0))

    def _handle_batch_alert(self, ev, cam, atype):
        self.mark_alert_as_read(ev)
        atext = "实警" if atype == "real" else "虚警"
        self.log_message(f"批量处理：事件 {ev} 标记为{atext}")
        target = self.real_alerts_dir if atype == "real" else self.false_alerts_dir
        cam_dir = self.get_safe_filename(cam)
        ev_dir = os.path.join(target, cam_dir, f"event_{ev:03d}")
        os.makedirs(ev_dir, exist_ok=True)

    # ========== 区域编辑器 ==========
    def select_region(self):
        if not self.is_running:
            messagebox.showwarning("警告", "请先启动检测")
            return
        self.show_region_editor()

    def show_region_editor(self):
        if hasattr(self, '_editor_window') and self._editor_window and self._editor_window.winfo_exists():
            self._editor_window.lift()
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("警告", "无法读取当前帧")
            return
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 1)

        win = tk.Toplevel(self.root)
        win.title("检测区域编辑器")
        win.geometry("1000x700")
        win.resizable(True, True)
        win.transient(self.root)
        win.grab_set()
        self._editor_window = win
        win.update_idletasks()
        win.geometry(f"+{self.root.winfo_x()+(self.root.winfo_width()-1000)//2}+{self.root.winfo_y()+(self.root.winfo_height()-700)//2}")

        main = ttk.Frame(win, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        disp = ttk.Frame(main)
        disp.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0,10))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.region_canvas = tk.Canvas(disp, bg="black", cursor="cross")
        self.region_canvas.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(disp, orient="vertical", command=self.region_canvas.yview)
        hsb = ttk.Scrollbar(disp, orient="horizontal", command=self.region_canvas.xview)
        self.region_canvas.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        disp.columnconfigure(0, weight=1)
        disp.rowconfigure(0, weight=1)

        self.region_inner = ttk.Frame(self.region_canvas)
        self.region_canvas.create_window((0,0), window=self.region_inner, anchor="nw", tags="region_frame")

        self.editor_frame = frame.copy()
        self.region_points = []
        self.preview_point = None
        self._display_editor_image()

        ctrl = ttk.LabelFrame(main, text="区域编辑控制", padding="10")
        ctrl.grid(row=0, column=2, sticky="nsew", padx=(10,0))
        main.columnconfigure(2, minsize=250)

        ttk.Label(ctrl, text="操作说明:", font=("Microsoft YaHei",10,"bold")).grid(row=0, column=0, sticky="w", pady=(0,10))
        for i, t in enumerate(["1. 左键点击添加点", "2. 右键点击删除上一个点", "3. 中键点击完成绘制",
                               "4. 双击图像清除所有点", "5. 至少需要3个点形成区域"]):
            ttk.Label(ctrl, text=t, wraplength=220).grid(row=i+1, column=0, sticky="w", pady=(0,5))

        bf = ttk.Frame(ctrl)
        bf.grid(row=6, column=0, pady=(20,10))
        self.undo_btn = ttk.Button(bf, text="↶ 撤销", command=self.undo_editor_point, state="disabled")
        self.undo_btn.grid(row=0, column=0, padx=(0,5))
        self.redo_btn = ttk.Button(bf, text="↷ 重做", command=self.redo_editor_point, state="disabled")
        self.redo_btn.grid(row=0, column=1, padx=5)
        self.clear_btn = ttk.Button(bf, text="🗑️ 清除", command=self.clear_editor_points)
        self.clear_btn.grid(row=0, column=2, padx=(5,0))

        self.point_count_label = ttk.Label(ctrl, text="点数: 0")
        self.point_count_label.grid(row=7, column=0, sticky="w", pady=(0,5))
        self.area_label = ttk.Label(ctrl, text="面积: 0 像素")
        self.area_label.grid(row=8, column=0, sticky="w", pady=(0,10))

        pf = ttk.LabelFrame(ctrl, text="已选点列表", padding="5")
        pf.grid(row=9, column=0, sticky="nsew", pady=(0,10))
        ctrl.rowconfigure(9, weight=1)
        self.points_text = scrolledtext.ScrolledText(pf, height=8, width=25)
        self.points_text.grid(row=0, column=0, sticky="nsew")
        pf.columnconfigure(0, weight=1)
        pf.rowconfigure(0, weight=1)

        btm = ttk.Frame(main)
        btm.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10,0))
        self.apply_btn = ttk.Button(btm, text="✓ 应用区域", command=lambda: self.apply_editor_region(win), state="disabled")
        self.apply_btn.grid(row=0, column=0, padx=(0,5))
        ttk.Button(btm, text="✗ 取消", command=win.destroy).grid(row=0, column=1, padx=5)
        ttk.Button(btm, text="👁️ 预览效果", command=self.preview_region_effect).grid(row=0, column=2, padx=(5,0))

        self.region_canvas.bind("<Button-1>", self._add_editor_point)
        self.region_canvas.bind("<Button-3>", self._remove_last_editor_point)
        self.region_canvas.bind("<Button-2>", self._complete_editor_region)
        self.region_canvas.bind("<Double-Button-1>", self.clear_editor_points)
        self.region_canvas.bind("<Motion>", self._preview_next_editor_point)
        self.region_canvas.bind("<Configure>", lambda e: self._update_editor_canvas())
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_editor_window(win))

        self.point_history = []
        self.redo_stack = []
        self._update_points_display()

    def _display_editor_image(self):
        if self.editor_frame is None:
            return
        for w in self.region_inner.winfo_children():
            w.destroy()
        disp = cv2.cvtColor(self.editor_frame.copy(), cv2.COLOR_BGR2RGB)
        self.editor_pil = Image.fromarray(disp)
        self.editor_photo = ImageTk.PhotoImage(self.editor_pil)
        self.region_image_label = tk.Label(self.region_inner, image=self.editor_photo, cursor="cross")
        self.region_image_label.image = self.editor_photo
        self.region_image_label.bind("<Button-1>", self._add_editor_point)
        self.region_image_label.bind("<Button-3>", self._remove_last_editor_point)
        self.region_image_label.bind("<Button-2>", self._complete_editor_region)
        self.region_image_label.bind("<Double-Button-1>", self.clear_editor_points)
        self.region_image_label.bind("<Motion>", self._preview_next_editor_point)
        self.region_image_label.pack()
        self._update_editor_canvas()

    def _add_editor_point(self, e):
        try:
            x, y = e.x, e.y
            self.point_history.append(self.region_points.copy())
            self.redo_stack.clear()
            self.region_points.append((x, y))
            self.update_editor_display()
            self._update_points_display()
            self.undo_btn.config(state="normal")
            self.redo_btn.config(state="disabled")
            if len(self.region_points) >= 3:
                self.apply_btn.config(state="normal")
        except:
            pass

    def _remove_last_editor_point(self, e):
        if not self.region_points:
            return
        self.point_history.append(self.region_points.copy())
        self.redo_stack.clear()
        self.region_points.pop()
        self.update_editor_display()
        self._update_points_display()
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="disabled")
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def _complete_editor_region(self, e):
        if len(self.region_points) < 3:
            messagebox.showwarning("警告", "至少需要3个点才能形成区域")
            return
        self.apply_btn.config(state="normal")

    def undo_editor_point(self):
        if not self.point_history:
            return
        self.redo_stack.append(self.region_points.copy())
        self.region_points = self.point_history.pop()
        self.update_editor_display()
        self._update_points_display()
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="normal" if self.redo_stack else "disabled")
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def redo_editor_point(self):
        if not self.redo_stack:
            return
        self.point_history.append(self.region_points.copy())
        self.region_points = self.redo_stack.pop()
        self.update_editor_display()
        self._update_points_display()
        self.undo_btn.config(state="normal")
        self.redo_btn.config(state="normal" if self.redo_stack else "disabled")
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def _preview_next_editor_point(self, e):
        if not self.region_points:
            return
        self.preview_point = (e.x, e.y)
        self.update_editor_display()

    def _update_editor_info(self):
        cnt = len(self.region_points)
        self.point_count_label.config(text=f"点数: {cnt}")
        if cnt >= 3:
            area = cv2.contourArea(np.array(self.region_points, np.int32))
            self.area_label.config(text=f"面积: {int(area)} 像素")
        else:
            self.area_label.config(text="面积: 0 像素")

    def _update_points_display(self):
        self.points_text.delete(1.0, tk.END)
        for i, (x, y) in enumerate(self.region_points):
            self.points_text.insert(tk.END, f"{i+1}. ({x}, {y})\n")

    def _update_editor_canvas(self):
        self.region_inner.update_idletasks()
        if self.region_image_label:
            w = self.region_image_label.winfo_reqwidth()
            h = self.region_image_label.winfo_reqheight()
            self.region_canvas.configure(scrollregion=(0,0,w,h), width=min(w,800), height=min(h,600))

    def _close_editor_window(self, win):
        win.destroy()
        self._editor_window = None

    def clear_editor_points(self, e=None):
        if not self.region_points:
            return
        self.point_history.append(self.region_points.copy())
        self.redo_stack.clear()
        self.region_points.clear()
        self.preview_point = None
        self.update_editor_display()
        self._update_points_display()
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="disabled")
        self.apply_btn.config(state="disabled")

    def preview_region_effect(self):
        if len(self.region_points) < 3:
            messagebox.showwarning("警告", "至少需要3个点才能预览")
            return
        mask = np.zeros(self.editor_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.region_points)], 255)
        masked = self.editor_frame.copy()
        masked[mask == 0] = 0
        win = tk.Toplevel(self.root)
        win.title("区域效果预览")
        win.geometry("800x600")
        cv = tk.Canvas(win, bg="black")
        cv.pack(fill=tk.BOTH, expand=True)
        preview = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(preview)
        photo = ImageTk.PhotoImage(pil)
        cv.create_image(0, 0, anchor="nw", image=photo)
        cv.image = photo
        win.update_idletasks()
        cv.config(scrollregion=cv.bbox("all"))

    def apply_editor_region(self, win):
        if len(self.region_points) < 3:
            messagebox.showwarning("警告", "至少需要3个点才能形成区域")
            return
        self.selected_region = self.region_points.copy()
        if win and win.winfo_exists():
            win.destroy()
        self._editor_window = None
        self.root.after(100, self._create_roi_mask)
        self._update_region_ui()
        self.status_var.set(f"检测区域设置完成（{len(self.selected_region)}个点）")
        self.log_message(f"检测区域已设置，包含 {len(self.selected_region)} 个点")
        if self.is_running:
            self.root.after(50, self.process_frame)

    def _create_roi_mask(self):
        if not self.selected_region or not self.is_running or not self.cap:
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 1)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = []
            for x, y in self.selected_region:
                pts.append((max(0, min(x, frame.shape[1]-1)), max(0, min(y, frame.shape[0]-1))))
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts)], 255)
                self.detector.roi_mask = mask
                self.detector.roi_bbox = cv2.boundingRect(np.array(pts))
        except:
            pass

    def _update_region_ui(self):
        if not self.selected_region:
            return
        cnt = len(self.selected_region)
        if cnt <= 4:
            txt = f"检测区域: {cnt}个点"
            for i, (x, y) in enumerate(self.selected_region[:4]):
                txt += f" P{i+1}({x},{y})"
            if cnt > 4:
                txt += " ..."
        else:
            txt = f"检测区域: {cnt}个多边形点 [已应用]"
        self.region_info_label.config(text=txt, foreground="green")
        self.clear_region_btn.config(state="normal")

    def clear_region(self):
        self.selected_region = None
        self.region_points = []
        if hasattr(self.detector, 'roi_mask'):
            self.detector.roi_mask = None
            self.detector.roi_bbox = None
        self.region_info_label.config(text="未设置检测区域", foreground="gray")
        self.clear_region_btn.config(state="disabled")
        self.status_var.set("检测区域已清除")
        self.log_message("检测区域已清除")
        if self.is_running:
            self.root.after(50, self.process_frame)

    # ========== 更新函数 ==========
    def update_alert_indicator(self):
        try:
            if any(self.camera_alert_status.values()):
                alerts = [n for n, s in self.camera_alert_status.items() if s]
                self.status_var.set(f"⚠️ 报警中: {', '.join(alerts[:2])}")
            self.root.after(2000, self.update_alert_indicator)
        except:
            self.root.after(1000, self.update_alert_indicator)

    def set_camera_alert_status(self, cam, alerting):
        old = self.camera_alert_status.get(cam, False)
        self.camera_alert_status[cam] = alerting
        if old != alerting:
            self.update_map_camera_status(cam, alerting)
            if alerting:
                self.flash_compact_circuit_camera_marker(cam)
        self.log_message(f"相机 '{cam}' {'触发报警' if alerting else '报警已解除'}", level="warning" if alerting else "info")

    def update_time_display(self):
        self.time_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_time_display)

    def update_all_displays(self):
        self.downsample_label.config(text=f"{self.downsample_var.get():.2f}")
        self.diff_threshold_label.config(text=str(self.diff_threshold_var.get()))
        self.min_area_label.config(text=str(self.min_area_var.get()))
        self.max_area_label.config(text=str(self.max_area_var.get()))
        for k, lbl in self.threshold_labels.items():
            v = self.threshold_vars[k]
            if k in ['min_speed_consistency', 'min_direction_consistency', 'min_area_stability', 'min_linearity', 'min_confidence']:
                lbl.config(text=f"{v.get()/10.0:.2f}")
            else:
                lbl.config(text=str(v.get()))

    def update_downsample_ratio(self):
        self.detector.downsample_ratio = self.downsample_var.get()
        self.downsample_label.config(text=f"{self.downsample_var.get():.2f}")

    def update_motion_method(self):
        self.detector.motion_method = self.motion_method_var.get()

    def update_diff_threshold(self):
        self.detector.frame_diff_threshold = self.diff_threshold_var.get()
        self.diff_threshold_label.config(text=str(self.diff_threshold_var.get()))

    def update_min_area(self):
        self.detector.min_motion_area = self.min_area_var.get()
        self.min_area_label.config(text=str(self.min_area_var.get()))

    def update_max_area(self):
        self.detector.max_motion_area = self.max_area_var.get()
        self.max_area_label.config(text=str(self.max_area_var.get()))

    def update_track_history(self):
        self.detector.track_history_length = self.track_history_var.get()
        self.track_history_label.config(text=str(self.track_history_var.get()))

    def update_min_track_duration(self):
        self.detector.min_track_duration = self.min_track_duration_var.get()
        self.min_track_duration_label.config(text=str(self.min_track_duration_var.get()))

    def update_max_track_speed(self):
        self.detector.max_track_speed = self.max_track_speed_var.get()
        self.max_track_speed_label.config(text=str(self.max_track_speed_var.get()))

    def update_bg_history(self):
        self.detector.bg_history = self.bg_history_var.get()
        self.bg_history_label.config(text=str(self.bg_history_var.get()))
        self.detector._init_background_subtractors()

    def update_detect_shadows(self):
        self.detector.detect_shadows = self.detect_shadows_var.get()
        self.detector._init_background_subtractors()

    def update_roi_enabled(self):
        self.detector.enable_roi = self.enable_roi_var.get()

    def update_speed_range(self):
        mn, mx = self.min_speed_var.get(), self.max_speed_var.get()
        if mn < mx:
            self.detector.thresholds['speed_range'] = (mn, mx)
            self.min_speed_label.config(text=str(mn))
            self.max_speed_label.config(text=str(mx))
        else:
            messagebox.showwarning("警告", "最小速度必须小于最大速度")
            self.min_speed_var.set(self.detector.thresholds['speed_range'][0])
            self.max_speed_var.set(self.detector.thresholds['speed_range'][1])

    def update_quality_check(self):
        self.detector.enable_quality_check = self.enable_quality_check_var.get()

    def update_skip_night_frames(self):
        self.detector.skip_night_frames = self.skip_night_frames_var.get()

    def update_quality_display(self):
        qi = self.detector.get_quality_info()
        if not qi:
            self.quality_status_var.set("状态: 等待评估")
            self.quality_detail_var.set("")
            return
        is_night = qi.get('is_night', False)
        msg = qi.get('message', '')
        if is_night:
            self.quality_status_var.set("🌙 状态: 天黑/过暗")
            self.quality_status_label.config(foreground="red")
        else:
            ov = qi.get('overall', 0)
            if ov > 0.6:
                self.quality_status_var.set(f"✅ 状态: 良好 ({ov:.2f})")
                self.quality_status_label.config(foreground="green")
            elif ov > 0.3:
                self.quality_status_var.set(f"⚠️ 状态: 一般 ({ov:.2f})")
                self.quality_status_label.config(foreground="orange")
            else:
                self.quality_status_var.set(f"❌ 状态: 较差 ({ov:.2f})")
                self.quality_status_label.config(foreground="red")
        self.quality_detail_var.set(msg)

    def update_screenshot_statistics(self):
        cnt = len(self.screenshots)
        self.screenshot_count_var.set(f"截图数量: {cnt}")
        self.clear_screenshots_btn.config(state="normal" if cnt else "disabled")

    def update_event_interval(self):
        self.event_interval = self.event_interval_var.get()

    def update_alert_enabled(self):
        self.alert_enabled = self.alert_enabled_var.get()

    # ========== 配置保存/加载 ==========
    def save_config(self):
        fp = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")],
                                         initialfile="airborne_detector_config.json")
        if not fp:
            return
        try:
            cfg = {
                'downsample_ratio': self.downsample_var.get(),
                'enable_roi': self.enable_roi_var.get(),
                'motion_method': self.motion_method_var.get(),
                'frame_diff_threshold': self.diff_threshold_var.get(),
                'min_motion_area': self.min_area_var.get(),
                'max_motion_area': self.max_area_var.get(),
                'bg_history': self.bg_history_var.get(),
                'detect_shadows': self.detect_shadows_var.get(),
                'track_history_length': self.track_history_var.get(),
                'min_track_duration': self.min_track_duration_var.get(),
                'max_track_speed': self.max_track_speed_var.get(),
                'thresholds': {}
            }
            for k in self.threshold_vars:
                if k in ['min_speed_consistency', 'min_direction_consistency', 'min_area_stability', 'min_linearity', 'min_confidence']:
                    cfg['thresholds'][k] = self.threshold_vars[k].get() / 10.0
                else:
                    cfg['thresholds'][k] = self.threshold_vars[k].get()
            cfg['thresholds']['speed_range'] = [self.min_speed_var.get(), self.max_speed_var.get()]
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            self.log_message(f"配置已保存到: {fp}")
            messagebox.showinfo("成功", "配置保存成功")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {str(e)}")

    def load_config(self):
        fp = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not fp:
            return
        try:
            if self.detector.load_configuration(fp):
                self.downsample_var.set(self.detector.downsample_ratio)
                self.enable_roi_var.set(self.detector.enable_roi)
                self.motion_method_var.set(self.detector.motion_method)
                self.diff_threshold_var.set(self.detector.frame_diff_threshold)
                self.min_area_var.set(self.detector.min_motion_area)
                self.max_area_var.set(self.detector.max_motion_area)
                self.bg_history_var.set(self.detector.bg_history)
                self.detect_shadows_var.set(self.detector.detect_shadows)
                self.track_history_var.set(self.detector.track_history_length)
                self.min_track_duration_var.set(self.detector.min_track_duration)
                self.max_track_speed_var.set(self.detector.max_track_speed)
                for k in self.threshold_vars:
                    if k in self.detector.thresholds and k != 'speed_range':
                        if k in ['min_speed_consistency', 'min_direction_consistency', 'min_area_stability', 'min_linearity', 'min_confidence']:
                            self.threshold_vars[k].set(int(self.detector.thresholds[k] * 10))
                        else:
                            self.threshold_vars[k].set(self.detector.thresholds[k])
                sr = self.detector.thresholds.get('speed_range', [1, 100])
                self.min_speed_var.set(sr[0])
                self.max_speed_var.set(sr[1])
                self.update_all_displays()
                self.log_message(f"配置已从文件加载: {fp}")
                messagebox.showinfo("成功", "配置加载成功")
            else:
                messagebox.showerror("错误", "加载配置失败")
        except Exception as e:
            messagebox.showerror("错误", f"加载配置失败: {str(e)}")

    def reset_parameters(self):
        if messagebox.askyesno("确认", "确定要重置所有参数为默认值吗？"):
            self.detector = AirborneDetector()
            self.detector.use_sky_detection = False
            self.downsample_var.set(self.detector.downsample_ratio)
            self.enable_roi_var.set(self.detector.enable_roi)
            self.motion_method_var.set(self.detector.motion_method)
            self.diff_threshold_var.set(self.detector.frame_diff_threshold)
            self.min_area_var.set(self.detector.min_motion_area)
            self.max_area_var.set(self.detector.max_motion_area)
            self.bg_history_var.set(self.detector.bg_history)
            self.detect_shadows_var.set(self.detector.detect_shadows)
            self.track_history_var.set(self.detector.track_history_length)
            self.min_track_duration_var.set(self.detector.min_track_duration)
            self.max_track_speed_var.set(self.detector.max_track_speed)
            for k in self.threshold_vars:
                if k in self.detector.thresholds and k != 'speed_range':
                    if k in ['min_speed_consistency', 'min_direction_consistency', 'min_area_stability', 'min_linearity', 'min_confidence']:
                        self.threshold_vars[k].set(int(self.detector.thresholds[k] * 10))
                    else:
                        self.threshold_vars[k].set(self.detector.thresholds[k])
            sr = self.detector.thresholds.get('speed_range', [1, 100])
            self.min_speed_var.set(sr[0])
            self.max_speed_var.set(sr[1])
            self.update_all_displays()
            self.log_message("所有参数已重置为默认值")
            messagebox.showinfo("成功", "参数重置完成")

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.log_message("日志已清空")

    def on_closing(self):
        self.stop_detection()
        for w in list(self.alert_windows.values()):
            if w and w.winfo_exists():
                try:
                    w.destroy()
                except:
                    pass
        self.root.destroy()

def main():
    root = tk.Tk()
    app = AirborneDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    import re
    main()