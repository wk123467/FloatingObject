"""
空飘物检测系统 GUI界面 - 修复版本
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
import math
# 使用颜色直方图对比检测到的空飘物
class AirborneDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("北京北站空飘物智能检测系统")
        self.root.geometry("1400x800")  # 减小初始高度
        self.root.minsize(1200, 700)    # 减小最小尺寸
        
        # 设置样式
        self.setup_styles()
        
        # 初始化检测器
        self.detector = AirborneDetector()

        self.detector.use_sky_detection = False   # 关闭天空检测器
        
        # 视频捕获
        self.cap = None
        self.is_running = False
        self.update_job = None
        
        # 区域选择
        self.selected_region = None
        self.is_selecting_region = False
        self.region_points = []

        # 添加这里：摄像头列表初始化
        self.camera_list = [
            {
                "name": "相机1 (网络)",
                "type": "network",
                "url": "rtsp://admin:admin123@192.168.1.101:554/Streaming/Channels/1",
                "index": None
            },
            {
                "name": "相机2 (网络)",
                "type": "network", 
                "url": "rtsp://admin:password@192.168.1.102:554/h264/ch1/main/av_stream",
                "index": None
            },
            {
                "name": "相机3 (网络)",
                "type": "network",
                "url": "rtsp://username:pass@192.168.1.103:554/stream1",
                "index": None
            },
            {
                "name": "相机4 (本地)",
                "type": "local",
                "url": None,
                "index": 0
            }
        ]
        self.camera_names = [cam["name"] for cam in self.camera_list]

        # ========== 新增：相机报警状态初始化（必须放在这里）==========
        self.camera_alert_status = {}  # 存储每个相机的报警状态
        for cam in self.camera_list:
            self.camera_alert_status[cam["name"]] = False

        # 截图管理
        self.screenshots = []  # 存储截图信息
        self.screenshot_dir = "screenshots"  # 截图保存目录
        self.current_camera_name = ""  # 当前摄像头名称

        # 区域编辑相关
        self._editor_window = None  # 编辑器窗口引用
        self.editor_frame = None    # 编辑器使用的图像帧
        self.region_points = []     # 存储多边形点
        self.preview_point = None   # 预览点
        self.point_history = []     # 用于撤销
        self.redo_stack = []        # 用于重做
        self.editor_scale = 1.0     # 图像缩放比例
        self.region_image_label = None  # 图像显示标签

        # 创建截图目录
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
        
        # 参数配置
        self.current_config = {}
        
        # 日志记录
        self.log_messages = []
        
        # ========== 新增：启动报警指示灯更新 ==========
        self.update_alert_indicator()

        # 创建界面
        self.create_widgets()
        
        # 初始化参数显示
        self.update_all_displays()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # ========== 新增：启动报警指示灯更新 ==========
        # 延迟启动，确保所有控件都已创建
        self.root.after(1000, self.update_alert_indicator)

        # 在 __init__ 方法中找到以下部分：
        # 截图管理
        self.screenshots = []  # 存储截图信息
        self.screenshot_dir = "screenshots"  # 截图保存目录
        self.current_camera_name = ""  # 当前摄像头名称

        # 在这之后添加：
        # 事件管理
        self.events = []  # 存储事件信息
        self.last_event_time = None  # 上次事件时间
        self.event_interval = 30  # 事件间隔秒数
        self.current_event_id = 0  # 当前事件ID
        self.alert_enabled = True  # 是否启用弹窗预警

        # ========== 新增：警报窗口管理 ==========
        self.alert_windows = {}  # 格式: {event_id: window} 存储当前打开的警报窗口


        # ========== 新增：未读警报管理 ==========
        self.unread_alerts = {}  # 格式: {camera_name: [event_id1, event_id2, ...]}
        self.real_alerts_dir = "alerts/real"  # 实警保存目录
        self.false_alerts_dir = "alerts/false"  # 虚警保存目录

        # 创建警报目录
        for dir_path in [self.real_alerts_dir, self.false_alerts_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        self.roi_points = None  # 存储多边形点
        self.roi_mask = None    # 预计算的掩码
        self.roi_bbox = None    # 区域边界框（用于快速判断）

        # 相似度判断配置
        self.similarity_threshold = 0.6      # 相似度阈值
        self.recent_time_threshold = 120     # 只考虑最近2分钟的事件（秒）
        self.enable_similarity_check = True  # 是否启用相似度检查

        # 在初始化完成后，启动演示模式（可选）
        # self.root.after(2000, self.demo_map_alert) 
        
    def update_alert_indicator(self):
        """更新报警指示灯（简化版本）"""
        try:
            # 检查是否有相机处于报警状态
            has_alert = any(status for status in self.camera_alert_status.values())
            
            # 更新地图显示（状态已经在 set_camera_alert_status 中更新）
            if has_alert:
                # 获取报警相机列表
                alert_cameras = [name for name, status in self.camera_alert_status.items() if status]
                if alert_cameras:
                    # 在状态栏显示报警信息
                    self.status_var.set(f"⚠️ 报警中: {', '.join(alert_cameras[:2])}")
            
            # 每2秒更新一次（保持闪烁效果）
            self.root.after(2000, self.update_alert_indicator)
            
        except Exception as e:
            print(f"更新指示灯时出错: {e}")
            self.root.after(1000, self.update_alert_indicator)
    
    def set_camera_alert_status(self, camera_name, is_alerting):
        """设置相机的报警状态"""
        try:
            # 确保 camera_alert_status 属性存在
            if not hasattr(self, 'camera_alert_status'):
                self.camera_alert_status = {}
            
            # 更新相机状态
            old_status = self.camera_alert_status.get(camera_name, False)
            self.camera_alert_status[camera_name] = is_alerting
            
            # 如果状态发生变化，更新地图显示
            if old_status != is_alerting:
                # 使用紧凑版电路图更新方法
                self.update_map_camera_status(camera_name, is_alerting)  # 这个方法已经调用了电路图版本
                
                # 如果变为报警状态，添加闪烁效果
                if is_alerting:
                    self.flash_compact_circuit_camera_marker(camera_name)
            
            # 记录日志
            if is_alerting:
                self.log_message(f"相机 '{camera_name}' 触发报警", level="warning")
            else:
                self.log_message(f"相机 '{camera_name}' 报警已解除", level="info")
                
        except Exception as e:
            print(f"设置相机报警状态时出错: {e}")
    
    # 修改检测方法，使用缓存的掩码
    def is_in_roi(self, x, y):
        """快速判断点是否在ROI内"""
        if self.roi_mask is None:
            return True  # 如果没有ROI，则全部区域都检测
        
        # 首先检查是否在边界框内（快速筛选）
        if self.roi_bbox:
            bx, by, bw, bh = self.roi_bbox
            if x < bx or x >= bx + bw or y < by or y >= by + bh:
                return False
        
        # 然后检查掩码
        return self.roi_mask[y, x] > 0
    
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置颜色
        self.colors = {
            'bg': '#2c3e50',
            'fg': '#ecf0f1',
            'accent': '#3498db',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'panel': '#34495e'
        }
        
        # 配置样式
        style.configure('Title.TLabel', font=('Microsoft YaHei', 16, 'bold'))
        style.configure('Header.TLabel', font=('Microsoft YaHei', 12, 'bold'))
        style.configure('Normal.TLabel', font=('Microsoft YaHei', 10))
        style.configure('Accent.TButton', font=('Microsoft YaHei', 10, 'bold'))
        
        # 配置主窗口背景
        self.root.configure(bg=self.colors['bg'])
    
    def create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 标题栏
        title_frame = ttk.Frame(main_container)
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="北京北站空飘物智能检测系统", 
                               style='Title.TLabel', foreground="black")
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        version_label = ttk.Label(title_frame, text="v1.0", 
                                 style='Normal.TLabel', foreground="black")
        version_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 主内容区域（左右布局）
        content_frame = ttk.Frame(main_container)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # 左侧视频显示区域
        left_panel = ttk.Frame(content_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        content_frame.columnconfigure(0, weight=2)
        content_frame.rowconfigure(0, weight=1)
        
        # 右侧控制面板
        right_panel = ttk.Frame(content_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(1, weight=1)
        
        # 创建左侧面板
        self.create_left_panel(left_panel)
        
        # 创建右侧面板（选项卡）
        self.create_right_panel(right_panel)
        
        # 状态栏
        self.create_status_bar(main_container)
    
    def create_left_panel(self, parent):
        """创建左侧视频显示面板"""
        # 视频显示区域
        video_frame = ttk.LabelFrame(parent, text="视频显示", padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # 原始视频显示
        self.video_label = ttk.Label(video_frame, text="等待启动检测...", 
                                    anchor="center", relief="solid", background="black")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=2)
        
        # 运动掩码显示
        mask_frame = ttk.LabelFrame(video_frame, text="运动检测掩码", padding="5")
        mask_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.rowconfigure(1, weight=1)
        
        self.mask_label = ttk.Label(mask_frame, text="运动掩码", 
                                   anchor="center", relief="solid", background="black")
        self.mask_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        mask_frame.columnconfigure(0, weight=1)
        mask_frame.rowconfigure(0, weight=1)
        
        # 视频控制按钮
        control_frame = ttk.Frame(video_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.start_btn = ttk.Button(control_frame, text="▶ 开始检测", 
                                   command=self.start_selected_camera, style='Accent.TButton')
        self.start_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_btn = ttk.Button(control_frame, text="⏸ 停止检测", 
                                  command=self.stop_detection, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        self.snapshot_btn = ttk.Button(control_frame, text="📸 保存快照", 
                                      command=self.save_snapshot, state="disabled")
        self.snapshot_btn.grid(row=0, column=2, padx=(5, 0))
    
    def create_right_panel(self, parent):
        """创建右侧控制面板"""
        # 创建选项卡
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # 创建各个选项卡
        self.create_input_tab(notebook)
        self.create_detection_tab(notebook)
        self.create_tracking_tab(notebook)
        self.create_threshold_tab(notebook)
        self.create_screenshot_tab(notebook)  # 新增截图选项卡
        self.create_info_tab(notebook)
    
    def put_chinese_text_cv(self, img, text, position, font_size=30, color=(0, 0, 255), 
                        bg_color=None, padding=5):
        """
        使用PIL在OpenCV图像上绘制中文（推荐方案）
        
        参数:
            img: OpenCV图像 (BGR格式)
            text: 要绘制的中文文本
            position: (x, y) 左上角坐标
            font_size: 字体大小
            color: 文字颜色 (B, G, R)
            bg_color: 背景颜色，None表示透明
            padding: 背景填充
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # 转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载字体
        try:
            # 常用字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
                "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
                "msyh.ttc",  # 微软雅黑
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            if font is None:
                # 回退到默认字体
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 计算文本大小
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制背景（可选）
        if bg_color:
            x, y = position
            bg_bbox = (x-padding, y-padding, x+text_width+padding, y+text_height+padding)
            draw.rectangle(bg_bbox, fill=bg_color)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color[::-1])  # RGB转BGR
        
        # 转换回OpenCV格式
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
    
    def create_input_tab(self, notebook):
        """创建输入源选项卡"""
        tab = ttk.Frame(notebook, padding="10")
        notebook.add(tab, text="输入源")
        
        # 摄像头选择 - 减小内边距
        camera_frame = ttk.LabelFrame(tab, text="摄像头选择", padding="8")
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # ========== 修改：清空初始相机列表，用户可以自己添加 ==========
        self.camera_list = []  # 完全清空，不保留任何相机
        self.camera_names = []
        # ========== 修改结束 ==========
        
        # 摄像头选择下拉列表
        ttk.Label(camera_frame, text="选择摄像头:").grid(row=0, column=0, sticky=tk.W, pady=(0, 3))
        
        self.camera_var = tk.StringVar(value="")
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, 
                                        values=self.camera_names, state="readonly", width=20)
        self.camera_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=(0, 3))
        camera_frame.columnconfigure(1, weight=1)
        
        # 摄像头详细信息显示
        self.camera_info_label = ttk.Label(camera_frame, text="请添加相机", foreground="gray")
        self.camera_info_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # 启动摄像头按钮（初始禁用）
        self.start_camera_btn = ttk.Button(camera_frame, text="启动摄像头", 
                                        command=self.start_selected_camera, state="disabled")
        self.start_camera_btn.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 3))
        
        # 自定义摄像头配置按钮
        self.custom_camera_btn = ttk.Button(camera_frame, text="自定义摄像头配置", 
                                        command=self.show_custom_camera_dialog)
        self.custom_camera_btn.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 摄像头状态指示灯
        self.camera_status_label = ttk.Label(camera_frame, text="状态: 未连接", foreground="red")
        self.camera_status_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(3, 0))
        
        # 绑定选择事件
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        # 视频文件选择
        file_frame = ttk.LabelFrame(tab, text="视频文件", padding="8")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.file_path_var = tk.StringVar(value="")
        ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly", 
                width=20).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        file_frame.columnconfigure(0, weight=1)
        
        ttk.Button(file_frame, text="浏览...", 
                command=self.browse_video_file).grid(row=0, column=1, padx=(3, 0), pady=(0, 3))
        
        self.video_file_btn = ttk.Button(file_frame, text="打开视频文件", 
                                        command=self.start_video_file)
        self.video_file_btn.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # ========== 区域设置 - 改为一行 ==========
        region_frame = ttk.LabelFrame(tab, text="检测区域设置", padding="8")
        region_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 将控件放在一行
        control_frame = ttk.Frame(region_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.enable_roi_var = tk.BooleanVar(value=self.detector.enable_roi)
        ttk.Checkbutton(control_frame, text="启用区域检测", variable=self.enable_roi_var,
                    command=self.update_roi_enabled).pack(side=tk.LEFT, padx=(0, 10))
        
        self.select_region_btn = ttk.Button(control_frame, text="🖱️ 绘制检测区域", 
                                        command=self.select_region, width=15)
        self.select_region_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_region_btn = ttk.Button(
            control_frame, 
            text="🗑️ 清除区域", 
            command=self.clear_region,
            state="disabled",
            width=12
        )
        self.clear_region_btn.pack(side=tk.LEFT)
        
        # 区域信息标签放在第二行
        self.region_info_label = ttk.Label(
            region_frame, 
            text="未设置检测区域", 
            foreground="gray",
            wraplength=350
        )
        self.region_info_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # ========== 可编辑的并联电路风格电子地图（简化版）==========
        map_frame = ttk.LabelFrame(tab, text="并联电路 - 相机网络 (可编辑)", padding="5")
        map_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # 创建Canvas用于绘制地图
        self.map_canvas = tk.Canvas(map_frame, width=450, height=200, bg="#1a1a1a", 
                                    highlightthickness=1, highlightbackground="#00ff00")
        self.map_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 地图编辑工具栏
        edit_toolbar = ttk.Frame(map_frame)
        edit_toolbar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(3, 0))
        
        # 添加相机按钮
        self.add_camera_btn = ttk.Button(edit_toolbar, text="➕ 添加相机", 
                                        command=self.start_add_camera_mode, width=12)
        self.add_camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 编辑模式标签
        self.edit_mode_label = ttk.Label(edit_toolbar, text="", foreground="#00ff00")
        self.edit_mode_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # 添加线段按钮
        self.add_line_btn = ttk.Button(edit_toolbar, text="〰️ 添加线段", 
                                    command=self.start_add_line_mode, width=12)
        self.add_line_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # 添加紧凑的图例
        legend_frame = ttk.Frame(map_frame)
        legend_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(3, 0))
        
        # 正常状态图例
        normal_legend = tk.Canvas(legend_frame, width=20, height=20, bg="#1a1a1a", highlightthickness=0)
        normal_legend.create_oval(2, 2, 18, 18, fill="#808080", outline="#404040", width=2)
        normal_legend.create_text(10, 8, text="📷", fill="#e0e0e0", font=("Arial", 8))
        normal_legend.grid(row=0, column=0, padx=(0, 2))
        ttk.Label(legend_frame, text="正常", foreground="#00ff00", background="#1a1a1a", 
                font=("Microsoft YaHei", 8)).grid(row=0, column=1, padx=(0, 8))

        # 报警状态图例
        alert_legend = tk.Canvas(legend_frame, width=20, height=20, bg="#1a1a1a", highlightthickness=0)
        alert_legend.create_oval(2, 2, 18, 18, fill="#ff0000", outline="#990000", width=2)
        alert_legend.create_text(10, 8, text="📷", fill="white", font=("Arial", 8))
        alert_legend.grid(row=0, column=2, padx=(0, 2))
        ttk.Label(legend_frame, text="报警", foreground="#ff0000", background="#1a1a1a", 
                font=("Microsoft YaHei", 8)).grid(row=0, column=3, padx=(0, 8))

        # 线段图例
        line_legend = tk.Canvas(legend_frame, width=20, height=20, bg="#1a1a1a", highlightthickness=0)
        line_legend.create_line(2, 10, 18, 10, fill="#00ff00", width=2)
        line_legend.grid(row=0, column=4, padx=(0, 2))
        ttk.Label(legend_frame, text="线段", foreground="#00ff00", background="#1a1a1a", 
                font=("Microsoft YaHei", 8)).grid(row=0, column=5)
        
        # 初始化地图元素列表
        self.map_lines = []  # 存储自定义线段 [(id, (x1,y1,x2,y2))]
        self.is_edit_mode = False
        self.is_line_mode = False
        self.line_start = None  # 线段起点
        self.temp_line = None   # 临时线段预览
        self.temp_camera = None # 临时相机预览
        
        # 绘制基础并联电路
        self.draw_parallel_circuit_background()
        
        # 初始化相机标记
        self.camera_markers = {}
        self.draw_parallel_circuit_camera_markers()
        
        # 绑定鼠标事件
        self.map_canvas.bind("<Button-1>", self.on_map_click)
        self.map_canvas.bind("<Motion>", self.on_map_motion)
        self.map_canvas.bind("<Button-3>", self.on_map_right_click)
        
        # 初始更新摄像头信息
        self.on_camera_selected(None)
    
    def draw_parallel_circuit_background(self):
        """绘制并联电路风格背景（简化版）- 完全清空背景"""
        # 删除所有背景元素
        for item in self.map_canvas.find_all():
            tags = self.map_canvas.gettags(item)
            # 只删除没有标签或者标签为"background"的元素
            if not tags or "background" in tags:
                self.map_canvas.delete(item)
        
        # 注意：这里不再绘制任何背景元素，地图完全空白
        # 用户可以自由添加相机和线段
        pass  # 空函数，什么都不画

    def draw_parallel_circuit_camera_markers(self):
        """绘制并联电路风格的相机标记"""
        # 只清除相机相关的元素
        for tag in ["camera", "camera_icon", "camera_label"]:
            for item in self.map_canvas.find_withtag(tag):
                self.map_canvas.delete(item)
        
        self.camera_markers.clear()
        
        # 为每个相机绘制标记
        for camera in self.camera_list:
            name = camera["name"]
            x, y = camera["position"]
            
            # 检查报警状态
            is_alert = self.camera_alert_status.get(name, False)
            
            # 根据状态选择颜色
            if is_alert:
                color = "#ff0000"  # 红色 - 报警
                outline_color = "#990000"
                text_color = "white"
            else:
                color = "#808080"  # 灰色 - 正常
                outline_color = "#404040"
                text_color = "#e0e0e0"
            
            # 绘制圆形相机标记
            marker_id = self.map_canvas.create_oval(
                x-12, y-12, x+12, y+12,
                fill=color,
                outline=outline_color,
                width=2,
                tags=("camera", f"camera_{name}")
            )
            
            # 添加相机图标
            icon_id = self.map_canvas.create_text(
                x, y-2,
                text="📷",
                fill=text_color,
                font=("Arial", 10),
                tags=("camera_icon", f"icon_{name}")
            )
            
            # 添加相机名称
            label_id = self.map_canvas.create_text(
                x, y+18,
                text=name,
                fill=text_color,
                font=("Courier", 7, "bold"),
                tags=("camera_label", f"label_{name}")
            )
            
            # 存储标记ID
            self.camera_markers[name] = {
                'marker': marker_id,
                'icon': icon_id,
                'label': label_id
            }
    
    def start_add_line_mode(self):
        """开始添加线段模式"""
        self.is_line_mode = True
        self.is_edit_mode = False
        self.edit_mode_label.config(text="点击起点开始绘制线段")
        self.map_canvas.config(cursor="crosshair")
        self.add_line_btn.config(text="✖️ 取消线段", command=self.cancel_add_line_mode)
        self.add_camera_btn.config(state="disabled")
        self.line_start = None
        
        # 清除所有临时标记
        self.map_canvas.delete("temp_point")
        self.map_canvas.delete("temp_line")
        self.map_canvas.delete("temp_camera")

    def cancel_add_line_mode(self):
        """取消添加线段模式"""
        self.is_line_mode = False
        self.edit_mode_label.config(text="")
        self.map_canvas.config(cursor="")
        self.add_line_btn.config(text="〰️ 添加线段", command=self.start_add_line_mode)
        self.add_camera_btn.config(state="normal")
        
        # 清除所有临时标记
        self.map_canvas.delete("temp_point")
        self.map_canvas.delete("temp_line")
        self.map_canvas.delete("temp_camera")
        self.line_start = None

    def on_map_click(self, event):
        """处理地图点击事件"""
        if self.is_line_mode:
            # 线段绘制模式
            self.handle_line_click(event)
        elif self.is_edit_mode:
            # 添加相机模式
            self.add_new_camera(event.x, event.y)
            self.cancel_add_camera_mode()
        else:
            # 检查是否点击了现有相机（用于编辑位置）
            self.check_camera_selection(event)

    def handle_line_click(self, event):
        """处理线段绘制点击"""
        x, y = event.x, event.y
        
        # 限制坐标在Canvas范围内
        x = max(0, min(x, 450))
        y = max(0, min(y, 200))
        
        if self.line_start is None:
            # 第一个点：设置起点
            self.line_start = (x, y)
            # 清除旧的临时标记
            self.map_canvas.delete("temp_point")
            self.map_canvas.delete("temp_line")
            # 显示起点标记
            self.map_canvas.create_oval(
                x-4, y-4, x+4, y+4, 
                fill="#00ff00", 
                outline="#00aa00",
                width=2,
                tags=("temp_point",)
            )
            # 显示提示文字
            self.edit_mode_label.config(text="点击终点完成线段")
        else:
            # 第二个点：绘制线段
            x1, y1 = self.line_start
            x2, y2 = x, y
            
            # 避免绘制长度为0的线段
            if abs(x1 - x2) < 5 and abs(y1 - y2) < 5:
                self.edit_mode_label.config(text="线段太短，请重新选择起点")
                self.line_start = None
                self.map_canvas.delete("temp_point")
                self.map_canvas.delete("temp_line")
                return
            
            # 生成唯一的线段ID
            line_index = len(self.map_lines)
            line_tag = f"line_{line_index}"
            
            # 创建线段
            line_id = self.map_canvas.create_line(
                x1, y1, x2, y2,
                fill="#00ff00",
                width=2,
                tags=("line", line_tag)
            )
            
            # 保存线段信息
            self.map_lines.append({
                'id': line_id,
                'tag': line_tag,
                'coords': (x1, y1, x2, y2)
            })
            
            # 清除临时标记和预览
            self.map_canvas.delete("temp_point")
            self.map_canvas.delete("temp_line")
            self.line_start = None
            
            # 更新提示
            self.edit_mode_label.config(text="点击起点开始新线段，或点击其他按钮退出")
            self.log_message(f"已添加线段 {line_index+1}", level="info")

    def on_map_motion(self, event):
        """鼠标移动事件"""
        x, y = event.x, event.y
        
        # 限制坐标在Canvas范围内
        x = max(0, min(x, 450))
        y = max(0, min(y, 200))
        
        if self.is_line_mode and self.line_start:
            # 线段绘制模式 - 显示预览线
            # 先清除旧的预览线
            self.map_canvas.delete("temp_line")
            
            x1, y1 = self.line_start
            self.temp_line = self.map_canvas.create_line(
                x1, y1, x, y,
                fill="#00ff00",
                width=2,
                dash=(4, 4),
                tags=("temp_line",)
            )
        elif self.is_edit_mode:
            # 添加相机模式 - 显示预览
            self.map_canvas.delete("temp_camera")
            
            # 显示预览圆形
            self.temp_camera = self.map_canvas.create_oval(
                x-12, y-12, x+12, y+12,
                fill="#00ff00",
                outline="#00aa00",
                width=2,
                stipple="gray50",
                tags=("temp_camera",)
            )
            self.map_canvas.create_text(
                x, y-2, 
                text="📷", 
                fill="black", 
                font=("Arial", 10), 
                tags=("temp_camera",)
            )
            self.map_canvas.create_text(
                x, y+18, 
                text="新相机", 
                fill="#00ff00", 
                font=("Courier", 7), 
                tags=("temp_camera",)
            )
        elif hasattr(self, 'dragging_camera'):
            # 拖动相机模式 - 实时更新位置
            # 先清除所有相机的临时状态
            for camera in self.camera_list:
                if camera["name"] == self.dragging_camera:
                    # 更新位置
                    camera["position"] = (x, y)
                    break
            
            # 重新绘制所有相机
            self.draw_parallel_circuit_camera_markers()

    def on_map_right_click(self, event):
        """右键点击 - 删除元素"""
        items = self.map_canvas.find_overlapping(event.x-5, event.y-5, event.x+5, event.y+5)
        for item in items:
            tags = self.map_canvas.gettags(item)
            for tag in tags:
                if tag.startswith("camera_"):
                    camera_name = tag[7:]
                    self.delete_camera(camera_name)
                    return
                elif tag.startswith("line_"):
                    self.delete_line(item, tag)
                    return
                elif tag == "line":
                    # 如果是通用line标签，需要找到具体的line_x标签
                    for t in tags:
                        if t.startswith("line_"):
                            self.delete_line(item, t)
                            return

    def delete_line(self, line_id, line_tag=None):
        """删除线段"""
        if messagebox.askyesno("确认", "确定要删除这条线段吗？"):
            self.map_canvas.delete(line_id)
            # 从列表中移除
            if line_tag:
                self.map_lines = [line for line in self.map_lines if line['tag'] != line_tag]
            else:
                self.map_lines = [line for line in self.map_lines if line['id'] != line_id]
            self.log_message("线段已删除", level="info")

    def start_add_camera_mode(self):
        """开始添加相机模式"""
        self.is_edit_mode = True
        self.is_line_mode = False
        self.edit_mode_label.config(text="点击地图添加新相机")
        self.map_canvas.config(cursor="crosshair")
        self.add_camera_btn.config(text="✖️ 取消添加", command=self.cancel_add_camera_mode)
        self.add_line_btn.config(state="disabled")
        
        # 清除线段绘制状态
        self.map_canvas.delete("temp_point")
        self.map_canvas.delete("temp_line")
        self.line_start = None

    def cancel_add_camera_mode(self):
        """取消添加相机模式"""
        self.is_edit_mode = False
        self.edit_mode_label.config(text="")
        self.map_canvas.config(cursor="")
        self.add_camera_btn.config(text="➕ 添加相机", command=self.start_add_camera_mode)
        self.add_line_btn.config(state="normal")
        
        # 清除临时预览
        self.map_canvas.delete("temp_camera")
        self.map_canvas.delete("temp_point")
        self.map_canvas.delete("temp_line")

    def check_camera_selection(self, event):
        """检查是否选中了相机（用于移动）"""
        items = self.map_canvas.find_overlapping(event.x-5, event.y-5, event.x+5, event.y+5)
        for item in items:
            tags = self.map_canvas.gettags(item)
            for tag in tags:
                if tag.startswith("camera_"):
                    camera_name = tag[7:]
                    # 开始拖动相机
                    self.start_drag_camera(camera_name, event)
                    return

    def start_drag_camera(self, camera_name, event):
        """开始拖动相机"""
        self.dragging_camera = camera_name
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        # 保存被拖动相机的原始位置
        for camera in self.camera_list:
            if camera["name"] == camera_name:
                self.original_position = camera["position"]
                break
        
        # 改变鼠标样式
        self.map_canvas.config(cursor="fleur")
        
        # 绑定拖动事件
        self.map_canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.map_canvas.bind("<ButtonRelease-1>", self.on_drag_release)

    def on_drag_motion(self, event):
        """拖动相机时的处理 - 解决拖影问题"""
        if hasattr(self, 'dragging_camera'):
            x, y = event.x, event.y
            
            # 只更新被拖动的相机位置
            for camera in self.camera_list:
                if camera["name"] == self.dragging_camera:
                    camera["position"] = (x, y)
                    break
            
            # 重新绘制所有相机（会清除旧的）
            self.draw_parallel_circuit_camera_markers()

    def on_drag_release(self, event):
        """释放拖动时的处理"""
        if hasattr(self, 'dragging_camera'):
            # 恢复鼠标样式
            self.map_canvas.config(cursor="")
            
            # 解除拖动绑定
            self.map_canvas.unbind("<B1-Motion>")
            self.map_canvas.unbind("<ButtonRelease-1>")
            delattr(self, 'dragging_camera')
            if hasattr(self, 'original_position'):
                delattr(self, 'original_position')
            
            # 确保最终位置正确
            self.draw_parallel_circuit_camera_markers()

    def add_new_camera(self, x, y):
        """添加新相机"""
        # 生成新相机名称
        existing_numbers = []
        for cam in self.camera_list:
            name = cam["name"]
            if name.startswith("相机"):
                try:
                    num = int(name[2:])
                    existing_numbers.append(num)
                except:
                    pass
        
        # 找到最小的可用编号
        new_num = 1
        while new_num in existing_numbers:
            new_num += 1
        new_name = f"相机{new_num}"
        
        # 创建新相机
        new_camera = {
            "name": new_name,
            "type": "network",
            "url": f"rtsp://camera{new_num}/stream",
            "index": None,
            "position": (x, y)
        }
        
        self.camera_list.append(new_camera)
        self.camera_names = [cam["name"] for cam in self.camera_list]
        self.camera_combo["values"] = self.camera_names
        
        # 初始化报警状态
        self.camera_alert_status[new_name] = False
        
        # 重绘地图
        self.draw_parallel_circuit_camera_markers()
        
        self.log_message(f"已添加新相机: {new_name}", level="info")

    def delete_camera(self, camera_name):
        """删除相机"""
        if len(self.camera_list) <= 1:
            messagebox.showwarning("警告", "至少保留一个相机")
            return
        
        if messagebox.askyesno("确认", f"确定要删除相机 '{camera_name}' 吗？"):
            # 从列表中移除
            self.camera_list = [cam for cam in self.camera_list if cam["name"] != camera_name]
            self.camera_names = [cam["name"] for cam in self.camera_list]
            self.camera_combo["values"] = self.camera_names
            
            # 从报警状态中移除
            if camera_name in self.camera_alert_status:
                del self.camera_alert_status[camera_name]
            
            # 重绘地图
            self.draw_parallel_circuit_camera_markers()
            
            self.log_message(f"已删除相机: {camera_name}", level="info")

    def flash_compact_circuit_camera_marker(self, camera_name):
        """让报警的相机标记闪烁（圆形版本）"""
        if camera_name in self.camera_markers:
            marker_id = self.camera_markers[camera_name]
            
            def toggle_flash(count=0):
                if count >= 4:  # 闪烁2次后停止
                    # 恢复到稳定状态（红色）
                    self.map_canvas.itemconfig(marker_id, fill="#ff0000", outline="#990000")
                    return
                
                if count % 2 == 0:
                    self.map_canvas.itemconfig(marker_id, fill="#ff9900", outline="#cc6600")  # 橙色闪烁
                else:
                    self.map_canvas.itemconfig(marker_id, fill="#ff0000", outline="#990000")  # 红色
                
                self.root.after(200, lambda: toggle_flash(count + 1))
            
            toggle_flash()

    def update_map_camera_status(self, camera_name, is_alerting):
        """更新地图上相机的状态（纯显示，无点击功能）"""
        # 更新报警状态
        self.camera_alert_status[camera_name] = is_alerting
        
        # 重绘相机标记 - 使用并联电路版本
        self.draw_parallel_circuit_camera_markers()
        
        # 添加闪烁效果
        if is_alerting:
            self.flash_parallel_circuit_camera_marker(camera_name)
    
    def flash_parallel_circuit_camera_marker(self, camera_name):
        """让报警的相机标记闪烁"""
        if camera_name in self.camera_markers:
            marker_id = self.camera_markers[camera_name]
            
            def toggle_flash(count=0):
                if count >= 4:  # 闪烁2次后停止
                    self.map_canvas.itemconfig(marker_id, fill="#ff0000", outline="#990000")
                    return
                
                if count % 2 == 0:
                    self.map_canvas.itemconfig(marker_id, fill="#ff9900", outline="#cc6600")
                else:
                    self.map_canvas.itemconfig(marker_id, fill="#ff0000", outline="#990000")
                
                self.root.after(200, lambda: toggle_flash(count + 1))
            
            toggle_flash()
    
    def draw_map_background(self):
        """绘制电子地图背景"""
        # 绘制铁路线
        self.map_canvas.create_line(50, 100, 500, 100, width=3, fill="#666666", dash=(5, 3))
        self.map_canvas.create_line(100, 200, 450, 200, width=3, fill="#666666", dash=(5, 3))
        
        # 绘制站台
        self.map_canvas.create_rectangle(80, 70, 200, 130, fill="#e0e0e0", outline="#999999")
        self.map_canvas.create_text(140, 100, text="1号站台", font=("Microsoft YaHei", 8))
        
        self.map_canvas.create_rectangle(250, 120, 370, 180, fill="#e0e0e0", outline="#999999")
        self.map_canvas.create_text(310, 150, text="2号站台", font=("Microsoft YaHei", 8))
        
        self.map_canvas.create_rectangle(400, 170, 520, 230, fill="#e0e0e0", outline="#999999")
        self.map_canvas.create_text(460, 200, text="3号站台", font=("Microsoft YaHei", 8))
        
        # 绘制建筑物
        self.map_canvas.create_rectangle(50, 250, 150, 320, fill="#cccccc", outline="#999999")
        self.map_canvas.create_text(100, 285, text="候车厅", font=("Microsoft YaHei", 8))
        
        self.map_canvas.create_rectangle(350, 280, 450, 350, fill="#cccccc", outline="#999999")
        self.map_canvas.create_text(400, 315, text="调度中心", font=("Microsoft YaHei", 8))
        
        # 绘制道路
        self.map_canvas.create_line(200, 350, 300, 350, width=2, fill="#aaaaaa")
        self.map_canvas.create_line(250, 350, 250, 250, width=2, fill="#aaaaaa")
        
        # 绘制标题
        self.map_canvas.create_text(275, 20, text="北京北站 电子地图", 
                                    font=("Microsoft YaHei", 12, "bold"))

    def draw_camera_markers(self):
        """绘制相机标记"""
        # 清除现有标记
        for marker_id in self.camera_markers.values():
            self.map_canvas.delete(marker_id)
        self.camera_markers.clear()
        
        # 为每个相机绘制标记
        for camera in self.camera_list:
            name = camera["name"]
            x, y = camera["position"]
            
            # 检查报警状态
            is_alert = self.camera_alert_status.get(name, False)
            is_selected = (self.camera_var.get() == name)
            
            # 根据状态选择颜色
            if is_alert:
                color = "red"
                outline = "darkred"
            elif is_selected:
                color = "yellow"
                outline = "orange"
            else:
                color = "green"
                outline = "darkgreen"
            
            # 绘制相机标记（使用圆形表示相机）
            marker_id = self.map_canvas.create_oval(
                x-12, y-12, x+12, y+12,
                fill=color, outline=outline, width=2,
                tags=("camera_marker", f"camera_{name}")
            )
            
            # 添加相机图标（摄像头符号）
            self.map_canvas.create_text(x, y-2, text="📷", font=("Microsoft YaHei", 10),
                                        tags=("camera_icon", f"icon_{name}"))
            
            # 添加相机名称
            self.map_canvas.create_text(x, y+20, text=name[:6], font=("Microsoft YaHei", 7),
                                        tags=("camera_label", f"label_{name}"))
            
            # 存储标记ID
            self.camera_markers[name] = marker_id


    def flash_camera_marker(self, camera_name):
        """让报警的相机标记闪烁"""
        if camera_name in self.camera_markers:
            marker_id = self.camera_markers[camera_name]
            
            def toggle_flash(count=0):
                if count >= 6:  # 闪烁3次后停止
                    self.draw_camera_markers()  # 恢复到稳定状态
                    return
                
                if count % 2 == 0:
                    self.map_canvas.itemconfig(marker_id, fill="orange", outline="red")
                else:
                    self.map_canvas.itemconfig(marker_id, fill="red", outline="darkred")
                
                self.root.after(200, lambda: toggle_flash(count + 1))
            
            toggle_flash()

    def demo_map_alert(self):
        """演示模式：让相机2变红"""
        if self.camera_list:
            # 设置相机2为报警状态
            self.update_map_camera_status("相机2 (网络)", True)
            
            # 显示演示提示
            self.status_var.set("演示模式：相机2触发报警")
            self.log_message("演示模式：相机2在地图上显示为红色", level="warning")
    
    def create_detection_tab(self, notebook):
        """创建检测参数选项卡"""
        tab = ttk.Frame(notebook, padding="15")
        notebook.add(tab, text="检测参数")
        
        # 预处理参数
        preprocess_frame = ttk.LabelFrame(tab, text="预处理参数", padding="10")
        preprocess_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(preprocess_frame, text="下采样比例:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.downsample_var = tk.DoubleVar(value=self.detector.downsample_ratio)
        downsample_scale = ttk.Scale(preprocess_frame, from_=0.1, to=1.0, 
                                    variable=self.downsample_var, orient=tk.HORIZONTAL)
        downsample_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        downsample_scale.bind("<ButtonRelease-1>", lambda e: self.update_downsample_ratio())
        self.downsample_label = ttk.Label(preprocess_frame, text=f"{self.downsample_var.get():.2f}")
        self.downsample_label.grid(row=0, column=2, padx=(5, 0), pady=(0, 5))
        
        # 运动检测方法
        method_frame = ttk.LabelFrame(tab, text="运动检测方法", padding="10")
        method_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.motion_method_var = tk.StringVar(value=self.detector.motion_method)
        
        methods = [
            ("帧差法", "frame_diff"),
            ("MOG2背景减法", "mog2"),
            ("KNN背景减法", "knn"),
            ("结合方法", "combine")
        ]
        
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(method_frame, text=text, variable=self.motion_method_var, 
                           value=value, command=self.update_motion_method).grid(row=i, column=0, sticky=tk.W, pady=(0, 5))
        
        # 帧差参数
        diff_frame = ttk.LabelFrame(tab, text="帧差参数", padding="10")
        diff_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(diff_frame, text="帧差阈值:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.diff_threshold_var = tk.IntVar(value=self.detector.frame_diff_threshold)
        diff_scale = ttk.Scale(diff_frame, from_=5, to=50, variable=self.diff_threshold_var, 
                              orient=tk.HORIZONTAL)
        diff_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        diff_scale.bind("<ButtonRelease-1>", lambda e: self.update_diff_threshold())
        self.diff_threshold_label = ttk.Label(diff_frame, text=f"{self.diff_threshold_var.get()}")
        self.diff_threshold_label.grid(row=0, column=2, padx=(5, 0), pady=(0, 5))
        
        # 区域面积参数
        area_frame = ttk.LabelFrame(tab, text="区域面积参数", padding="10")
        area_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(area_frame, text="最小面积:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.min_area_var = tk.IntVar(value=self.detector.min_motion_area)
        min_area_scale = ttk.Scale(area_frame, from_=10, to=500, variable=self.min_area_var, 
                                  orient=tk.HORIZONTAL)
        min_area_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        min_area_scale.bind("<ButtonRelease-1>", lambda e: self.update_min_area())
        self.min_area_label = ttk.Label(area_frame, text=f"{self.min_area_var.get()}")
        self.min_area_label.grid(row=0, column=2, padx=(5, 0), pady=(0, 5))
        
        ttk.Label(area_frame, text="最大面积:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.max_area_var = tk.IntVar(value=self.detector.max_motion_area)
        max_area_scale = ttk.Scale(area_frame, from_=1000, to=20000, variable=self.max_area_var, 
                                  orient=tk.HORIZONTAL)
        max_area_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        max_area_scale.bind("<ButtonRelease-1>", lambda e: self.update_max_area())
        self.max_area_label = ttk.Label(area_frame, text=f"{self.max_area_var.get()}")
        self.max_area_label.grid(row=1, column=2, padx=(5, 0), pady=(0, 5))
    
    def create_tracking_tab(self, notebook):
        """创建跟踪参数选项卡"""
        tab = ttk.Frame(notebook, padding="15")
        notebook.add(tab, text="跟踪参数")
        
        # 跟踪器参数
        tracker_frame = ttk.LabelFrame(tab, text="轨迹跟踪参数", padding="10")
        tracker_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(tracker_frame, text="轨迹历史长度:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.track_history_var = tk.IntVar(value=self.detector.track_history_length)
        track_history_scale = ttk.Scale(tracker_frame, from_=10, to=100, variable=self.track_history_var, 
                                       orient=tk.HORIZONTAL)
        track_history_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        track_history_scale.bind("<ButtonRelease-1>", lambda e: self.update_track_history())
        self.track_history_label = ttk.Label(tracker_frame, text=f"{self.track_history_var.get()}")
        self.track_history_label.grid(row=0, column=2, padx=(5, 0), pady=(0, 5))
        
        ttk.Label(tracker_frame, text="最小跟踪持续帧数:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.min_track_duration_var = tk.IntVar(value=self.detector.min_track_duration)
        min_track_scale = ttk.Scale(tracker_frame, from_=3, to=30, variable=self.min_track_duration_var, 
                                   orient=tk.HORIZONTAL)
        min_track_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        min_track_scale.bind("<ButtonRelease-1>", lambda e: self.update_min_track_duration())
        self.min_track_duration_label = ttk.Label(tracker_frame, text=f"{self.min_track_duration_var.get()}")
        self.min_track_duration_label.grid(row=1, column=2, padx=(5, 0), pady=(0, 5))
        
        ttk.Label(tracker_frame, text="最大跟踪速度:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.max_track_speed_var = tk.IntVar(value=self.detector.max_track_speed)
        max_speed_scale = ttk.Scale(tracker_frame, from_=10, to=200, variable=self.max_track_speed_var, 
                                   orient=tk.HORIZONTAL)
        max_speed_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        max_speed_scale.bind("<ButtonRelease-1>", lambda e: self.update_max_track_speed())
        self.max_track_speed_label = ttk.Label(tracker_frame, text=f"{self.max_track_speed_var.get()}")
        self.max_track_speed_label.grid(row=2, column=2, padx=(5, 0), pady=(0, 5))
        
        # 背景减法器参数
        bg_frame = ttk.LabelFrame(tab, text="背景减法器参数", padding="10")
        bg_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(bg_frame, text="历史帧数:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.bg_history_var = tk.IntVar(value=self.detector.bg_history)
        bg_history_scale = ttk.Scale(bg_frame, from_=10, to=500, variable=self.bg_history_var, 
                                    orient=tk.HORIZONTAL)
        bg_history_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        bg_history_scale.bind("<ButtonRelease-1>", lambda e: self.update_bg_history())
        self.bg_history_label = ttk.Label(bg_frame, text=f"{self.bg_history_var.get()}")
        self.bg_history_label.grid(row=0, column=2, padx=(5, 0), pady=(0, 5))
        
        self.detect_shadows_var = tk.BooleanVar(value=self.detector.detect_shadows)
        ttk.Checkbutton(bg_frame, text="检测阴影", variable=self.detect_shadows_var,
                       command=self.update_detect_shadows).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
    
    def create_threshold_tab(self, notebook):
        """创建阈值参数选项卡"""
        tab = ttk.Frame(notebook, padding="15")
        notebook.add(tab, text="判定阈值")
        
        # 空飘物判定阈值
        threshold_frame = ttk.LabelFrame(tab, text="空飘物判定阈值", padding="10")
        threshold_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        thresholds = [
            ("min_duration", "最小持续时间(帧):", 3, 30),
            ("min_speed_consistency", "速度一致性:", 0.1, 1.0),
            ("min_direction_consistency", "方向一致性:", 0.1, 1.0),
            ("min_area_stability", "面积稳定性:", 0.1, 1.0),
            ("min_linearity", "轨迹线性度:", 0.1, 1.0),
            ("min_confidence", "最小置信度:", 0.1, 1.0)
        ]
        
        self.threshold_vars = {}
        self.threshold_labels = {}
        
        for i, (key, label, min_val, max_val) in enumerate(thresholds):
            ttk.Label(threshold_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=(0, 5))
            
            current_val = self.detector.thresholds[key]
            if isinstance(current_val, float):
                # 对于浮点数，使用缩放因子
                scale_factor = 10  # 将0.1-1.0放大到1-10
                var = tk.IntVar(value=int(current_val * scale_factor))
                format_str = "{:.2f}"
                # 创建更新函数
                scale_cmd = lambda val, k=key, sf=scale_factor: self.update_threshold_scale(k, float(val)/sf)
            else:
                var = tk.IntVar(value=current_val)
                format_str = "{}"
                scale_cmd = lambda val, k=key: self.update_threshold_scale(k, int(val))
            
            self.threshold_vars[key] = var
            
            # 创建Scale控件
            if isinstance(current_val, float):
                scale_min = int(min_val * scale_factor)
                scale_max = int(max_val * scale_factor)
            else:
                scale_min = min_val
                scale_max = max_val
                
            scale = ttk.Scale(threshold_frame, from_=scale_min, to=scale_max, 
                             variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
            scale.bind("<ButtonRelease-1>", lambda e, cmd=scale_cmd, v=var: cmd(v.get()))
            
            # 显示当前值
            label_widget = ttk.Label(threshold_frame, text=format_str.format(current_val))
            self.threshold_labels[key] = label_widget
            label_widget.grid(row=i, column=2, padx=(5, 0), pady=(0, 5))
        
        # 速度范围
        speed_range_frame = ttk.LabelFrame(tab, text="速度范围参数", padding="10")
        speed_range_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(speed_range_frame, text="最小速度:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.min_speed_var = tk.IntVar(value=self.detector.thresholds['speed_range'][0])
        min_speed_scale = ttk.Scale(speed_range_frame, from_=1, to=100, variable=self.min_speed_var, 
                                   orient=tk.HORIZONTAL)
        min_speed_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        min_speed_scale.bind("<ButtonRelease-1>", lambda e: self.update_speed_range())
        self.min_speed_label = ttk.Label(speed_range_frame, text=f"{self.min_speed_var.get()}")
        self.min_speed_label.grid(row=0, column=2, padx=(5, 0), pady=(0, 5))
        
        ttk.Label(speed_range_frame, text="最大速度:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.max_speed_var = tk.IntVar(value=self.detector.thresholds['speed_range'][1])
        max_speed_scale = ttk.Scale(speed_range_frame, from_=10, to=200, variable=self.max_speed_var, 
                                   orient=tk.HORIZONTAL)
        max_speed_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        max_speed_scale.bind("<ButtonRelease-1>", lambda e: self.update_speed_range())
        self.max_speed_label = ttk.Label(speed_range_frame, text=f"{self.max_speed_var.get()}")
        self.max_speed_label.grid(row=1, column=2, padx=(5, 0), pady=(0, 5))
    
    def reset_events(self):
        """重置事件计数"""
        if messagebox.askyesno("确认", "确定要重置事件计数吗？当前事件列表将被清空。"):
            self.events.clear()
            self.current_event_id = 0
            self.last_event_time = None
            self.update_screenshot_statistics()
            self.reset_events_btn.config(state="disabled")
            self.log_message("事件计数已重置")
    
    def batch_process_unread_alerts(self):
        """批量处理未读警报"""
        if not self.unread_alerts:
            messagebox.showinfo("信息", "没有未读警报需要处理")
            return
        
        # 创建批量处理窗口
        batch_window = tk.Toplevel(self.root)
        batch_window.title("批量处理未读警报")
        batch_window.geometry("600x400")
        
        # 主框架
        main_frame = ttk.Frame(batch_window, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        batch_window.columnconfigure(0, weight=1)
        batch_window.rowconfigure(0, weight=1)
        
        # 标题
        ttk.Label(
            main_frame,
            text="未读警报批量处理",
            font=("Microsoft YaHei", 14, "bold")
        ).grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # 创建警报列表
        columns = ("相机", "事件ID", "状态", "操作")
        alerts_tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            alerts_tree.heading(col, text=col)
            alerts_tree.column(col, width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=alerts_tree.yview)
        alerts_tree.configure(yscrollcommand=scrollbar.set)
        
        alerts_tree.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=3, sticky=(tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 填充未读警报
        alert_items = []
        for camera_name, event_ids in self.unread_alerts.items():
            for event_id in event_ids:
                item_id = alerts_tree.insert("", "end", values=(
                    camera_name, event_id, "未处理", ""
                ))
                alert_items.append((item_id, camera_name, event_id))
        
        # 操作按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=(15, 0))
        
        def process_selected_as_real():
            """将选中的警报标记为实警"""
            for item in alerts_tree.selection():
                values = alerts_tree.item(item, "values")
                camera_name = values[0]
                event_id = int(values[1])
                
                # 调用处理函数
                self.handle_batch_alert(event_id, camera_name, "real", batch_window)
                
                # 更新列表状态
                alerts_tree.item(item, values=(camera_name, event_id, "已标记为实警", "✅"))
        
        def process_selected_as_false():
            """将选中的警报标记为虚警"""
            for item in alerts_tree.selection():
                values = alerts_tree.item(item, "values")
                camera_name = values[0]
                event_id = int(values[1])
                
                # 调用处理函数
                self.handle_batch_alert(event_id, camera_name, "false", batch_window)
                
                # 更新列表状态
                alerts_tree.item(item, values=(camera_name, event_id, "已标记为虚警", "❌"))
        
        def mark_all_as_read():
            """将所有未读标记为已读（不分类）"""
            if messagebox.askyesno("确认", "确定将所有未读警报标记为已读吗？"):
                for item_id, camera_name, event_id in alert_items:
                    self.mark_alert_as_read(event_id)
                
                batch_window.destroy()
                messagebox.showinfo("完成", "所有未读警报已标记为已读")
        
        # 创建按钮
        ttk.Button(
            button_frame,
            text="标记选中为实警",
            command=process_selected_as_real
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="标记选中为虚警",
            command=process_selected_as_false
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="全部标记为已读",
            command=mark_all_as_read
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="关闭",
            command=batch_window.destroy
        ).pack(side=tk.LEFT, padx=(5, 0))

    def handle_batch_alert(self, event_id, camera_name, alert_type, batch_window):
        """批量处理中的单个警报处理"""
        # 这里可以调用之前的 handle_alert_type 方法，但需要调整参数
        # 简化版本，直接调用处理逻辑
        try:
            # 标记为已读
            self.mark_alert_as_read(event_id)
            
            # 根据警报类型分类
            target_dir = self.real_alerts_dir if alert_type == "real" else self.false_alerts_dir
            alert_type_text = "实警" if alert_type == "real" else "虚警"
            
            # 创建目录
            camera_dir_name = self.get_safe_filename(camera_name)
            camera_dir = os.path.join(target_dir, camera_dir_name)
            event_dir = os.path.join(camera_dir, f"event_{event_id:03d}")
            
            if not os.path.exists(event_dir):
                os.makedirs(event_dir, exist_ok=True)
            
            # 这里可以添加实际的截图复制逻辑
            # 为了简化，这里只记录操作
            
            self.log_message(f"批量处理：事件 {event_id} 标记为{alert_type_text}", level="info")
            
        except Exception as e:
            self.log_message(f"批量处理失败: {str(e)}", level="error")
    
    def create_screenshot_tab(self, notebook):
        """创建检测截图选项卡"""
        tab = ttk.Frame(notebook, padding="10")
        notebook.add(tab, text="检测截图")
        
        # 主容器
        main_frame = ttk.Frame(tab)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        
        # 工具栏
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 添加批量处理按钮
        self.batch_process_btn = ttk.Button(
            toolbar_frame,
            text="批量处理未读警报",
            command=self.batch_process_unread_alerts,
            state="disabled"
        )
        self.batch_process_btn.grid(row=0, column=0, padx=(0, 5))

        
        # 在其他按钮后面检查是否需要启用批量处理按钮
        # if self.unread_alerts:
        #     self.batch_process_btn.config(state="normal")
        
        self.reset_events_btn = ttk.Button(
            toolbar_frame, 
            text="🔄 重置事件",
            command=self.reset_events,
            state="disabled" 
        )
        self.reset_events_btn.grid(row=0, column=5, padx=(10, 0))
        
        # 操作按钮
        self.clear_screenshots_btn = ttk.Button(
            toolbar_frame, 
            text="清空截图", 
            command=self.clear_screenshots,
            state="disabled"
        )
        self.clear_screenshots_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.open_screenshot_dir_btn = ttk.Button(
            toolbar_frame, 
            text="打开截图文件夹", 
            command=self.open_screenshot_dir
        )
        self.open_screenshot_dir_btn.grid(row=0, column=1, padx=5)
        
        self.auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar_frame, 
            text="自动保存截图", 
            variable=self.auto_save_var
        ).grid(row=0, column=2, padx=(5, 0))
        
        # === 修改这里：改为使用Treeview显示树形结构 ===
        
        # 创建Treeview框架
        tree_frame = ttk.LabelFrame(main_frame, text="截图目录结构", padding="5")
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 创建Treeview（树形结构）
        self.screenshot_tree = ttk.Treeview(
            tree_frame, 
            columns=("type", "count", "time"), 
            show="tree headings",
            height=15
        )
        
        # 设置列
        self.screenshot_tree.heading("#0", text="名称", anchor="w")
        self.screenshot_tree.heading("type", text="类型", anchor="w")
        self.screenshot_tree.heading("count", text="数量", anchor="w")
        self.screenshot_tree.heading("time", text="时间", anchor="w")
        
        self.screenshot_tree.column("#0", width=250, minwidth=200)
        self.screenshot_tree.column("type", width=80, minwidth=80)
        self.screenshot_tree.column("count", width=60, minwidth=60)
        self.screenshot_tree.column("time", width=120, minwidth=120)
        
        # 添加滚动条
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.screenshot_tree.yview)
        self.screenshot_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # 布局
        self.screenshot_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # 预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="截图预览", padding="5")
        preview_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(2, weight=2)
        
        # 创建Canvas用于显示预览
        self.preview_canvas = tk.Canvas(preview_frame, bg="white", highlightthickness=0)
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_canvas.yview)
        
        # 内部Frame用于放置预览图
        self.preview_inner_frame = ttk.Frame(self.preview_canvas)
        
        # 配置Canvas
        self.preview_canvas.configure(yscrollcommand=preview_scrollbar.set)
        self.preview_canvas.create_window((0, 0), window=self.preview_inner_frame, anchor="nw")
        
        # 布局
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # 绑定鼠标滚轮事件
        self.preview_canvas.bind_all("<MouseWheel>", lambda e: self._on_preview_mousewheel(e))
        
        # 绑定Treeview选择事件
        self.screenshot_tree.bind("<<TreeviewSelect>>", self.on_tree_item_selected)
        self.screenshot_tree.bind("<Double-Button-1>", self.on_tree_item_double_click)
        
        # 底部信息栏
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        # 统计信息 - 修改为显示未读警报
        def update_tree_info():
            total_cameras = len(set(s['camera'] for s in self.screenshots))
            total_events = len(set(s['event_id'] for s in self.screenshots))
            total_screenshots = len(self.screenshots)
            total_unread = sum(len(alerts) for alerts in self.unread_alerts.values())
            
            info_text = f"相机: {total_cameras} | 事件: {total_events} | 截图: {total_screenshots} | 未读警报: {total_unread}"
            if total_unread > 0:
                info_text += " 🔴"  # 添加未读标记
            self.tree_info_var.set(info_text)
        
        # 统计信息
        self.tree_info_var = tk.StringVar(value="相机: 0 | 事件: 0 | 截图: 0 | 未读警报: 0")
        ttk.Label(info_frame, textvariable=self.tree_info_var).pack(side=tk.LEFT)
        
        # 事件管理设置（保持不变）
        event_frame = ttk.LabelFrame(main_frame, text="事件管理设置", padding="10")
        event_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 事件间隔设置
        ttk.Label(event_frame, text="事件间隔(秒):").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.event_interval_var = tk.IntVar(value=30)
        interval_spinbox = ttk.Spinbox(event_frame, from_=10, to=300, textvariable=self.event_interval_var, width=10)
        interval_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 5))
        interval_spinbox.bind("<FocusOut>", lambda e: self.update_event_interval())
        
        # 预警开关
        self.alert_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            event_frame, 
            text="启用弹窗预警", 
            variable=self.alert_enabled_var,
            command=self.update_alert_enabled
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # 添加右键菜单
        self.setup_tree_context_menu()
    
    def setup_tree_context_menu(self):
        """设置Treeview的右键菜单"""
        self.tree_menu = tk.Menu(self.root, tearoff=0)
        self.tree_menu.add_command(label="处理未读警报", command=self.process_selected_unread_alert)
        self.tree_menu.add_command(label="打开文件夹", command=self.open_selected_folder)
        self.tree_menu.add_command(label="刷新显示", command=self.refresh_screenshot_tree)
        self.tree_menu.add_separator()
        self.tree_menu.add_command(label="删除选中项", command=self.delete_selected_tree_item)
        
        # 绑定右键事件
        self.screenshot_tree.bind("<Button-3>", self.show_tree_context_menu)

    def process_selected_unread_alert(self):
        """处理选中的未读警报"""
        selection = self.screenshot_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_tags = self.screenshot_tree.item(item, "tags")
        item_text = self.screenshot_tree.item(item, "text")
        
        import re
        
        # 检查是否为未读事件
        if "unread_event" in item_tags:
            # 提取事件ID
            event_match = re.search(r'事件\s*(\d+)', item_text)
            if event_match:
                event_id = int(event_match.group(1))
                
                # 获取父节点（相机节点）
                parent = self.screenshot_tree.parent(item)
                if parent:
                    camera_text = self.screenshot_tree.item(parent, "text")
                    # 从相机文本中提取相机名称（移除未读标记）
                    camera_name = re.sub(r'📷\s*|🔴.*', '', camera_text).strip()
                    
                    # 检查是否确实是未读警报
                    if camera_name in self.unread_alerts and event_id in self.unread_alerts[camera_name]:
                        # 弹出警报处理对话框
                        self.show_alert_processing_dialog(event_id, camera_name)
                        return
        
        messagebox.showinfo("提示", "请选择未读警报事件")

    def show_tree_context_menu(self, event):
        """显示右键菜单"""
        # 选中右键点击的项目
        item = self.screenshot_tree.identify_row(event.y)
        if item:
            self.screenshot_tree.selection_set(item)
            self.tree_menu.post(event.x_root, event.y_root)

    def refresh_screenshot_tree(self):
        """刷新树形结构显示"""
        self.update_screenshot_tree()

    # ========== 修复2：更新 update_screenshot_tree 方法，确保未读标记为红色 ==========
    def update_screenshot_tree(self):
        """更新树形结构显示（添加未读标记）"""
        # 清空现有树
        for item in self.screenshot_tree.get_children():
            self.screenshot_tree.delete(item)
        
        # 配置标签样式（确保颜色生效）
        self.screenshot_tree.tag_configure("unread_camera", foreground="red")
        self.screenshot_tree.tag_configure("unread_event", foreground="red")
        
        # 按相机分组
        cameras_data = {}
        for screenshot in self.screenshots:
            camera_name = screenshot['camera']
            event_id = screenshot['event_id']
            
            if camera_name not in cameras_data:
                cameras_data[camera_name] = {}
            
            if event_id not in cameras_data[camera_name]:
                cameras_data[camera_name][event_id] = []
            
            cameras_data[camera_name][event_id].append(screenshot)
        
        # 添加相机节点
        total_cameras = len(cameras_data)
        total_events = 0
        total_screenshots = len(self.screenshots)
        
        for camera_name, events_data in cameras_data.items():
            # 计算该相机的统计
            event_count = len(events_data)
            screenshot_count = sum(len(screenshots) for screenshots in events_data.values())
            
            # 检查未读警报
            unread_count = self.get_unread_alert_count(camera_name)
            
            # 相机节点文本（添加未读标记）
            if unread_count > 0:
                camera_text = f"📷 {camera_name} 🔴({unread_count})"  # 红色未读标记
                camera_tags = ("camera", "unread_camera")
            else:
                camera_text = f"📷 {camera_name}"
                camera_tags = ("camera",)
            
            # 添加相机节点
            camera_id = self.screenshot_tree.insert(
                "", "end", 
                text=camera_text, 
                values=("相机", f"{event_count}事件", ""),
                tags=camera_tags
            )
            
            total_events += event_count
            
            # 添加事件节点
            for event_id, screenshots in events_data.items():
                # 获取事件信息
                event_info = None
                for event in self.events:
                    if event['event_id'] == event_id:
                        event_info = event
                        break
                
                event_time = event_info['start_time'] if event_info else "未知时间"
                
                # 检查是否为未读事件
                is_unread = camera_name in self.unread_alerts and event_id in self.unread_alerts[camera_name]
                
                # 事件节点文本
                if is_unread:
                    event_text = f"📁 事件 {event_id} 🔴"  # 红色未读标记
                    event_tags = ("event", "unread_event")
                else:
                    event_text = f"📁 事件 {event_id}"
                    event_tags = ("event",)
                
                # 添加事件节点
                event_id_str = self.screenshot_tree.insert(
                    camera_id, "end",
                    text=event_text,
                    values=("事件", f"{len(screenshots)}张", event_time),
                    tags=event_tags
                )
                
                # 添加截图节点
                if len(screenshots) <= 10:
                    for i, screenshot in enumerate(screenshots[:10]):
                        timestamp_display = screenshot['timestamp'][9:17]
                        self.screenshot_tree.insert(
                            event_id_str, "end",
                            text=f"🖼️ {screenshot['filename'][:30]}...",
                            values=("截图", f"{i+1}/{len(screenshots)}", timestamp_display),
                            tags=("screenshot", screenshot['filepath'])
                        )
                else:
                    self.screenshot_tree.insert(
                        event_id_str, "end",
                        text=f"📊 {len(screenshots)}张截图",
                        values=("截图集", f"{len(screenshots)}张", ""),
                        tags=("screenshot_summary",)
                    )
        
        # 更新统计信息（添加未读警报统计）
        total_unread = sum(len(alerts) for alerts in self.unread_alerts.values())
        self.tree_info_var.set(f"相机: {total_cameras} | 事件: {total_events} | 截图: {total_screenshots} | 未读警报: {total_unread}")
        
        # 如果没有数据，显示提示
        if not cameras_data:
            self.screenshot_tree.insert("", "end", text="暂无检测截图", values=("", "", ""))
        
        # 展开所有相机节点
        for camera_id in self.screenshot_tree.get_children():
            self.screenshot_tree.item(camera_id, open=True)

    def on_tree_item_double_click(self, event):
        """树形项目双击事件 - 修复未读警报处理"""
        selection = self.screenshot_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_tags = self.screenshot_tree.item(item, "tags")
        item_text = self.screenshot_tree.item(item, "text")
        
        # 1. 首先检查是否为未读事件（包含"unread_event"标签）
        if "unread_event" in item_tags:
            # 提取相机名称和事件ID
            import re
            
            # 从事件文本中提取事件ID
            event_match = re.search(r'事件\s*(\d+)', item_text)
            if event_match:
                event_id = int(event_match.group(1))
                
                # 获取父节点（相机节点）
                parent = self.screenshot_tree.parent(item)
                if parent:
                    camera_text = self.screenshot_tree.item(parent, "text")
                    # 从相机文本中提取相机名称（移除未读标记）
                    camera_name = re.sub(r'📷\s*|🔴.*', '', camera_text).strip()
                    
                    # 检查是否确实是未读警报
                    if camera_name in self.unread_alerts and event_id in self.unread_alerts[camera_name]:
                        # 弹出警报处理对话框
                        self.show_alert_processing_dialog(event_id, camera_name)
                        return
        
        # 2. 原有的双击处理逻辑
        if "screenshot" in item_tags:
            # 双击截图：显示大图
            for tag in item_tags:
                if tag.startswith('/') or '\\' in tag:
                    self.show_screenshot_detail_by_path(tag)
                    break
        elif "event" in item_tags or "unread_event" in item_tags:
            # 双击事件：展开/折叠
            if self.screenshot_tree.item(item, "open"):
                self.screenshot_tree.item(item, open=False)
            else:
                self.screenshot_tree.item(item, open=True)
        elif "camera" in item_tags or "unread_camera" in item_tags:
            # 双击相机：展开/折叠
            if self.screenshot_tree.item(item, "open"):
                self.screenshot_tree.item(item, open=False)
            else:
                self.screenshot_tree.item(item, open=True)


    def show_alert_processing_dialog(self, event_id, camera_name):
        """显示警报处理对话框（用于处理未读警报）- 修复版本"""
        try:
            print(f"DEBUG: 显示警报处理对话框 - 事件 {event_id}, 相机 {camera_name}")
            
            # 检查是否已在警报窗口中打开
            if event_id in self.alert_windows:
                window = self.alert_windows[event_id]
                if window and window.winfo_exists():
                    print(f"DEBUG: 窗口已存在，提升到前面")
                    window.lift()
                    window.focus_set()
                    return
            
            print(f"DEBUG: 创建新窗口")
            
            # 创建对话框
            dialog = tk.Toplevel(self.root)
            dialog.title(f"处理未读警报 - 事件 {event_id}")
            dialog.geometry("500x400")
            dialog.resizable(False, False)
            dialog.transient(self.root)
            dialog.grab_set()
            
            # 保存窗口引用
            self.alert_windows[event_id] = dialog
            
            # 居中显示
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
            dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            # 主框架
            main_frame = ttk.Frame(dialog, padding="20")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            dialog.columnconfigure(0, weight=1)
            dialog.rowconfigure(0, weight=1)
            
            # 标题
            title_label = ttk.Label(
                main_frame,
                text="🔴 处理未读警报",
                font=("Microsoft YaHei", 14, "bold"),
                foreground="red"
            )
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
            
            # 警报信息
            info_frame = ttk.LabelFrame(main_frame, text="警报信息", padding="10")
            info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
            
            current_time = datetime.now().strftime("%H:%M:%S")
            info_lines = [
                f"事件编号: {event_id}",
                f"来源相机: {camera_name}",
                f"处理时间: {current_time}",
                f"状态: 未处理"
            ]
            
            for i, line in enumerate(info_lines):
                ttk.Label(
                    info_frame,
                    text=line,
                    font=("Microsoft YaHei", 10)
                ).grid(row=i, column=0, sticky=tk.W, pady=2)
            
            # 处理选项
            options_frame = ttk.LabelFrame(main_frame, text="请选择处理方式", padding="15")
            options_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
            
            # 实警按钮
            def mark_as_real():
                print(f"DEBUG: 标记为实警 - 事件 {event_id}")
                self.handle_alert_type(event_id, camera_name, "real", dialog)
            
            real_btn = ttk.Button(
                options_frame,
                text="✅ 标记为实警",
                command=mark_as_real,
                width=20
            )
            real_btn.grid(row=0, column=0, padx=(0, 10), pady=10)
            
            # 虚警按钮
            def mark_as_false():
                print(f"DEBUG: 标记为虚警 - 事件 {event_id}")
                self.handle_alert_type(event_id, camera_name, "false", dialog)
            
            false_btn = ttk.Button(
                options_frame,
                text="❌ 标记为虚警",
                command=mark_as_false,
                width=20
            )
            false_btn.grid(row=0, column=1, pady=10)
            
            # 稍后处理按钮
            def mark_as_read_later():
                print(f"DEBUG: 稍后处理 - 事件 {event_id}")
                # 只标记为已读，不分类
                self.mark_alert_as_read(event_id)
                self.log_message(f"事件 {event_id} 标记为已读（未分类）")
                self.close_alert_window(event_id)
                # 更新树形显示
                self.update_screenshot_tree()
            
            later_btn = ttk.Button(
                options_frame,
                text="查看事件详情",
                command=lambda: self.show_event_details(event_id, dialog),
                width=20
            )
            later_btn.grid(row=1, column=0, columnspan=2, pady=(10, 0))
            
            # 底部按钮
            bottom_frame = ttk.Frame(main_frame)
            bottom_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
            
            # 查看详情按钮
            ttk.Button(
                bottom_frame,
                text="查看事件详情",
                command=lambda: self.show_event_details(event_id, dialog)
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            # 关闭按钮
            ttk.Button(
                bottom_frame,
                text="关闭",
                command=lambda: self.close_alert_window(event_id)
            ).pack(side=tk.LEFT)
            
            # 配置列权重
            options_frame.columnconfigure(0, weight=1)
            options_frame.columnconfigure(1, weight=1)
            
            # 绑定窗口关闭事件
            dialog.protocol("WM_DELETE_WINDOW", lambda: None)
            
            print(f"DEBUG: 对话框创建完成")
            
        except Exception as e:
            print(f"DEBUG: 显示警报处理对话框失败: {str(e)}")
            self.log_message(f"显示警报处理对话框失败: {str(e)}", level="error")

    def on_tree_item_selected(self, event):
        """树形项目被选中时的处理"""
        selection = self.screenshot_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_tags = self.screenshot_tree.item(item, "tags")
        
        # 根据标签类型处理
        if "screenshot" in item_tags:
            # 获取文件路径
            for tag in item_tags:
                if tag.startswith('/') or '\\' in tag:  # 判断是否为文件路径
                    filepath = tag
                    # 在预览区域显示该截图
                    self.display_selected_screenshot(filepath)
                    break
        elif "event" in item_tags:
            # 显示事件的所有截图
            self.display_event_screenshots(item)
        elif "camera" in item_tags:
            # 显示相机的所有截图
            self.display_camera_screenshots(item)

    def on_tree_item_double_click(self, event):
        """树形项目双击事件"""
        selection = self.screenshot_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_tags = self.screenshot_tree.item(item, "tags")
        
        if "screenshot" in item_tags:
            # 双击截图：显示大图
            for tag in item_tags:
                if tag.startswith('/') or '\\' in tag:
                    self.show_screenshot_detail_by_path(tag)
                    break
        elif "event" in item_tags:
            # 双击事件：展开/折叠
            if self.screenshot_tree.item(item, "open"):
                self.screenshot_tree.item(item, open=False)
            else:
                self.screenshot_tree.item(item, open=True)
        elif "camera" in item_tags:
            # 双击相机：展开/折叠
            if self.screenshot_tree.item(item, "open"):
                self.screenshot_tree.item(item, open=False)
            else:
                self.screenshot_tree.item(item, open=True)

    def display_selected_screenshot(self, filepath):
        """在预览区域显示选中的截图"""
        try:
            # 清空预览区域
            for widget in self.preview_inner_frame.winfo_children():
                widget.destroy()
            
            # 加载并显示图像
            if os.path.exists(filepath):
                image = cv2.imread(filepath)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image)
                    
                    # 调整大小以适应预览区域
                    max_width = 300
                    max_height = 200
                    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # 创建标签显示图像
                    label = tk.Label(self.preview_inner_frame, image=photo)
                    label.image = photo
                    label.pack(pady=10)
                    
                    # 显示文件名
                    filename = os.path.basename(filepath)
                    ttk.Label(
                        self.preview_inner_frame,
                        text=filename,
                        font=("Microsoft YaHei", 9)
                    ).pack()
                    
                    # 更新Canvas区域
                    self.update_preview_canvas()
        except Exception as e:
            print(f"显示截图预览失败: {e}")

    def display_event_screenshots(self, event_item):
        """显示事件的所有截图"""
        try:
            # 清空预览区域
            for widget in self.preview_inner_frame.winfo_children():
                widget.destroy()
            
            # 获取事件ID
            item_text = self.screenshot_tree.item(event_item, "text")
            import re
            match = re.search(r'事件 (\d+)', item_text)
            if match:
                event_id = int(match.group(1))
                
                # 获取该事件的所有截图
                event_screenshots = [s for s in self.screenshots if s['event_id'] == event_id]
                
                if not event_screenshots:
                    ttk.Label(
                        self.preview_inner_frame,
                        text="该事件没有截图",
                        foreground="gray"
                    ).pack(pady=50)
                    return
                
                # 显示事件信息
                event_info = None
                for event in self.events:
                    if event['event_id'] == event_id:
                        event_info = event
                        break
                
                if event_info:
                    info_text = f"事件 {event_id} - {event_info['camera']}\n"
                    info_text += f"开始时间: {event_info['start_time']}\n"
                    info_text += f"检测次数: {event_info['detection_count']}\n"
                    info_text += f"截图数量: {event_info['screenshot_count']}"
                    
                    ttk.Label(
                        self.preview_inner_frame,
                        text=info_text,
                        font=("Microsoft YaHei", 10, "bold"),
                        justify="left"
                    ).pack(pady=(10, 20))
                
                # 显示截图缩略图网格
                self.display_screenshot_grid(event_screenshots[:20])  # 最多显示20张
            
        except Exception as e:
            print(f"显示事件截图失败: {e}")

    def display_camera_screenshots(self, camera_item):
        """显示相机的所有截图"""
        try:
            # 清空预览区域
            for widget in self.preview_inner_frame.winfo_children():
                widget.destroy()
            
            # 获取相机名称
            item_text = self.screenshot_tree.item(camera_item, "text")
            camera_name = item_text.replace("📷 ", "")
            
            # 获取该相机的所有截图
            camera_screenshots = [s for s in self.screenshots if s['camera'] == camera_name]
            
            if not camera_screenshots:
                ttk.Label(
                    self.preview_inner_frame,
                    text="该相机没有截图",
                    foreground="gray"
                ).pack(pady=50)
                return
            
            # 显示相机信息
            info_text = f"📷 {camera_name}\n"
            info_text += f"总截图数: {len(camera_screenshots)}\n"
            
            # 统计事件
            event_ids = set(s['event_id'] for s in camera_screenshots)
            info_text += f"事件数量: {len(event_ids)}"
            
            ttk.Label(
                self.preview_inner_frame,
                text=info_text,
                font=("Microsoft YaHei", 10, "bold"),
                justify="left"
            ).pack(pady=(10, 20))
            
            # 显示最近的事件截图
            recent_screenshots = sorted(
                camera_screenshots,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:20]  # 最多显示20张
            
            self.display_screenshot_grid(recent_screenshots)
            
        except Exception as e:
            print(f"显示相机截图失败: {e}")

    def display_screenshot_grid(self, screenshots):
        """以网格形式显示截图"""
        # 创建网格框架
        grid_frame = ttk.Frame(self.preview_inner_frame)
        grid_frame.pack(fill="both", expand=True)
        
        # 每行显示3个缩略图
        for i, screenshot in enumerate(screenshots):
            row = i // 3
            col = i % 3
            
            # 创建缩略图容器
            thumb_frame = ttk.Frame(grid_frame, relief="solid", borderwidth=1, padding=2)
            thumb_frame.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            grid_frame.columnconfigure(col, weight=1)
            
            try:
                # 加载缩略图
                thumbnail = screenshot['thumbnail']
                pil_image = Image.fromarray(thumbnail)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 创建标签
                label = tk.Label(thumb_frame, image=photo, cursor="hand2")
                label.image = photo
                label.pack()
                
                # 绑定点击事件
                label.bind("<Button-1>", 
                    lambda e, path=screenshot['filepath']: self.show_screenshot_detail_by_path(path))
                
                # 显示简要信息
                info_text = f"事件 {screenshot['event_id']}\n"
                info_text += f"检测 {screenshot['id']}\n"
                info_text += f"{screenshot['timestamp'][9:17]}"
                
                ttk.Label(
                    thumb_frame,
                    text=info_text,
                    font=("Microsoft YaHei", 7),
                    justify="center"
                ).pack(pady=(2, 0))
                
            except Exception as e:
                print(f"创建缩略图失败: {e}")
        
        # 更新预览Canvas
        self.update_preview_canvas()

    def update_preview_canvas(self):
        """更新预览Canvas区域"""
        self.preview_inner_frame.update_idletasks()
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

    def _on_preview_mousewheel(self, event):
        """处理预览区域的鼠标滚轮事件"""
        if self.preview_canvas.winfo_exists():
            self.preview_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def show_screenshot_detail_by_path(self, filepath):
        """通过文件路径显示截图详情"""
        # 查找对应的截图信息
        screenshot_info = None
        for screenshot in self.screenshots:
            if screenshot['filepath'] == filepath:
                screenshot_info = screenshot
                break
        
        if screenshot_info:
            self.show_screenshot_detail(screenshot_info)

    def open_selected_folder(self):
        """打开选中的文件夹"""
        selection = self.screenshot_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_tags = self.screenshot_tree.item(item, "tags")
        
        try:
            if "camera" in item_tags:
                # 打开相机文件夹
                item_text = self.screenshot_tree.item(item, "text")
                camera_name = item_text.replace("📷 ", "")
                camera_dir_name = self.get_safe_filename(camera_name)
                camera_path = os.path.join(self.screenshot_dir, camera_dir_name)
                
                if os.path.exists(camera_path):
                    os.startfile(camera_path)
                else:
                    messagebox.showinfo("提示", f"文件夹不存在: {camera_path}")
                    
            elif "event" in item_tags:
                # 打开事件文件夹
                parent = self.screenshot_tree.parent(item)
                if parent:
                    camera_text = self.screenshot_tree.item(parent, "text")
                    camera_name = camera_text.replace("📷 ", "")
                    camera_dir_name = self.get_safe_filename(camera_name)
                    
                    item_text = self.screenshot_tree.item(item, "text")
                    import re
                    match = re.search(r'事件 (\d+)', item_text)
                    if match:
                        event_id = match.group(1)
                        event_path = os.path.join(self.screenshot_dir, camera_dir_name, f"event_{int(event_id):03d}")
                        
                        if os.path.exists(event_path):
                            os.startfile(event_path)
                        else:
                            messagebox.showinfo("提示", f"文件夹不存在: {event_path}")
            
            elif "screenshot" in item_tags:
                # 打开截图所在文件夹
                for tag in item_tags:
                    if tag.startswith('/') or '\\' in tag:
                        folder_path = os.path.dirname(tag)
                        if os.path.exists(folder_path):
                            os.startfile(folder_path)
                        break
        
        except Exception as e:
            messagebox.showerror("错误", f"打开文件夹失败: {str(e)}")

    def delete_selected_tree_item(self):
        """删除选中的树形项目"""
        selection = self.screenshot_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_tags = self.screenshot_tree.item(item, "tags")
        item_text = self.screenshot_tree.item(item, "text")
        
        if "camera" in item_tags:
            # 删除整个相机
            camera_name = item_text.replace("📷 ", "")
            if messagebox.askyesno("确认", f"确定要删除相机 '{camera_name}' 的所有截图吗？"):
                self.delete_camera_screenshots(camera_name)
                
        elif "event" in item_tags:
            # 删除事件
            parent = self.screenshot_tree.parent(item)
            if parent:
                camera_text = self.screenshot_tree.item(parent, "text")
                camera_name = camera_text.replace("📷 ", "")
                
                import re
                match = re.search(r'事件 (\d+)', item_text)
                if match:
                    event_id = int(match.group(1))
                    if messagebox.askyesno("确认", f"确定要删除事件 {event_id} 的所有截图吗？"):
                        self.delete_event_screenshots(camera_name, event_id)
        
        elif "screenshot" in item_tags:
            # 删除单个截图
            for tag in item_tags:
                if tag.startswith('/') or '\\' in tag:
                    filename = os.path.basename(tag)
                    if messagebox.askyesno("确认", f"确定要删除截图 '{filename}' 吗？"):
                        self.delete_single_screenshot(tag)
                    break

    def delete_camera_screenshots(self, camera_name):
        """删除相机的所有截图"""
        try:
            # 获取相机对应的截图
            camera_screenshots = [s for s in self.screenshots if s['camera'] == camera_name]
            
            # 删除文件
            for screenshot in camera_screenshots:
                if os.path.exists(screenshot['filepath']):
                    os.remove(screenshot['filepath'])
            
            # 从列表中移除
            self.screenshots = [s for s in self.screenshots if s['camera'] != camera_name]
            
            # 更新树形显示
            self.update_screenshot_tree()
            
            # 更新统计
            self.update_screenshot_statistics()
            
            self.log_message(f"相机 '{camera_name}' 的所有截图已删除")
            
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {str(e)}")

    def delete_event_screenshots(self, camera_name, event_id):
        """删除事件的所有截图"""
        try:
            # 获取事件对应的截图
            event_screenshots = [s for s in self.screenshots if s['camera'] == camera_name and s['event_id'] == event_id]
            
            # 删除文件
            for screenshot in event_screenshots:
                if os.path.exists(screenshot['filepath']):
                    os.remove(screenshot['filepath'])
            
            # 从列表中移除
            self.screenshots = [s for s in self.screenshots if not (s['camera'] == camera_name and s['event_id'] == event_id)]
            
            # 更新树形显示
            self.update_screenshot_tree()
            
            # 更新统计
            self.update_screenshot_statistics()
            
            self.log_message(f"事件 {event_id} 的所有截图已删除")
            
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {str(e)}")

    def delete_single_screenshot(self, filepath):
        """删除单个截图"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # 从列表中移除
            self.screenshots = [s for s in self.screenshots if s['filepath'] != filepath]
            
            # 更新树形显示
            self.update_screenshot_tree()
            
            # 更新统计
            self.update_screenshot_statistics()
            
            self.log_message(f"截图已删除: {os.path.basename(filepath)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {str(e)}")

    def get_safe_filename(self, filename):
        """获取安全的文件名（移除非法字符）"""
        import re
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    def update_event_interval(self):
        """更新事件间隔"""
        self.event_interval = self.event_interval_var.get()
        self.log_message(f"事件间隔已更新为 {self.event_interval} 秒")

    def update_alert_enabled(self):
        """更新预警开关"""
        self.alert_enabled = self.alert_enabled_var.get()
        status = "启用" if self.alert_enabled else "禁用"
        self.log_message(f"弹窗预警已{status}")

    def update_screenshot_statistics(self):
        """更新截图统计"""
        # 截图数量
        self.screenshot_count_var.set(f"截图数量: {len(self.screenshots)}")
        
        # 事件统计
        event_count = len(self.events)
        self.event_count_var.set(f"事件总数: {event_count}")
        
        if self.events:
            last_event = self.events[-1]
            self.last_event_var.set(f"上次事件: #{last_event['event_id']} ({last_event['camera']})")
        
        if len(self.screenshots) == 0:
            self.clear_screenshots_btn.config(state="disabled")
        else:
            self.clear_screenshots_btn.config(state="normal")

    def show_event_list(self):
        """显示事件列表"""
        if not self.events:
            messagebox.showinfo("事件列表", "暂无事件记录")
            return
        
        # 创建事件列表窗口
        event_window = tk.Toplevel(self.root)
        event_window.title("事件列表")
        event_window.geometry("800x600")
        
        # 主框架
        main_frame = ttk.Frame(event_window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        event_window.columnconfigure(0, weight=1)
        event_window.rowconfigure(0, weight=1)
        
        # 事件表格
        columns = ("事件ID", "相机", "开始时间", "检测次数", "截图数")
        event_tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            event_tree.heading(col, text=col)
            event_tree.column(col, width=120)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=event_tree.yview)
        event_tree.configure(yscrollcommand=scrollbar.set)
        
        event_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 填充数据
        for event in reversed(self.events):  # 最新的在前
            event_tree.insert("", "end", values=(
                event['event_id'],
                event['camera'],
                event['start_time'],
                event['detection_count'],
                event['screenshot_count']
            ))
        
        # 双击查看详情
        def on_double_click(event):
            selection = event_tree.selection()
            if selection:
                item = selection[0]
                values = event_tree.item(item, "values")
                event_id = int(values[0])
                self.show_event_details(event_id, None)
        
        event_tree.bind("<Double-1>", on_double_click)
        
        # 底部按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="导出事件报告",
            command=self.export_event_report
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="关闭",
            command=event_window.destroy
        ).pack(side=tk.LEFT)

    def export_event_report(self):
        """导出事件报告"""
        if not self.events:
            messagebox.showwarning("警告", "没有事件可以导出")
            return
        
        try:
            import csv
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"event_report_{timestamp}.csv"
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
                initialfile=filename
            )
            
            if file_path:
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(["事件ID", "相机名称", "开始时间", "检测次数", "截图数量"])
                    
                    for event in self.events:
                        writer.writerow([
                            event['event_id'],
                            event['camera'],
                            event['start_time'],
                            event['detection_count'],
                            event['screenshot_count']
                        ])
                
                self.log_message(f"事件报告已导出到: {file_path}")
                messagebox.showinfo("成功", f"事件报告已导出到:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"导出事件报告失败: {str(e)}")
    def _update_screenshot_canvas_region(self):
        """更新Canvas区域"""
        self.screenshot_canvas.configure(scrollregion=self.screenshot_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        if self.screenshot_canvas.winfo_exists():
            self.screenshot_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_info_tab(self, notebook):
        """创建信息显示选项卡"""
        tab = ttk.Frame(notebook, padding="15")
        notebook.add(tab, text="信息显示")

        # 图片质量状态显示（新增）
        quality_frame = ttk.LabelFrame(tab, text="图片质量状态", padding="10")
        quality_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # 质量状态标签
        self.quality_status_var = tk.StringVar(value="状态: 等待评估")
        self.quality_status_label = ttk.Label(
            quality_frame, 
            textvariable=self.quality_status_var,
            font=("Microsoft YaHei", 10, "bold")
        )
        self.quality_status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 质量详细信息
        self.quality_detail_var = tk.StringVar(value="")
        self.quality_detail_label = ttk.Label(
            quality_frame,
            textvariable=self.quality_detail_var,
            foreground="gray"
        )
        self.quality_detail_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # 质量控制选项
        quality_control_frame = ttk.Frame(quality_frame)
        quality_control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 启用质量检查
        self.enable_quality_check_var = tk.BooleanVar(value=self.detector.enable_quality_check)
        ttk.Checkbutton(
            quality_control_frame, 
            text="启用质量检查",
            variable=self.enable_quality_check_var,
            command=self.update_quality_check
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # 跳过天黑帧
        self.skip_night_frames_var = tk.BooleanVar(value=self.detector.skip_night_frames)
        ttk.Checkbutton(
            quality_control_frame, 
            text="天黑时暂停检测",
            variable=self.skip_night_frames_var,
            command=self.update_skip_night_frames
        ).pack(side=tk.LEFT)
        
        # 统计信息
        stats_frame = ttk.LabelFrame(tab, text="实时统计", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # 使用Canvas创建更好的显示
        self.stats_canvas = tk.Canvas(stats_frame, height=150, bg="white")
        self.stats_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        stats_frame.columnconfigure(0, weight=1)
        
        # 检测详情
        details_frame = ttk.LabelFrame(tab, text="检测详情", padding="10")
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        
        # 使用Treeview显示检测详情
        columns = ("ID", "位置", "速度", "置信度", "轨迹长度", "状态")
        self.detection_tree = ttk.Treeview(details_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=scrollbar.set)
        
        self.detection_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        # 日志显示
        log_frame = ttk.LabelFrame(tab, text="系统日志", padding="10")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tab.rowconfigure(2, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 配置按钮
        config_frame = ttk.Frame(tab)
        config_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(config_frame, text="保存配置", command=self.save_config).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(config_frame, text="加载配置", command=self.load_config).grid(row=0, column=1, padx=5)
        ttk.Button(config_frame, text="重置参数", command=self.reset_parameters).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(config_frame, text="清空日志", command=self.clear_log).grid(row=0, column=3, padx=(5, 0))
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="系统就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief="sunken", anchor=tk.W)
        status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(0, weight=1)
        
        # 添加时间显示
        self.time_var = tk.StringVar()
        time_label = ttk.Label(status_frame, textvariable=self.time_var, 
                              relief="sunken", anchor=tk.E, width=20)
        time_label.grid(row=0, column=1, sticky=tk.E)
        
        # 更新时间显示
        self.update_time_display()

    def update_quality_check(self):
        """更新质量检查设置"""
        self.detector.enable_quality_check = self.enable_quality_check_var.get()
        status = "启用" if self.enable_quality_check_var.get() else "禁用"
        self.log_message(f"图片质量检查已{status}")
    
    def update_skip_night_frames(self):
        """更新跳过天黑帧设置"""
        self.detector.skip_night_frames = self.skip_night_frames_var.get()
        status = "启用" if self.skip_night_frames_var.get() else "禁用"
        self.log_message(f"天黑暂停检测已{status}")
    
    def update_quality_display(self):
        """更新质量信息显示"""
        quality_info = self.detector.get_quality_info()
        
        if not quality_info:
            self.quality_status_var.set("状态: 等待评估")
            self.quality_detail_var.set("")
            return
        
        is_night = quality_info.get('is_night', False)
        message = quality_info.get('message', '')
        night_frames = quality_info.get('night_frames', 0)
        consecutive_night = quality_info.get('consecutive_night_frames', 0)
        
        # 更新状态标签
        if is_night:
            self.quality_status_var.set("🌙 状态: 天黑/过暗")
            self.quality_status_label.config(foreground="red")
        else:
            overall = quality_info.get('overall', 0)
            if overall > 0.6:
                self.quality_status_var.set(f"✅ 状态: 良好 ({overall:.2f})")
                self.quality_status_label.config(foreground="green")
            elif overall > 0.3:
                self.quality_status_var.set(f"⚠️ 状态: 一般 ({overall:.2f})")
                self.quality_status_label.config(foreground="orange")
            else:
                self.quality_status_var.set(f"❌ 状态: 较差 ({overall:.2f})")
                self.quality_status_label.config(foreground="red")
        
        # 更新详细信息
        detail_text = f"{message}"
        if night_frames > 0:
            detail_text += f" | 天黑帧数: {night_frames}"
        if consecutive_night > 0:
            detail_text += f" | 连续天黑: {consecutive_night}帧"
        
        self.quality_detail_var.set(detail_text)

    def save_and_display_screenshot(self, frame, detection):
        """保存并显示空飘物截图 - 使用事件ID索引确保正确更新"""
        try:
            current_time = time.time()
            camera_name = self.current_camera_name
            
            if not camera_name:
                camera_name = "Unknown_Camera"
            
            # 提取ROI
            x, y, w, h = detection['bbox']
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # 提取特征
            current_features = self._extract_image_features(roi)
            
            # ========== 创建事件ID到事件对象的映射 ==========
            event_id_map = {}
            for event in self.events:
                eid = event.get('event_id')
                if eid is not None:
                    event_id_map[eid] = event
            
            # 搜索相似事件
            event_id = None
            matched_event = None
            highest_similarity = 0
            
            for event in self.events:
                # 检查条件
                if event.get('camera') != camera_name:
                    continue
                
                last_time = event.get('last_detection_time', 0)
                if last_time <= 0:
                    continue
                
                time_diff = current_time - last_time   # 只与最近的两分钟事件进行相似度判断
                if time_diff > self.recent_time_threshold:
                    continue
                
                event_features = event.get('latest_features')
                if event_features is None:
                    continue
                
                # 计算相似度
                similarity = self._calculate_image_similarity(current_features, event_features)
                
                if similarity >= self.similarity_threshold and similarity > highest_similarity:
                    highest_similarity = similarity
                    event_id = event.get('event_id')
                    matched_event = event
            
            # 判断是否为新事件
            is_new_event = (event_id is None or highest_similarity < self.similarity_threshold)
            
            # 处理事件
            if is_new_event:
                # 创建新事件
                self.current_event_id += 1
                event_id = self.current_event_id
                
                new_event = {
                    'event_id': event_id,
                    'camera': camera_name,
                    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'detection_count': 1,
                    'screenshot_count': 0,
                    'last_detection_time': current_time,
                    'latest_features': current_features,
                    'has_alerted': False,
                    'similarity_score': 1.0,
                    'created_time': current_time
                }
                
                self.events.append(new_event)
                event_id_map[event_id] = new_event
                
                # 报警
                if self.alert_enabled:
                    self.show_alert_dialog(event_id, camera_name)
                    new_event['has_alerted'] = True
                    
                    # ========== 修改：改为相机1触发报警 ==========
                    if camera_name == "相机1":
                        self.update_map_camera_status(camera_name, True)
                        self.log_message(f"相机1触发报警，地图变红", level="info")
                    # ========== 修改结束 ==========
                
                print(f"🆕 新事件 {event_id}，总事件数: {len(self.events)}")
                
            else:
                # 更新现有事件 - 通过映射确保更新正确的事件
                if matched_event and event_id in event_id_map:
                    target_event = event_id_map[event_id]
                    
                    # 记录旧值用于日志
                    old_time = target_event.get('last_detection_time', 0)
                    old_count = target_event.get('detection_count', 0)
                    
                    # 更新
                    target_event['detection_count'] = old_count + 1
                    target_event['last_detection_time'] = old_time
                    target_event['latest_features'] = current_features
                    target_event['similarity_score'] = highest_similarity
                    
                    # 报警（如果需要）
                    if not target_event.get('has_alerted', False):
                        if self.alert_enabled:
                            self.show_alert_dialog(event_id, camera_name)
                        target_event['has_alerted'] = True
                        
                        # ========== 修改：改为相机1触发报警 ==========
                        if camera_name == "相机1":
                            self.update_map_camera_status(camera_name, True)
                            self.log_message(f"相机1触发报警，地图变红", level="info")
                        # ========== 修改结束 ==========
                    
                    print(f"🔄 更新事件 {event_id}，时间差: {current_time - old_time:.1f}秒")
                else:
                    print(f"⚠️ 错误：找不到事件 {event_id} 进行更新")
            
            # ========== 截图保存功能 ==========
            # 检查该事件已有多少张截图
            existing_screenshots = [s for s in self.screenshots if s['event_id'] == event_id]
            
            # 如果已经有3张或更多截图，直接返回不保存
            if len(existing_screenshots) >= 3:
                return
            
            # 按相机和事件创建目录结构
            import re
            camera_dir_name = re.sub(r'[<>:"/\\|?*]', '_', camera_name)
            
            # 创建目录结构: screenshots/相机名称/事件_序号/
            event_dir_name = f"event_{event_id:03d}"
            event_dir = os.path.join(self.screenshot_dir, camera_dir_name, event_dir_name)
            
            if not os.path.exists(event_dir):
                os.makedirs(event_dir, exist_ok=True)
            
            # 保存截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{event_id:03d}_{detection['id']}_{len(existing_screenshots)+1}_{timestamp}.jpg"
            filepath = os.path.join(event_dir, filename)
            
            cv2.imwrite(filepath, roi)
            
            # 创建缩略图
            thumbnail_size = (120, 90)
            thumbnail = cv2.resize(roi, thumbnail_size)
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            
            # 添加到截图列表
            screenshot_info = {
                'event_id': event_id,
                'id': detection['id'],
                'filepath': filepath,
                'filename': filename,
                'timestamp': timestamp,
                'camera': camera_name,
                'camera_dir': camera_dir_name,
                'event_dir': event_dir_name,
                'confidence': detection['features']['confidence'],
                'thumbnail': thumbnail,
                'original_size': (roi.shape[1], roi.shape[0]),
                'image_index': len(existing_screenshots) + 1
            }
            
            self.screenshots.append(screenshot_info)
            
            # 更新事件截图数量
            for event in self.events:
                if event['event_id'] == event_id:
                    event['screenshot_count'] = len(existing_screenshots) + 1
                    break
            
            # 更新截图显示
            self.update_screenshot_tree()
            
            # 更新统计
            self.update_screenshot_statistics()
            
        except Exception as e:
            self.log_message(f"保存截图失败: {str(e)}", level="error")
    
    def debug_event_timestamps(self):
        """调试事件时间戳"""
        current_time = time.time()
        print(f"\n=== 事件时间戳调试 ===")
        print(f"当前时间: {current_time}")
        
        if not self.events:
            print("没有事件记录")
            return
        
        for i, event in enumerate(self.events):
            event_id = event.get('event_id', '未知')
            last_time = event.get('last_detection_time', 0)
            created_time = event.get('created_time', 0)
            camera = event.get('camera', '未知')
            
            time_diff_last = current_time - last_time if last_time > 0 else float('inf')
            time_diff_created = current_time - created_time if created_time > 0 else float('inf')
            
            print(f"事件[{i}] ID:{event_id} 相机:{camera}")
            print(f"  最后检测时间: {last_time} ({time_diff_last:.1f}秒前)")
            print(f"  创建时间: {created_time} ({time_diff_created:.1f}秒前)")
            print(f"  检测次数: {event.get('detection_count', 0)}")
            print(f"  有特征: {'latest_features' in event}")
            print(f"  已报警: {event.get('has_alerted', False)}")
            print("-" * 40)
    
    # 可以在GUI中添加一个调试按钮
    def add_debug_button(self):
        """添加调试按钮"""
        debug_frame = ttk.Frame(self.root)
        debug_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(
            debug_frame,
            text="调试事件时间戳",
            command=self.debug_event_timestamps
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            debug_frame,
            text="测试相似度",
            command=self.test_similarity
        ).pack(side=tk.LEFT, padx=5)

    def test_similarity(self):
        """测试相似度计算"""
        if len(self.events) < 2:
            print("需要至少2个事件来测试相似度")
            return
        
        event1 = self.events[-1]  # 最新事件
        event2 = self.events[-2]  # 次新事件
        
        if 'latest_features' in event1 and 'latest_features' in event2:
            similarity = self._calculate_image_similarity(
                event1['latest_features'], 
                event2['latest_features']
            )
            print(f"事件 {event1['event_id']} 和事件 {event2['event_id']} 的相似度: {similarity:.2f}")
        else:
            print("事件缺少特征数据")

    def _extract_image_features(self, image):
        """提取图像特征 - 改进版"""
        try:
            if image is None or image.size == 0:
                return None
            
            features = {}
            
            # 1. 颜色直方图（BGR）
            if len(image.shape) == 3:
                color_features = []
                for i in range(3):  # BGR通道
                    hist = cv2.calcHist([image], [i], None, [16], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    color_features.extend(hist)
                features['color'] = np.array(color_features)
            
            # 2. 转换为HSV空间提取特征
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180])
                s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
                v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
                
                h_hist = cv2.normalize(h_hist, h_hist).flatten()
                s_hist = cv2.normalize(s_hist, s_hist).flatten()
                v_hist = cv2.normalize(v_hist, v_hist).flatten()
                
                features['hsv'] = np.hstack([h_hist, s_hist, v_hist])
            
            # 3. 灰度图像纹理特征
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # 计算梯度直方图
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 归一化并计算直方图
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
                hist = cv2.calcHist([magnitude], [0], None, [8], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features['texture'] = hist
            else:
                features['texture'] = np.zeros(8)
            
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None

    def _calculate_image_similarity(self, features1, features2):
        """计算图像相似度 - 加权综合"""
        try:
            if features1 is None or features2 is None:
                return 0.0
            
            total_similarity = 0.0
            total_weight = 0.0
            
            # 特征权重
            weights = {
                'color': 0.4,
                'hsv': 0.3,
                'texture': 0.3
            }
            
            for feature_type in weights.keys():
                if feature_type in features1 and feature_type in features2:
                    f1 = features1[feature_type]
                    f2 = features2[feature_type]
                    
                    # 确保长度一致
                    min_len = min(len(f1), len(f2))
                    if min_len == 0:
                        continue
                        
                    f1 = f1[:min_len]
                    f2 = f2[:min_len]
                    
                    # 计算余弦相似度
                    dot_product = np.dot(f1, f2)
                    norm1 = np.linalg.norm(f1)
                    norm2 = np.linalg.norm(f2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        similarity = max(0.0, min(1.0, similarity))
                        
                        total_similarity += similarity * weights[feature_type]
                        total_weight += weights[feature_type]
            
            # 计算加权平均
            if total_weight > 0:
                return total_similarity / total_weight
            else:
                return 0.0
                
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return 0.0
    
    def debug_similarity(self, image1, image2):
        """调试相似度计算"""
        features1 = self._extract_image_features(image1)
        features2 = self._extract_image_features(image2)
        similarity = self._calculate_image_similarity(features1, features2)
        
        print(f"相似度: {similarity:.3f}")
        return similarity

    def debug_event_comparison(self):
        """调试事件比较逻辑"""
        current_time = time.time()
        print(f"\n=== 事件比较调试 ===")
        print(f"当前时间: {current_time}")
        print(f"总事件数: {len(self.events)}")
        
        recent_events = []
        old_events = []
        
        for event in self.events:
            event_time = event.get('last_detection_time', 0)
            time_diff = current_time - event_time
            
            if time_diff <= 120:
                recent_events.append(event)
            else:
                old_events.append(event)
        
        print(f"最近2分钟内事件: {len(recent_events)}个")
        print(f"超过2分钟事件: {len(old_events)}个")
        
        for event in recent_events[:5]:  # 显示前5个最近事件
            event_time = event.get('last_detection_time', 0)
            time_diff = current_time - event_time
            print(f"  事件 {event['event_id']}: {time_diff:.1f}秒前")
        
        return recent_events, old_events
    
    def display_screenshot_by_camera(self):
        """按相机分类显示截图"""
        try:
            # 清空当前显示
            for widget in self.screenshot_inner_frame.winfo_children():
                widget.destroy()
            
            if not self.screenshots:
                self.empty_label = ttk.Label(
                    self.screenshot_inner_frame, 
                    text="暂无检测截图\n检测到的空飘物将在这里显示",
                    foreground="gray",
                    justify="center"
                )
                self.empty_label.pack(expand=True, fill="both", padx=20, pady=50)
                return
            
            # 按相机分组截图
            screenshots_by_camera = {}
            for screenshot in self.screenshots:
                camera = screenshot['camera']
                if camera not in screenshots_by_camera:
                    screenshots_by_camera[camera] = []
                screenshots_by_camera[camera].append(screenshot)
            
            # 创建可折叠的面板
            row_index = 0
            
            for camera_name, camera_screenshots in screenshots_by_camera.items():
                # 创建相机面板
                camera_frame = ttk.Frame(self.screenshot_inner_frame, relief="solid", borderwidth=1)
                camera_frame.grid(row=row_index, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
                self.screenshot_inner_frame.columnconfigure(0, weight=1)
                
                # 相机标题栏（可点击折叠）
                title_frame = ttk.Frame(camera_frame)
                title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
                title_frame.columnconfigure(1, weight=1)
                
                # 折叠/展开按钮
                expand_var = tk.BooleanVar(value=True)
                expand_btn = ttk.Button(
                    title_frame, 
                    text="▼" if expand_var.get() else "▶",
                    width=2,
                    command=lambda var=expand_var, btn=expand_btn: self.toggle_camera_panel(var, btn)
                )
                expand_btn.grid(row=0, column=0, padx=(0, 5))
                
                # 相机名称
                camera_label = ttk.Label(
                    title_frame,
                    text=f"📷 {camera_name}",
                    font=("Microsoft YaHei", 10, "bold")
                )
                camera_label.grid(row=0, column=1, sticky=tk.W)
                
                # 截图数量
                count_label = ttk.Label(
                    title_frame,
                    text=f"({len(camera_screenshots)}张截图)",
                    font=("Microsoft YaHei", 9)
                )
                count_label.grid(row=0, column=2, sticky=tk.E, padx=(5, 0))
                
                # 截图内容区域
                content_frame = ttk.Frame(camera_frame)
                content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=(0, 5))
                
                # 显示该相机的截图（每行4个）
                for i, screenshot in enumerate(camera_screenshots):
                    self.create_screenshot_thumbnail(content_frame, screenshot, i)
                
                row_index += 1
            
            # 更新Canvas区域
            self._update_screenshot_canvas_region()
            
        except Exception as e:
            self.log_message(f"显示截图失败: {str(e)}", level="error")

    def toggle_camera_panel(self, expand_var, button):
        """切换相机面板的折叠/展开"""
        expand_var.set(not expand_var.get())
        button.config(text="▼" if expand_var.get() else "▶")
        # 这里需要更新显示，但为了简单起见，我们先刷新整个显示
        self.display_screenshot_by_camera()

    def create_screenshot_thumbnail(self, parent_frame, screenshot_info, index):
        """创建单个截图缩略图"""
        try:
            # 计算位置
            row = index // 4
            col = index % 4
            
            # 创建缩略图容器
            thumbnail_frame = ttk.Frame(
                parent_frame, 
                relief="solid", 
                borderwidth=1,
                padding=5
            )
            thumbnail_frame.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E))
            
            # 转换为PhotoImage
            thumbnail_pil = Image.fromarray(screenshot_info['thumbnail'])
            thumbnail_photo = ImageTk.PhotoImage(thumbnail_pil)
            
            # 创建缩略图标签
            thumbnail_label = tk.Label(
                thumbnail_frame,
                image=thumbnail_photo,
                cursor="hand2"
            )
            thumbnail_label.image = thumbnail_photo  # 保持引用
            thumbnail_label.pack()
            
            # 绑定点击事件
            thumbnail_label.bind("<Button-1>", 
                lambda e, info=screenshot_info: self.show_screenshot_detail(info))
            
            # 显示信息
            info_text = f"事件: {screenshot_info['event_id']}\n"
            info_text += f"检测: {screenshot_info['id']}\n"
            info_text += f"图: {screenshot_info['image_index']}\n"
            info_text += f"置信度: {screenshot_info['confidence']*100:.1f}%"
            
            info_label = ttk.Label(
                thumbnail_frame,
                text=info_text,
                font=("Microsoft YaHei", 8),
                justify="left"
            )
            info_label.pack(pady=(5, 0))
            
            # 配置网格权重
            parent_frame.columnconfigure(col, weight=1)
            
        except Exception as e:
            print(f"创建缩略图失败: {e}")

    def display_screenshot_thumbnail(self, screenshot_info):
        """显示截图缩略图"""
        try:
            # 创建缩略图容器
            thumbnail_frame = ttk.Frame(
                self.screenshot_inner_frame, 
                relief="solid", 
                borderwidth=1,
                padding=5
            )
            
            # 转换为PhotoImage
            thumbnail_pil = Image.fromarray(screenshot_info['thumbnail'])
            thumbnail_photo = ImageTk.PhotoImage(thumbnail_pil)
            
            # 创建缩略图标签
            thumbnail_label = tk.Label(
                thumbnail_frame,
                image=thumbnail_photo,
                cursor="hand2"  # 鼠标手型
            )
            thumbnail_label.image = thumbnail_photo  # 保持引用
            thumbnail_label.pack()
            
            # 绑定点击事件（放大显示）
            thumbnail_label.bind("<Button-1>", 
                lambda e, info=screenshot_info: self.show_screenshot_detail(info))
            
            # 显示信息
            info_text = f"ID: {screenshot_info['id']}\n"
            info_text += f"时间: {screenshot_info['timestamp'][9:17]}\n"
            info_text += f"置信度: {screenshot_info['confidence']*100:.1f}%"
            
            info_label = ttk.Label(
                thumbnail_frame,
                text=info_text,
                font=("Microsoft YaHei", 8),
                justify="left"
            )
            info_label.pack(pady=(5, 0))
            
            # 添加到网格布局
            current_count = len(self.screenshots)
            row = (current_count - 1) // 4  # 每行4个
            col = (current_count - 1) % 4
            
            thumbnail_frame.grid(
                row=row, 
                column=col, 
                padx=5, 
                pady=5, 
                sticky=(tk.W, tk.E)
            )
            
            # 配置网格权重
            self.screenshot_inner_frame.columnconfigure(col, weight=1)
            
            # 更新Canvas区域
            self._update_screenshot_canvas_region()
            
        except Exception as e:
            self.log_message(f"显示缩略图失败: {str(e)}", level="error")

    def show_screenshot_detail(self, screenshot_info):
        """显示截图详情（放大）"""
        try:
            # 创建详情窗口
            detail_window = tk.Toplevel(self.root)
            detail_window.title(f"空飘物检测详情 - {screenshot_info['filename']}")
            detail_window.geometry("800x600")
            
            # 居中显示
            detail_window.update_idletasks()
            width = detail_window.winfo_width()
            height = detail_window.winfo_height()
            x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
            detail_window.geometry(f"{width}x{height}+{x}+{y}")
            
            # 主框架
            main_frame = ttk.Frame(detail_window, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            detail_window.columnconfigure(0, weight=1)
            detail_window.rowconfigure(0, weight=1)
            
            # 图像显示区域
            image_frame = ttk.Frame(main_frame)
            image_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(0, weight=1)
            
            # 加载原始图像
            original_image = cv2.imread(screenshot_info['filepath'])
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 转换为PhotoImage（调整大小以适应窗口）
            pil_image = Image.fromarray(original_image)
            
            # 计算合适的显示尺寸
            max_width = 780
            max_height = 450
            
            width_ratio = max_width / pil_image.width
            height_ratio = max_height / pil_image.height
            scale = min(width_ratio, height_ratio, 1.0)  # 不超过原尺寸
            
            new_width = int(pil_image.width * scale)
            new_height = int(pil_image.height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            # 创建标签显示图像
            image_label = ttk.Label(image_frame, image=photo)
            image_label.image = photo  # 保持引用
            image_label.pack(expand=True)
            
            # 信息显示区域
            info_frame = ttk.LabelFrame(main_frame, text="检测信息", padding="10")
            info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
            
            # 详细信息
            info_text = f"文件名: {screenshot_info['filename']}\n"
            info_text += f"相机: {screenshot_info['camera']}\n"
            info_text += f"检测ID: {screenshot_info['id']}\n"
            info_text += f"时间: {screenshot_info['timestamp'][:8]} {screenshot_info['timestamp'][9:15]}\n"
            info_text += f"置信度: {screenshot_info['confidence']*100:.1f}%\n"
            info_text += f"原始尺寸: {screenshot_info['original_size'][0]}x{screenshot_info['original_size'][1]}"
            info_text = f"文件名: {screenshot_info['filename']}\n"
            info_text += f"相机: {screenshot_info['camera']}\n"
            info_text += f"事件ID: {screenshot_info['event_id']}\n"  # 新增
            info_text += f"检测ID: {screenshot_info['id']}\n"
            info_text += f"图片序号: {screenshot_info['image_index']}\n"  # 新增
            info_text += f"时间: {screenshot_info['timestamp'][:8]} {screenshot_info['timestamp'][9:15]}\n"
            info_text += f"置信度: {screenshot_info['confidence']*100:.1f}%\n"
            info_text += f"原始尺寸: {screenshot_info['original_size'][0]}x{screenshot_info['original_size'][1]}"
            
            info_label = ttk.Label(info_frame, text=info_text, justify="left")
            info_label.pack(anchor="w")
            
            # 操作按钮区域
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
            
            # 操作按钮
            ttk.Button(
                button_frame, 
                text="打开原图",
                command=lambda: self.open_original_image(screenshot_info['filepath']),
                width=15
            ).pack(pady=(0, 5))
            
            ttk.Button(
                button_frame, 
                text="复制到剪贴板",
                command=lambda: self.copy_image_to_clipboard(screenshot_info['filepath']),
                width=15
            ).pack(pady=(0, 5))
            
            ttk.Button(
                button_frame, 
                text="删除截图",
                command=lambda: self.delete_screenshot(screenshot_info, detail_window),
                width=15
            ).pack(pady=(0, 5))
            
            ttk.Button(
                button_frame, 
                text="关闭",
                command=detail_window.destroy,
                width=15
            ).pack()
            
        except Exception as e:
            messagebox.showerror("错误", f"显示详情失败: {str(e)}")

    def open_original_image(self, filepath):
        """用系统默认程序打开原图"""
        try:
            os.startfile(filepath)  # Windows
        except:
            try:
                import subprocess
                subprocess.run(['xdg-open', filepath])  # Linux
            except:
                try:
                    import subprocess
                    subprocess.run(['open', filepath])  # macOS
                except:
                    messagebox.showwarning("提示", f"无法打开文件: {filepath}")

    def copy_image_to_clipboard(self, filepath):
        """复制图像到剪贴板"""
        try:
            # 这种方法在不同系统上可能不同
            # 这里提供一个简单的实现
            import shutil
            import tempfile
            
            # 创建一个临时文件用于复制
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, "temp_screenshot.jpg")
            shutil.copy2(filepath, temp_file)
            
            # 这里可以添加系统特定的剪贴板操作
            # 暂时显示一个消息
            messagebox.showinfo("提示", "图像路径已复制到剪贴板（功能待完善）")
            
        except Exception as e:
            messagebox.showerror("错误", f"复制失败: {str(e)}")

    def delete_screenshot(self, screenshot_info, detail_window=None):
        """删除截图"""
        if messagebox.askyesno("确认", f"确定要删除截图 '{screenshot_info['filename']}' 吗？"):
            try:
                # 删除文件
                if os.path.exists(screenshot_info['filepath']):
                    os.remove(screenshot_info['filepath'])
                
                # 从列表中移除
                self.screenshots = [s for s in self.screenshots if s['filepath'] != screenshot_info['filepath']]
                
                # 更新显示
                self.refresh_screenshot_display()
                
                # 更新统计
                self.screenshot_count_var.set(f"截图数量: {len(self.screenshots)}")
                
                if len(self.screenshots) == 0:
                    self.clear_screenshots_btn.config(state="disabled")
                
                self.log_message(f"截图已删除: {screenshot_info['filename']}")
                
                # 关闭详情窗口
                if detail_window and detail_window.winfo_exists():
                    detail_window.destroy()
                    
            except Exception as e:
                messagebox.showerror("错误", f"删除截图失败: {str(e)}")

    def refresh_screenshot_display(self):
        """刷新截图显示 - 改为更新树形结构"""
        self.update_screenshot_tree()
    
    def show_alert_dialog(self, event_id, camera_name):
        """显示预警弹窗 - 叠加显示多个窗口"""
        try:
            # 更新电子地图状态（变红）
            self.update_map_camera_status(camera_name, True)
            
            # 1. 检查是否已有相同事件的预警窗口打开
            if event_id in self.alert_windows:
                window = self.alert_windows[event_id]
                if window and window.winfo_exists():
                    window.lift()  # 将现有窗口提到前面
                    window.focus_set()  # 设置焦点
                    return
            
            # 2. 创建预警窗口（不再限制数量）
            self._create_alert_window(event_id, camera_name)
            
        except Exception as e:
            self.log_message(f"显示预警弹窗失败: {str(e)}", level="error")

    def _create_alert_window(self, event_id, camera_name):
        """创建单个预警窗口 - 在屏幕中心显示，多个窗口向固定右下方向偏移"""
        try:
            # 创建预警窗口
            alert_window = tk.Toplevel(self.root)
            alert_window.title(f"⚠️ 空飘物检测预警 - 事件 {event_id}")
            alert_window.geometry("400x400")
            alert_window.resizable(False, False)
            
            # 设置模态
            alert_window.transient(self.root)
            
            # 保存窗口引用
            self.alert_windows[event_id] = alert_window
            
            # 计算已打开的窗口数量（用于叠加偏移）
            window_index = len(self.alert_windows) - 1
            
            # 计算屏幕尺寸
            screen_width = alert_window.winfo_screenwidth()
            screen_height = alert_window.winfo_screenheight()
            
            window_width = 400
            window_height = 400
            
            # 基础位置（稍微偏离中心，给偏移留空间）
            base_x = screen_width // 2 - window_width // 2 - 50
            base_y = screen_height // 2 - window_height // 2 - 50
            
            # 固定向右下方向偏移（每个窗口偏移25像素）
            offset_x = window_index * 25
            offset_y = window_index * 25
            
            # 最终位置
            x = base_x + offset_x
            y = base_y + offset_y
            
            # 确保窗口在屏幕内
            x = max(0, min(x, screen_width - window_width))
            y = max(0, min(y, screen_height - window_height))
            
            # 设置窗口位置
            alert_window.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y)}")
            
            # 主框架
            main_frame = ttk.Frame(alert_window, padding="20")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            alert_window.columnconfigure(0, weight=1)
            alert_window.rowconfigure(0, weight=1)
            
            # 预警图标和标题
            title_frame = ttk.Frame(main_frame)
            title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 15))
            
            icon_label = ttk.Label(
                title_frame,
                text="⚠️",
                font=("Microsoft YaHei", 36),
                foreground="orange"
            )
            icon_label.grid(row=0, column=0, padx=(0, 10))
            
            title_text = ttk.Label(
                title_frame,
                text="空飘物检测预警",
                font=("Microsoft YaHei", 16, "bold"),
                foreground="red"
            )
            title_text.grid(row=0, column=1)
            
            # 预警信息
            info_frame = ttk.Frame(main_frame)
            info_frame.grid(row=1, column=0, columnspan=2, pady=(0, 15))
            
            current_time = datetime.now().strftime("%H:%M:%S")
            
            info_lines = [
                f"预警时间: {current_time}",
                f"事件编号: {event_id}",
                f"来源相机: {camera_name}",
                f"检测类型: 空飘物入侵",
                f"状态: 等待处理"
            ]
            
            for i, line in enumerate(info_lines):
                ttk.Label(
                    info_frame,
                    text=line,
                    font=("Microsoft YaHei", 10)
                ).grid(row=i, column=0, sticky=tk.W, pady=2)
            
            # 信息提示
            alert_info_frame = ttk.LabelFrame(main_frame, text="警报信息", padding="10")
            alert_info_frame.grid(row=2, column=0, columnspan=2, pady=(10, 15), sticky=(tk.W, tk.E))
            
            info_text = "请在事件详情页面处理此警报"
            
            ttk.Label(
                alert_info_frame,
                text=info_text,
                font=("Microsoft YaHei", 10),
                justify="center"
            ).grid(row=0, column=0, columnspan=2, pady=5)
            
            # 控制按钮
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
            
            # 查看详情按钮
            ttk.Button(
                button_frame,
                text="查看详情",
                command=lambda: self.show_event_details(event_id, alert_window),
                width=12
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            # 稍后处理按钮
            ttk.Button(
                button_frame,
                text="稍后处理",
                command=lambda: self.close_alert_window(event_id, mark_read=False),
                width=12
            ).pack(side=tk.LEFT)
            
            # 配置列权重
            main_frame.columnconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            
            # 绑定窗口关闭事件
            alert_window.protocol("WM_DELETE_WINDOW", lambda: None)
            
            # 播放提示音
            self.play_alert_sound()
            
            # 记录为未读警报
            self.mark_alert_as_unread(event_id, camera_name)
            
            # 记录预警
            self.log_message(f"事件 {event_id} - {camera_name} 触发预警", level="warning")
            
        except Exception as e:
            self.log_message(f"创建预警窗口失败: {str(e)}", level="error")
        
    def close_alert_window(self, event_id, mark_read=True):
        """关闭预警窗口"""
        try:
            # 1. 关闭当前窗口
            if event_id in self.alert_windows:
                window = self.alert_windows[event_id]
                camera_name = None
                
                # 获取该事件对应的相机名称
                for cam, alerts in self.unread_alerts.items():
                    if event_id in alerts:
                        camera_name = cam
                        break
                
                if window and window.winfo_exists():
                    if mark_read:
                        self.mark_alert_as_read(event_id)
                    window.destroy()
                
                # 从窗口字典中移除
                del self.alert_windows[event_id]
                
                # ========== 修改：改为相机1 ==========
                if camera_name == "相机1":
                    has_other_alerts = False
                    for cam, alerts in self.unread_alerts.items():
                        if cam == camera_name and alerts:
                            has_other_alerts = True
                            break
                    
                    if not has_other_alerts:
                        self.update_map_camera_status(camera_name, False)
                        self.log_message(f"相机1所有警报已处理，地图状态恢复为灰色", level="info")
                # ========== 修改结束 ==========
            
            # 2. 重新排列剩余窗口的位置（可选）
            if not self.alert_windows:
                for camera_name in self.camera_alert_status:
                    self.camera_alert_status[camera_name] = False
                
        except Exception as e:
            self.log_message(f"关闭警报窗口失败: {str(e)}", level="error")

    def mark_alert_as_unread(self, event_id, camera_name):
        """标记警报为未读"""
        if camera_name not in self.unread_alerts:
            self.unread_alerts[camera_name] = []
        
        if event_id not in self.unread_alerts[camera_name]:
            self.unread_alerts[camera_name].append(event_id)
        
        # 更新树形显示（添加未读标记）
        self.update_screenshot_tree()

    def mark_alert_as_read(self, event_id):
        """标记警报为已读"""
        for camera_name, alerts in list(self.unread_alerts.items()):
            if event_id in alerts:
                alerts.remove(event_id)
                if not alerts:  # 如果列表为空，删除该相机条目
                    del self.unread_alerts[camera_name]
        
        # 更新树形显示
        self.update_screenshot_tree()

    def get_unread_alert_count(self, camera_name):
        """获取指定相机的未读警报数量"""
        return len(self.unread_alerts.get(camera_name, []))
    
    def handle_alert_type(self, event_id, camera_name, alert_type, parent_window=None):
        """处理警报类型分类（实警/虚警）- 修复关闭窗口逻辑"""
        try:
            # 标记为已读
            self.mark_alert_as_read(event_id)
            
            # 根据警报类型分类处理
            if alert_type == "real":
                target_dir = self.real_alerts_dir
                alert_type_text = "实警"
                self.log_message(f"事件 {event_id} - {camera_name} 标记为实警", level="info")
            else:
                target_dir = self.false_alerts_dir
                alert_type_text = "虚警"
                self.log_message(f"事件 {event_id} - {camera_name} 标记为虚警", level="info")
            
            # 创建按相机和事件分组的目录
            camera_dir_name = self.get_safe_filename(camera_name)
            camera_dir = os.path.join(target_dir, camera_dir_name)
            event_dir = os.path.join(camera_dir, f"event_{event_id:03d}")
            
            if not os.path.exists(event_dir):
                os.makedirs(event_dir, exist_ok=True)
            
            # 查找该事件的所有截图
            event_screenshots = [s for s in self.screenshots if s['event_id'] == event_id]
            
            # 复制截图到相应目录
            copied_files = []
            for screenshot in event_screenshots:
                if os.path.exists(screenshot['filepath']):
                    # 创建新的文件名
                    filename = f"{alert_type_text}_{event_id:03d}_{os.path.basename(screenshot['filepath'])}"
                    target_path = os.path.join(event_dir, filename)
                    
                    try:
                        # 复制文件
                        import shutil
                        shutil.copy2(screenshot['filepath'], target_path)
                        copied_files.append(target_path)
                        
                        # 记录警报信息
                        alert_info = {
                            'event_id': event_id,
                            'camera': camera_name,
                            'alert_type': alert_type,
                            'original_path': screenshot['filepath'],
                            'saved_path': target_path,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 保存警报信息到JSON文件
                        info_file = os.path.join(event_dir, f"alert_info_{event_id}.json")
                        with open(info_file, 'w', encoding='utf-8') as f:
                            json.dump(alert_info, f, indent=2, ensure_ascii=False)
                            
                    except Exception as e:
                        self.log_message(f"复制截图失败: {str(e)}", level="error")
            
            # 更新事件记录中的警报类型
            for event in self.events:
                if event['event_id'] == event_id:
                    event['alert_type'] = alert_type
                    event['processed_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
            
            # 显示处理结果
            if parent_window and parent_window.winfo_exists():
                messagebox.showinfo("处理完成", 
                                f"事件 {event_id} 已标记为{alert_type_text}\n"
                                f"已保存到: {event_dir}\n"
                                f"共处理 {len(copied_files)} 张截图",
                                parent=parent_window)
                # 关闭窗口
                self.close_alert_window(event_id)
            else:
                messagebox.showinfo("处理完成", 
                                f"事件 {event_id} 已标记为{alert_type_text}\n"
                                f"已保存到: {event_dir}\n"
                                f"共处理 {len(copied_files)} 张截图")
            
            # 更新树形显示
            self.update_screenshot_tree()

            # 关闭窗口
            if parent_window and parent_window.winfo_exists():
                print(f"DEBUG: 关闭父窗口")
                parent_window.destroy()
            
            # 从窗口管理器中移除
            if event_id in self.alert_windows:
                del self.alert_windows[event_id]
            
            # ========== 修改：检查相机1是否还有未处理的警报 ==========
            # 检查该相机是否还有其他未处理的警报
            has_other_alerts = False
            for cam, alerts in self.unread_alerts.items():
                if cam == camera_name and alerts:
                    has_other_alerts = True
                    break
            
            # 如果是相机1且没有其他未处理警报，恢复为灰色
            if camera_name == "相机1" and not has_other_alerts:
                self.update_map_camera_status(camera_name, False)
                self.log_message(f"相机1所有警报已处理，地图状态恢复为灰色", level="info")
            # ========== 修改结束 ==========
            
        except Exception as e:
            print(f"DEBUG: 处理警报类型失败: {str(e)}")
            self.log_message(f"处理警报类型失败: {str(e)}", level="error")
            
    
    def get_unread_alert_count(self, camera_name=None):
        """获取未读警报数量"""
        if camera_name:
            return len(self.unread_alerts.get(camera_name, []))
        else:
            total = 0
            for alerts in self.unread_alerts.values():
                total += len(alerts)
            return total

    def cleanup_closed_windows(self):
        """清理已关闭的窗口引用"""
        closed_windows = []
        for event_id, window in list(self.alert_windows.items()):
            if not window or not window.winfo_exists():
                closed_windows.append(event_id)
        
        for event_id in closed_windows:
            del self.alert_windows[event_id]
        
        return len(closed_windows)

    def play_alert_sound(self):
        """播放预警提示音（简单实现）"""
        try:
            import winsound
            winsound.Beep(1000, 500)  # 频率1000Hz，持续500ms
        except:
            # 如果winsound不可用，使用其他方式或跳过
            pass
    
    def create_event_screenshot_thumbnail(self, parent_frame, screenshot_info, index):
        """为事件详情页面创建截图缩略图"""
        try:
            # 创建缩略图容器
            thumb_frame = ttk.Frame(
                parent_frame, 
                relief="solid", 
                borderwidth=1,
                padding=5
            )
            thumb_frame.pack(pady=5, padx=5, fill=tk.X)
            
            # 加载并显示缩略图
            if 'thumbnail' in screenshot_info:
                thumbnail = screenshot_info['thumbnail']
                pil_image = Image.fromarray(thumbnail)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 创建标签
                label = tk.Label(thumb_frame, image=photo, cursor="hand2")
                label.image = photo  # 保持引用
                label.pack(side=tk.LEFT, padx=(0, 10))
                
                # 绑定点击事件
                label.bind("<Button-1>", 
                    lambda e, path=screenshot_info['filepath']: self.show_screenshot_detail_by_path(path))
            else:
                # 如果没有缩略图，尝试从文件加载
                try:
                    if os.path.exists(screenshot_info['filepath']):
                        # 加载原始图像并创建缩略图
                        image = cv2.imread(screenshot_info['filepath'])
                        if image is not None:
                            # 创建缩略图
                            thumbnail_size = (120, 90)
                            thumbnail = cv2.resize(image, thumbnail_size)
                            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(thumbnail)
                            photo = ImageTk.PhotoImage(pil_image)
                            
                            # 创建标签
                            label = tk.Label(thumb_frame, image=photo, cursor="hand2")
                            label.image = photo  # 保持引用
                            label.pack(side=tk.LEFT, padx=(0, 10))
                            
                            # 绑定点击事件
                            label.bind("<Button-1>", 
                                lambda e, path=screenshot_info['filepath']: self.show_screenshot_detail_by_path(path))
                except:
                    pass
            
            # 显示信息
            info_frame = ttk.Frame(thumb_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # 文件名
            filename_label = ttk.Label(
                info_frame,
                text=f"文件: {os.path.basename(screenshot_info['filepath'])[:40]}...",
                font=("Microsoft YaHei", 9),
                anchor="w"
            )
            filename_label.pack(anchor="w", pady=(0, 5))
            
            # 时间信息
            time_text = f"时间: {screenshot_info['timestamp'][:8]} {screenshot_info['timestamp'][9:15]}"
            time_label = ttk.Label(
                info_frame,
                text=time_text,
                font=("Microsoft YaHei", 9),
                anchor="w"
            )
            time_label.pack(anchor="w", pady=(0, 5))
            
            # 操作按钮
            button_frame = ttk.Frame(info_frame)
            button_frame.pack(anchor="w", pady=(5, 0))
            
            # 打开按钮
            ttk.Button(
                button_frame,
                text="打开",
                command=lambda path=screenshot_info['filepath']: self.open_original_image(path),
                width=8
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            # 查看详情按钮
            ttk.Button(
                button_frame,
                text="详情",
                command=lambda path=screenshot_info['filepath']: self.show_screenshot_detail_by_path(path),
                width=8
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            return thumb_frame
            
        except Exception as e:
            print(f"创建事件缩略图失败: {e}")
            # 创建错误提示框架
            error_frame = ttk.Frame(parent_frame)
            error_frame.pack(pady=5, padx=5, fill=tk.X)
            
            ttk.Label(
                error_frame,
                text=f"加载截图失败: {os.path.basename(screenshot_info.get('filepath', '未知'))}",
                foreground="red"
            ).pack()
            
            return error_frame

    # ========== 修复 show_event_details 方法 ==========
    def show_event_details(self, event_id, alert_window=None):
        """显示事件详情"""
        # 如果有关联的预警窗口，先关闭它
        if alert_window and alert_window.winfo_exists():
            alert_window.destroy()
        
        # 查找事件信息
        event_info = None
        for event in self.events:
            if event['event_id'] == event_id:
                event_info = event
                break
        
        if not event_info:
            messagebox.showinfo("信息", f"未找到事件 {event_id} 的详细信息")
            return
        
        # 创建详情窗口
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"事件详情 - 事件 {event_id}")
        detail_window.geometry("800x600")
        
        # 主框架
        main_frame = ttk.Frame(detail_window, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        detail_window.columnconfigure(0, weight=1)
        detail_window.rowconfigure(0, weight=1)
        
        # 事件信息
        info_frame = ttk.LabelFrame(main_frame, text="事件基本信息", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 检查是否已处理
        alert_type = event_info.get('alert_type', '未处理')
        processed_time = event_info.get('processed_time', '未处理')
        
        info_text = f"事件ID: {event_info['event_id']}\n"
        info_text += f"相机: {event_info['camera']}\n"
        info_text += f"开始时间: {event_info['start_time']}\n"
        info_text += f"处理状态: {alert_type}\n"
        if alert_type != '未处理':
            info_text += f"处理时间: {processed_time}\n"
        info_text += f"检测次数: {event_info['detection_count']}\n"
        info_text += f"截图数量: {event_info['screenshot_count']}"
        
        ttk.Label(
            info_frame,
            text=info_text,
            justify="left",
            font=("Microsoft YaHei", 10)
        ).pack(anchor="w")
        
        # 截图列表
        shots_frame = ttk.LabelFrame(main_frame, text="事件截图", padding="10")
        shots_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(1, weight=1)
        
        # 获取该事件的所有截图（最多3张）
        event_screenshots = [s for s in self.screenshots if s['event_id'] == event_id]
        # 按时间排序，只取最新的3张
        event_screenshots = sorted(event_screenshots, key=lambda x: x['timestamp'], reverse=True)[:3]
        
        if not event_screenshots:
            ttk.Label(
                shots_frame,
                text="该事件没有截图",
                foreground="gray"
            ).pack(expand=True)
        else:
            # 使用Canvas和滚动条显示截图
            canvas_frame = ttk.Frame(shots_frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(canvas_frame, bg="white")
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            
            inner_frame = ttk.Frame(canvas)
            
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")
            
            # 显示所有截图
            for i, screenshot in enumerate(event_screenshots):
                thumb_frame = self.create_event_screenshot_thumbnail(inner_frame, screenshot, i)
            
            # 更新Canvas区域
            inner_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            
            # 绑定Canvas大小变化
            def on_frame_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
            
            inner_frame.bind("<Configure>", on_frame_configure)
        
        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 导出按钮
        if event_screenshots:
            ttk.Button(
                bottom_frame,
                text="导出事件截图",
                command=lambda: self.export_event_screenshots(event_id)
            ).pack(side=tk.LEFT, padx=(0, 10))
        
        # 处理按钮（如果未处理）
        if alert_type == '未处理':
            ttk.Button(
                bottom_frame,
                text="处理此警报",
                command=lambda: self.show_alert_processing_dialog(event_id, event_info['camera'])
            ).pack(side=tk.LEFT, padx=(0, 10))
        
        # 关闭按钮
        ttk.Button(
            bottom_frame,
            text="关闭",
            command=detail_window.destroy
        ).pack(side=tk.LEFT)


    # ========== 添加导出事件截图的方法 ==========
    def export_event_screenshots(self, event_id):
        """导出事件的所有截图"""
        try:
            # 获取事件的所有截图
            event_screenshots = [s for s in self.screenshots if s['event_id'] == event_id]
            
            if not event_screenshots:
                messagebox.showwarning("警告", "该事件没有截图")
                return
            
            # 选择保存目录
            import tkinter.filedialog as filedialog
            save_dir = filedialog.askdirectory(title="选择保存目录")
            
            if not save_dir:
                return
            
            # 创建事件目录
            event_dir = os.path.join(save_dir, f"event_{event_id:03d}")
            if not os.path.exists(event_dir):
                os.makedirs(event_dir, exist_ok=True)
            
            # 复制所有截图
            copied_count = 0
            for screenshot in event_screenshots:
                if os.path.exists(screenshot['filepath']):
                    filename = os.path.basename(screenshot['filepath'])
                    target_path = os.path.join(event_dir, filename)
                    
                    try:
                        import shutil
                        shutil.copy2(screenshot['filepath'], target_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"复制文件失败: {e}")
            
            # 创建信息文件
            info_file = os.path.join(event_dir, f"event_{event_id}_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"事件ID: {event_id}\n")
                f.write(f"截图数量: {len(event_screenshots)}\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"导出数量: {copied_count}\n\n")
                
                for screenshot in event_screenshots:
                    f.write(f"文件: {os.path.basename(screenshot['filepath'])}\n")
                    f.write(f"时间: {screenshot['timestamp']}\n")
                    f.write(f"相机: {screenshot['camera']}\n")
                    f.write("-" * 40 + "\n")
            
            messagebox.showinfo("成功", f"已导出 {copied_count}/{len(event_screenshots)} 张截图到:\n{event_dir}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出截图失败: {str(e)}")

    def clear_screenshots(self):
        """清空所有截图"""
        if not self.screenshots:
            return
        
        if messagebox.askyesno("确认", f"确定要清空所有{len(self.screenshots)}个截图吗？"):
            try:
                # 删除截图文件
                for screenshot in self.screenshots:
                    if os.path.exists(screenshot['filepath']):
                        os.remove(screenshot['filepath'])
                
                # 尝试删除空目录
                import shutil
                if os.path.exists(self.screenshot_dir):
                    # 删除所有子目录
                    for item in os.listdir(self.screenshot_dir):
                        item_path = os.path.join(self.screenshot_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                
                # 清空列表和数据
                self.screenshots.clear()
                self.events.clear()
                self.current_event_id = 0
                self.last_event_time = None
                
                # 更新树形显示
                self.update_screenshot_tree()
                
                # 更新统计
                self.update_screenshot_statistics()
                
                self.log_message("所有截图和事件记录已清空")
                
            except Exception as e:
                messagebox.showerror("错误", f"清空截图失败: {str(e)}")

    def open_screenshot_dir(self):
        """打开截图文件夹"""
        try:
            if not os.path.exists(self.screenshot_dir):
                os.makedirs(self.screenshot_dir)
            
            os.startfile(self.screenshot_dir)  # Windows
        except:
            try:
                import subprocess
                subprocess.run(['xdg-open', self.screenshot_dir])  # Linux
            except:
                try:
                    import subprocess
                    subprocess.run(['open', self.screenshot_dir])  # macOS
                except:
                    messagebox.showwarning("提示", f"无法打开文件夹: {self.screenshot_dir}")
    
    def update_time_display(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(current_time)
        self.root.after(1000, self.update_time_display)
    
    def update_threshold_scale(self, key, value):
        """更新阈值参数（用于Scale控件）"""
        # 对于浮点数参数，确保范围正确
        if key in ['min_speed_consistency', 'min_direction_consistency', 
                  'min_area_stability', 'min_linearity', 'min_confidence']:
            value = max(0.1, min(1.0, value))
            value = round(value, 2)  # 保留两位小数
        
        self.detector.thresholds[key] = value
        
        # 更新标签显示
        if isinstance(value, float):
            self.threshold_labels[key].config(text=f"{value:.2f}")
        else:
            self.threshold_labels[key].config(text=f"{value}")
        
        self.log_message(f"阈值 '{key}' 更新为: {value}")
    
    # ========== 参数更新方法 ==========
    
    def update_all_displays(self):
        """更新所有显示"""
        # 更新标签显示
        self.downsample_label.config(text=f"{self.downsample_var.get():.2f}")
        self.diff_threshold_label.config(text=f"{self.diff_threshold_var.get()}")
        self.min_area_label.config(text=f"{self.min_area_var.get()}")
        self.max_area_label.config(text=f"{self.max_area_var.get()}")
        self.track_history_label.config(text=f"{self.track_history_var.get()}")
        self.min_track_duration_label.config(text=f"{self.min_track_duration_var.get()}")
        self.max_track_speed_label.config(text=f"{self.max_track_speed_var.get()}")
        self.bg_history_label.config(text=f"{self.bg_history_var.get()}")
        self.min_speed_label.config(text=f"{self.min_speed_var.get()}")
        self.max_speed_label.config(text=f"{self.max_speed_var.get()}")
        
        # 更新阈值标签
        for key, label in self.threshold_labels.items():
            var = self.threshold_vars[key]
            if key in ['min_speed_consistency', 'min_direction_consistency', 
                      'min_area_stability', 'min_linearity', 'min_confidence']:
                # 浮点数参数
                value = var.get() / 10.0  # 缩放回0.1-1.0范围
                label.config(text=f"{value:.2f}")
            else:
                label.config(text=f"{var.get()}")
    
    def update_downsample_ratio(self):
        """更新下采样比例"""
        self.detector.downsample_ratio = self.downsample_var.get()
        self.downsample_label.config(text=f"{self.downsample_var.get():.2f}")
        self.log_message(f"下采样比例更新为: {self.downsample_var.get():.2f}")
    
    def update_motion_method(self):
        """更新运动检测方法"""
        self.detector.motion_method = self.motion_method_var.get()
        self.log_message(f"运动检测方法更新为: {self.motion_method_var.get()}")
    
    def update_diff_threshold(self):
        """更新帧差阈值"""
        self.detector.frame_diff_threshold = self.diff_threshold_var.get()
        self.diff_threshold_label.config(text=f"{self.diff_threshold_var.get()}")
        self.log_message(f"帧差阈值更新为: {self.diff_threshold_var.get()}")
    
    def update_min_area(self):
        """更新最小面积"""
        self.detector.min_motion_area = self.min_area_var.get()
        self.min_area_label.config(text=f"{self.min_area_var.get()}")
        self.log_message(f"最小检测面积更新为: {self.min_area_var.get()}")
    
    def update_max_area(self):
        """更新最大面积"""
        self.detector.max_motion_area = self.max_area_var.get()
        self.max_area_label.config(text=f"{self.max_area_var.get()}")
        self.log_message(f"最大检测面积更新为: {self.max_area_var.get()}")
    
    def update_track_history(self):
        """更新轨迹历史长度"""
        self.detector.track_history_length = self.track_history_var.get()
        self.track_history_label.config(text=f"{self.track_history_var.get()}")
        self.log_message(f"轨迹历史长度更新为: {self.track_history_var.get()}")
    
    def update_min_track_duration(self):
        """更新最小跟踪持续帧数"""
        self.detector.min_track_duration = self.min_track_duration_var.get()
        self.min_track_duration_label.config(text=f"{self.min_track_duration_var.get()}")
        self.log_message(f"最小跟踪持续帧数更新为: {self.min_track_duration_var.get()}")
    
    def update_max_track_speed(self):
        """更新最大跟踪速度"""
        self.detector.max_track_speed = self.max_track_speed_var.get()
        self.max_track_speed_label.config(text=f"{self.max_track_speed_var.get()}")
        self.log_message(f"最大跟踪速度更新为: {self.max_track_speed_var.get()}")
    
    def update_bg_history(self):
        """更新背景历史帧数"""
        self.detector.bg_history = self.bg_history_var.get()
        self.bg_history_label.config(text=f"{self.bg_history_var.get()}")
        self.detector._init_background_subtractors()
        self.log_message(f"背景历史帧数更新为: {self.bg_history_var.get()}")
    
    def update_detect_shadows(self):
        """更新阴影检测设置"""
        self.detector.detect_shadows = self.detect_shadows_var.get()
        self.detector._init_background_subtractors()
        status = "启用" if self.detect_shadows_var.get() else "禁用"
        self.log_message(f"阴影检测已{status}")
    
    def update_roi_enabled(self):
        """更新ROI启用状态"""
        self.detector.enable_roi = self.enable_roi_var.get()
        status = "启用" if self.enable_roi_var.get() else "禁用"
        self.log_message(f"区域检测已{status}")
    
    def update_sky_detection(self):
        """更新天空检测设置"""
        self.detector.use_sky_detection = self.enable_sky_detection_var.get()
        status = "启用" if self.enable_sky_detection_var.get() else "禁用"
        self.log_message(f"天空区域检测已{status}")
    
    def update_speed_range(self):
        """更新速度范围"""
        min_speed = self.min_speed_var.get()
        max_speed = self.max_speed_var.get()
        
        if min_speed < max_speed:
            self.detector.thresholds['speed_range'] = (min_speed, max_speed)
            self.min_speed_label.config(text=f"{min_speed}")
            self.max_speed_label.config(text=f"{max_speed}")
            self.log_message(f"速度范围更新为: {min_speed}-{max_speed}")
        else:
            messagebox.showwarning("警告", "最小速度必须小于最大速度")
            self.min_speed_var.set(self.detector.thresholds['speed_range'][0])
            self.max_speed_var.set(self.detector.thresholds['speed_range'][1])
    def update_camera_fields(self, type_var, url_entry, index_spinbox, url_label, index_label, rtsp_help=None):
        """根据摄像头类型更新字段状态"""
        if type_var.get() == "network":
            # 显示网络摄像头字段
            url_entry.config(state="normal")
            url_label.config(state="normal")
            if rtsp_help is not None:  # 添加None检查
                rtsp_help.config(state="normal")
            
            # 隐藏本地摄像头字段
            index_spinbox.config(state="disabled")
            index_label.config(state="disabled")
        else:
            # 显示本地摄像头字段
            index_spinbox.config(state="normal")
            index_label.config(state="normal")
            
            # 隐藏网络摄像头字段
            url_entry.config(state="disabled")
            url_label.config(state="disabled")
            if rtsp_help is not None:  # 添加None检查
                rtsp_help.config(state="disabled")
    def on_camera_selected(self, event):
        """摄像头选择事件 - 添加到所有 update_xxx 方法后面"""
        selected_index = self.camera_combo.current()
        if selected_index >= 0 and selected_index < len(self.camera_list):
            camera = self.camera_list[selected_index]
            
            # 更新摄像头信息显示
            if camera["type"] == "network":
                info_text = f"类型: 网络摄像头 | RTSP: {camera['url'][:50]}..."
                self.camera_info_label.config(text=info_text, foreground="blue")
            else:
                info_text = f"类型: 本地摄像头 | 索引: {camera['index']}"
                self.camera_info_label.config(text=info_text, foreground="green")
            
            # 更新状态
            status_text = "状态: 就绪" if self.cap is None else "状态: 已连接"
            self.camera_status_label.config(text=status_text, 
                                        foreground="green" if self.cap else "red")

    def start_selected_camera(self):
        """启动选中的摄像头 - 添加到 on_camera_selected 后面"""
        if self.is_running:
            messagebox.showwarning("警告", "请先停止当前检测")
            return
        
        selected_index = self.camera_combo.current()
        if selected_index < 0 or selected_index >= len(self.camera_list):
            messagebox.showwarning("警告", "请选择摄像头")
            return
        
        camera = self.camera_list[selected_index]
        self.current_camera_name = camera["name"]  # 记录当前摄像头名称
        
        try:
            if camera["type"] == "network":
                # 网络摄像头
                self.cap = cv2.VideoCapture(camera["url"])
                if not self.cap.isOpened():
                    # 尝试不同的RTSP传输协议
                    protocols = [
                        camera["url"],
                        camera["url"].replace("rtsp://", "rtsp://@"),
                        camera["url"] + "?transport=tcp",  # 使用TCP传输
                        camera["url"] + "?transport=udp",  # 使用UDP传输
                    ]
                    
                    for protocol in protocols:
                        self.cap = cv2.VideoCapture(protocol)
                        if self.cap.isOpened():
                            camera["working_url"] = protocol  # 保存有效的URL
                            break
                    
                    if not self.cap.isOpened():
                        messagebox.showerror("错误", f"无法连接网络摄像头: {camera['name']}")
                        return
            else:
                # 本地摄像头
                camera_index = camera["index"]
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    # 尝试其他索引
                    for i in range(0, 5):
                        self.cap = cv2.VideoCapture(i)
                        if self.cap.isOpened():
                            camera["index"] = i  # 更新有效的索引
                            break
                    
                    if not self.cap.isOpened():
                        messagebox.showerror("错误", f"无法打开本地摄像头")
                        return
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 更新状态
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.snapshot_btn.config(state="normal")
            
            self.camera_status_label.config(text="状态: 已连接", foreground="green")
            self.status_var.set(f"{camera['name']} 已启动")
            self.log_message(f"{camera['name']} 已启动")
            
            # 开始处理帧
            self.process_frame()
            
        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {str(e)}")

    def show_custom_camera_dialog(self):
        """显示自定义摄像头配置对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("自定义摄像头配置")
        dialog.geometry("750x600")  # 调整为合适尺寸
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 中心对齐
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # 主框架
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        
        # 摄像头列表管理
        list_frame = ttk.LabelFrame(main_frame, text="摄像头列表", padding="10")
        list_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 创建Treeview显示摄像头列表
        columns = ("序号", "名称", "类型", "地址/索引")
        camera_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=6)
        
        column_widths = [50, 120, 70, 300]
        for i, col in enumerate(columns):
            camera_tree.heading(col, text=col)
            camera_tree.column(col, width=column_widths[i], minwidth=50)
            
        
        # 添加滚动条
        tree_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=camera_tree.yview)
        camera_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        camera_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # 填充摄像头列表
        for i, camera in enumerate(self.camera_list):
            url_display = camera["url"] if camera["type"] == "network" else f"索引: {camera['index']}"
            camera_tree.insert("", "end", values=(i+1, camera["name"], camera["type"], url_display))
        
        # 摄像头配置
        config_frame = ttk.LabelFrame(main_frame, text="添加/编辑摄像头", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        # 名称
        ttk.Label(config_frame, text="摄像头名称:").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        name_var = tk.StringVar(value="新摄像头")
        name_entry = ttk.Entry(config_frame, textvariable=name_var, width=35)
        name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 8), padx=(10, 0))
        
        # 类型选择
        ttk.Label(config_frame, text="摄像头类型:").grid(row=1, column=0, sticky=tk.W, pady=(0, 8))
        type_var = tk.StringVar(value="network")
        
        type_frame = ttk.Frame(config_frame)
        type_frame.grid(row=1, column=1, sticky=tk.W, pady=(0, 8), padx=(10, 0))
        
        ttk.Radiobutton(type_frame, text="网络摄像头", variable=type_var, 
                    value="network", command=lambda: self.update_camera_fields(type_var, url_entry, index_spinbox, url_label, index_label)).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(type_frame, text="本地摄像头", variable=type_var, 
                    value="local", command=lambda: self.update_camera_fields(type_var, url_entry, index_spinbox, url_label, index_label)).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # RTSP地址（网络摄像头）
        url_label = ttk.Label(config_frame, text="RTSP地址:")
        url_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 8))
        
        url_var = tk.StringVar(value="rtsp://username:password@192.168.1.100:554/stream")
        url_entry = ttk.Entry(config_frame, textvariable=url_var, width=40)
        url_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(0, 8), padx=(10, 0))
        
        # RTSP地址帮助文本（不需要改变状态，所以不传递到update方法）
        rtsp_help = ttk.Label(config_frame, text="格式: rtsp://用户名:密码@IP地址:端口/路径", 
                            foreground="gray", font=("Microsoft YaHei", 8))
        rtsp_help.grid(row=3, column=1, sticky=tk.W, pady=(0, 8), padx=(10, 0))
        
        # 摄像头索引（本地摄像头）
        index_label = ttk.Label(config_frame, text="摄像头索引:")
        index_label.grid(row=4, column=0, sticky=tk.W, pady=(0, 8))
        
        index_var = tk.IntVar(value=0)
        index_spinbox = ttk.Spinbox(config_frame, from_=0, to=10, textvariable=index_var, width=15)
        index_spinbox.grid(row=4, column=1, sticky=tk.W, pady=(0, 8), padx=(10, 0))
        
        # 初始更新字段状态
        self.update_camera_fields(type_var, url_entry, index_spinbox, url_label, index_label)
        
        # 操作按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 测试连接按钮
        def test_connection():
            """测试摄像头连接"""
            if type_var.get() == "network":
                url = url_var.get().strip()
                if not url:
                    messagebox.showwarning("警告", "请输入RTSP地址")
                    return
                cap = cv2.VideoCapture(url)
            else:
                cap = cv2.VideoCapture(index_var.get())
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    messagebox.showinfo("测试成功", "摄像头连接成功！\n已成功读取到视频帧。")
                else:
                    messagebox.showwarning("警告", "摄像头已连接但无法读取帧")
                cap.release()
            else:
                messagebox.showerror("测试失败", "无法连接摄像头")
        
        # 添加摄像头按钮
        def add_camera():
            """添加新摄像头"""
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("警告", "请输入摄像头名称")
                return
            
            new_camera = {
                "name": name,
                "type": type_var.get(),
                "url": url_var.get().strip() if type_var.get() == "network" else None,
                "index": index_var.get() if type_var.get() == "local" else None
            }
            
            # 检查名称是否重复
            for cam in self.camera_list:
                if cam["name"] == name:
                    if not messagebox.askyesno("确认", f"摄像头名称 '{name}' 已存在，是否覆盖？"):
                        return
                    # 移除已有的同名摄像头
                    self.camera_list = [cam for cam in self.camera_list if cam["name"] != name]
                    break
            
            self.camera_list.append(new_camera)
            self.camera_names = [cam["name"] for cam in self.camera_list]
            self.camera_combo["values"] = self.camera_names
            self.camera_combo.set(name)
            
            # 更新列表显示
            for item in camera_tree.get_children():
                camera_tree.delete(item)
            
            for i, camera in enumerate(self.camera_list):
                url_display = camera["url"] if camera["type"] == "network" else f"索引: {camera['index']}"
                camera_tree.insert("", "end", values=(i+1, camera["name"], camera["type"], url_display))
            
            self.log_message(f"摄像头 '{name}' 已添加到列表")
            messagebox.showinfo("成功", f"摄像头 '{name}' 已添加")
        
        # 删除摄像头按钮
        def delete_camera():
            """删除选中的摄像头"""
            selection = camera_tree.selection()
            if not selection:
                messagebox.showwarning("警告", "请选择要删除的摄像头")
                return
            
            for item in selection:
                values = camera_tree.item(item, "values")
                camera_name = values[1]
                
                if messagebox.askyesno("确认", f"确定要删除摄像头 '{camera_name}' 吗？"):
                    # 从列表中移除
                    self.camera_list = [cam for cam in self.camera_list if cam["name"] != camera_name]
                    camera_tree.delete(item)
            
            self.camera_names = [cam["name"] for cam in self.camera_list]
            self.camera_combo["values"] = self.camera_names
            if self.camera_names:
                self.camera_combo.set(self.camera_names[0])
            else:
                self.camera_combo.set("")
        
        # 保存配置按钮
        def save_config():
            """保存摄像头配置"""
            try:
                config = {
                    "camera_list": self.camera_list,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                default_dir = os.path.join(os.path.expanduser("~"), "Documents")
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    initialfile="camera_config.json",
                    initialdir=default_dir
                )
                
                if file_path:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    
                    messagebox.showinfo("成功", f"摄像头配置已保存到:\n{file_path}")
                    dialog.destroy()
                    
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败: {str(e)}")
        
        # 加载配置按钮
        def load_config():
            """加载摄像头配置"""
            try:
                default_dir = os.path.join(os.path.expanduser("~"), "Documents")
                file_path = filedialog.askopenfilename(
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    initialdir=default_dir
                )
                
                if file_path:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    self.camera_list = config.get("camera_list", self.camera_list)
                    self.camera_names = [cam["name"] for cam in self.camera_list]
                    self.camera_combo["values"] = self.camera_names
                    self.camera_combo.set(self.camera_names[0] if self.camera_names else "")
                    
                    # 更新列表显示
                    for item in camera_tree.get_children():
                        camera_tree.delete(item)
                    
                    for i, camera in enumerate(self.camera_list):
                        url_display = camera["url"] if camera["type"] == "network" else f"索引: {camera['index']}"
                        camera_tree.insert("", "end", values=(i+1, camera["name"], camera["type"], url_display))
                    
                    messagebox.showinfo("成功", f"摄像头配置已从 {os.path.basename(file_path)} 加载")
                    
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {str(e)}")
        
        # 创建按钮
        buttons = [
            ("测试连接", test_connection),
            ("添加摄像头", add_camera),
            ("删除选中", delete_camera),
            ("保存配置", save_config),
            ("加载配置", load_config),
            ("关闭", dialog.destroy)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(button_frame, text=text, command=command)
            btn.grid(row=0, column=i, padx=5, sticky=tk.W+tk.E)
            button_frame.columnconfigure(i, weight=1)
        
        # 配置列权重
        config_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 绑定双击编辑事件
        def on_tree_double_click(event):
            item = camera_tree.selection()
            if item:
                item = item[0]
                values = camera_tree.item(item, "values")
                camera_name = values[1]
                
                # 查找对应的摄像头
                for camera in self.camera_list:
                    if camera["name"] == camera_name:
                        name_var.set(camera["name"])
                        type_var.set(camera["type"])
                        if camera["type"] == "network":
                            url_var.set(camera["url"] if camera["url"] else "")
                        else:
                            index_var.set(camera["index"] if camera["index"] is not None else 0)
                        # 更新字段状态
                        self.update_camera_fields(type_var, url_entry, index_spinbox, url_label, index_label)
                        break
        
        camera_tree.bind("<Double-1>", on_tree_double_click)
        
        # 设置焦点
        name_entry.focus_set()

    # ========== 视频输入方法 ==========
    def browse_video_file(self):
        """浏览视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
    
    def start_video_file(self):
        """打开视频文件"""
        if self.is_running:
            messagebox.showwarning("警告", "请先停止当前检测")
            return
        
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showwarning("警告", "请选择有效的视频文件")
            return
        
        try:
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                messagebox.showerror("错误", f"无法打开视频文件: {file_path}")
                return
            
            # ========== 修改：使用"相机1"作为报警相机 ==========
            # 检查相机1是否存在，如果不存在则创建
            camera1_exists = False
            for cam in self.camera_list:
                if cam["name"] == "相机1":
                    camera1_exists = True
                    break
            
            if not camera1_exists:
                # 创建相机1
                self.camera_list.append({
                    "name": "相机1",
                    "type": "network",
                    "url": "rtsp://admin:admin123@192.168.1.101:554/Streaming/Channels/1",
                    "index": None,
                    "position": (150, 80)
                })
                self.camera_names = [cam["name"] for cam in self.camera_list]
                self.camera_combo["values"] = self.camera_names
                
                # 初始化报警状态
                self.camera_alert_status["相机1"] = False
                
                # 重绘地图
                self.draw_parallel_circuit_camera_markers()
            
            self.current_camera_name = "相机1"  # 使用相机1
            # ========== 修改结束 ==========
            
            # 获取视频信息
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.snapshot_btn.config(state="normal")
            
            self.status_var.set("视频文件已加载")
            self.log_message(f"视频文件已加载: {file_path}")
            self.log_message(f"视频信息: {fps:.1f} FPS, {frame_count}帧, {duration:.1f}秒")
            self.log_message("当前使用相机1进行检测，报警时将在地图上显示为红色")
            
            # 开始处理帧
            self.process_frame()
            
        except Exception as e:
            messagebox.showerror("错误", f"打开视频文件失败: {str(e)}")
    
    def start_detection(self):
        """开始检测"""
        if self.cap is None:
            messagebox.showwarning("警告", "请先选择输入源")
            return
        
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.snapshot_btn.config(state="normal")
            
            self.status_var.set("检测已启动")
            self.log_message("检测已启动")
            
            # 开始处理帧
            self.process_frame()
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        
        if self.update_job:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.snapshot_btn.config(state="disabled")
        
        self.video_label.config(image='', text="等待启动检测...")
        self.mask_label.config(image='', text="运动掩码")
        
        self.status_var.set("检测已停止")
        self.log_message("检测已停止")
        
        # 显示统计信息
        stats = self.detector.get_statistics()
        self.log_message("=== 检测统计 ===")
        self.log_message(f"总处理帧数: {stats['total_frames']}")
        self.log_message(f"总检测次数: {stats['total_detections']}")
        self.log_message(f"平均FPS: {stats['current_fps']:.1f}")
        self.log_message(f"运行时间: {stats['uptime_seconds']:.1f}秒")
        
        # 重置检测器
        self.detector.reset()
    
    def on_mouse_move(self, event):
        """处理鼠标移动事件（预览下一个点）"""
        if not self.is_selecting_region or not self.region_points:
            return
        
        # 获取当前鼠标位置
        x, y = event.x, event.y
        
        # 获取图像显示尺寸
        if hasattr(self.video_label, 'image'):
            img_width = self.video_label.image.width()
            img_height = self.video_label.image.height()
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width > 0 and label_height > 0:
                # 计算图像偏移
                offset_x = (label_width - img_width) // 2 if label_width > img_width else 0
                offset_y = (label_height - img_height) // 2 if label_height > img_height else 0
                
                # 调整坐标
                x = x - offset_x
                y = y - offset_y
                
                # 绘制包含鼠标位置的预览
                self.draw_temporary_region_with_preview(x, y)

    def draw_temporary_region_with_preview(self, preview_x, preview_y):
        """绘制包含鼠标位置预览的临时区域"""
        if not self.region_points or not self.is_running:
            return
        
        # 读取一帧
        ret, frame = self.cap.read()
        if ret:
            # 保存当前位置
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            # 绘制已选择的点
            for i, (x, y) in enumerate(self.region_points):
                # 将原始坐标转换为显示坐标
                display_x = int(x * self.video_label.image.width() / frame.shape[1])
                display_y = int(y * self.video_label.image.height() / frame.shape[0])
                
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # 红色实心点
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)  # 白色外圈
                cv2.putText(frame, f"{i+1}", (x+15, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"({x},{y})", (x+15, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 绘制连线
            if len(self.region_points) > 1:
                points = np.array(self.region_points, np.int32)
                cv2.polylines(frame, [points], False, (0, 255, 0), 2)
            
            # 绘制预览线（从最后一个点到鼠标位置）
            if len(self.region_points) > 0:
                last_point = self.region_points[-1]
                # 将鼠标位置转换为原始坐标
                if preview_x >= 0 and preview_y >= 0:
                    preview_original_x = int(preview_x * frame.shape[1] / self.video_label.image.width())
                    preview_original_y = int(preview_y * frame.shape[0] / self.video_label.image.height())
                    cv2.line(frame, last_point, (preview_original_x, preview_original_y), 
                            (255, 255, 0), 1, cv2.LINE_AA)  # 青色虚线预览
            
            # 恢复位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            
            # 显示
            self.display_frame(frame, self.video_label)
    
    def draw_temporary_region(self):
        """绘制临时区域"""
        self.draw_temporary_region_with_preview(-1, -1)
    
    def show_region_editor(self):
        """显示区域编辑器窗口"""
        if not self.is_running:
            messagebox.showwarning("警告", "请先启动检测")
            return
        
        # 检查是否已有编辑器窗口打开
        if hasattr(self, '_editor_window') and self._editor_window and self._editor_window.winfo_exists():
            self._editor_window.lift()  # 将现有窗口提到前面
            return
        
        # 读取当前帧
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("警告", "无法读取当前帧")
            return
        
        # 保存当前位置
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
        
        # 创建编辑器窗口
        editor = tk.Toplevel(self.root)
        editor.title("检测区域编辑器")
        editor.geometry("1000x700")
        editor.resizable(True, True)
        editor.transient(self.root)
        editor.grab_set()
        
        # 保存窗口引用
        self._editor_window = editor
        
        # 居中显示
        editor.update_idletasks()
        width = editor.winfo_width()
        height = editor.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        editor.geometry(f"{width}x{height}+{x}+{y}")
        
        # 主框架
        main_frame = ttk.Frame(editor, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(0, weight=1)
        
        # 上部分：图像显示和编辑区域
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 创建Canvas用于显示和编辑
        self.region_canvas = tk.Canvas(display_frame, bg="black", cursor="cross")
        self.region_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        v_scrollbar = ttk.Scrollbar(display_frame, orient="vertical", command=self.region_canvas.yview)
        h_scrollbar = ttk.Scrollbar(display_frame, orient="horizontal", command=self.region_canvas.xview)
        self.region_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 内部Frame用于放置图像
        self.region_inner_frame = ttk.Frame(self.region_canvas)
        self.canvas_window = self.region_canvas.create_window(
            (0, 0), window=self.region_inner_frame, anchor="nw", tags="region_frame"
        )
        
        # 加载图像到Canvas
        self.editor_frame = frame.copy()
        self.region_points = []  # 存储多边形点
        self.preview_point = None  # 预览点
        
        # 显示图像
        self.display_editor_image()
        
        # 右侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="区域编辑控制", padding="10")
        control_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        main_frame.columnconfigure(2, minsize=250)
        
        # 操作说明
        ttk.Label(control_frame, text="操作说明:", font=("Microsoft YaHei", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        instructions = [
            "1. 左键点击添加点",
            "2. 右键点击删除上一个点", 
            "3. 中键点击完成绘制",
            "4. 双击图像清除所有点",
            "5. 至少需要3个点形成区域"
        ]
        
        for i, text in enumerate(instructions):
            ttk.Label(control_frame, text=text, wraplength=220).grid(row=i+1, column=0, sticky=tk.W, pady=(0, 5))
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(20, 10))
        
        # 撤销按钮
        self.undo_btn = ttk.Button(button_frame, text="↶ 撤销", 
                                command=self.undo_editor_point, state="disabled")
        self.undo_btn.grid(row=0, column=0, padx=(0, 5))
        
        # 重做按钮
        self.redo_btn = ttk.Button(button_frame, text="↷ 重做", 
                                command=self.redo_editor_point, state="disabled")
        self.redo_btn.grid(row=0, column=1, padx=5)
        
        # 清除按钮
        self.clear_btn = ttk.Button(button_frame, text="🗑️ 清除", 
                                command=self.clear_editor_points)
        self.clear_btn.grid(row=0, column=2, padx=(5, 0))
        
        # 点数和面积显示
        self.point_count_label = ttk.Label(control_frame, text="点数: 0", font=("Microsoft YaHei", 9))
        self.point_count_label.grid(row=7, column=0, sticky=tk.W, pady=(0, 5))
        
        self.area_label = ttk.Label(control_frame, text="面积: 0 像素", font=("Microsoft YaHei", 9))
        self.area_label.grid(row=8, column=0, sticky=tk.W, pady=(0, 10))
        
        # 点列表
        points_frame = ttk.LabelFrame(control_frame, text="已选点列表", padding="5")
        points_frame.grid(row=9, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        control_frame.rowconfigure(9, weight=1)
        
        # 点列表文本框
        self.points_text = scrolledtext.ScrolledText(points_frame, height=8, width=25)
        self.points_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        points_frame.columnconfigure(0, weight=1)
        points_frame.rowconfigure(0, weight=1)
        
        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 应用按钮
        self.apply_btn = ttk.Button(bottom_frame, text="✓ 应用区域", 
                                command=lambda: self.apply_editor_region(editor), state="disabled")
        self.apply_btn.grid(row=0, column=0, padx=(0, 5))
        
        # 取消按钮
        ttk.Button(bottom_frame, text="✗ 取消", 
                command=editor.destroy).grid(row=0, column=1, padx=5)
        
        # 预览按钮
        ttk.Button(bottom_frame, text="👁️ 预览效果", 
                command=self.preview_region_effect).grid(row=0, column=2, padx=(5, 0))
        
        # 绑定事件
        self.region_canvas.bind("<Button-1>", self.add_editor_point)  # 左键添加点
        self.region_canvas.bind("<Button-3>", self.remove_last_editor_point)  # 右键删除点
        self.region_canvas.bind("<Button-2>", self.complete_editor_region)  # 中键完成
        self.region_canvas.bind("<Double-Button-1>", self.clear_editor_points)  # 双击清除
        self.region_canvas.bind("<Motion>", self.preview_next_editor_point)  # 鼠标移动预览
        
        # 绑定Canvas大小变化
        self.region_canvas.bind("<Configure>", lambda e: self.update_editor_canvas())
        
        # 绑定窗口关闭事件
        editor.protocol("WM_DELETE_WINDOW", lambda: self.close_editor_window(editor))
        
        # 初始化历史记录
        self.point_history = []  # 用于撤销/重做
        self.redo_stack = []
        
        # 初始更新
        self.update_points_display()
        
        return editor
    
    def redo_editor_point(self):
        """重做操作"""
        if not self.redo_stack:
            return
        
        # 保存当前状态到历史
        self.point_history.append(self.region_points.copy())
        
        # 恢复到重做状态
        self.region_points = self.redo_stack.pop()
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal")
        self.redo_btn.config(state="normal" if self.redo_stack else "disabled")
        
        # 检查应用按钮状态
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def preview_next_editor_point(self, event):
        """预览下一个点"""
        if not self.region_points:
            return
        
        # 获取鼠标位置
        self.preview_point = (event.x, event.y)
        
        # 更新显示
        self.update_editor_display()

    def update_editor_display(self):
        """更新编辑器显示（绘制点和连线）"""
        if not hasattr(self, 'editor_frame') or self.editor_frame is None:
            return
        
        try:
            # 创建图像副本用于绘制
            display_frame = cv2.cvtColor(self.editor_frame.copy(), cv2.COLOR_BGR2RGB)
            
            # 绘制点
            for i, (x, y) in enumerate(self.region_points):
                # 绘制点
                cv2.circle(display_frame, (x, y), 6, (255, 0, 0), -1)  # 蓝色实心点
                cv2.circle(display_frame, (x, y), 8, (255, 255, 255), 2)  # 白色外圈
                
                # 绘制编号
                cv2.putText(display_frame, str(i+1), (x+12, y-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 绘制坐标（可选，如果点太多可能会拥挤）
                if len(self.region_points) <= 10:
                    cv2.putText(display_frame, f"({x},{y})", (x+12, y+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 绘制连线
            if len(self.region_points) > 1:
                points = np.array(self.region_points, np.int32)
                cv2.polylines(display_frame, [points], False, (0, 255, 0), 2)  # 绿色线
            
            # 绘制闭合多边形（如果有足够点）
            if len(self.region_points) >= 3:
                points = np.array(self.region_points, np.int32)
                # 绘制半透明填充
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [points], (0, 255, 0, 128))
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            
            # 绘制预览线
            if self.region_points and self.preview_point:
                last_point = self.region_points[-1]
                cv2.line(display_frame, last_point, self.preview_point, 
                        (255, 255, 0), 2, cv2.LINE_AA)  # 青色虚线
            
            # 更新图像显示
            pil_image = Image.fromarray(display_frame)
            new_photo = ImageTk.PhotoImage(pil_image)
            
            if hasattr(self, 'region_image_label') and self.region_image_label:
                self.region_image_label.config(image=new_photo)
                self.region_image_label.image = new_photo
            
            # 更新点数和面积信息
            self.update_editor_info()
            
        except Exception as e:
            print(f"更新编辑器显示时出错: {e}")

    def update_editor_info(self):
        """更新编辑器信息显示"""
        point_count = len(self.region_points)
        if hasattr(self, 'point_count_label'):
            self.point_count_label.config(text=f"点数: {point_count}")
        
        # 计算并显示区域面积
        if point_count >= 3:
            points = np.array(self.region_points, np.int32)
            area = cv2.contourArea(points)
            if hasattr(self, 'area_label'):
                self.area_label.config(text=f"面积: {int(area)} 像素")
        elif hasattr(self, 'area_label'):
            self.area_label.config(text="面积: 0 像素")

    def update_points_display(self):
        """更新点列表显示"""
        if not hasattr(self, 'points_text'):
            return
        
        self.points_text.delete(1.0, tk.END)
        
        for i, (x, y) in enumerate(self.region_points):
            self.points_text.insert(tk.END, f"{i+1}. ({x}, {y})\n")

    def complete_editor_region(self, event):
        """完成区域绘制"""
        if len(self.region_points) < 3:
            messagebox.showwarning("警告", "至少需要3个点才能形成区域")
            return
        
        # 这里可以选择自动闭合多边形
        # 当前实现需要用户手动闭合
        
        self.apply_btn.config(state="normal")
    
    def remove_last_editor_point(self, event):
        """删除最后一个点"""
        if not self.region_points:
            return
        
        # 保存到历史
        self.point_history.append(self.region_points.copy())
        self.redo_stack.clear()
        
        # 删除最后一个点
        self.region_points.pop()
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="disabled")
        
        # 检查应用按钮状态
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")
    
    def undo_editor_point(self):
        """撤销操作"""
        if not self.point_history:
            return
        
        # 保存当前状态到重做栈
        self.redo_stack.append(self.region_points.copy())
        
        # 恢复到上一个状态
        self.region_points = self.point_history.pop()
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="normal" if self.redo_stack else "disabled")
        
        # 检查应用按钮状态
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def close_editor_window(self, editor_window):
        """关闭编辑器窗口"""
        editor_window.destroy()
        if hasattr(self, '_editor_window'):
            self._editor_window = None

    def display_editor_frame(self):
        """显示编辑器的帧"""
        if self.editor_frame is None:
            return
        
        # 清除Canvas
        for widget in self.region_inner_frame.winfo_children():
            widget.destroy()
        
        # 转换为RGB
        display_frame = cv2.cvtColor(self.editor_frame.copy(), cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 创建标签显示图像
        image_label = tk.Label(self.region_inner_frame, image=photo, cursor="cross")
        image_label.image = photo  # 保持引用
        
        # 绑定点击事件到图像标签
        image_label.bind("<Button-1>", self.add_editor_point)
        image_label.bind("<Button-3>", self.remove_last_point)
        image_label.bind("<Button-2>", self.complete_editor_region)
        image_label.bind("<Double-Button-1>", self.clear_editor_points)
        image_label.bind("<Motion>", self.preview_next_point)
        
        image_label.pack()
        
        # 更新Canvas区域
        self.update_editor_canvas()

    def update_editor_canvas(self):
        """更新编辑器Canvas"""
        self.region_inner_frame.update_idletasks()
        if self.region_image_label:
            width = self.region_image_label.winfo_reqwidth()
            height = self.region_image_label.winfo_reqheight()
            self.region_canvas.configure(
                scrollregion=(0, 0, width, height),
                width=min(width, 800),  # 限制最大宽度
                height=min(height, 600)  # 限制最大高度
            )

    def add_editor_point(self, event):
        """在编辑器中添加点"""
        # 获取相对于图像的坐标
        x = event.x
        y = event.y
        
        # 保存到历史
        self.point_history.append(self.region_points.copy())
        self.redo_stack.clear()
        
        # 添加新点
        self.region_points.append((x, y))
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal")
        self.redo_btn.config(state="disabled")
        
        # 如果有足够点，启用应用按钮
        if len(self.region_points) >= 3:
            self.apply_btn.config(state="normal")

    def remove_last_point(self, event):
        """删除最后一个点"""
        if not self.region_points:
            return
        
        # 保存到历史
        self.point_history.append(self.region_points.copy())
        self.redo_stack.clear()
        
        # 删除最后一个点
        self.region_points.pop()
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="disabled")
        
        # 检查应用按钮状态
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def undo_point(self):
        """撤销操作"""
        if not self.point_history:
            return
        
        # 保存当前状态到重做栈
        self.redo_stack.append(self.region_points.copy())
        
        # 恢复到上一个状态
        self.region_points = self.point_history.pop()
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="normal" if self.redo_stack else "disabled")
        
        # 检查应用按钮状态
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def redo_point(self):
        """重做操作"""
        if not self.redo_stack:
            return
        
        # 保存当前状态到历史
        self.point_history.append(self.region_points.copy())
        
        # 恢复到重做状态
        self.region_points = self.redo_stack.pop()
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal")
        self.redo_btn.config(state="normal" if self.redo_stack else "disabled")
        
        # 检查应用按钮状态
        if len(self.region_points) < 3:
            self.apply_btn.config(state="disabled")

    def clear_editor_points(self, event=None):
        """清除所有点"""
        if not self.region_points:
            return
        
        # 保存到历史
        self.point_history.append(self.region_points.copy())
        self.redo_stack.clear()
        
        # 清除所有点
        self.region_points.clear()
        
        # 更新显示
        self.update_editor_display()
        self.update_points_display()
        
        # 更新按钮状态
        self.undo_btn.config(state="normal" if self.point_history else "disabled")
        self.redo_btn.config(state="disabled")
        self.apply_btn.config(state="disabled")

    def apply_region(self, editor_window):
        """应用区域到检测器"""
        if len(self.region_points) < 3:
            messagebox.showwarning("警告", "至少需要3个点才能形成区域")
            return
        
        try:
            # 关闭编辑器窗口
            if editor_window and editor_window.winfo_exists():
                editor_window.destroy()
                if hasattr(self, '_editor_window'):
                    self._editor_window = None
            
            # 保存选择的区域
            self.selected_region = self.region_points.copy()
            
            # 检查是否有摄像头正在运行
            if not self.is_running or self.cap is None:
                messagebox.showwarning("警告", "请先启动摄像头或视频")
                return
            
            # 读取一帧用于创建掩码
            ret, temp_frame = self.cap.read()
            if not ret:
                messagebox.showwarning("警告", "无法读取当前帧")
                return
            
            # 恢复读取位置
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            
            # 创建掩码
            mask = np.zeros(temp_frame.shape[:2], dtype=np.uint8)
            
            # 将点坐标转换为原始图像坐标
            # 注意：编辑器中的点已经在原始坐标中
            original_points = []
            for x, y in self.selected_region:
                # 确保坐标在图像范围内
                x = max(0, min(x, temp_frame.shape[1] - 1))
                y = max(0, min(y, temp_frame.shape[0] - 1))
                original_points.append((x, y))
            
            # 创建多边形掩码
            if len(original_points) >= 3:
                points = np.array(original_points, np.int32)
                cv2.fillPoly(mask, [points], 255)
                
                # 应用ROI掩码到检测器
                self.detector.roi_mask = mask
                
                # 更新区域信息显示（使用正确的属性）
                point_count = len(original_points)
                if hasattr(self, 'region_info_label'):
                    if point_count <= 6:
                        coords_text = f"检测区域: {point_count}个点"
                        for i, (x, y) in enumerate(original_points[:3]):  # 只显示前3个点，避免太长
                            coords_text += f" P{i+1}({x},{y})"
                        if point_count > 3:
                            coords_text += f" ...等{point_count}个点"
                    else:
                        coords_text = f"检测区域: {point_count}个多边形点 [已应用]"
                    
                    self.region_info_label.config(text=coords_text, foreground="green")
                
                # 启用清除按钮
                if hasattr(self, 'clear_region_btn'):
                    self.clear_region_btn.config(state="normal")
                
                self.status_var.set(f"检测区域设置完成（{point_count}个点）")
                self.log_message(f"检测区域已设置，包含 {point_count} 个点")
                
                # 使用after延迟处理下一帧，避免递归调用
                self.root.after(50, self.process_frame)
            else:
                messagebox.showwarning("警告", "区域点无效")
                
        except Exception as e:
            print(f"应用区域时出错: {e}")
            self.log_message(f"应用区域失败: {str(e)}", level="error")
            messagebox.showerror("错误", f"应用区域失败: {str(e)}")

    def preview_region_effect(self):
        """预览区域效果"""
        if len(self.region_points) < 3:
            messagebox.showwarning("警告", "至少需要3个点才能预览")
            return
        
        # 创建临时掩码
        mask = np.zeros(self.editor_frame.shape[:2], dtype=np.uint8)
        points = np.array(self.region_points, np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # 应用掩码到图像
        masked_frame = self.editor_frame.copy()
        masked_frame[mask == 0] = 0
        
        # 在新窗口中显示预览
        preview = tk.Toplevel(self.root)
        preview.title("区域效果预览")
        preview.geometry("800x600")
        
        # 创建Canvas显示预览
        canvas = tk.Canvas(preview, bg="black")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # 转换为RGB并显示
        preview_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(preview_frame)
        photo = ImageTk.PhotoImage(pil_image)
        
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # 保持引用
        
        # 调整窗口大小
        preview.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    # ========== 修改 select_region 方法 ==========

    def select_region(self):
        """选择检测区域（使用新的编辑器）"""
        if not self.is_running:
            messagebox.showwarning("警告", "请先启动检测")
            return
        
        # 使用新的区域编辑器
        self.show_region_editor()

    def display_editor_image(self):
        """显示编辑器的图像"""
        if self.editor_frame is None:
            return
        
        # 清除Canvas内部Frame的内容
        for widget in self.region_inner_frame.winfo_children():
            widget.destroy()
        
        # 转换图像格式
        display_frame = cv2.cvtColor(self.editor_frame.copy(), cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        self.editor_pil_image = Image.fromarray(display_frame)
        self.editor_photo = ImageTk.PhotoImage(self.editor_pil_image)
        
        # 创建标签显示图像
        self.region_image_label = tk.Label(self.region_inner_frame, image=self.editor_photo, cursor="cross")
        self.region_image_label.image = self.editor_photo  # 保持引用
        
        # 绑定事件到图像标签
        self.region_image_label.bind("<Button-1>", self.add_editor_point)
        self.region_image_label.bind("<Button-3>", self.remove_last_editor_point)
        self.region_image_label.bind("<Button-2>", self.complete_editor_region)
        self.region_image_label.bind("<Double-Button-1>", self.clear_editor_points)
        self.region_image_label.bind("<Motion>", self.preview_next_editor_point)
        
        self.region_image_label.pack()
        
        # 更新Canvas区域
        self.update_editor_canvas()

    def add_editor_point(self, event):
        """在编辑器中添加点"""
        try:
            # 获取相对于图像的坐标
            x = event.x
            y = event.y
            
            # 调整坐标为原始图像坐标（考虑缩放）
            if hasattr(self, 'editor_scale') and self.editor_scale < 1.0:
                x = int(x / self.editor_scale)
                y = int(y / self.editor_scale)
            
            # 保存到历史
            self.point_history.append(self.region_points.copy())
            self.redo_stack.clear()
            
            # 添加新点
            self.region_points.append((x, y))
            
            # 更新显示
            self.update_editor_display()
            
            # 更新按钮状态
            self.undo_btn.config(state="normal")
            self.redo_btn.config(state="disabled")
            
            # 如果有足够点，启用应用按钮
            if len(self.region_points) >= 3:
                self.apply_btn.config(state="normal")
                
        except Exception as e:
            print(f"添加点时出错: {e}")

    def update_region_ui(self):
        """更新区域UI显示"""
        if not self.selected_region:
            return
        
        point_count = len(self.selected_region)
        
        # 更新区域信息标签
        if point_count <= 4:
            coords_text = f"检测区域: {point_count}个点"
            for i, (x, y) in enumerate(self.selected_region[:4]):
                coords_text += f" P{i+1}({x},{y})"
            if point_count > 4:
                coords_text += f" ..."
        else:
            coords_text = f"检测区域: {point_count}个多边形点 [已应用]"
        
        if hasattr(self, 'region_info_label'):
            self.region_info_label.config(text=coords_text, foreground="green")
        
        # 启用清除按钮
        if hasattr(self, 'clear_region_btn'):
            self.clear_region_btn.config(state="normal")

    def create_and_cache_roi_mask(self):
        """创建并缓存ROI掩码（提高性能）"""
        if not self.selected_region or not self.is_running or not self.cap:
            return
        
        try:
            # 读取一帧用于创建掩码
            ret, temp_frame = self.cap.read()
            if not ret:
                return
            
            # 恢复读取位置
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            
            # 创建掩码
            mask = np.zeros(temp_frame.shape[:2], dtype=np.uint8)
            
            # 确保点坐标在图像范围内
            original_points = []
            for x, y in self.selected_region:
                x = max(0, min(x, temp_frame.shape[1] - 1))
                y = max(0, min(y, temp_frame.shape[0] - 1))
                original_points.append((x, y))
            
            # 创建多边形掩码
            if len(original_points) >= 3:
                points = np.array(original_points, np.int32)
                cv2.fillPoly(mask, [points], 255)
                
                # 缓存掩码和边界框
                self.detector.roi_mask = mask
                self.detector.roi_bbox = cv2.boundingRect(points)
                
                # 更新区域信息显示
                self.update_region_ui()
                
        except Exception as e:
            print(f"创建ROI掩码时出错: {e}")

    def apply_editor_region(self, editor_window):
        """应用编辑好的区域到检测器"""
        try:
            if len(self.region_points) < 3:
                messagebox.showwarning("警告", "至少需要3个点才能形成区域")
                return
            
            # 保存选择的区域(点)
            self.selected_region = self.region_points.copy()
            
            # 关闭编辑器窗口
            if editor_window and editor_window.winfo_exists():
                editor_window.destroy()
            
            # 清除编辑器引用
            self._editor_window = None
            
            # 异步创建ROI掩码
            self.root.after(100, self.create_and_cache_roi_mask)
            
            # 更新UI
            self.update_region_ui()
            
            self.status_var.set(f"检测区域设置完成（{len(self.selected_region)}个点）")
            self.log_message(f"检测区域已设置，包含 {len(self.selected_region)} 个点")
            
            # 重新处理一帧以显示区域
            if self.is_running:
                self.root.after(50, self.process_frame)
            
        except Exception as e:
            print(f"应用区域时出错: {e}") 
            messagebox.showerror("错误", f"应用区域失败: {str(e)}")

    def create_and_apply_roi_mask(self):
        """创建并应用ROI掩码（在后台线程中执行）"""
        try:
            # 读取当前帧
            if not self.cap or not self.is_running:
                return
            
            ret, temp_frame = self.cap.read()
            if not ret:
                messagebox.showwarning("警告", "无法读取当前帧")
                return
            
            # 恢复读取位置
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            
            # 创建掩码
            mask = np.zeros(temp_frame.shape[:2], dtype=np.uint8)
            
            # 将点坐标转换为原始图像坐标
            # 注意：编辑器中的点已经在原始坐标中
            original_points = []
            for x, y in self.selected_region:
                # 确保坐标在图像范围内
                x = max(0, min(x, temp_frame.shape[1] - 1))
                y = max(0, min(y, temp_frame.shape[0] - 1))
                original_points.append((x, y))
            
            # 创建多边形掩码
            if len(original_points) >= 3:
                points = np.array(original_points, np.int32)
                cv2.fillPoly(mask, [points], 255)
                
                # 应用ROI掩码到检测器
                self.detector.roi_mask = mask
                
                # 更新区域信息显示
                point_count = len(original_points)
                if point_count <= 4:
                    coords_text = f"检测区域: {point_count}个点"
                    for i, (x, y) in enumerate(original_points[:4]):
                        coords_text += f" P{i+1}({x},{y})"
                    if point_count > 4:
                        coords_text += f" ..."
                else:
                    coords_text = f"检测区域: {point_count}个多边形点 [已应用]"
                
                self.region_info_label.config(text=coords_text, foreground="black")
                self.clear_region_btn.config(state="normal")
                
                self.status_var.set(f"检测区域设置完成（{point_count}个点）")
                self.log_message(f"检测区域已设置，包含 {point_count} 个点")
                
                # 使用after延迟处理下一帧，避免递归调用
                self.root.after(50, self.process_frame)
            else:
                messagebox.showwarning("警告", "区域点无效")
                
        except Exception as e:
            print(f"创建ROI掩码时出错: {e}")
            self.log_message(f"创建ROI掩码失败: {str(e)}", level="error")

    def remove_last_point(self, event):
        """删除最后一个点"""
        try:
            if not self.region_points:
                return
            
            # 保存到历史
            self.point_history.append(self.region_points.copy())
            self.redo_stack.clear()
            
            # 删除最后一个点
            self.region_points.pop()
            
            # 更新显示
            self.update_editor_display()
            
            # 更新按钮状态
            self.undo_btn.config(state="normal" if self.point_history else "disabled")
            self.redo_btn.config(state="disabled")
            
            # 检查应用按钮状态
            if len(self.region_points) < 3:
                self.apply_btn.config(state="disabled")
                
        except Exception as e:
            print(f"删除点时出错: {e}")

    def clear_editor_points(self, event=None):
        """清除所有点"""
        try:
            if not self.region_points:
                return
            
            # 保存到历史
            self.point_history.append(self.region_points.copy())
            self.redo_stack.clear()
            
            # 清除所有点
            self.region_points.clear()
            self.preview_point = None
            
            # 更新显示
            self.update_editor_display()
            
            # 更新按钮状态
            self.undo_btn.config(state="normal" if self.point_history else "disabled")
            self.redo_btn.config(state="disabled")
            self.apply_btn.config(state="disabled")
            
        except Exception as e:
            print(f"清除点时出错: {e}")

    def preview_next_point(self, event):
        """预览下一个点"""
        try:
            if not self.region_points:
                return
            
            # 获取鼠标位置
            self.preview_point = (event.x, event.y)
            
            # 更新显示
            self.update_editor_display()
            
        except Exception as e:
            print(f"预览点时出错: {e}")

    # ========== 在 create_input_tab 中修改按钮 ==========

    # 在 create_input_tab 方法中，修改区域设置部分：

    def clear_region(self):
        """清除检测区域"""
        self.selected_region = None
        self.region_points = []
        
        # 清除检测器中的ROI设置
        if hasattr(self.detector, 'roi_mask'):
            self.detector.roi_mask = None
            self.detector.roi_points = None
            self.detector.roi_bbox = None
        
        # 更新区域信息显示
        if hasattr(self, 'region_info_label'):
            self.region_info_label.config(text="未设置检测区域", foreground="gray")
        
        # 禁用清除按钮
        if hasattr(self, 'clear_region_btn'):
            self.clear_region_btn.config(state="disabled")
        
        self.status_var.set("检测区域已清除")
        self.log_message("检测区域已清除")
        
        # 重新处理一帧以更新显示
        if self.is_running:
            self.root.after(50, self.process_frame)
    
    def complete_region_selection(self):
        """完成区域选择"""
        self.is_selecting_region = False
        self.selected_region = self.region_points.copy()
        
        # 解绑鼠标事件
        self.video_label.unbind("<Button-1>")
        
        # 更新区域信息
        coords_text = "区域坐标: "
        for i, (x, y) in enumerate(self.region_points):
            coords_text += f"P{i+1}({x},{y}) "
        
        self.region_info_label.config(text=coords_text, foreground="black")
        self.clear_region_btn.config(state="normal")
        
        self.status_var.set("检测区域设置完成")
        self.log_message(f"检测区域已设置，包含 {len(self.region_points)} 个点")
    
    def clear_region(self):
        """清除检测区域"""
        self.selected_region = None
        self.region_points = []
        self.region_info_label.config(text="未设置检测区域", foreground="gray")
        self.clear_region_btn.config(state="disabled")
        
        self.status_var.set("检测区域已清除")
        self.log_message("检测区域已清除")
    
    def process_frame(self):
        """处理视频帧"""
        if not self.is_running or self.cap is None:
            return
        
        try:
            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                # 如果是视频文件，循环播放
                if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.stop_detection()
                        return
                else:
                    self.stop_detection()
                    return
            
            # ========== 新增：绘制检测区域（如果已设置） ==========
            start_time = time.time()
            if self.selected_region and len(self.selected_region) >= 3:
                try:
                    # 将点列表转换为numpy数组
                    points = np.array(self.selected_region, np.int32)
                    
                    # 绘制区域轮廓
                    cv2.polylines(frame, [points], True, (0, 255, 0), 2)  # 绿色轮廓
                    
                    # 在区域中心显示"检测区域"文字
                    if len(points) > 0:
                        # 计算区域中心
                        center_x = int(np.mean(points[:, 0]))
                        center_y = int(np.mean(points[:, 1]))
                        
                        # 在轮廓上添加标签
                        text_position = (points[0][0] - 20, points[0][1] - 10)
                        cv2.putText(frame, "ROI", text_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # 在第一个点上添加编号
                        for i, (x, y) in enumerate(self.selected_region):
                            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                            cv2.putText(frame, str(i+1), (x+8, y-8), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"绘制检测区域时出错: {e}")
            end_time = time.time()
            #print(f"绘制区域花费时间： {end_time - start_time} s")
            # ========== 新增结束 ==========
            
            # 获取质量信息 
            quality_time1 = time.time()
            quality_info = self.detector.get_quality_info()
            is_night = quality_info.get('is_night', False)
            
            # 如果天黑且设置了跳过，则不进行检测
            if is_night and self.detector.skip_night_frames:
                # 在天黑帧上添加提示
                h, w = frame.shape[:2]
                
                # 添加红色半透明覆盖层
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # 添加天黑提示文字
                text = "🌙 天黑/过暗 - 检测暂停"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2
                
                # cv2.putText(frame, text, (text_x, text_y), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                frame = self.put_chinese_text_cv(frame, "  天黑/过暗 - 检测暂停", 
                                 (text_x, text_y),
                                 font_size=30, 
                                 color=(0, 0, 255),
                                 bg_color=(255, 255, 255))  # 白色背景
                
                # 添加原因说明
                reason = quality_info.get('message', '')
                # cv2.putText(frame, reason, (text_x, text_y + 50), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                frame = self.put_chinese_text_cv(frame, reason, 
                                 (text_x, text_y + 50),
                                 font_size=30, 
                                 color=(0, 0, 255),
                                 bg_color=(255, 255, 255))  # 白色背景
                
                # 设置空检测结果
                detections = []
                motion_mask = None
                result_frame = frame
            else:
                # 正常检测
                detections, motion_mask = self.detector.detect(frame, self.selected_region)
                
                # 保存截图
                if detections and self.auto_save_var.get():
                    for detection in detections:
                        if detection['features']['confidence'] > 0.7:
                            self.save_and_display_screenshot(frame.copy(), detection)
                
                # 绘制检测结果
                result_frame = self.detector.draw_detections(frame, detections)
            
            # 在右上角添加质量状态指示器
            h, w = result_frame.shape[:2]
            
            if is_night:
                # 红色月亮图标 - 天黑
                indicator_x = w - 60
                indicator_y = 20
                #cv2.circle(result_frame, (indicator_x, indicator_y), 20, (0, 0, 255), 2)
                # cv2.putText(result_frame, "🌙", (indicator_x - 15, indicator_y + 10),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                result_frame = self.put_chinese_text_cv(result_frame, "图片过暗", 
                                 (indicator_x - 80, indicator_y + 10),
                                 font_size=30, 
                                 color=(0, 0, 255),
                                 bg_color=(255, 255, 255))  # 白色背景
            else:
                # 绿色太阳图标 - 正常
                indicator_x = w - 60
                indicator_y = 20
                #cv2.circle(result_frame, (indicator_x, indicator_y), 20, (0, 255, 0), 2)
                # cv2.putText(result_frame, "☀", (indicator_x - 15, indicator_y + 10),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                result_frame = self.put_chinese_text_cv(result_frame, "图片正常", 
                                 (indicator_x - 80, indicator_y + 10),
                                 font_size=30, 
                                 color=(0, 0, 255),
                                 bg_color=(255, 255, 255))  # 白色背景
            quality_time2 = time.time()
            # print(f"计算图片质量花费: {quality_time2 - quality_time1} s")

            relax_time1 = time.time()
            # 显示帧
            self.display_frame(result_frame, self.video_label)
            
            # 显示运动掩码
            if motion_mask is not None:
                if len(motion_mask.shape) == 2:
                    motion_mask_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                else:
                    motion_mask_display = motion_mask
                self.display_frame(motion_mask_display, self.mask_label)
            
            # 更新信息显示（包括质量信息）
            self.update_info_displays(detections)
            
            # 继续处理下一帧
            if self.is_running:
                self.update_job = self.root.after(30, self.process_frame)
            relax_time2 = time.time()
            #print(f"剩余耗时: {relax_time2 - relax_time1} s")
            
        except Exception as e:
            self.log_message(f"处理帧时出错: {str(e)}", level="error")
            if self.is_running:
                self.update_job = self.root.after(100, self.process_frame)
    
    def display_frame(self, frame, label):
        """显示帧到标签"""
        try:
            if frame is None or frame.size == 0:
                return
            
            # 转换颜色空间
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # BGR转RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应标签
            label_width = label.winfo_width()
            label_height = label.winfo_height()
            
            if label_width > 10 and label_height > 10:
                # 计算保持宽高比的尺寸
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                
                if label_width / label_height > aspect_ratio:
                    new_height = label_height
                    new_width = int(label_height * aspect_ratio)
                else:
                    new_width = label_width
                    new_height = int(label_width / aspect_ratio)
                
                # 调整大小
                if new_width > 0 and new_height > 0:
                    frame = cv2.resize(frame, (new_width, new_height))
            
            # 转换为PhotoImage
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            # 更新标签
            label.config(image=photo)
            label.image = photo
            
        except Exception as e:
            print(f"显示帧错误: {e}")
    
    def update_info_displays(self, detections):
        """更新信息显示"""
        # 更新统计信息
        stats = self.detector.get_statistics()
        
        # 更新质量信息（新增）
        self.update_quality_display()
        
        # 更新状态栏（添加质量状态）
        quality_info = self.detector.get_quality_info()
        is_night = quality_info.get('is_night', False)
        quality_status = "🌙天黑" if is_night else "☀正常"
        
        self.status_var.set(f"检测中 | FPS: {stats['current_fps']:.1f} | 检测数: {len(detections)} | 光照: {quality_status}")
        
        # 更新检测详情表格
        self.update_detection_tree(detections)
        
        # 更新统计图表
        self.update_stats_chart(stats)
    
    def update_detection_tree(self, detections):
        """更新检测详情表格"""
        # 清空表格
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
        
        # 添加新数据
        for detection in detections:
            features = detection['features']
            
            # 格式化数据
            position = f"({detection['center'][0]}, {detection['center'][1]})"
            speed = f"{features['mean_speed']:.1f}"
            confidence = f"{features['confidence']*100:.1f}%"
            track_length = str(detection['track_length'])
            status = detection.get('status', 'confirmed')
            
            # 插入行
            self.detection_tree.insert("", "end", values=(
                detection['id'],
                position,
                speed,
                confidence,
                track_length,
                status
            ))
    
    def update_stats_chart(self, stats):
        """更新统计图表"""
        self.stats_canvas.delete("all")
        
        # 绘制背景
        width = self.stats_canvas.winfo_width()
        height = self.stats_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # 绘制标题
        self.stats_canvas.create_text(width//2, 15, text="实时统计", 
                                     font=("Microsoft YaHei", 12, "bold"))
        
        # 绘制统计信息
        info_lines = [
            f"FPS: {stats['current_fps']:.1f}",
            f"检测数: {stats['current_detections']}",
            f"总检测: {stats['total_detections']}",
            f"活动轨迹: {stats['active_tracks']}",
            f"处理时间: {stats['processing_time_ms']:.1f}ms"
        ]
        
        y_pos = 40
        for line in info_lines:
            self.stats_canvas.create_text(20, y_pos, text=line, 
                                         font=("Microsoft YaHei", 10), anchor="w")
            y_pos += 20
    
    def save_snapshot(self):
        """保存快照"""
        if not self.is_running or self.cap is None:
            return
        
        try:
            # 读取当前帧
            ret, frame = self.cap.read()
            if ret:
                # 进行检测
                detections, _ = self.detector.detect(frame, self.selected_region)
                
                # 绘制检测结果
                result_frame = self.detector.draw_detections(frame, detections)
                
                # 保存文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                
                cv2.imwrite(filename, result_frame)
                
                # 恢复读取位置
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                
                self.status_var.set(f"快照已保存: {filename}")
                self.log_message(f"快照已保存: {filename}")
                
        except Exception as e:
            messagebox.showerror("错误", f"保存快照失败: {str(e)}")
    
    def save_config(self):
        """保存配置"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="airborne_detector_config.json"
        )
        
        if file_path:
            try:
                # 收集当前配置
                config = {
                    'downsample_ratio': self.downsample_var.get(),
                    'use_sky_detection': self.enable_sky_detection_var.get(),
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
                
                # 添加阈值参数
                for key in self.threshold_vars:
                    if key in ['min_speed_consistency', 'min_direction_consistency', 
                              'min_area_stability', 'min_linearity', 'min_confidence']:
                        # 浮点数参数，需要缩放
                        config['thresholds'][key] = self.threshold_vars[key].get() / 10.0
                    else:
                        config['thresholds'][key] = self.threshold_vars[key].get()
                
                # 添加速度范围
                config['thresholds']['speed_range'] = [
                    self.min_speed_var.get(),
                    self.max_speed_var.get()
                ]
                
                # 保存到文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.log_message(f"配置已保存到: {file_path}")
                messagebox.showinfo("成功", "配置保存成功")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败: {str(e)}")
    
    def load_config(self):
        """加载配置"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if self.detector.load_configuration(file_path):
                    # 更新UI控件
                    self.downsample_var.set(self.detector.downsample_ratio)
                    self.enable_sky_detection_var.set(self.detector.use_sky_detection)
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
                    
                    # 更新阈值参数
                    for key in self.threshold_vars:
                        if key in self.detector.thresholds and key != 'speed_range':
                            if key in ['min_speed_consistency', 'min_direction_consistency', 
                                      'min_area_stability', 'min_linearity', 'min_confidence']:
                                # 浮点数参数，需要缩放
                                self.threshold_vars[key].set(int(self.detector.thresholds[key] * 10))
                            else:
                                self.threshold_vars[key].set(self.detector.thresholds[key])
                    
                    # 更新速度范围
                    if 'speed_range' in self.detector.thresholds:
                        speed_range = self.detector.thresholds['speed_range']
                        self.min_speed_var.set(speed_range[0])
                        self.max_speed_var.set(speed_range[1])
                    
                    # 更新显示
                    self.update_all_displays()
                    
                    self.log_message(f"配置已从文件加载: {file_path}")
                    messagebox.showinfo("成功", "配置加载成功")
                else:
                    messagebox.showerror("错误", "加载配置失败")
                    
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {str(e)}")
    
    def reset_parameters(self):
        """重置参数为默认值"""
        if messagebox.askyesno("确认", "确定要重置所有参数为默认值吗？"):
            # 重新初始化检测器
            self.detector = AirborneDetector()
            
            # 更新UI控件
            self.downsample_var.set(self.detector.downsample_ratio)
            self.enable_sky_detection_var.set(self.detector.use_sky_detection)
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
            
            # 更新阈值参数
            for key in self.threshold_vars:
                if key in self.detector.thresholds and key != 'speed_range':
                    if key in ['min_speed_consistency', 'min_direction_consistency', 
                              'min_area_stability', 'min_linearity', 'min_confidence']:
                        self.threshold_vars[key].set(int(self.detector.thresholds[key] * 10))
                    else:
                        self.threshold_vars[key].set(self.detector.thresholds[key])
            
            # 更新速度范围
            self.min_speed_var.set(self.detector.thresholds['speed_range'][0])
            self.max_speed_var.set(self.detector.thresholds['speed_range'][1])
            
            # 更新显示
            self.update_all_displays()
            
            self.log_message("所有参数已重置为默认值")
            messagebox.showinfo("成功", "参数重置完成")
    
    def log_message(self, message, level="info"):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 设置颜色
        if level == "error":
            color = "red"
            prefix = "[错误]"
        elif level == "warning":
            color = "orange"
            prefix = "[警告]"
        else:
            color = "black"
            prefix = "[信息]"
        
        log_entry = f"[{timestamp}] {prefix} {message}\n"
        
        # 添加到日志文本
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)  # 滚动到底部
        
        # 限制日志行数
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 200:
            self.log_text.delete(1.0, "2.0")
        
        # 保存到消息列表
        self.log_messages.append(log_entry)
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        self.log_messages.clear()
        self.log_message("日志已清空")
    
    def on_closing(self):
        """关闭窗口时的处理"""
        self.stop_detection()
        
        # 关闭所有警报窗口
        if hasattr(self, 'alert_windows'):
            for window in self.alert_windows[:]:  # 使用副本遍历
                if window and window.winfo_exists():
                    try:
                        window.destroy()
                    except:
                        pass
        
        # 清理截图相关的引用
        if hasattr(self, 'screenshot_canvas'):
            try:
                self.screenshot_canvas.unbind_all("<MouseWheel>")
            except:
                pass
        
        self.root.destroy()

def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置窗口图标（可选）
    try:
        root.iconbitmap('icon.ico')  # 如果有图标文件
    except:
        pass
    
    app = AirborneDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()