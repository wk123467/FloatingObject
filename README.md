# 空飘物检测

## 空飘物检测代码流程图

### 1.创建ROI掩码

- 按照在图中选取的三个点，构成一个ROI掩码，掩码区域的数值设为255

### 2.天空检测

- 利用HSV色彩空间，提取图片中的白色和蓝色区域（使用or符号，将两组颜色合并到一个掩码区域）
- 使用形态学工具，利用腐蚀和膨胀技术，修补两组颜色区域之间存在的空隙
- 利用连通域确保天空是连通在一起的
- 返回检测到的天空

### 3.运动检测

- 下采样加速：对图片进行缩小，加速运算

1. 将前一帧和当前帧转换为灰度图
2. 进行适当的高斯模糊，忽略细节带来的影响
3. 计算当前帧和前一帧之间的差值，并提取大于一定阈值的区域
4. 再次利用形态学工具（腐蚀和膨胀），修补帧差带来的间隙
5. 返回运动检测区域

### 4.从运动掩码中提取运动区域

1. 从运动掩码中提取轮廓
2. 计算每个轮廓的面积，提取符合阈值的运动轮廓
3. 计算每个轮廓的中心点和宽高，确保中心点落在ROI区域内，并宽高比在0.2~5的区间内

### 5.跟踪

1. 如果某个物体1秒钟都没有检测到，则清理该物体的跟踪ID和跟踪器
2. 检测到的物体和原有物体进行匹配
   - 尝试与现有轨迹进行关联-
     - 关联规则：当检测到的轨迹与现有轨迹的欧式距离小于**最大跟踪速度**的2倍；遍历所有轨迹选择距离最小的轨迹
     - 长度限制：存储**最大跟踪长度**
     - **未关联的检测物体，生成新的跟踪器**
   - 计算运动特征
     - 计算速度，利用同个物体前后两个轨迹中心点的（dx，dy）的欧氏距离测算
     - 计算位移方向：利用（dx，dy）构成的夹角获得
3. 计算有效轨迹
   - 分析轨迹特征
     - 计算速度一致性
     - 计算角度方差
     - 面积稳定性
     - 轨迹线性度
   - 分析是否为空漂物
     - 是否检测到8帧
     - 速度范围是否在阈值内
     - 置信度是否大于阈值

## 空飘物检测界面

### 打开界面

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img1.png" width="100%">
</div>

### 报警界面

- 下图所示，当发现空飘物后，会有报警弹窗

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img2.png" width="100%">
</div>

- 点击报警弹窗下的”查看详情“按钮，会出现报警细节，在详情界面中，有”实警“和”虚警“按钮，可人为点击，进行判断

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img5.png" width="100%">
</div>

- 若在报警弹窗下点击”稍后处理“，则会在”检测截图“选项卡上对应报警事件中显示未读消息，如下图所示

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img3.png" width="100%">
</div>

- 右击，可处理未读报警事件，处理报警事件的弹窗如下所示

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img4.png" width="100%">
</div>

## 其他功能

- 添加相机信息，分为网络相机和本地相机两部分

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img6.png" width="100%">
</div>

- log日志，保存log信息

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img7.png" width="100%">
</div>

- 检测功能参数

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img8.png" width="100%">
</div>

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img9.png" width="100%">
</div>

<div align=center>
<img src="https://github.com/wk123467/FloatingObject/blob/main/img/img10.png" width="100%">
</div>

