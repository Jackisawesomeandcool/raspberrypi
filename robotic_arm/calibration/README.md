# 机械臂标定工具

本目录提供相机、工作平面和关节标定工具：

| 脚本 | 职责 |
| --- | --- |
| `calibrate_camera.py` | 计算相机内参与畸变参数 |
| `calibrate_plane.py` | 建立图像像素到 J2–J4 侧视运动平面的映射 |
| `calibrate_joints.py` | 从静态图片测量 J2–J4，并拟合控制角到实际关节角的映射 |
| `calibrate_workspace.py` | 建立图像像素到 J1 基座工作平面的映射 |
| `auto_calibrate_joints.py` | 按配置采集关节样本并拟合静态映射 |

`generate_checkerboards.py` 只用于重新生成 `checkerboards/` 中的打印文件。

## 使用边界

- J1 手动标定，不进入侧视相机流程。
- 正式上位机载荷固定为 `J1,J2,J3,J4,J6`；J2 是逻辑肩关节字段。实体映射应以“舵机 2/配对舵机”描述，不得写成“J2 镜像”。
- J5 固定为 `q5=0`，不得下发运动命令。
- 会移动机械臂的脚本默认只预览，只有显式传入 `--send` 才发送命令。

## 环境

```bash
conda run -n codex python -c "import cv2; print(cv2.__version__)"
```

## 1. 打印棋盘格

| 文件 | 方格 | OpenCV 内角点 |
| --- | --- | --- |
| `checkerboards/a4_8x6_25mm.svg` | 8×6 | 7×5 |
| `checkerboards/a3_12x8_25mm.svg` | 12×8 | 11×7 |
| `checkerboards/a2_16x12_25mm.svg` | 16×12 | 15×11 |

优先打印 SVG，选择横向、`100%` 或“实际大小”，关闭自动缩放。打印后跨越多个方格测量平均边长，后续 `--square-mm` 使用实测值。

A2 无法整张打印时，使用 `checkerboards/a2_tiles/` 中的四张 A4。沿裁切标记裁剪，在平整硬板上对接并从背面固定。

重新生成全部文件：

```bash
conda run -n codex python calibration/generate_checkerboards.py
```

## 2. 相机内参

使用已有图片：

```bash
conda run -n codex python calibration/calibrate_camera.py \
  --images "calibration/data/intrinsics/*.png" \
  --square-mm 25.0
```

也可以通过 `--camera-index 0` 直接采集。默认按 A4 棋盘的 7×5 内角点处理；使用 A3 或 A2 时，分别传入：

```text
A3: --pattern-cols 11 --pattern-rows 7
A2: --pattern-cols 15 --pattern-rows 11
```

默认输出：`calibration/output/camera_intrinsics.json`

## 3. 侧视平面

棋盘格应与连杆标记位于同一侧视运动平面：

```bash
conda run -n codex python calibration/calibrate_plane.py \
  --image calibration/data/side_plane.png \
  --intrinsics calibration/output/camera_intrinsics.json \
  --square-mm 25.0
```

默认输出：

```text
calibration/output/side_plane.json
calibration/output/side_plane_annotated.png
```

完成后可以移走棋盘格，但不能移动相机。

## 4. J2–J4 角度修正

复制并填写 `joint_samples.example.csv`。每张图片必须在机械臂完全静止后拍摄，一次只改变 J2、J3 或 J4 中的一个逻辑关节。

`yellow_tape.example.json` 中的 HSV 范围只是示例，应根据实际画面修改。

```bash
conda run -n codex python calibration/calibrate_joints.py \
  --samples calibration/data/joint_samples.csv \
  --intrinsics calibration/output/camera_intrinsics.json \
  --side-plane calibration/output/side_plane.json \
  --marker-config calibration/yellow_tape.example.json
```

默认输出：

```text
calibration/output/joint_calibration.json
calibration/output/joint_samples_measured.csv
calibration/output/joint_calibration_plot.png
```

拟合关系为：

```text
q_actual_deg = slope * logical_command_deg + intercept
```

更换镜头、分辨率或图像裁切方式后需要重新标定内参；移动固定相机后需要重新标定侧视平面。
