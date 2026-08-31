#!/usr/bin/env python3
"""
YOLOv5 缺陷检测 ROS2 节点 (Task 26, 71)
========================================
订阅无人机相机图像 → YOLOv5 GPU推理 → 发布检测结果

发布话题:
  /vision/detection  (DefectDetectionArray) - 检测到的缺陷列表
  /vision/annotated   (Image)               - 标注后图像(可选)

参数:
  model_path:      模型权重路径
  conf_threshold:  置信度阈值 (默认0.25)
  device:          推理设备 (cuda/cpu)
  image_topic:     订阅的相机话题
"""
import sys
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# YOLOv5路径
YOLOV5_ROOT = '/home/developer/yolov5'
if os.path.isdir(YOLOV5_ROOT):
    sys.path.insert(0, YOLOV5_ROOT)

import torch
_old_load = torch.load
torch.load = lambda *a, **kw: _old_load(*a, **{'weights_only': False, **kw})

from wind_turbine_interfaces.msg import DefectDetection, DefectDetectionArray

CLASS_NAMES = [
    'craze',           # 龟裂
    'corrosion',       # 腐蚀
    'surface_injure',  # 表面损伤
    'thunderstrike',   # 雷击损伤
    'crack',           # 裂纹
    'hide_craze',      # 隐蔽龟裂
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]


class YoloDetector(Node):
    """YOLOv5 风机叶片缺陷检测节点"""

    def __init__(self):
        super().__init__('yolo_detector')

        # 参数
        self.declare_parameter('model_path',
                               '/home/developer/yolov5/runs/train/wt_blade4/weights/best.pt')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('img_size', [640, 640])
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('image_topic', '/drone/camera/image_raw')
        self.declare_parameter('publish_annotated', False)
        # V9.2 相机帧（检测消息 header.frame_id 与目标点链一致）
        self.declare_parameter('camera_frame', 'ar0234_camera_optical_frame')

        self.bridge = CvBridge()
        self.frame_count = 0
        self.inference_times = []
        self.loaded = False

        # 订阅相机
        img_topic = self.get_parameter('image_topic').value
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, img_topic, self._on_image, qos)

        # 发布检测结果
        self.det_pub = self.create_publisher(
            DefectDetectionArray, '/vision/detection', 10)

        if self.get_parameter('publish_annotated').value:
            self.ann_pub = self.create_publisher(
                Image, '/vision/annotated', 10)
        else:
            self.ann_pub = None

        # 加载模型
        self._load_model()

    def _load_model(self):
        model_path = self.get_parameter('model_path').value
        device_str = self.get_parameter('device').value

        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if device_str == 'cuda':
                self.get_logger().warn('CUDA不可用，回退CPU')

        self.conf_thres = self.get_parameter('conf_threshold').value
        self.iou_thres = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('img_size').value

        if not os.path.exists(model_path):
            self.get_logger().error(f'模型不存在: {model_path}')
            return

        try:
            from models.common import DetectMultiBackend
            self.model = DetectMultiBackend(
                model_path, device=self.device, fp16=False)
            self.stride = self.model.stride
            self.model.warmup(imgsz=(1, 3, *self.imgsz))
            self.loaded = True
            self.get_logger().info(
                f'YOLOv5 加载成功 | 设备: {self.device} | '
                f'置信度: {self.conf_thres} | 订阅: {self.get_parameter("image_topic").value}')
        except Exception as e:
            self.get_logger().error(f'模型加载失败: {e}')

    def _on_image(self, msg: Image):
        if not self.loaded:
            return

        self.frame_count += 1
        try:
            im0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            return

        h0, w0 = im0.shape[:2]
        t0 = time.time()

        # 预处理
        r = min(self.imgsz[0] / h0, self.imgsz[1] / w0)
        new_h, new_w = int(h0 * r), int(w0 * r)
        dh, dw = self.imgsz[0] - new_h, self.imgsz[1] - new_w
        top, left = dh // 2, dw // 2

        im = cv2.resize(im0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, top, dh - top, left, dw - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device).float() / 255.0
        im = im.unsqueeze(0)

        # 推理
        pred = self.model(im)
        from utils.general import non_max_suppression
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=100)
        det = pred[0]

        t_infer = (time.time() - t0) * 1000

        # 缩放回原图
        if len(det):
            gain_w = new_w / w0
            gain_h = new_h / h0
            det[:, [0, 2]] = (det[:, [0, 2]] - left) / gain_w
            det[:, [1, 3]] = (det[:, [1, 3]] - top) / gain_h
            det[:, [0, 2]] = det[:, [0, 2]].clamp(0, w0)
            det[:, [1, 3]] = det[:, [1, 3]].clamp(0, h0)

        # 发布
        self._publish(det, msg.header)

        # 标注图
        if self.ann_pub is not None:
            annotated = self._draw(im0, det)
            ros_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ros_img.header = msg.header
            self.ann_pub.publish(ros_img)

        # 性能日志
        self.inference_times.append(t_infer)
        if self.frame_count % 200 == 0:
            avg = np.mean(self.inference_times[-200:])
            self.get_logger().info(
                f'Frame {self.frame_count} | {avg:.1f}ms | '
                f'{1000/avg:.0f}FPS | 检测到 {len(det)} 个目标')

    def _publish(self, det, header):
        msg = DefectDetectionArray()
        msg.header = header
        msg.header.frame_id = self.get_parameter('camera_frame').value

        if det is not None and len(det):
            for *xyxy, conf, cls in det.cpu().numpy():
                d = DefectDetection()
                d.header = header
                d.header.frame_id = self.get_parameter('camera_frame').value
                d.class_id = int(cls)
                d.class_name = CLASS_NAMES[int(cls)] if int(cls) < 6 else 'unknown'
                d.confidence = float(conf)
                d.bbox_x_min = float(xyxy[0])
                d.bbox_y_min = float(xyxy[1])
                d.bbox_x_max = float(xyxy[2])
                d.bbox_y_max = float(xyxy[3])
                d.has_3d = False
                msg.detections.append(d)

        self.det_pub.publish(msg)

    def _draw(self, im0, det):
        im = im0.copy()
        if det is not None and len(det):
            for *xyxy, conf, cls in det.cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                cid = int(cls)
                color = COLORS[cid % len(COLORS)]
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                label = f'{CLASS_NAMES[cid]} {conf:.2f}'
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 2)
                cv2.rectangle(im, (x1, y1-th-4), (x1+tw, y1), color, -1)
                cv2.putText(im, label, (x1, y1-2), FONT, 0.5,
                            (255, 255, 255), 1)
        return im


def main():
    rclpy.init()
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
