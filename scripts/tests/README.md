# V9.2 视觉链离线测试

PM 交付物⑦：5 项离线测试。**纯 numpy + 标准库 unittest，无需 ROS 环境**，
在任意装有 numpy 的机器（含 Windows 开发机）可直接运行。

```bash
python3 tests/test_v9_2_localization.py        # 汇总输出
python3 tests/test_v9_2_localization.py -v     # 逐项详细输出
```

当前测试输出留存于 `tests/OFFLINE_TEST_OUTPUT.txt`（24 项用例全部 OK，2026-08-31）。

## 覆盖的 5 项（对应测试类）

| # | 测试类 | 内容 |
|---|--------|------|
| 1 | `TestLinkToOptical` | ar0234_camera_link(FLU: X前/Y左/Z上) → optical(RDF: Z前/X右/Y下) 轴映射；与 V9.2 静态 TF `rpy(-π/2, 0, -π/2)` 旋转矩阵的**等价性断言**；正交性；往返变换 |
| 2 | `TestProjection` | CameraInfo 内参 K 投影：已知点、光轴点、相机后方/退化点返回 (-1,-1)、反投影往返一致 |
| 3 | `TestLidarDepth` | 点云深度查询：墙面点云取中位数；离群点稳健（±50% 窗口）；空区域/空点云/NaN 点返回 None |
| 4 | `TestTimeSkew` | `stamps_within_skew` 0.5s 边界（恰好 0.5 接受 / 超 1ns 拒绝 / 负方向 / nanosec 进位）；零时间戳与 None 拒绝 |
| 5 | `TestPC2Parse` | PointCloud2 解析：字段乱序按 offset 读取、gz 12 字节紧凑布局、非对齐慢路径、不支持类型/缺字段抛错、空点云 |
