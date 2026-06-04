# Heart Python Versions

## 运行环境

- Conda 环境：`D:\Program\Anaconda\envs\gomoku_ai`

## 版本

| 文件 | 说明 |
|------|------|
| `li_xun_heart_v2.py` | **推荐** — 真3D体积心脏，HDR Bloom，电影级相机，支持 500K+ 粒子 |
| `li_xun_heart.py` | 原始版本 — 2D 参数方程挤出伪3D，单通道渲染，~50K 粒子 |

## 运行

```powershell
# v2 增强版（推荐）
python li_xun_heart_v2.py

# v2 自定义粒子数
python li_xun_heart_v2.py --particles 300000

# v2 无 Bloom（轻量模式）
python li_xun_heart_v2.py --no-bloom --no-trails

# 原始版本
python li_xun_heart.py
```

## v2 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--width` | 1280 | 窗口宽度 |
| `--height` | 960 | 窗口高度 |
| `--seed` | 7 | 随机种子 |
| `--particles` | 200000 | 粒子总数（最大 ~500000） |
| `--bloom` / `--no-bloom` | on | HDR Bloom 后处理 |
| `--trails` / `--no-trails` | on | 运动拖尾累积 |
| `--fps` | off | 窗口标题显示 FPS |
| `--self-test` | off | 无窗口自检模式 |

## 控制

- `Esc`：关闭窗口

## v2 特性

- 真实 3D 隐式曲面心脏（Taubin Heart Surface）
- 3D Simplex 噪声调制粒子密度，云雾状有机结构
- 5 层粒子：轮廓 / 壳体 / 填充 / 核心 / 轨道
- HDR 渲染 + 亮度提取 + 高斯模糊 + 合成（5-Pass 管线）
- 温度场颜色系统（核心白粉 → 边缘暗红 → 轨道金橙）
- 三级心跳脉冲 + 冲击波传播
- 电影级多频相机（非重复环绕/推拉/仰俯）
- 8 字形轨道粒子 + 运动拖尾累积
- 深度雾化 + Reinhard 色调映射 + 暗角
