import tkinter as tk
import math
import random
import colorsys

class HeartAnimation:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2  # 重新居中

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("李峋爱心代码 - 极细粒子优化版")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg='black')
        self.root.resizable(False, False)

        # 创建Canvas
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 粒子列表
        self.particles = []
        self.halo_particles = []  # 独立的底部光晕粒子

        # 初始化粒子
        self.init_particles()

        # 动画控制变量
        self.frame = 0
        self.beat_phase = 0
        self.beat_intensity = 0.09  # 心跳强度
        self.beat_speed = 0.13      # 心跳速度
        self.noise_intensity = 1.8  # 抖动强度（增强电子噪点效果）
        
    def heart_function(self, t, scale=10):
        """心形参数方程 - 李峋版本"""
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        return x * scale, y * scale

    def hsv_to_hex(self, h, s, v):
        """将HSV颜色值转换为十六进制颜色代码"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def init_particles(self):
        """初始化粒子 - 极细粒子优化版本"""
        # 清空现有粒子
        self.particles = []
        self.halo_particles = []

        # 1. 使用"随机点填充算法"填满整个心形区域（解决空心网格感）
        # 粒子总数保持5000个，获得更好的粉尘质感
        for i in range(5000):
            # 使用优化的反向对数分布填充算法，避免中心结块
            t = random.uniform(0, 2 * math.pi)
            
            # 优化的反向对数分布：ratio = -k * log(random())
            # 调整参数让粒子更均匀地散布在心形轮廓和中心点之间的区域
            # 使用更小的k值(0.1)来减少中心堆积，让分布更均匀
            ratio = -0.1 * math.log(random.uniform(0.001, 1.0))
            ratio = min(ratio, 1.0)  # 限制最大值
            
            x, y = self.heart_function(t, ratio * 10)
            x = x + self.center_x
            y = y + self.center_y

            # 使用渐变色增加视觉效果
            hue = random.uniform(0.9, 1.0)  # 红色系
            saturation = random.uniform(0.8, 1.0)
            value = random.uniform(0.7, 1.0)
            color = self.hsv_to_hex(hue, saturation, value)

            particle = {
                'original_x': x,
                'original_y': y,
                'x': x,
                'y': y,
                'type': 'filled',
                'size': random.uniform(1, 1.5),  # 粒子大小严格限制在1-1.5px之间，实现微粒化
                'color': color,
                'alpha': random.uniform(0.6, 1.0),  # 透明度变化
                'velocity_x': 0,
                'velocity_y': 0
            }
            self.particles.append(particle)

        # 2. 创建独立的底部光晕粒子（修复底部蓝色光环的位置与形态）
        # 减少粒子数量到400个，使粒子稍微稀疏一点
        for i in range(400):
            # 椭圆分布算法
            angle = random.uniform(0, 2 * math.pi)
            # 极薄椭圆盘效果：大幅拉大X轴范围，极度压缩Y轴范围
            radius_x = random.uniform(150, 250)  # X轴扩散更大，范围150-250
            radius_y = random.uniform(5, 15)     # Y轴极度压缩，范围5-15（极薄）
            
            # 下移：位于爱心最尖端的下方
            x = self.center_x + math.cos(angle) * radius_x
            y = self.center_y + 200 + math.sin(angle) * radius_y  # 下移200像素
            
            # 使用青蓝色系
            hue = random.uniform(0.5, 0.6)  # 青蓝色系
            saturation = random.uniform(0.7, 1.0)
            value = random.uniform(0.6, 0.9)
            color = self.hsv_to_hex(hue, saturation, value)

            halo_particle = {
                'original_x': x,
                'original_y': y,
                'x': x,
                'y': y,
                'size': random.uniform(1, 1.5),  # 粒子大小也限制在1-1.5px
                'color': color,
                'alpha': random.uniform(0.3, 0.7),  # 降低透明度范围，制造朦胧感
                'velocity_x': 0,
                'velocity_y': 0,
                # 添加随机扩散范围，制造虚化效果
                'spread_radius': random.uniform(0, 20)
            }
            self.halo_particles.append(halo_particle)
    
    def update_particles(self):
        """更新粒子位置和效果 - 优化版本"""
        # 心脏跳动效果 - 更接近李峋的"收缩-扩张"节奏
        self.beat_phase += self.beat_speed

        # 使用复合正弦波模拟真实心跳（快吸慢呼）
        # 快速收缩，缓慢扩张
        base_pulse = math.sin(self.beat_phase)
        fast_pulse = math.sin(self.beat_phase * 3.0) * 0.3
        self.scale = 1.0 + self.beat_intensity * (base_pulse + fast_pulse)

        # 添加呼吸效果 - 模拟真实心跳的不规则性
        breath_effect = 0.01 * math.sin(self.beat_phase * 0.3)
        self.scale += breath_effect

        # 更新心形粒子
        for particle in self.particles:
            # 基于心跳缩放调整位置
            dx = particle['original_x'] - self.center_x
            dy = particle['original_y'] - self.center_y
            scaled_x = self.center_x + dx * self.scale
            scaled_y = self.center_y + dy * self.scale

            # 添加物理运动效果
            particle['velocity_x'] = (scaled_x - particle['x']) * 0.12 + particle['velocity_x'] * 0.88
            particle['velocity_y'] = (scaled_y - particle['y']) * 0.12 + particle['velocity_y'] * 0.88

            particle['x'] += particle['velocity_x']
            particle['y'] += particle['velocity_y']

            # 添加高斯抖动 - 模拟电流滋滋作响的震动感（增强电子噪点质感）
            noise_x = random.uniform(-1, 1)
            noise_y = random.uniform(-1, 1)
            particle['x'] += noise_x
            particle['y'] += noise_y

            # 添加微小的随机闪烁效果
            if random.random() < 0.15:  # 15%概率闪烁
                particle['alpha'] = max(0.2, min(1.0, particle['alpha'] + random.uniform(-0.2, 0.2)))

            # 根据心跳调整粒子大小
            beat_effect = 0.03 * math.sin(self.beat_phase * 2.0)
            particle['size'] = max(0.1, particle['size'] + beat_effect)

        # 更新底部光晕粒子（静态或微动背景底座）
        for particle in self.halo_particles:
            # 微小的随机移动
            particle['x'] += random.uniform(-0.5, 0.5)
            particle['y'] += random.uniform(-0.3, 0.3)
            
            # 保持在一定范围内
            dx = particle['x'] - particle['original_x']
            dy = particle['y'] - particle['original_y']
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > particle['spread_radius']:
                # 如果超出范围，向初始位置靠拢
                particle['x'] -= dx * 0.05
                particle['y'] -= dy * 0.05

        self.frame += 1
    
    def draw_particles(self):
        """绘制所有粒子 - 优化版本"""
        # 清空画布
        self.canvas.delete("all")

        # 先绘制底部光晕粒子
        for particle in self.halo_particles:
            x, y = particle['x'], particle['y']
            size = particle['size']

            # 确保粒子在画布范围内
            if -size <= x <= self.width + size and -size <= y <= self.height + size:
                # 绘制粒子
                self.canvas.create_oval(
                    x - size, y - size, x + size, y + size,
                    fill=particle['color'], outline='', tags="halo_particle"
                )

        # 再绘制心形粒子
        for particle in self.particles:
            x, y = particle['x'], particle['y']
            size = particle['size']

            # 确保粒子在画布范围内
            if -size <= x <= self.width + size and -size <= y <= self.height + size:
                # 绘制粒子
                self.canvas.create_oval(
                    x - size, y - size, x + size, y + size,
                    fill=particle['color'], outline='', tags="heart_particle"
                )
    
    def animate(self):
        """动画循环 - 优化帧率"""
        self.update_particles()
        self.draw_particles()
        self.root.after(16, self.animate)  # 约60FPS，更流畅
    
    def run(self):
        """运行动画"""
        self.animate()
        self.root.mainloop()

# 运行程序
if __name__ == "__main__":
    heart_anim = HeartAnimation()
    heart_anim.run()