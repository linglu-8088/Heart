import tkinter as tk
import math
import random
import colorsys

class HeartAnimation:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2 + 20  # 稍微向下移动

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("李峋爱心代码 - 优化版")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg='black')
        self.root.resizable(False, False)

        # 创建Canvas
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 粒子列表
        self.particles = []

        # 初始化粒子
        self.init_particles()

        # 动画控制变量
        self.frame = 0
        self.beat_phase = 0
        self.beat_intensity = 0.08  # 心跳强度
        self.beat_speed = 0.12      # 心跳速度
        self.noise_intensity = 1.5  # 抖动强度
        
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
        """初始化粒子 - 优化版本，更接近李峋效果"""
        # 清空现有粒子
        self.particles = []

        # 增加粒子总数到3000个，获得更好的粉尘质感
        total_particles = 3000

        # 1. 核心轮廓粒子 (1000个) - 紧贴心形公式
        for i in range(1000):
            t = random.uniform(0, 2 * math.pi)
            x, y = self.heart_function(t, 10)
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
                'type': 'core',
                'size': random.uniform(1.2, 2.8),
                'color': color,
                'alpha': random.uniform(0.8, 1.0),  # 透明度变化
                'velocity_x': 0,
                'velocity_y': 0
            }
            self.particles.append(particle)

        # 2. 内扩散粒子 (1500个) - 填充心形内部
        for i in range(1500):
            # 使用更复杂的分布算法
            t = random.uniform(0, 2 * math.pi)
            # 使用beta分布控制内部填充密度
            beta_scale = random.betavariate(1.5, 2.5) * 0.8 + 0.2  # 0.2-1.0范围
            x, y = self.heart_function(t, beta_scale * 10)
            x = x + self.center_x
            y = y + self.center_y

            # 使用渐变色增加视觉效果
            hue = random.uniform(0.9, 1.0)  # 红色系
            saturation = random.uniform(0.7, 0.9)
            value = random.uniform(0.6, 0.9)
            color = self.hsv_to_hex(hue, saturation, value)

            particle = {
                'original_x': x,
                'original_y': y,
                'x': x,
                'y': y,
                'type': 'inner',
                'size': random.uniform(0.8, 2.2),
                'color': color,
                'alpha': random.uniform(0.6, 0.9),
                'velocity_x': 0,
                'velocity_y': 0
            }
            self.particles.append(particle)

        # 3. 外光晕粒子 (500个) - 制造发光朦胧感
        for i in range(500):
            t = random.uniform(0, 2 * math.pi)
            # 外晕粒子更稀疏，分布更广
            scale_factor = random.uniform(1.05, 1.7)
            x, y = self.heart_function(t, scale_factor * 10)
            x = x + self.center_x
            y = y + self.center_y

            # 使用渐变色增加视觉效果
            hue = random.uniform(0.9, 1.0)  # 红色系
            saturation = random.uniform(0.6, 0.8)
            value = random.uniform(0.4, 0.8)
            color = self.hsv_to_hex(hue, saturation, value)

            particle = {
                'original_x': x,
                'original_y': y,
                'x': x,
                'y': y,
                'type': 'halo',
                'size': random.uniform(0.5, 1.8),
                'color': color,
                'alpha': random.uniform(0.3, 0.7),
                'velocity_x': 0,
                'velocity_y': 0
            }
            self.particles.append(particle)
    
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

        # 更新每个粒子
        for particle in self.particles:
            # 基于心跳缩放调整位置
            dx = particle['original_x'] - self.center_x
            dy = particle['original_y'] - self.center_y
            scaled_x = self.center_x + dx * self.scale
            scaled_y = self.center_y + dy * self.scale

            # 添加物理运动效果
            particle['velocity_x'] = (scaled_x - particle['x']) * 0.1 + particle['velocity_x'] * 0.9
            particle['velocity_y'] = (scaled_y - particle['y']) * 0.1 + particle['velocity_y'] * 0.9

            particle['x'] += particle['velocity_x']
            particle['y'] += particle['velocity_y']

            # 添加高斯抖动 - 模拟电流滋滋作响的震动感
            # 不同类型粒子有不同的抖动强度
            if particle['type'] == 'core':
                noise_factor = self.noise_intensity * 0.8
            elif particle['type'] == 'inner':
                noise_factor = self.noise_intensity * 1.0
            else:  # halo
                noise_factor = self.noise_intensity * 1.3

            noise_x = random.gauss(0, noise_factor)
            noise_y = random.gauss(0, noise_factor)
            particle['x'] += noise_x
            particle['y'] += noise_y

            # 添加微小的随机闪烁效果
            if random.random() < 0.08:  # 8%概率闪烁
                particle['alpha'] = max(0.2, min(1.0, particle['alpha'] + random.uniform(-0.15, 0.15)))

            # 根据心跳调整粒子大小
            beat_effect = 0.02 * math.sin(self.beat_phase * 2.0)
            particle['size'] = max(0.1, particle['size'] + beat_effect)

        self.frame += 1
    
    def draw_particles(self):
        """绘制所有粒子 - 优化版本"""
        # 清空画布
        self.canvas.delete("all")

        # 绘制每个粒子（按类型排序以获得正确的层次感）
        # 先绘制外晕粒子，再绘制内部粒子，最后绘制核心粒子
        draw_order = ['halo', 'inner', 'core']

        for particle_type in draw_order:
            for particle in self.particles:
                if particle['type'] == particle_type:
                    x, y = particle['x'], particle['y']
                    size = particle['size']

                    # 确保粒子在画布范围内
                    if -size <= x <= self.width + size and -size <= y <= self.height + size:
                        # 绘制粒子（Tkinter不支持真正的alpha透明度，所以直接使用原始颜色）
                        self.canvas.create_oval(
                            x - size, y - size, x + size, y + size,
                            fill=particle['color'], outline='', tags="particle"
                        )
    
    def animate(self):
        """动画循环 - 优化帧率"""
        self.update_particles()
        self.draw_particles()
        self.root.after(20, self.animate)  # 约50FPS，更流畅
    
    def run(self):
        """运行动画"""
        self.animate()
        self.root.mainloop()

# 运行程序
if __name__ == "__main__":
    heart_anim = HeartAnimation()
    heart_anim.run()