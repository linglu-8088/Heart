# 爱心动画升级实现计划

## 项目目标
将现有的爱心动画升级为：底座悬浮式爱心，带粒子喷射效果，整体更饱满

## 当前状态
- 文件：`li_xun_heart.py`
- 粒子总数：3160个（glow:250, fill:2550, core:360）
- 已有功能：心跳缩放、粒子闪烁、高光效果
- 缺少功能：底座、悬浮、粒子喷射

---

## Phase 1: 底座与悬浮效果（基础）

### 步骤 1.1：添加悬浮动画变量
**目标**：让爱心产生上下浮动效果

**修改位置**：`HeartAnimation.__init__()` 方法（第31-39行）

**添加变量**：
```python
self.float_amplitude = 20.0  # 悬浮幅度（像素）
self.float_speed = 0.8       # 悬浮速度
self.float_offset = 0.0      # 当前悬浮偏移
```

**验收标准**：
- [ ] 变量成功添加到 `__init__` 方法
- [ ] 程序可以正常运行

---

### 步骤 1.2：实现悬浮偏移计算
**目标**：计算每帧的悬浮偏移量

**修改位置**：`update_frame()` 方法（第338行附近）

**添加代码**：
```python
# 在 self.time += 0.045 之后添加
self.float_offset = math.sin(self.time * self.float_speed) * self.float_amplitude
```

**验收标准**：
- [ ] 悬浮偏移计算正确
- [ ] 偏移值在 -20 到 +20 之间

---

### 步骤 1.3：应用悬浮偏移到爱心粒子
**目标**：让所有爱心粒子跟随悬浮偏移移动

**修改位置**：`update_frame()` 方法中的爱心粒子更新部分（第365-409行）

**修改方式**：
在计算 `target_y` 时加上悬浮偏移：
```python
target_y = self.center_y + dy * scale + self.float_offset
```

**验收标准**：
- [ ] 爱心整体上下浮动
- [ ] 浮动平滑自然
- [ ] 心跳效果不受影响

---

### 步骤 1.4：应用悬浮偏移到高光层
**目标**：让高光层跟随爱心一起悬浮

**修改位置**：`update_glow_layers()` 方法（第321-336行）

**修改方式**：
在计算高光位置的 `y` 时加上悬浮偏移：
```python
y = self.center_y + offset_y - pulse_offset + self.float_offset
```

**验收标准**：
- [ ] 高光层跟随爱心悬浮
- [ ] 高光效果不变形

---

### 步骤 1.5：绘制底座
**目标**：在爱心下方绘制一个发光底座

**新增方法**：在 `HeartAnimation` 类中添加

**代码位置**：在 `_draw_background()` 方法之后（第253行附近）

**方法内容**：
```python
def _draw_pedestal(self) -> None:
    """绘制底座"""
    pedestal_y = self.center_y + 280  # 底座Y坐标
    pedestal_radius = 120  # 底座半径
    
    # 外圈发光效果
    for i in range(3):
        r = pedestal_radius + i * 15
        alpha_hex = format(int(30 - i * 10), '02x')
        color = f"#4a7bff{alpha_hex}"
        self.canvas.create_oval(
            self.center_x - r, pedestal_y - r//4,
            self.center_x + r, pedestal_y + r//4,
            fill=color, outline="", tags="pedestal"
        )
    
    # 主体
    self.canvas.create_oval(
        self.center_x - pedestal_radius, pedestal_y - pedestal_radius//4,
        self.center_x + pedestal_radius, pedestal_y + pedestal_radius//4,
        fill="#1a2f4f", outline="#4a7bff", width=2, tags="pedestal"
    )
```

**调用位置**：在 `_build_scene()` 方法中，`_draw_background()` 之后调用

**验收标准**：
- [ ] 底座显示在爱心下方
- [ ] 底座有发光效果
- [ ] 底座不遮挡爱心

---

## Phase 2: 粒子喷射系统

### 步骤 2.1：定义喷射粒子数据结构
**目标**：创建喷射粒子的数据类

**修改位置**：文件顶部，`Particle` 类之后（第26行附近）

**添加代码**：
```python
@dataclass
class FountainParticle:
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    x: float
    y: float
    progress: float  # 0.0 到 1.0
    speed: float
    color: str
    size: float
    item_id: int = 0
```

**验收标准**：
- [ ] 数据类定义正确
- [ ] 可以创建实例

---

### 步骤 2.2：初始化喷射粒子列表
**目标**：添加存储喷射粒子的列表

**修改位置**：`__init__()` 方法（第56-60行附近）

**添加代码**：
```python
self.fountain_particles: list[FountainParticle] = []
self.pedestal_y = self.center_y + 280  # 底座Y坐标
self.fountain_spawn_rate = 2  # 每帧生成的粒子数
```

**验收标准**：
- [ ] 列表初始化成功
- [ ] 底座Y坐标正确

---

### 步骤 2.3：创建喷射粒子生成方法
**目标**：实现从底座生成粒子的逻辑

**新增方法**：在类中添加

**代码位置**：在 `make_spark_particle()` 方法之后（第224行附近）

**方法内容**：
```python
def create_fountain_particle(self) -> FountainParticle:
    """创建一个从底座喷射的粒子"""
    # 起点：底座表面随机位置
    angle = random.uniform(0, math.tau)
    radius = random.uniform(20, 100)
    start_x = self.center_x + math.cos(angle) * radius
    start_y = self.pedestal_y
    
    # 终点：爱心边缘随机位置
    t = random.uniform(0, math.tau)
    px, py = self.heart_point(t, scale=random.uniform(0.9, 1.1))
    target_x = self.center_x + px * 10.5
    target_y = self.center_y + py * 10.5
    
    # 颜色：从蓝色渐变到红色
    color = self.hsv_to_hex(
        random.uniform(0.55, 0.65),  # 蓝紫色调
        random.uniform(0.6, 0.8),
        random.uniform(0.7, 0.9)
    )
    
    return FountainParticle(
        start_x=start_x,
        start_y=start_y,
        target_x=target_x,
        target_y=target_y,
        x=start_x,
        y=start_y,
        progress=0.0,
        speed=random.uniform(0.015, 0.025),
        color=color,
        size=random.uniform(1.0, 2.0)
    )
```

**验收标准**：
- [ ] 方法可以成功创建粒子
- [ ] 起点在底座范围内
- [ ] 终点在爱心范围内

---

### 步骤 2.4：实现喷射粒子更新逻辑
**目标**：更新喷射粒子的位置和状态

**新增方法**：

**代码位置**：在 `create_fountain_particle()` 之后

**方法内容**：
```python
def update_fountain_particles(self) -> None:
    """更新喷射粒子"""
    # 生成新粒子
    for _ in range(self.fountain_spawn_rate):
        particle = self.create_fountain_particle()
        particle.item_id = self.canvas.create_oval(
            0, 0, 0, 0, fill=particle.color, outline=""
        )
        self.fountain_particles.append(particle)
    
    # 更新现有粒子
    to_remove = []
    for particle in self.fountain_particles:
        particle.progress += particle.speed
        
        if particle.progress >= 1.0:
            to_remove.append(particle)
            self.canvas.delete(particle.item_id)
            continue
        
        # 使用缓动函数（ease-out）
        t = particle.progress
        eased_t = 1 - (1 - t) ** 2
        
        # 计算当前位置
        particle.x = particle.start_x + (particle.target_x - particle.start_x) * eased_t
        particle.y = particle.start_y + (particle.target_y - particle.start_y) * eased_t
        
        # 添加悬浮偏移
        particle.y += self.float_offset
        
        # 更新画布
        size = particle.size * (1 - t * 0.5)  # 逐渐变小
        self.canvas.coords(
            particle.item_id,
            particle.x - size, particle.y - size,
            particle.x + size, particle.y + size
        )
    
    # 移除完成的粒子
    for particle in to_remove:
        self.fountain_particles.remove(particle)
```

**验收标准**：
- [ ] 粒子从底座向爱心移动
- [ ] 粒子到达后消失
- [ ] 粒子轨迹平滑

---

### 步骤 2.5：集成喷射粒子到主循环
**目标**：在每帧更新中调用喷射粒子更新

**修改位置**：`update_frame()` 方法（第338行附近）

**添加调用**：
在更新火花粒子之后（第411-426行之后）添加：
```python
# 更新喷射粒子
self.update_fountain_particles()
```

**验收标准**：
- [ ] 喷射粒子持续生成
- [ ] 动画流畅
- [ ] 无性能问题

---

## Phase 3: 爱心饱满度优化

### 步骤 3.1：增加粒子数量
**目标**：让爱心看起来更饱满

**修改位置**：`_build_scene()` 方法（第287-304行）

**修改数量**：
```python
for _ in range(400):  # 原250
    self.heart_particles.append(self.make_heart_particle("glow"))
for _ in range(3500):  # 原2550
    self.heart_particles.append(self.make_heart_particle("fill"))
for _ in range(500):  # 原360
    self.heart_particles.append(self.make_heart_particle("core"))
```

**验收标准**：
- [ ] 爱心更饱满
- [ ] 帧率保持流畅（>30fps）

---

### 步骤 3.2：调整粒子大小
**目标**：增大粒子尺寸

**修改位置**：`make_heart_particle()` 方法（第125-171行）

**修改范围**：
- core: `base_size = random.uniform(1.2, 2.2)`（原1.0-1.8）
- fill: `base_size = random.uniform(1.0, 1.6)`（原0.8-1.35）
- glow: `base_size = random.uniform(1.2, 2.4)`（原0.9-1.8）

**验收标准**：
- [ ] 粒子更大更明显
- [ ] 不会过度重叠

---

### 步骤 3.3：增强颜色饱和度
**目标**：让爱心颜色更鲜艳

**修改位置**：`make_heart_particle()` 方法中的 core 部分（第137-144行）

**修改参数**：
```python
color = self.hsv_to_hex(
    random.uniform(0.96, 1.0),
    random.uniform(0.6, 0.85),  # 原0.45-0.75
    random.uniform(0.95, 1.0)   # 原0.92-1.0
)
```

**验收标准**：
- [ ] 爱心颜色更鲜艳
- [ ] 核心区域更亮

---

## 实施顺序

1. **先做 Phase 1**（底座与悬浮）- 建立基础
2. **再做 Phase 2**（粒子喷射）- 添加动态效果
3. **最后 Phase 3**（优化）- 提升视觉质量

## 测试检查点

每完成一个 Phase，运行程序检查：
- [ ] 程序无报错
- [ ] 动画流畅（FPS > 30）
- [ ] 视觉效果符合预期
- [ ] 所有交互功能正常（Esc、Space、R、鼠标点击）

## 注意事项

1. **性能**：增加粒子后注意帧率，如果卡顿需要减少数量
2. **颜色协调**：底座颜色应与爱心协调
3. **悬浮幅度**：不要太大，保持优雅
4. **粒子生成率**：根据性能调整 `fountain_spawn_rate`

---

## 完成标志

- [ ] 爱心在底座上方悬浮
- [ ] 底座持续向爱心喷射粒子
- [ ] 爱心饱满、颜色鲜艳
- [ ] 心跳效果保持正常
- [ ] 整体动画流畅自然
