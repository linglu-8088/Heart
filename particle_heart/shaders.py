QUAD_VS = """#version 330
const vec2 POS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0)
);
out vec2 v_uv;
void main() {
    vec2 p = POS[gl_VertexID];
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv = p * 0.5 + 0.5;
}
"""

LEGACY_PEDESTAL_VS_UNUSED = """#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;
uniform float u_beat;
uniform float u_point_scale;
uniform vec3 u_ped_tint[4];

in vec3 in_position;
in float in_size;
in float in_phase;
in float in_layer;
in float in_alpha;

out vec3 v_color;
out float v_alpha;
out float v_seed;
out float v_layer;

float saturate(float v) { return clamp(v, 0.0, 1.0); }

void main() {
    vec3 pos = in_position;
    float pulse = u_beat;
    int layer = int(in_layer + 0.5);

    float layer_norm = in_layer / 3.0;
    float layer_pulse_xy = mix(0.028, 0.004, layer_norm);
    float layer_pulse_z = mix(0.036, 0.005, layer_norm);
    pos.xy *= 1.0 + pulse * layer_pulse_xy;
    pos.z *= 1.0 + pulse * layer_pulse_z;

    float breath_speed = (layer == 0) ? 0.56 : (layer == 1) ? 0.68 : (layer == 2) ? 0.44 : 0.30;
    float breath_amp = (layer == 0) ? 0.0030 : (layer == 1) ? 0.0032 : (layer == 2) ? 0.0016 : 0.0008;
    pos.y += sin(u_time * breath_speed + in_phase * 2.1) * breath_amp;

    float flow = sin(u_time * 0.18 + in_phase) * 0.0012;
    float ca = cos(flow), sa = sin(flow);
    pos.x = in_position.x * ca - in_position.z * sa;
    pos.z = in_position.x * sa + in_position.z * ca;

    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float depth = -view_pos.z;
    float frontness = saturate(pos.z * 1.15 + 0.5);
    float depth_scale = 1.55 / max(2.0, depth);
    float pulse_size = 1.0 + pulse * 0.018;
    float layer_size = (layer == 0) ? 1.20 : (layer == 1) ? 1.12 : (layer == 2) ? 1.00 : 0.86;
    float front_size = mix(0.94, 1.08, frontness);
    gl_PointSize = clamp(in_size * u_point_scale * depth_scale * pulse_size * layer_size * front_size, 1.8, 42.0);

    vec3 base_color = u_ped_tint[layer];
    float shimmer = 0.5 + 0.5 * sin(u_time * 0.55 + in_phase * 0.9);
    float shim_light = mix(0.985, 1.015, shimmer);
    vec3 depth_tint = mix(vec3(0.70, 0.76, 0.97), vec3(1.08, 0.97, 0.87), frontness);
    float depth_light = saturate(1.18 - 0.08 * abs(depth));

    float center_glow = 1.0;
    if (layer == 0) {
        float cd = min(1.0, length(in_position.xz) / 0.38);
        center_glow = mix(1.10, 1.0, cd);
    }

    v_color = base_color * depth_tint * depth_light * shim_light * center_glow;
    v_alpha = in_alpha * mix(0.94, 1.06, frontness) * mix(0.98, 1.02, shimmer);
    v_seed = in_phase;
    v_layer = in_layer;
}
"""

LEGACY_PEDESTAL_FS_UNUSED = """#version 330
in vec3 v_color;
in float v_alpha;
in float v_seed;
in float v_layer;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float radius = sqrt(r2);
    float angle = atan(uv.y, uv.x);

    float wobble = 0.95
        + 0.02 * sin(angle * 5.0 + v_seed * 1.6)
        + 0.01 * sin(angle * 9.0 - v_seed * 1.1)
        + 0.005 * sin(angle * 13.0 + v_seed * 2.4);
    float particle_mask = 1.0 - smoothstep(wobble - 0.05, wobble, radius);

    float dense = exp(-r2 * 5.0);
    float soft = exp(-r2 * 2.2);
    float core = exp(-r2 * 10.0);

    float alpha_shape;
    float glow;
    int layer = int(v_layer + 0.5);

    if (layer == 0) {
        alpha_shape = mix(0.74, 1.0, dense) * pow(1.0 - radius, 1.15);
        glow = 0.98 + core * 0.18;
    } else if (layer == 1) {
        alpha_shape = mix(0.70, 1.0, dense) * pow(1.0 - radius, 1.0);
        glow = 0.96 + core * 0.22;
    } else if (layer == 2) {
        alpha_shape = mix(0.58, 1.0, soft) * pow(1.0 - radius, 1.45);
        glow = 0.90 + core * 0.14;
    } else {
        alpha_shape = mix(0.26, 1.0, soft) * pow(1.0 - radius, 2.1);
        glow = 0.78 + core * 0.08;
    }

    alpha_shape *= particle_mask;

    vec3 hdr = v_color * glow;
    fragColor = vec4(hdr, v_alpha * alpha_shape);
}
"""

PARTICLE_VS = """#version 330

uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;
uniform float u_beat;
uniform float u_point_scale;
uniform float u_shockwave;
uniform vec3 u_theme_tint[5];
uniform vec3 u_light_dir;

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_phase;
in float in_kind;
in float in_radial;
in float in_alpha;
in float in_speed;

out vec3 v_color;
out float v_alpha;
out float v_brightness;
out float v_kind;
out float v_core_mix;
out float v_seed;

float saturate(float v) { return clamp(v, 0.0, 1.0); }

vec3 rot_y(vec3 v, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
}
vec3 rot_x(vec3 v, float a) {
    float c = cos(a), s = sin(a);
    return vec3(v.x, c*v.y - s*v.z, s*v.y + c*v.z);
}

void main() {
    vec3 pos = in_position;

    if (in_kind < 3.5) {
        pos.y += 0.030 * sin(u_time * 1.20 + in_phase * 0.45);
        pos.x += 0.012 * sin(u_time * 1.50 + in_phase * 0.35 + 0.8);
        pos.z += 0.010 * cos(u_time * 0.90 + in_phase * 0.50);
        pos.y += 0.015 * sin(u_time * 0.65 + in_phase * 0.30);
    }

    float edge_lock = smoothstep(0.78, 1.0, in_radial);
    float core_mix = 1.0 - smoothstep(0.25, 0.95, in_radial);
    float pulse = u_beat;
    float shock = u_shockwave;
    float shimmer = 0.5 + 0.5 * sin(u_time * (1.2 + in_speed * 0.35) + in_phase);

    if (in_kind < 0.5) {
        pos.xy *= 1.0 + pulse * 0.032;
        pos.z += 0.008 * sin(u_time * 0.75 + in_phase * 0.7);
    } else if (in_kind < 1.5) {
        pos.xy *= 1.0 + pulse * 0.038;
        pos.z  *= 1.0 + pulse * 0.045;
    } else if (in_kind < 2.5) {
        pos.xy *= 1.0 + pulse * (0.045 + 0.025 * core_mix);
        pos.z  *= 1.0 + pulse * 0.055;
        pos    *= 1.0 + shock * 0.005 * (1.0 - abs(in_radial - 0.5) * 2.0);
    } else if (in_kind < 3.5) {
        pos.xy *= 1.0 + pulse * (0.055 + 0.035 * core_mix);
        pos.z  *= 1.0 + pulse * 0.065;
        pos    *= 1.0 + shock * 0.008 * core_mix;
    } else if (in_kind < 4.5) {
        float orbit = u_time * (0.85 + in_speed * 0.40) + in_phase;
        float c = cos(orbit), s = sin(orbit);
        float denom = 1.0 + s * s;
        float blend = 0.5 + 0.5 * sin(in_phase * 2.7);
        vec3 fig8 = vec3(
            in_position.x + 0.10 * c / denom,
            in_position.y + 0.10 * s * c / denom * 0.7,
            in_position.z + 0.08 * s / denom
        );
        vec3 circ = vec3(
            in_position.x + 0.04 * cos(orbit),
            in_position.y + 0.02 * sin(orbit * 1.2),
            in_position.z + 0.04 * sin(orbit)
        );
        pos = mix(circ, fig8, blend);
        pos *= 1.0 + pulse * 0.030;
        pos.x += 0.006 * sin(u_time * 3.5 + in_phase * 5.0);
        pos.y += 0.006 * cos(u_time * 4.0 + in_phase * 4.5);
        pos.z += 0.005 * sin(u_time * 3.0 + in_phase * 5.5);
    } else {
        float drift = u_time * (0.45 + in_speed * 0.30) + in_phase;
        float cx = cos(drift);
        float sx = sin(drift);
        pos.x = in_position.x + 0.05 * cx;
        pos.y = in_position.y + 0.04 * sx * cx * 0.6;
        pos.z = in_position.z + 0.03 * sx;
        pos *= 1.0 + pulse * 0.025;
        pos.x += 0.004 * sin(u_time * 5.0 + in_phase * 7.0);
        pos.y += 0.004 * cos(u_time * 4.5 + in_phase * 6.0);
    }

    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float depth = -view_pos.z;
    float frontness = saturate(pos.z * 1.15 + 0.5);
    float depth_scale = 1.55 / max(2.0, depth);
    float pulse_size = 1.0 + pulse * 0.06;
    float front_size = mix(0.84, 1.22, frontness);
    gl_PointSize = clamp(in_size * u_point_scale * depth_scale * pulse_size * front_size, 1.2, 35.0);

    float depth_light = saturate(1.34 - 0.12 * abs(depth));
    float sil_light = mix(0.90, 0.96, edge_lock);
    float shim_light = mix(0.95, 1.04, shimmer);
    vec3 depth_tint = mix(vec3(0.65, 0.72, 0.98), vec3(1.12, 1.00, 0.90), frontness);
    if (in_kind > 3.5) sil_light *= 1.03;
    float dir_light = dot(normalize(in_position), u_light_dir) * 0.5 + 0.5;
    dir_light = mix(0.64, 1.0, dir_light);
    v_color = in_color * depth_tint * depth_light * sil_light * shim_light * dir_light
              * u_theme_tint[int(in_kind)];

    v_brightness = 1.06 + core_mix * (0.62 + u_beat * 0.42);

    float fog = exp(-depth * 0.08);
    v_alpha = in_alpha * fog * mix(0.65, 1.25, frontness) * mix(0.90, 1.12, shimmer);
    v_kind = in_kind;
    v_core_mix = core_mix;
    v_seed = in_phase;
}
"""

PARTICLE_FS = """#version 330

in vec3 v_color;
in float v_alpha;
in float v_brightness;
in float v_kind;
in float v_core_mix;
in float v_seed;

out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float radius = sqrt(r2);
    float angle = atan(uv.y, uv.x);
    float wobble = 0.86
        + 0.06 * sin(angle * 5.0 + v_seed * 1.9)
        + 0.03 * sin(angle * 9.0 - v_seed * 1.3)
        + 0.02 * sin(angle * 13.0 + v_seed * 2.7);
    float particle_mask = 1.0 - smoothstep(wobble - 0.11, wobble, radius);
    float dense = exp(-r2 * 7.0);
    float soft = exp(-r2 * 3.0);
    float halo = exp(-r2 * 2.0);

    float alpha_shape = particle_mask * mix(0.62, 1.0, dense);
    float glow = mix(0.95, 1.06, halo);
    float soft_falloff = 1.0;

    if (v_kind < 1.5) {
        alpha_shape *= mix(0.84, 1.0, dense);
        glow = mix(0.96, 1.03, halo);
        soft_falloff = pow(1.0 - radius, 1.3);
    } else if (v_kind < 3.5) {
        alpha_shape *= mix(0.86, 1.0, soft);
        glow = mix(0.94, 1.04, halo);
        soft_falloff = pow(1.0 - radius, 1.8);
    } else if (v_kind < 4.5) {
        alpha_shape *= mix(0.82, 1.0, soft);
        glow = mix(0.95, 1.10, halo);
        soft_falloff = pow(1.0 - radius, 1.3);
    } else {
        alpha_shape *= mix(0.76, 1.0, soft);
        glow = mix(0.98, 1.10, halo);
        soft_falloff = pow(1.0 - radius, 1.1);
    }
    alpha_shape *= soft_falloff;
    glow *= mix(1.0, 1.06, v_core_mix);
    float glow_var = 0.85 + 0.30 * sin(v_seed * 7.3);
    glow *= glow_var;

    float inner_glow = exp(-r2 * 12.0);
    float outer_glow = exp(-r2 * 1.5);
    vec3 inner_color = vec3(1.0, 0.70, 0.30);
    vec3 outer_color = vec3(0.70, 0.00, 0.20);

    vec3 hdr = v_color * glow * v_brightness
             + inner_color * inner_glow * 0.22
             + outer_color * outer_glow * 0.045;
    fragColor = vec4(hdr, v_alpha * alpha_shape);
}
"""

BRIGHT_PASS_FS = """#version 330
uniform sampler2D u_scene_tex;
uniform float u_threshold;
uniform float u_intensity;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 color = texture(u_scene_tex, v_uv).rgb;
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float contrib = smoothstep(u_threshold - 0.15, u_threshold, lum);
    fragColor = vec4(color * contrib * u_intensity, 1.0);
}
"""

BLUR_H_FS = """#version 330
uniform sampler2D u_input_tex;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 fragColor;

const float W[9] = float[9](
    0.0136, 0.0476, 0.1172, 0.2028, 0.2378, 0.2028, 0.1172, 0.0476, 0.0136
);

void main() {
    vec3 result = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        vec2 uv = v_uv + vec2(u_texel_size.x * float(i - 4), 0.0);
        result += texture(u_input_tex, uv).rgb * W[i];
    }
    fragColor = vec4(result, 1.0);
}
"""

BLUR_V_FS = """#version 330
uniform sampler2D u_input_tex;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 fragColor;

const float W[9] = float[9](
    0.0136, 0.0476, 0.1172, 0.2028, 0.2378, 0.2028, 0.1172, 0.0476, 0.0136
);

void main() {
    vec3 result = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        vec2 uv = v_uv + vec2(0.0, u_texel_size.y * float(i - 4));
        result += texture(u_input_tex, uv).rgb * W[i];
    }
    fragColor = vec4(result, 1.0);
}
"""

COMPOSITE_FS = """#version 330
uniform sampler2D u_scene_tex;
uniform sampler2D u_bloom_tex;
uniform sampler2D u_trail_tex;
uniform float u_bloom_strength;
uniform float u_trail_strength;
uniform float u_time;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 scene = texture(u_scene_tex, v_uv).rgb;
    vec3 bloom = texture(u_bloom_tex, v_uv).rgb;
    vec3 trail = texture(u_trail_tex, v_uv).rgb;

    vec3 hdr = scene + bloom * u_bloom_strength + trail * u_trail_strength;
    vec3 ldr = hdr / (hdr + vec3(1.0));

    float vignette = 1.0 - 0.3 * length(v_uv * 2.0 - 1.0);
    vignette = smoothstep(0.0, 1.0, vignette);
    ldr *= vignette;

    float dist = length(v_uv * 2.0 - 1.0);
    vec3 bg_glow = vec3(0.015, 0.004, 0.030) * (1.0 - dist * 0.7);
    ldr += bg_glow;

    float grain = fract(sin(dot(v_uv, vec2(12.9898, 78.233))) * 43758.5453);
    grain = (grain - 0.5) * 0.006;
    ldr += grain;

    ldr = smoothstep(0.0, 1.0, ldr);

    fragColor = vec4(ldr, 1.0);
}
"""

TRAIL_DECAY_FS = """#version 330
uniform sampler2D u_input_tex;
uniform float u_decay;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    fragColor = texture(u_input_tex, v_uv) * u_decay;
}
"""

STAR_VS = """#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_phase;
in float in_kind;
in float in_radial;
in float in_alpha;
in float in_speed;

out vec3 v_color;
out float v_alpha;

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    gl_Position = u_proj * view_pos;
    gl_PointSize = in_size * 2.5 / max(1.0, -view_pos.z);

    float twinkle = 0.6 + 0.4 * sin(u_time * in_speed + in_phase);
    v_color = in_color * (0.95 + 0.05 * in_kind * 0.0);
    float rad_factor = 1.0 - 0.02 * in_radial;
    v_alpha = in_alpha * twinkle * rad_factor;
}
"""

STAR_FS = """#version 330
in vec3 v_color;
in float v_alpha;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = length(uv);
    if (d > 1.0) discard;
    float soft = exp(-d * d * 6.0);
    fragColor = vec4(v_color, v_alpha * soft);
}
"""

SPARK_VS = """#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;

in vec3 in_origin;
in vec3 in_direction;
in float in_speed;
in float in_phase_off;
in float in_particle_size;
in float in_alpha_val;

out vec3 v_color;
out float v_alpha;

void main() {
    float period = 60.0 / 72.0;
    float phase = mod(u_time / period, 1.0);
    float activation = 0.10 + in_phase_off;
    float elapsed = phase - activation;
    float is_active = step(0.0, elapsed);

    float age = clamp(elapsed / 0.30, 0.0, 1.0);
    float life = 1.0 - age;

    vec3 pos = in_origin + in_direction * in_speed * age * 2.5;

    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float spark_alpha = is_active * smoothstep(0.0, 0.15, life) * (1.0 - smoothstep(0.5, 1.0, age));
    gl_PointSize = max(0.5, in_particle_size * (1.0 - age * 0.7)) * 2.5 / max(1.0, -view_pos.z);

    vec3 core_gold = vec3(1.0, 0.88, 0.50);
    vec3 tip_white = vec3(1.0, 0.96, 0.82);
    v_color = mix(core_gold, tip_white, age);
    v_alpha = in_alpha_val * spark_alpha;
}
"""

SPARK_FS = """#version 330
in vec3 v_color;
in float v_alpha;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = length(uv);
    if (d > 1.0) discard;
    float soft = exp(-d * d * 4.0);
    float glow = exp(-d * d * 8.0);
    vec3 col = v_color * (1.0 + glow * 0.6);
    fragColor = vec4(col, v_alpha * soft);
}
"""

PEDESTAL_VS = """#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;
uniform float u_beat;
uniform vec3 u_ped_tint[4];

in vec3 in_position;
in float in_size;
in float in_phase;
in float in_layer;
in float in_alpha;

out vec3 v_color;
out float v_alpha;
out float v_seed;
out float v_layer;

float saturate(float v) { return clamp(v, 0.0, 1.0); }

void main() {
    vec3 pos = in_position;

    float pulse = u_beat;
    int layer = int(in_layer + 0.5);

    // Crystal heartbeat — gentle but clear pulse through the gem body
    float layer_pulse_xy = mix(0.026, 0.008, in_layer / 3.0);
    float layer_pulse_z  = mix(0.034, 0.010, in_layer / 3.0);
    pos.xy *= 1.0 + pulse * layer_pulse_xy;
    pos.z  *= 1.0 + pulse * layer_pulse_z;

    // Per-layer breathing: warm, slow drift like light refracting through crystal
    float breath_speed = (layer == 0) ? 0.70 : (layer == 1) ? 0.85 : 0.50;
    float breath_amp   = (layer == 0) ? 0.005 : (layer == 1) ? 0.006 : 0.003;
    pos.y += sin(u_time * breath_speed + in_phase * 2.1) * breath_amp;

    // Subtle radial breathing — crystal seems to pulse with internal light
    float expand = sin(u_time * 0.90 + in_phase * 1.3) * 0.003;
    pos.x += pos.x * expand;
    pos.z += pos.z * expand;

    float flow = sin(u_time * 0.30 + in_phase) * 0.002;
    float ca = cos(flow), sa = sin(flow);
    pos.x = in_position.x * ca - in_position.z * sa;
    pos.z = in_position.x * sa + in_position.z * ca;

    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float depth = -view_pos.z;
    float frontness = saturate(pos.z * 1.15 + 0.5);
    float depth_scale = 1.55 / max(2.0, depth);
    float pulse_size = 1.0 + pulse * 0.032;
    float layer_size = 1.0 + (3.0 - in_layer) * 0.12;
    float front_size = mix(0.94, 1.14, frontness);
    gl_PointSize = clamp(in_size * depth_scale * pulse_size * layer_size * front_size, 1.5, 36.0);

    vec3 base_color = u_ped_tint[layer];

    // Golden shimmer — warm, gem-like sparkle
    float shimmer = 0.5 + 0.5 * sin(u_time * 1.4 + in_phase * 1.8);
    float shim_light = mix(0.94, 1.08, shimmer);

    // Warm golden depth tint — crystal refracts warm light
    vec3 depth_tint = mix(vec3(0.85, 0.72, 0.55), vec3(1.12, 1.00, 0.78), frontness);
    float depth_light = saturate(1.28 - 0.06 * abs(depth));

    // Layer-specific glow: core brightest, rim sparkles, body soft, skirt faint
    float glow_mult = (layer == 0) ? 1.20 : (layer == 1) ? 1.35 : (layer == 2) ? 1.05 : 0.92;

    // Center glow — internal light source radiating from the crystal core
    float cd = min(1.0, length(in_position.xz) / 0.40);
    float center_glow = mix(glow_mult, 1.0, cd);

    v_color = base_color * depth_tint * depth_light * shim_light * center_glow;
    v_alpha = in_alpha * mix(0.86, 1.16, frontness) * mix(0.94, 1.06, shimmer);
    v_seed = in_phase;
    v_layer = in_layer;
}
"""

PEDESTAL_FS = """#version 330
in vec3 v_color;
in float v_alpha;
in float v_seed;
in float v_layer;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float radius = sqrt(r2);
    float angle = atan(uv.y, uv.x);

    // Crystal-cut edge — slightly faceted, gem-like
    float facet = 0.92
        + 0.04 * sin(angle * 6.0 + v_seed * 2.0)
        + 0.02 * sin(angle * 10.0 - v_seed * 1.4)
        + 0.01 * sin(angle * 14.0 + v_seed * 2.8);
    float crystal_edge = 1.0 - smoothstep(facet - 0.08, facet, radius);

    // Multi-layer glow: core, body, halo
    float core = exp(-r2 * 6.0);
    float body = exp(-r2 * 2.5);
    float halo = exp(-r2 * 1.2);

    int layer = int(v_layer + 0.5);
    float alpha_shape;
    float glow;

    if (layer == 0) {
        // Crystal face — dense core with internal golden light
        alpha_shape = mix(0.65, 1.0, core) * pow(1.0 - radius, 1.05);
        glow = 1.02 + core * 0.40 + halo * 0.12;
    } else if (layer == 1) {
        // Decorative rim — brightest, gem-like sparkle burst
        alpha_shape = mix(0.70, 1.0, core) * pow(1.0 - radius, 0.85);
        glow = 1.10 + core * 0.50 + halo * 0.10;
    } else if (layer == 2) {
        // Crystal body — translucent, internal glow visible through volume
        alpha_shape = mix(0.48, 1.0, body) * pow(1.0 - radius, 1.30);
        glow = 0.95 + core * 0.28 + halo * 0.14;
    } else {
        // Ground halo — very soft, wide golden glow on the floor
        alpha_shape = mix(0.32, 1.0, body) * pow(1.0 - radius, 1.70);
        glow = 0.88 + core * 0.18 + halo * 0.16;
    }

    alpha_shape *= crystal_edge;

    // Golden inner glow — warm light trapped inside the crystal
    vec3 gold_inner = vec3(1.00, 0.72, 0.28);
    vec3 gold_outer = vec3(0.90, 0.55, 0.15);
    float inner = exp(-r2 * 10.0);
    float outer = exp(-r2 * 1.8);
    vec3 gem_glow = gold_inner * inner * 0.20 + gold_outer * outer * 0.06;

    // Rim highlight — bright edge reflection on crystal facets
    float rim_light = smoothstep(0.70, 0.95, radius) * crystal_edge;
    vec3 rim_color = vec3(1.15, 0.95, 0.50) * rim_light * 0.12;

    vec3 hdr = v_color * glow + gem_glow + rim_color;
    fragColor = vec4(hdr, v_alpha * alpha_shape);
}
"""
