# Heart Python Versions

## Layout

- `particle_heart/`: main 3D particle heart package.
- `legacy/li_xun_heart.py`: archived v1 heart renderer.
- `docs/heart_upgrade_plan.md`: older design notes kept for reference.
- `gomoku.py`, `gomoku_models/`, `diff_test.py`: unchanged non-heart project files.

## Environment

- Python environment: `D:\Program\Anaconda\envs\gomoku_ai`
- Core dependencies: `glfw`, `moderngl`, `numpy`

## Run

```powershell
python -m particle_heart
python -m particle_heart --quality low
python -m particle_heart --quality high --no-bloom
python -m particle_heart --particles 300000
python -m particle_heart --no-bloom --no-trails
python -m particle_heart --no-pedestal
python -m particle_heart --self-test
python -m particle_heart --self-test --quality low
```

## CLI Options

- `--width`: window width, default `1280`
- `--height`: window height, default `960`
- `--seed`: random seed, default `7`
- `--particles`: total particle count, default `70000`
- `--bloom` / `--no-bloom`: toggle bloom
- `--trails` / `--no-trails`: toggle trails
- `--fps`: show FPS in the window title
- `--self-test`: hidden-window smoke test
- `--theme`: theme index
- `--quality`: preset: `low` (40k particles, no trails), `medium` (70k, full effects), `high` (90k, full effects)
- `--pedestal` / `--no-pedestal`: toggle pedestal layer beneath the heart

## Controls

- `Esc`: quit
- `Space`: pause / resume
- `C`: toggle auto and manual camera
- `T`: cycle themes
- `1`: toggle bloom
- `2`: toggle trails
- `F`: toggle FPS title display
- `R`: reset manual camera
- Mouse left drag: orbit camera
- Mouse right drag: pan target
- Mouse wheel: zoom
