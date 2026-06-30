# OpenCV Stereo Vision Pipeline

A stereo disparity pipeline built with OpenCV and Python. Supports the Middlebury 2014 stereo dataset with auto-download, custom stereo image pairs, depth measurement, point cloud export, and more.

## Requirements

```bash
pip install opencv-contrib-python numpy open3d
```

## Usage

### Middlebury Dataset

```bash
# Default scene (Shopvac)
python cv.py

# List all 23 available scenes
python cv.py --list

# Load a specific scene (auto-downloads if not on disk)
python cv.py --scene Motorcycle
python cv.py --scene Piano
```

### Custom Stereo Images

```bash
# Basic (no metric depth)
python cv.py --left left.jpg --right right.jpg

# With metric depth (focal length in pixels, baseline in mm)
python cv.py --left left.jpg --right right.jpg --focal 800 --baseline 60
```

## Options

| Flag | Description |
|---|---|
| `--scene NAME` | Middlebury 2014 scene to load (default: Shopvac) |
| `--list` | Show all available scenes and which are downloaded |
| `--wls` | Apply WLS filter for cleaner disparity (recommended) |
| `--inpaint` | Fill invalid pixels via inpainting |
| `--open3d` | View point cloud in Open3D interactive viewer |
| `--export FILE` | Save point cloud as .ply (default: output.ply) |
| `--voxel M` | Voxel downsample before export, size in metres (e.g. 0.01) |
| `--max-depth MM` | Max depth cutoff for point cloud in mm (default: 5000) |
| `--focal PX` | Camera focal length in pixels (custom images) |
| `--baseline MM` | Camera separation in mm (custom images) |
| `--ndisp N` | Override number of disparities (must be multiple of 16) |

## Features

- **Disparity heatmap** — colour-coded near (warm) to far (cool)
- **Click-to-measure** — click anywhere on the disparity window to print depth at that pixel
- **WLS filter** — weighted least squares post-processing for cleaner edges and fewer holes
- **Ground truth comparison** — error map vs Middlebury ground truth disparity
- **Inpainting** — fills invalid/black regions using TELEA algorithm
- **Point cloud** — reprojects disparity to 3D, viewable in Open3D or exportable as .ply

## Available Scenes

Adirondack, Backpack, Bicycle1, Cable, Classroom1, Couch, Flowers, Jadeplant, Mask, Motorcycle, Piano, Pipes, Playroom, Playtable, Recycle, Shelves, Shopvac, Sticks, Storage, Sword1, Sword2, Umbrella, Vintage

Scenes are downloaded automatically from the [Middlebury Stereo Dataset](https://vision.middlebury.edu/stereo/data/scenes2014/) on first use (~50–100MB each).

## Example Commands

```bash
# Best quality disparity with WLS filter
python cv.py --scene Motorcycle --wls

# View point cloud interactively
python cv.py --open3d --voxel 0.01

# Export downsampled point cloud
python cv.py --export output.ply --voxel 0.01 --max-depth 3000

# Custom images with full pipeline
python cv.py --left left.jpg --right right.jpg --focal 800 --baseline 60 --wls --open3d
```

## Point Cloud Export Formats

Supported extensions: `.ply`, `.pcd`, `.xyz`, `.xyzrgb`, `.pts`

```bash
python cv.py --export cloud.ply
python cv.py --export cloud.xyz
```
