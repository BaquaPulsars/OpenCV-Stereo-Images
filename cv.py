import argparse
import os
import re
import urllib.request
import zipfile
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Usage:
#   python cv.py                         # default scene (Shopvac)
#   python cv.py --scene Motorcycle      # pick a different scene
#   python cv.py --list                  # show all available scenes
#   python cv.py --wls                   # WLS disparity filter
#   python cv.py --inpaint               # fill invalid pixels
#   python cv.py --open3d                # view point cloud in Open3D
#   python cv.py --export output.ply     # save point cloud
#   python cv.py --left a.jpg --right b.jpg --focal 800 --baseline 60
#
# Click anywhere on the disparity window to measure depth at that point.
# ---------------------------------------------------------------------------

SCENES_DIR  = r"c:\Users\ohlri\Downloads\Py\OpenCV"
BASE_URL    = "https://vision.middlebury.edu/stereo/data/scenes2014/zip"
SCALE       = 0.25
DISP_WINDOW = "Disparity (click to measure)"

SCENES = [
    "Adirondack", "Backpack",  "Bicycle1",  "Cable",     "Classroom1",
    "Couch",      "Flowers",   "Jadeplant", "Mask",      "Motorcycle",
    "Piano",      "Pipes",     "Playroom",  "Playtable", "Recycle",
    "Shelves",    "Shopvac",   "Sticks",    "Storage",   "Sword1",
    "Sword2",     "Umbrella",  "Vintage",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",     default="Shopvac",   choices=SCENES, metavar="NAME",
                                             help="Middlebury 2014 scene name (use --list to see all)")
    p.add_argument("--list",      action="store_true", help="List all scenes and exit")
    p.add_argument("--left",      default=None,        help="Left image path (custom stereo pair)")
    p.add_argument("--right",     default=None,        help="Right image path (custom stereo pair)")
    p.add_argument("--focal",     type=float,          help="Focal length in pixels (custom images)")
    p.add_argument("--baseline",  type=float,          help="Baseline in mm (custom images)")
    p.add_argument("--ndisp",     type=int,            help="Number of disparities (multiple of 16)")
    p.add_argument("--wls",       action="store_true", help="Apply WLS disparity filter")
    p.add_argument("--inpaint",   action="store_true", help="Fill invalid disparity pixels")
    p.add_argument("--open3d",    action="store_true", help="View point cloud in Open3D")
    p.add_argument("--voxel",     type=float, default=0.0,  metavar="M",
                                             help="Voxel downsample size in metres (e.g. 0.01 = 1cm)")
    p.add_argument("--max-depth", type=float, default=5000, metavar="MM",
                                             help="Max depth in mm for point cloud (default: 5000)")
    p.add_argument("--export",    nargs="?", const="output.ply", metavar="FILE",
                                             help="Save point cloud as .ply (default: output.ply)")
    return p.parse_args()


def scene_dir(name):
    return os.path.join(SCENES_DIR, f"{name}-imperfect")


def is_downloaded(name):
    d = scene_dir(name)
    return os.path.isfile(os.path.join(d, "im0.png"))


def download_scene(name):
    url      = f"{BASE_URL}/{name}-imperfect.zip"
    zip_path = os.path.join(SCENES_DIR, f"{name}-imperfect.zip")
    print(f"Downloading {name} from Middlebury...")

    def progress(count, block, total):
        pct = min(100, count * block * 100 // total)
        print(f"  {pct}%", end="\r")

    urllib.request.urlretrieve(url, zip_path, reporthook=progress)
    print()
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(SCENES_DIR)
    os.remove(zip_path)
    print(f"Ready: {scene_dir(name)}")


def parse_calib(path):
    text = open(path).read()
    cam0     = re.search(r"cam0=\[([^\]]+)\]", text).group(1).replace(";", " ").split()
    baseline = float(re.search(r"baseline=([0-9.]+)", text).group(1))
    ndisp    = int(re.search(r"ndisp=([0-9]+)", text).group(1))
    return {
        "focal":    float(cam0[0]),
        "cx":       float(cam0[2]),
        "cy":       float(cam0[5]),
        "baseline": baseline,
        "ndisp":    ndisp,
    }


def read_pfm(path):
    with open(path, "rb") as f:
        f.readline()
        w, h  = map(int, f.readline().decode().strip().split())
        scale = float(f.readline().decode().strip())
        data  = np.frombuffer(f.read(), dtype=np.float32).reshape((h, w))
        if scale < 0:
            data = data[::-1]
    return data


def to_heatmap(arr, colormap=cv2.COLORMAP_TURBO):
    valid = (arr > 0) & np.isfinite(arr)
    norm  = np.zeros_like(arr, dtype=np.uint8)
    if valid.any():
        norm[valid] = cv2.normalize(
            arr[valid], None, 1, 255, cv2.NORM_MINMAX
        ).flatten().astype(np.uint8)
    color         = cv2.applyColorMap(norm, colormap)
    color[~valid] = 0
    return color


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)


def build_point_cloud(disp, img_bgr, focal_s, base_mm, cx_s, cy_s, max_depth_mm=5000):
    h, w   = disp.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    d_min  = focal_s * base_mm / max_depth_mm
    valid  = disp >= d_min
    d      = disp[valid]
    Z = focal_s * base_mm / d
    X = (uu[valid] - cx_s) * Z / focal_s
    Y = (vv[valid] - cy_s) * Z / focal_s
    xyz = np.stack([X, Y, Z], axis=1) / 1000.0
    rgb = img_bgr[valid][:, ::-1].astype(np.float64) / 255.0
    return xyz, rgb


# =============================================================================
args = parse_args()

# --list
if args.list:
    print(f"\n{'Scene':<14}  Status")
    print("-" * 28)
    for name in SCENES:
        status = "downloaded" if is_downloaded(name) else "not downloaded"
        marker = "*" if name == args.scene else " "
        print(f" {marker} {name:<14}  {status}")
    print(f"\nDefault: Shopvac   Use: python cv.py --scene <NAME>")
    raise SystemExit(0)

custom_mode = args.left is not None and args.right is not None

# --- Load images ---
print("Loading images...")
if custom_mode:
    img0        = load_image(args.left)
    img1        = load_image(args.right)
    focal_px    = args.focal
    baseline_mm = args.baseline
    cx_px       = img0.shape[1] / 2
    cy_px       = img0.shape[0] / 2
    ndisp_full  = args.ndisp or 128
    gt_small    = None
else:
    if not is_downloaded(args.scene):
        download_scene(args.scene)

    d    = scene_dir(args.scene)
    cal  = parse_calib(os.path.join(d, "calib.txt"))
    img0 = cv2.resize(cv2.imread(os.path.join(d, "im0.png")), (0, 0), fx=SCALE, fy=SCALE)
    img1 = cv2.resize(cv2.imread(os.path.join(d, "im1.png")), (0, 0), fx=SCALE, fy=SCALE)

    focal_px    = cal["focal"]
    baseline_mm = cal["baseline"]
    cx_px       = cal["cx"]
    cy_px       = cal["cy"]
    ndisp_full  = args.ndisp or cal["ndisp"]
    gt_full     = read_pfm(os.path.join(d, "disp0.pfm"))
    gt_small    = cv2.resize(gt_full, (0, 0), fx=SCALE, fy=SCALE) * SCALE
    print(f"Scene: {args.scene}  |  baseline={baseline_mm:.1f}mm  ndisp={ndisp_full}")

gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

cv2.imshow("Left image",  img0)
cv2.imshow("Right image", img1)
cv2.waitKey(1)

# --- Compute disparity ---
ndisp = max(16, int(ndisp_full * SCALE / 16) * 16)
print(f"Computing disparity (ndisp={ndisp})...")
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=ndisp,
    blockSize=5,
    P1=8  * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)
disp_left = stereo.compute(gray0, gray1).astype(np.float32) / 16.0

# --- WLS filter ---
if args.wls:
    print("Applying WLS filter...")
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    disp_right    = right_matcher.compute(gray1, gray0).astype(np.float32) / 16.0
    wls           = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls.setLambda(8000)
    wls.setSigmaColor(1.5)
    disp_active   = np.clip(wls.filter(disp_left, gray0, disparity_map_right=disp_right), 0, None)
    cv2.imshow("Disparity — raw",          to_heatmap(disp_left[:, ndisp:]))
    cv2.imshow("Disparity — WLS filtered", to_heatmap(disp_active[:, ndisp:]))
    cv2.waitKey(1)
else:
    disp_active = disp_left

# Crop left strip (no valid stereo correspondence there)
c      = ndisp
disp_c = disp_active[:, c:]
img0_c = img0[:, c:]
gt_c   = gt_small[:, c:] if gt_small is not None else None

# --- Calibration-dependent quantities ---
has_calib    = focal_px is not None and baseline_mm is not None
focal_scaled = focal_px * SCALE if has_calib else None
cx_scaled    = cx_px    * SCALE if has_calib else None
cy_scaled    = cy_px    * SCALE if has_calib else None

if has_calib:
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_mm = np.where(disp_c > 0, focal_scaled * baseline_mm / disp_c, 0)
    h, w = depth_mm.shape
    print(f"Depth at centre: {depth_mm[h//2, w//2]:.0f} mm  ({depth_mm[h//2, w//2]/1000:.2f} m)")
else:
    depth_mm = None
    print("No focal/baseline — pass --focal and --baseline for metric depth.")

# --- Click-to-measure callback ---
def on_mouse(event, x, y, flags, param):
    del flags
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    d_map, f_s, b_mm = param
    if not (0 <= y < d_map.shape[0] and 0 <= x < d_map.shape[1]):
        return
    d = d_map[y, x]
    if d > 0 and f_s and b_mm:
        z = f_s * b_mm / d
        print(f"  [{x:4d}, {y:4d}]  disp={d:6.1f} px   depth={z:7.0f} mm  ({z/1000:.2f} m)")
    else:
        print(f"  [{x:4d}, {y:4d}]  no valid disparity")

# --- Main disparity display ---
disp_color = to_heatmap(disp_c)
if has_calib and depth_mm is not None:
    h, w = depth_mm.shape
    cv2.circle(disp_color, (w//2, h//2), 8, (255, 255, 255), 2)
    cv2.putText(disp_color, f"{depth_mm[h//2, w//2]/1000:.2f} m",
                (w//2 + 12, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow("Left image (valid region)", img0_c)
cv2.imshow(DISP_WINDOW, disp_color)
cv2.setMouseCallback(DISP_WINDOW, on_mouse, (disp_c, focal_scaled, baseline_mm))
print(f'Click on "{DISP_WINDOW}" to measure depth at any point.')

# --- Inpainting ---
if args.inpaint:
    print("Inpainting invalid pixels...")
    valid_mask   = disp_active > 0
    disp_u8      = np.zeros(disp_active.shape, dtype=np.uint8)
    if valid_mask.any():
        disp_u8[valid_mask] = cv2.normalize(
            disp_active[valid_mask], None, 1, 255, cv2.NORM_MINMAX
        ).flatten().astype(np.uint8)
    inpaint_mask = (~valid_mask).astype(np.uint8) * 255
    disp_filled  = cv2.inpaint(disp_u8, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    cv2.imshow("Inpainted disparity (full width)", cv2.applyColorMap(disp_filled, cv2.COLORMAP_TURBO))
    n = int(inpaint_mask.sum() / 255)
    print(f"Filled {n:,} pixels ({100*n/disp_active.size:.1f}%). Left-strip values are extrapolated, not real.")

# --- Ground truth + error map (dataset only) ---
if gt_c is not None:
    valid    = (gt_c > 0) & (disp_c > 0) & np.isfinite(gt_c) & np.isfinite(disp_c)
    error    = np.abs(disp_c - gt_c)
    print(f"Mean disparity error vs ground truth: {error[valid].mean():.2f} px")
    cv2.imshow("Ground truth disparity",      to_heatmap(gt_c))
    cv2.imshow("Error map (vs ground truth)", to_heatmap(np.where(valid, error, 0), cv2.COLORMAP_HOT))

# --- Point cloud ---
if (args.open3d or args.export) and has_calib:
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed: pip install open3d")
        args.open3d = args.export = None

    if args.open3d or args.export:
        print("Building point cloud...")
        xyz, rgb = build_point_cloud(disp_active, img0, focal_scaled, baseline_mm,
                                     cx_scaled, cy_scaled, max_depth_mm=args.max_depth)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        print(f"Point cloud: {len(xyz):,} points")

        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
            print(f"After voxel downsample ({args.voxel}m): {len(pcd.points):,} points")

        print("Estimating normals...")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

        if args.export:
            supported = {".ply", ".pcd", ".xyz", ".xyzrgb", ".pts"}
            ext = "." + args.export.rsplit(".", 1)[-1].lower() if "." in args.export else ""
            if ext not in supported:
                print(f"Unsupported format '{ext}'. Use: {', '.join(supported)}")
            else:
                o3d.io.write_point_cloud(args.export, pcd, write_ascii=True)
                print(f"Saved: {args.export}")

        if args.open3d:
            print("Opening Open3D viewer... (close to continue)")
            o3d.visualization.draw_geometries(
                [pcd], window_name=f"Point Cloud — {args.scene}",
                zoom=0.5, front=[0, 0, -1], lookat=[0, 0, 1], up=[0, -1, 0],
            )
elif (args.open3d or args.export) and not has_calib:
    print("Point cloud requires --focal and --baseline.")

print("Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
