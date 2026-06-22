"""Project KL HD-map lane polylines onto the DOM GeoTIFF.

This is a lightweight inspection tool for ``data/kl_8/map``. It reads lane
centerlines and boundaries from the Apollo-style ``base_map.txt`` text map,
converts the local map x/y coordinates through ``map_origin.yaml`` into
WGS84 lon/lat, then uses GeoTIFF tags from the DOM image to draw those lanes
on top of the orthophoto.

Example:
    python3 tools/analysis_tools/visualize_hdmap_on_dom.py \
        --map-dir data/kl_8/map
"""

import argparse
import json
import math
import os
import os.path as osp
import re
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:
    yaml = None


NUM_RE = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
POINT_RE = re.compile(
    r'point\s*\{[^{}]*?\bx:\s*(' + NUM_RE + r')'
    r'[^{}]*?\by:\s*(' + NUM_RE + r')',
    re.S)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Overlay KL base_map lanes on the DOM GeoTIFF.')
    parser.add_argument('--map-dir', default='data/kl_8/map',
                        help='directory containing base_map.txt/result.tif')
    parser.add_argument('--base-map', default=None,
                        help='Apollo text map path; defaults to map-dir/base_map.txt')
    parser.add_argument('--lane-npz', default=None,
                        help='optional parsed lane npz path')
    parser.add_argument('--source', default='txt', choices=['txt', 'npz'],
                        help='read full-resolution lanes from txt or cached lanes from npz')
    parser.add_argument('--dom', default=None,
                        help='DOM GeoTIFF path; defaults to map-dir/result.tif')
    parser.add_argument('--dom-json', default=None,
                        help='fallback DOM bounds json if GeoTIFF tags are missing')
    parser.add_argument('--origin', default=None,
                        help='map origin yaml; defaults to map-dir/map_origin.yaml')
    parser.add_argument('--output', default=None,
                        help='full-resolution overlay output image')
    parser.add_argument('--preview-output', default=None,
                        help='downscaled preview output image')
    parser.add_argument('--preview-width', type=int, default=3200,
                        help='preview width in pixels; 0 disables preview')
    parser.add_argument('--local-model', default='wgs84',
                        choices=['utm', 'enu', 'wgs84', 'sphere'],
                        help=('local x/y metres to lon/lat model: enu is an '
                              'exact WGS84 ENU tangent-plane conversion; '
                              'utm treats x/y as UTM metre offsets from the '
                              'map origin; wgs84/sphere are linear '
                              'approximations'))
    parser.add_argument('--utm-zone', type=int, default=None,
                        help='UTM zone for --local-model utm; auto from origin lon if omitted')
    parser.add_argument('--utm-south', action='store_true',
                        help='use southern-hemisphere UTM false northing')
    parser.add_argument('--pixel-shift-x', type=float, default=0.0,
                        help='optional manual image-space shift in pixels')
    parser.add_argument('--pixel-shift-y', type=float, default=0.0,
                        help='optional manual image-space shift in pixels')
    parser.add_argument('--line-thickness', type=int, default=5,
                        help='lane line thickness on the full-resolution image')
    parser.add_argument('--halo-thickness', type=int, default=3,
                        help='extra black halo thickness for readability')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='lane overlay opacity in [0, 1]')
    parser.add_argument('--no-legend', action='store_true',
                        help='do not draw the small color legend')
    return parser.parse_args()


def _default_path(path: Optional[str], map_dir: str, name: str) -> str:
    return path if path else osp.join(map_dir, name)


def read_origin(path: str) -> Tuple[float, float]:
    if yaml is not None:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = float(value.strip())

    lat = data.get('local_map_orignal_latitude',
                   data.get('local_map_original_latitude'))
    lon = data.get('local_map_orignal_longitude',
                   data.get('local_map_original_longitude'))
    if lat is None or lon is None:
        raise KeyError(
            'map origin must contain local_map_orignal_latitude and '
            'local_map_orignal_longitude')
    return float(lat), float(lon)


def iter_named_blocks(text: str, name: str) -> Iterable[str]:
    needle = name + ' {'
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return
        brace = text.find('{', start)
        if brace < 0:
            return
        depth = 0
        for idx in range(brace, len(text)):
            ch = text[idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    yield text[brace + 1:idx]
                    cursor = idx + 1
                    break
        else:
            return


def find_field_block(block: str, field: str) -> Optional[str]:
    match = re.search(r'\b' + re.escape(field) + r'\s*\{', block)
    if match is None:
        return None
    brace = match.end() - 1
    depth = 0
    for idx in range(brace, len(block)):
        ch = block[idx]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return block[brace + 1:idx]
    return None


def extract_points(block: Optional[str]) -> Optional[np.ndarray]:
    if not block:
        return None
    points = [(float(x), float(y)) for x, y in POINT_RE.findall(block)]
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.float64)


def load_lanes_from_txt(path: str) -> List[Dict[str, Optional[np.ndarray]]]:
    with open(path, 'r') as f:
        text = f.read()

    lanes = []
    for lane_block in iter_named_blocks(text, 'lane'):
        central = extract_points(find_field_block(lane_block, 'central_curve'))
        if central is None:
            continue
        lanes.append({
            'central': central,
            'left': extract_points(find_field_block(lane_block,
                                                    'left_boundary')),
            'right': extract_points(find_field_block(lane_block,
                                                     'right_boundary')),
        })
    return lanes


def load_lanes_from_npz(path: str) -> List[Dict[str, Optional[np.ndarray]]]:
    data = np.load(path, allow_pickle=True)
    lanes = []
    for lane in data['lanes']:
        lanes.append({
            'central': lane.get('central'),
            'left': lane.get('left'),
            'right': lane.get('right'),
        })
    return lanes


def geodetic_meters_per_degree(lat_deg: float,
                               local_model: str) -> Tuple[float, float]:
    lat = math.radians(lat_deg)
    if local_model == 'sphere':
        radius = 6378137.0
        meters_per_deg_lat = math.pi * radius / 180.0
        meters_per_deg_lon = meters_per_deg_lat * math.cos(lat)
        return meters_per_deg_lat, meters_per_deg_lon

    semi_major = 6378137.0
    eccentricity_sq = 6.69437999014e-3
    sin_lat = math.sin(lat)
    denom = 1.0 - eccentricity_sq * sin_lat * sin_lat
    meridian = semi_major * (1.0 - eccentricity_sq) / (denom ** 1.5)
    prime_vertical = semi_major / math.sqrt(denom)
    meters_per_deg_lat = math.pi * meridian / 180.0
    meters_per_deg_lon = math.pi * prime_vertical * math.cos(lat) / 180.0
    return meters_per_deg_lat, meters_per_deg_lon


def enu_xy_to_lonlat(points: np.ndarray, origin_lat: float,
                     origin_lon: float, origin_alt: float = 0.0) -> np.ndarray:
    """Convert local ENU x/y metres to WGS84 lon/lat using ECEF exactly."""
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)

    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)
    prime_vertical = semi_major / math.sqrt(
        1.0 - eccentricity_sq * sin_lat * sin_lat)

    origin = np.array([
        (prime_vertical + origin_alt) * cos_lat * cos_lon,
        (prime_vertical + origin_alt) * cos_lat * sin_lon,
        (prime_vertical * (1.0 - eccentricity_sq) + origin_alt) * sin_lat,
    ], dtype=np.float64)

    enu_to_ecef = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [0.0, cos_lat, sin_lat],
    ], dtype=np.float64)

    enu = np.zeros((points.shape[0], 3), dtype=np.float64)
    enu[:, :2] = points[:, :2]
    ecef = origin[None, :] + enu @ enu_to_ecef.T

    x = ecef[:, 0]
    y = ecef[:, 1]
    z = ecef[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - eccentricity_sq))
    for _ in range(8):
        sin_lat = np.sin(lat)
        prime_vertical = semi_major / np.sqrt(
            1.0 - eccentricity_sq * sin_lat * sin_lat)
        height = p / np.cos(lat) - prime_vertical
        lat = np.arctan2(
            z,
            p * (1.0 - eccentricity_sq * prime_vertical /
                 (prime_vertical + height)))

    return np.stack([np.degrees(lon), np.degrees(lat)], axis=1)


def utm_zone_from_lon(lon_deg: float) -> int:
    return int(math.floor((lon_deg + 180.0) / 6.0)) + 1


def utm_forward(lat_deg: float, lon_deg: float,
                zone: int) -> Tuple[float, float]:
    """WGS84 lat/lon to UTM easting/northing for one point."""
    semi_major = 6378137.0
    eccentricity_sq = 6.69437999014e-3
    scale = 0.9996
    eccentricity_prime_sq = eccentricity_sq / (1.0 - eccentricity_sq)

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon_origin = math.radians((zone - 1) * 6 - 180 + 3)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    n = semi_major / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = eccentricity_prime_sq * cos_lat * cos_lat
    a = cos_lat * (lon - lon_origin)
    m = semi_major * (
        (1.0 - eccentricity_sq / 4.0 - 3.0 * eccentricity_sq**2 / 64.0 -
         5.0 * eccentricity_sq**3 / 256.0) * lat -
        (3.0 * eccentricity_sq / 8.0 +
         3.0 * eccentricity_sq**2 / 32.0 +
         45.0 * eccentricity_sq**3 / 1024.0) * math.sin(2.0 * lat) +
        (15.0 * eccentricity_sq**2 / 256.0 +
         45.0 * eccentricity_sq**3 / 1024.0) * math.sin(4.0 * lat) -
        (35.0 * eccentricity_sq**3 / 3072.0) * math.sin(6.0 * lat))

    easting = scale * n * (
        a + (1.0 - t + c) * a**3 / 6.0 +
        (5.0 - 18.0 * t + t**2 + 72.0 * c -
         58.0 * eccentricity_prime_sq) * a**5 / 120.0) + 500000.0
    northing = scale * (
        m + n * tan_lat *
        (a**2 / 2.0 +
         (5.0 - t + 9.0 * c + 4.0 * c**2) * a**4 / 24.0 +
         (61.0 - 58.0 * t + t**2 + 600.0 * c -
          330.0 * eccentricity_prime_sq) * a**6 / 720.0))
    if lat_deg < 0:
        northing += 10000000.0
    return easting, northing


def utm_inverse(easting: np.ndarray, northing: np.ndarray, zone: int,
                northern: bool = True) -> np.ndarray:
    """WGS84 UTM easting/northing arrays to lon/lat."""
    semi_major = 6378137.0
    eccentricity_sq = 6.69437999014e-3
    scale = 0.9996
    eccentricity_prime_sq = eccentricity_sq / (1.0 - eccentricity_sq)

    x = np.asarray(easting, dtype=np.float64) - 500000.0
    y = np.asarray(northing, dtype=np.float64)
    if not northern:
        y = y - 10000000.0

    lon_origin = math.radians((zone - 1) * 6 - 180 + 3)
    m = y / scale
    mu = m / (semi_major * (
        1.0 - eccentricity_sq / 4.0 -
        3.0 * eccentricity_sq**2 / 64.0 -
        5.0 * eccentricity_sq**3 / 256.0))

    e1 = (1.0 - math.sqrt(1.0 - eccentricity_sq)) / (
        1.0 + math.sqrt(1.0 - eccentricity_sq))
    fp = (mu +
          (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * np.sin(2.0 * mu) +
          (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) *
          np.sin(4.0 * mu) +
          (151.0 * e1**3 / 96.0) * np.sin(6.0 * mu) +
          (1097.0 * e1**4 / 512.0) * np.sin(8.0 * mu))

    sin_fp = np.sin(fp)
    cos_fp = np.cos(fp)
    tan_fp = np.tan(fp)
    c1 = eccentricity_prime_sq * cos_fp * cos_fp
    t1 = tan_fp * tan_fp
    n1 = semi_major / np.sqrt(1.0 - eccentricity_sq * sin_fp * sin_fp)
    r1 = (semi_major * (1.0 - eccentricity_sq) /
          (1.0 - eccentricity_sq * sin_fp * sin_fp)**1.5)
    d = x / (n1 * scale)

    lat = fp - (n1 * tan_fp / r1) * (
        d**2 / 2.0 -
        (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 -
         9.0 * eccentricity_prime_sq) * d**4 / 24.0 +
        (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 -
         252.0 * eccentricity_prime_sq - 3.0 * c1**2) *
        d**6 / 720.0)
    lon = lon_origin + (
        d - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0 +
        (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 +
         8.0 * eccentricity_prime_sq + 24.0 * t1**2) *
        d**5 / 120.0) / cos_fp

    return np.stack([np.degrees(lon), np.degrees(lat)], axis=1)


def utm_offset_xy_to_lonlat(points: np.ndarray, origin_lat: float,
                            origin_lon: float, zone: int,
                            northern: bool) -> np.ndarray:
    origin_easting, origin_northing = utm_forward(origin_lat, origin_lon, zone)
    easting = origin_easting + points[:, 0]
    northing = origin_northing + points[:, 1]
    return utm_inverse(easting, northing, zone, northern=northern)


def local_xy_to_lonlat(points: np.ndarray, origin_lat: float,
                       origin_lon: float, local_model: str,
                       utm_zone: Optional[int] = None,
                       utm_south: bool = False) -> np.ndarray:
    if local_model == 'utm':
        zone = utm_zone or utm_zone_from_lon(origin_lon)
        return utm_offset_xy_to_lonlat(
            points, origin_lat, origin_lon, zone, northern=not utm_south)
    if local_model == 'enu':
        return enu_xy_to_lonlat(points, origin_lat, origin_lon)
    meters_per_deg_lat, meters_per_deg_lon = geodetic_meters_per_degree(
        origin_lat, local_model)
    lon = origin_lon + points[:, 0] / meters_per_deg_lon
    lat = origin_lat + points[:, 1] / meters_per_deg_lat
    return np.stack([lon, lat], axis=1)


class DomTransform:

    def __init__(self, width: int, height: int, tie_col: float,
                 tie_row: float, tie_lon: float, tie_lat: float,
                 lon_per_px: float, lat_per_px: float):
        self.width = int(width)
        self.height = int(height)
        self.tie_col = float(tie_col)
        self.tie_row = float(tie_row)
        self.tie_lon = float(tie_lon)
        self.tie_lat = float(tie_lat)
        self.lon_per_px = float(lon_per_px)
        self.lat_per_px = float(lat_per_px)

    def lonlat_to_pixel(self, lonlat: np.ndarray) -> np.ndarray:
        col = self.tie_col + (lonlat[:, 0] - self.tie_lon) / self.lon_per_px
        row = self.tie_row + (self.tie_lat - lonlat[:, 1]) / self.lat_per_px
        return np.stack([col, row], axis=1)


def _read_lon_lat(obj: dict) -> Tuple[float, float]:
    lon = obj.get('longtitude', obj.get('longitude'))
    lat = obj.get('latitude')
    if lon is None or lat is None:
        raise KeyError('DOM corner must contain longitude/longtitude and latitude')
    return float(lon), float(lat)


def dom_transform_from_json(json_path: str, width: int,
                            height: int) -> DomTransform:
    with open(json_path, 'r') as f:
        data = json.load(f)
    west, south = _read_lon_lat(data['southWest'])
    east, north = _read_lon_lat(data['northEast'])
    return DomTransform(
        width=width,
        height=height,
        tie_col=0.0,
        tie_row=0.0,
        tie_lon=west,
        tie_lat=north,
        lon_per_px=(east - west) / float(width),
        lat_per_px=(north - south) / float(height))


def read_dom_transform(dom_path: str,
                       fallback_json: Optional[str]) -> DomTransform:
    with Image.open(dom_path) as image:
        width, height = image.size
        tags = image.tag_v2
        pixel_scale = tags.get(33550)
        tie_points = tags.get(33922)

    if pixel_scale is not None and tie_points is not None:
        tie_col, tie_row, _, tie_lon, tie_lat, _ = [float(v)
                                                    for v in tie_points[:6]]
        return DomTransform(
            width=width,
            height=height,
            tie_col=tie_col,
            tie_row=tie_row,
            tie_lon=tie_lon,
            tie_lat=tie_lat,
            lon_per_px=float(pixel_scale[0]),
            lat_per_px=float(pixel_scale[1]))

    if fallback_json:
        return dom_transform_from_json(fallback_json, width, height)

    raise ValueError(
        'DOM image has no GeoTIFF pixel scale/tiepoint tags. Pass '
        '--dom-json with the corner lon/lat metadata.')


def clip_visible(points: np.ndarray, width: int, height: int,
                 margin: int = 16) -> bool:
    if points.shape[0] < 2:
        return False
    xs = points[:, 0]
    ys = points[:, 1]
    return bool(xs.max() >= -margin and xs.min() < width + margin and
                ys.max() >= -margin and ys.min() < height + margin)


def draw_polyline_layer(layer: np.ndarray, points: np.ndarray,
                        color: Tuple[int, int, int], thickness: int,
                        halo_thickness: int):
    pix = np.round(points).astype(np.int32)
    if pix.shape[0] < 2:
        return
    if halo_thickness > 0:
        cv2.polylines(layer, [pix], False, (0, 0, 0),
                      max(1, thickness + halo_thickness),
                      lineType=cv2.LINE_AA)
    cv2.polylines(layer, [pix], False, color, max(1, thickness),
                  lineType=cv2.LINE_AA)


def draw_legend(image: np.ndarray):
    items = [
        ((0, 0, 255), 'left boundary'),
        ((0, 255, 255), 'centerline'),
        ((255, 0, 0), 'right boundary'),
    ]
    x0, y0 = 18, 18
    row_h = 34
    width = 245
    height = 18 + row_h * len(items)
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height),
                  (0, 0, 0), -1)
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height),
                  (230, 230, 230), 1)
    y = y0 + 30
    for color, label in items:
        cv2.line(image, (x0 + 14, y - 6), (x0 + 70, y - 6),
                 color, 6, cv2.LINE_AA)
        cv2.putText(image, label, (x0 + 84, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += row_h


def output_default(map_dir: str, local_model: str) -> str:
    return osp.join(map_dir, 'result',
                    'base_map_lanes_on_dom_{}.png'.format(local_model))


def preview_default(output: str) -> str:
    stem, _ = osp.splitext(output)
    return stem + '_preview.jpg'


def save_preview(image: np.ndarray, path: str, width: int):
    if width <= 0:
        return
    scale = min(1.0, float(width) / float(image.shape[1]))
    if scale < 1.0:
        preview = cv2.resize(image, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    else:
        preview = image
    params = []
    if path.lower().endswith(('.jpg', '.jpeg')):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    cv2.imwrite(path, preview, params)


def main():
    args = parse_args()
    map_dir = args.map_dir
    base_map = _default_path(args.base_map, map_dir, 'base_map.txt')
    lane_npz = _default_path(args.lane_npz, map_dir, 'base_map.txt.parsed.npz')
    dom_path = _default_path(args.dom, map_dir, 'result.tif')
    dom_json = args.dom_json
    if dom_json is None:
        candidate = osp.join(map_dir, 'result.json')
        dom_json = candidate if osp.exists(candidate) else None
    origin_path = _default_path(args.origin, map_dir, 'map_origin.yaml')
    output = args.output or output_default(map_dir, args.local_model)
    preview_output = args.preview_output or preview_default(output)

    if args.source == 'txt':
        lanes = load_lanes_from_txt(base_map)
    else:
        lanes = load_lanes_from_npz(lane_npz)
    if not lanes:
        raise RuntimeError('no lanes were loaded')

    origin_lat, origin_lon = read_origin(origin_path)
    dom_transform = read_dom_transform(dom_path, dom_json)

    image = cv2.imread(dom_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('failed to read DOM image: {}'.format(dom_path))

    overlay = np.zeros_like(image)
    stats = {
        'lanes': len(lanes),
        'central_drawn': 0,
        'left_drawn': 0,
        'right_drawn': 0,
        'points_total': 0,
        'points_inside': 0,
    }
    colors = {
        'left': (0, 0, 255),
        'central': (0, 255, 255),
        'right': (255, 0, 0),
    }

    for lane in lanes:
        for key in ('left', 'right', 'central'):
            local = lane.get(key)
            if local is None or len(local) < 2:
                continue
            lonlat = local_xy_to_lonlat(
                np.asarray(local, dtype=np.float64),
                origin_lat,
                origin_lon,
                args.local_model,
                utm_zone=args.utm_zone,
                utm_south=args.utm_south)
            pix = dom_transform.lonlat_to_pixel(lonlat)
            if args.pixel_shift_x or args.pixel_shift_y:
                pix[:, 0] += args.pixel_shift_x
                pix[:, 1] += args.pixel_shift_y
            stats['points_total'] += int(pix.shape[0])
            inside = ((pix[:, 0] >= 0) & (pix[:, 0] < image.shape[1]) &
                      (pix[:, 1] >= 0) & (pix[:, 1] < image.shape[0]))
            stats['points_inside'] += int(inside.sum())
            if not clip_visible(pix, image.shape[1], image.shape[0]):
                continue
            draw_polyline_layer(overlay, pix, colors[key],
                                args.line_thickness, args.halo_thickness)
            stats[key + '_drawn'] += 1

    alpha = min(1.0, max(0.0, float(args.alpha)))
    drawn = np.any(overlay != 0, axis=2)
    image[drawn] = cv2.addWeighted(
        image[drawn], 1.0 - alpha, overlay[drawn], alpha, 0.0)
    if not args.no_legend:
        draw_legend(image)

    os.makedirs(osp.dirname(output) or '.', exist_ok=True)
    cv2.imwrite(output, image)
    if args.preview_width > 0:
        save_preview(image, preview_output, args.preview_width)

    inside_ratio = (float(stats['points_inside']) /
                    float(stats['points_total'] or 1))
    print('source: {}'.format(args.source))
    print('local_model: {}'.format(args.local_model))
    print('lanes loaded: {}'.format(stats['lanes']))
    print('polylines drawn: central={} left={} right={}'.format(
        stats['central_drawn'], stats['left_drawn'], stats['right_drawn']))
    print('points inside DOM: {}/{} ({:.2%})'.format(
        stats['points_inside'], stats['points_total'], inside_ratio))
    print('output: {}'.format(output))
    if args.preview_width > 0:
        print('preview: {}'.format(preview_output))


if __name__ == '__main__':
    main()
