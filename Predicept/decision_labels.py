import numpy as np
import torch

from .channels import _corridor_arrays, _project

LON_CLASSES = ['remain_stopped', 'stop_quickly', 'stop_gently', 'slow_quickly', 'slow_gently',
               'accel_quickly', 'accel_gently', 'maintain', 'reverse']
LAT_CLASSES = ['turn_left', 'turn_right', 'lane_change_left', 'lane_change_right',
               'inlane_left', 'inlane_right', 'no_lateral']
NUM_LON = len(LON_CLASSES)
NUM_LAT = len(LAT_CLASSES)

LON_CE_WEIGHT = [0.49, 10.0, 5.80, 9.34, 1.80, 1.15, 0.42, 0.35, 1.0]
LAT_CE_WEIGHT = [0.47, 0.97, 10.0, 9.82, 5.46, 4.64, 0.31]

LON_MERGE_MAP = [0, 1, 1, 2, 2, 3, 3, 4, 5]
LON_MERGED_CLASSES = ['remain_stopped', 'stop', 'slow', 'accel', 'maintain', 'reverse']
NUM_LON_MERGED = len(LON_MERGED_CLASSES)
LON_MERGED_CE_WEIGHT = [0.74, 7.31, 2.26, 0.46, 0.53, 1.0]

# ---------------------------------------------------------------------------
LON5_MAP = [0, 1, 1, 2, 2, 3, 3, 4, 0]
LON5_CLASSES = ['remain_stopped', 'stop', 'slow', 'accel', 'maintain']
NUM_LON5 = len(LON5_CLASSES)
LON5_CE_WEIGHT = [0.89, 8.77, 2.71, 0.55, 0.63]
LAT5_MAP = [0, 1, 2, 3, 4, 4, 4]
LAT5_CLASSES = ['turn_left', 'turn_right', 'lane_change_left', 'lane_change_right', 'none']
NUM_LAT5 = len(LAT5_CLASSES)
LAT5_CE_WEIGHT = [0.65, 1.35, 10.0, 10.0, 0.39]

LON5_FAMILY = [1, 0, 0, 2, 2]
LON_FAMILY_NAMES = ['brake', 'hold', 'cruise']
LAT5_FAMILY = [0, 1, 0, 1, 2]
LAT_FAMILY_NAMES = ['left', 'right', 'none']
NUM_FAMILIES = 3

# ---------------------------------------------------------------------------
LON4_MAP = [0, 0, 0, 1, 1, 2, 2, 3, 0]
LON4_CLASSES = ['stop', 'slow', 'accel', 'maintain']
NUM_LON4 = len(LON4_CLASSES)
LON4_CE_WEIGHT = [1.01, 3.39, 0.69, 0.79]
LAT5V_MAP = [0, 1, 2, 3, 2, 3, 4]
LAT5V_CLASSES = ['turn_left', 'turn_right', 'to_left', 'to_right', 'none']
NUM_LAT5V = len(LAT5V_CLASSES)
LAT5V_CE_WEIGHT = [0.65, 1.35, 5.28, 4.41, 0.43]

LAT5L_MAP = [0, 1, 2, 3, 4, 4, 4]
LAT5L_CLASSES = ['turn_left', 'turn_right', 'lane_change_left', 'lane_change_right', 'none']
NUM_LAT5L = len(LAT5L_CLASSES)
LAT5L_CE_WEIGHT = [0.65, 1.35, 10.0, 10.0, 0.39]

DT = 0.1
LON_WINDOW = 40
V_STOP = 0.5
V_MOVING = 1.0
DV_BAND = 1.0
A_HARD = 1.5
A_SMOOTH_W = 5
REV_X = -0.5
LC_DLAT = 2.0
INLANE_DLAT = 0.6
LC_PAR_RAD = 0.26
LC_NET_ROT = 0.35
MIN_ARC = 3.0
TURN_C_LO, TURN_C_HI = 0.03, 0.18
TURN_HDIFF = 0.2


def _resample_arc(xy, n):
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.repeat(xy[:1], n, axis=0)
    cum = cum / cum[-1]
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])], axis=1)


def _turn_class(xy, yaw):
    valid = ~np.all(xy == 0, axis=1)
    xy, yaw = xy[valid], yaw[valid]
    if len(xy) < 2:
        return None
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if length < MIN_ARC:
        return None
    pts = _resample_arc(xy, max(int(length), 2))
    tan = np.diff(pts, axis=0)
    tan = tan / np.clip(np.linalg.norm(tan, axis=1, keepdims=True), 1e-8, None)
    if len(tan) < 2:
        return None
    ang = np.arccos(np.clip((tan[:-1] * tan[1:]).sum(1), -1.0, 1.0))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    curv = ang / np.clip(seg[:-1], 1e-8, None)
    sign = np.sign(np.cross(tan[:-1], tan[1:]))
    i = int(np.argmax(curv))
    c = round(float(curv[i]), 2)
    s = float(sign[i])
    diff = round(float(abs(yaw[0] - yaw[-1])), 2)
    turning = (TURN_C_LO < c < TURN_C_HI and diff > TURN_HDIFF) or (0.1 < c < TURN_C_HI)
    uturn = c >= TURN_C_HI
    if not (turning or uturn):
        return None
    if s == 1.0:
        return 'left'
    if s == -1.0:
        return 'right'
    return None


def _lon_one(xy, yaw):
    w = xy[:LON_WINDOW]
    valid = ~np.all(w == 0, axis=1)
    if valid.sum() < 2:
        return LON_CLASSES.index('remain_stopped')
    w = w[valid]
    v = np.linalg.norm(np.diff(w, axis=0), axis=1) / DT                     # [T-1]
    v0 = float(v[:5].mean()) if len(v) >= 5 else float(v.mean())
    v_end = float(v[-5:].mean()) if len(v) >= 5 else float(v.mean())
    if w[-1, 0] < REV_X and v.max() > V_STOP:
        return LON_CLASSES.index('reverse')
    if v.max() < V_MOVING and v0 < V_STOP and v_end < V_STOP:
        return LON_CLASSES.index('remain_stopped')
    a = np.diff(v) / DT
    if len(a) >= A_SMOOTH_W:
        kern = np.ones(A_SMOOTH_W) / A_SMOOTH_W
        a = np.convolve(a, kern, mode='valid')
    hard = bool(np.abs(a).max() >= A_HARD) if len(a) else False
    dv = v_end - v0
    if v0 >= V_MOVING and v_end < V_STOP:
        return LON_CLASSES.index('stop_quickly' if hard else 'stop_gently')
    if dv <= -DV_BAND:
        return LON_CLASSES.index('slow_quickly' if hard else 'slow_gently')
    if dv >= DV_BAND:
        return LON_CLASSES.index('accel_quickly' if hard else 'accel_gently')
    return LON_CLASSES.index('maintain')


def _lat_one(xy, yaw, dlat=None, dth=None):
    turn = _turn_class(xy, yaw)
    if dlat is not None and dth is not None:
        v = ~np.all(xy == 0, axis=1)
        if v.sum() >= 2:
            dl, dt = dlat[v], dth[v]
            k = max(len(dl) // 8, 1)
            delta_c = float(np.median(dl[-k:]) - np.median(dl[:k]))
            end_par = abs(float(np.median(dt[-k:]))) <= LC_PAR_RAD
            yv = yaw[v]
            net = float(yv[-1] - yv[0])
            net = abs(np.arctan2(np.sin(net), np.cos(net)))
            if abs(delta_c) >= LC_DLAT and end_par and net <= LC_NET_ROT:
                return LAT_CLASSES.index('lane_change_left' if delta_c > 0
                                         else 'lane_change_right')
    if turn is not None:
        return LAT_CLASSES.index('turn_left' if turn == 'left' else 'turn_right')
    valid = ~np.all(xy == 0, axis=1)
    if valid.sum() < 2:
        return LAT_CLASSES.index('no_lateral')
    length = float(np.linalg.norm(np.diff(xy[valid], axis=0), axis=1).sum())
    if length < MIN_ARC:
        return LAT_CLASSES.index('no_lateral')
    if dlat is not None:
        dl = dlat[valid]
        delta = float(np.median(dl[-max(len(dl) // 8, 1):]) - np.median(dl[:max(len(dl) // 8, 1)]))
    else:
        if abs(float(yaw[valid][-1] - yaw[valid][0])) > TURN_HDIFF:
            return LAT_CLASSES.index('no_lateral')
        delta = float(xy[valid][-1, 1] - xy[valid][0, 1])
    if delta >= LC_DLAT:
        return LAT_CLASSES.index('lane_change_left')
    if delta <= -LC_DLAT:
        return LAT_CLASSES.index('lane_change_right')
    if delta >= INLANE_DLAT:
        return LAT_CLASSES.index('inlane_left')
    if delta <= -INLANE_DLAT:
        return LAT_CLASSES.index('inlane_right')
    return LAT_CLASSES.index('no_lateral')


def decision_labels(ego_future, ref_path=None, turn_fix=True):
    ef = ego_future.detach().cpu().numpy()
    B = ef.shape[0]
    dlat_np = dth_np = None
    if ref_path is not None:
        cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
        pts = ego_future[..., :2].float().to(ref_path.device)               # [B,80,2]
        _, d_lat, tyaw, _ = _project(pts, cxy, cyaw, ccum, cvalid)          # [B,80]
        has_corr = cvalid.any(dim=1).cpu().numpy()
        dlat_np = d_lat.detach().cpu().numpy()
        if turn_fix:
            dth = ego_future[..., 2].float().to(ref_path.device) - tyaw
            dth = torch.atan2(torch.sin(dth), torch.cos(dth))
            dth_np = dth.detach().cpu().numpy()
    lon, lat = [], []
    for b in range(B):
        lo = _lon_one(ef[b, :, :2], ef[b, :, 2])
        dl = dlat_np[b] if (dlat_np is not None and has_corr[b]) else None
        dt = dth_np[b] if (dth_np is not None and has_corr[b]) else None
        la = _lat_one(ef[b, :, :2], ef[b, :, 2], dl, dt)
        lon.append(lo)
        lat.append(la)
    dev = ego_future.device
    return (torch.tensor(lon, dtype=torch.long, device=dev),
            torch.tensor(lat, dtype=torch.long, device=dev))


def decision_labels_single(ego_future_np, ref_path_np):
    ef = torch.from_numpy(np.ascontiguousarray(ego_future_np)).float().unsqueeze(0)
    rp = torch.from_numpy(np.ascontiguousarray(ref_path_np)).float().unsqueeze(0)
    lon, lat = decision_labels(ef, rp)
    return int(lon[0]), int(lat[0])
