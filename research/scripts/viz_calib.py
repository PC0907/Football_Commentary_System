"""Overlay the pitch line model through a calibration onto a frame (visual check)."""
import sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcs.geometry.calib_provider import sequence_calibrations
from fcs.geometry.camera import apply_h
from fcs.tracking.data import SNMOT_ROOT, load_sequence
from fcs.geometry.keypoint_calib import correspondences
L, W = 105.0, 68.0
def pitch_lines():
    segs = [((0,0),(L,0)),((0,W),(L,W)),((0,0),(0,W)),((L,0),(L,W)),((L/2,0),(L/2,W)),
            ((0,13.84),(16.5,13.84)),((16.5,13.84),(16.5,54.16)),((0,54.16),(16.5,54.16)),
            ((L,13.84),(L-16.5,13.84)),((L-16.5,13.84),(L-16.5,54.16)),((L,54.16),(L-16.5,54.16)),
            ((0,24.84),(5.5,24.84)),((5.5,24.84),(5.5,43.16)),((0,43.16),(5.5,43.16)),
            ((L,24.84),(L-5.5,24.84)),((L-5.5,24.84),(L-5.5,43.16)),((L,43.16),(L-5.5,43.16))]
    a = np.linspace(0, 2*np.pi, 60); circ = np.c_[L/2 + 9.15*np.cos(a), W/2 + 9.15*np.sin(a)]
    segs += [(tuple(circ[i]), tuple(circ[i+1])) for i in range(59)]
    return segs
name, t, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seq = load_sequence(SNMOT_ROOT/'test'/name); c = np.load(f'cache/snmot/test/{name}.npz')
cal = sequence_calibrations(mode, c['kp'], c['M'], seq.width, seq.height)
img = cv2.imread(str(seq.img_path(t))); H = cal[t][0]
if H is not None:
    G = np.linalg.inv(H)
    for p, q in pitch_lines():
        P = apply_h(G, np.linspace(p, q, 20))
        for a, b in zip(P, P[1:]):
            if np.all(np.abs(np.r_[a, b]) < 5000):
                cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 2)
kp = c['kp'][t]
for x, y, s in kp:
    if s > 0.5: cv2.circle(img, (int(x), int(y)), 6, (255, 255, 0), -1)
src, dst, w = correspondences(kp)
for (x, y), (X, Y) in zip(src, dst):
    cv2.circle(img, (int(x), int(y)), 9, (0, 255, 255), 2)
    cv2.putText(img, f"{X:.0f},{Y:.0f}", (int(x)+8, int(y)-8), 0, 0.7, (0, 255, 255), 2)
out = f'outputs/debug/{name}_{t}_{mode}.jpg'
cv2.imwrite(out, cv2.resize(img, None, fx=0.5, fy=0.5)); print(out)
