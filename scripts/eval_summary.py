"""Quick per-frame RMSE inspection."""
import json
import sys

d = json.load(open(sys.argv[1]))
F = len(d['ball_pos_rmse'])
dt = d['frame_dt']
print(f'F={F}, dt={dt}, total duration={F*dt:.3f}s')
print('frame -> ball_pos RMSE (m), ball_vel RMSE (m/s), net RMSE (mm):')
for f in [0, 30, 60, 90, 120, 150, 180, 210, 240, 300, 360, 420, 480, 540, 600]:
    if f < F:
        print(f'  f={f:3d}  t={f*dt:.3f}s   '
              f'ball_pos={d["ball_pos_rmse"][f]:6.3f}m  '
              f'ball_vel={d["ball_vel_rmse"][f]:5.3f}  '
              f'net={d["net_rmse"][f]*1000:5.2f}mm')
