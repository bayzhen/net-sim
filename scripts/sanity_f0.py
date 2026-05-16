"""Sanity-check the model at frame 0: with t_norm=0 the inputs are
literally the ball's initial state, and we should get position outputs
that match input_position[:2] (and z=1.5 always)."""
import json
import sys
import numpy as np
import torch
import h5py

sys.path.insert(0, '/data3/netsim/net-sim')
from model import GoalNetMLP, encode_input_features, input_dim_for

ckpt_path = '/data3/netsim/runs/mlp_v2/best.pt'
h5_path = '/data3/netsim/dataset_v2/dataset.h5'

device = torch.device('cuda')
state = torch.load(ckpt_path, map_location=device, weights_only=False)
cfg = state['config']
n_time_freq = cfg.get('n_time_freq', 4)
in_dim = input_dim_for(n_time_freq)
model = GoalNetMLP(
    in_dim=in_dim,
    n_particles=cfg['particle_count'],
    hidden=tuple(cfg['hidden']),
    activation=cfg.get('activation', 'gelu'),
    predict_velocity=True,
    dropout=0.0,
).to(device)
model.load_state_dict(state['model_state'])
model.eval()

f = h5py.File(h5_path, 'r')
clean = f['quality_clean'][:].astype(bool)
idx = np.flatnonzero(clean)[:8]
print(f'Inspecting {len(idx)} clean samples at frame 0:')

for i in idx:
    pos = f['input_position'][i]
    vel = f['input_velocity'][i]
    ang = f['input_angular'][i]
    rad = float(f['input_radius'][i])
    mas = float(f['input_mass'][i])
    bp_gt = f['ball_position'][i, 0]
    bv_gt = f['ball_velocity'][i, 0]
    
    # Sanity: at frame 0, ball_position should == input_position (it's
    # the initial state).
    print(f'\n  sample {i}:')
    print(f'    input_position = {pos}')
    print(f'    ball_pos[f=0]  = {bp_gt}     (delta = {np.linalg.norm(pos - bp_gt):.4f} m)')
    print(f'    input_velocity = {vel}')
    print(f'    ball_vel[f=0]  = {bv_gt}     (delta = {np.linalg.norm(vel - bv_gt):.4f} m/s)')
    
    # Run model
    pos_xy = torch.tensor(pos[:2], device=device, dtype=torch.float32).unsqueeze(0)
    vel_t = torch.tensor(vel, device=device, dtype=torch.float32).unsqueeze(0)
    ang_t = torch.tensor(ang, device=device, dtype=torch.float32).unsqueeze(0)
    rad_t = torch.tensor([rad], device=device, dtype=torch.float32)
    mas_t = torch.tensor([mas], device=device, dtype=torch.float32)
    t0 = torch.tensor([0.0], device=device, dtype=torch.float32)
    
    x = encode_input_features(pos_xy=pos_xy, vel=vel_t, ang=ang_t,
                               radius=rad_t, mass=mas_t, t_norm=t0,
                               n_time_freq=n_time_freq)
    with torch.no_grad():
        out = model(x, return_normalized=False)
    
    pred_pos = out['ball_pos'][0].cpu().numpy()
    pred_vel = out['ball_vel'][0].cpu().numpy()
    print(f'    pred_pos       = {pred_pos}     (err = {np.linalg.norm(pred_pos - bp_gt):.4f} m)')
    print(f'    pred_vel       = {pred_vel}     (err = {np.linalg.norm(pred_vel - bv_gt):.4f} m/s)')

# Print model normalization stats summary
print('\n--- model normalization stats ---')
print('in_mean[:6]:', model.in_mean[:6].cpu().numpy())
print('in_std [:6]:', model.in_std[:6].cpu().numpy())
print('ball_pos_mean:', model.ball_pos_mean.cpu().numpy())
print('ball_pos_std :', model.ball_pos_std.cpu().numpy())
print('ball_vel_mean:', model.ball_vel_mean.cpu().numpy())
print('ball_vel_std :', model.ball_vel_std.cpu().numpy())
