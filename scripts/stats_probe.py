"""Quick statistics dump on a clean subset to design normalization."""
import h5py
import numpy as np

f = h5py.File('/data3/netsim/dataset_v2/dataset.h5', 'r')
clean = f['quality_clean'][:].astype(bool)
idx = np.flatnonzero(clean)[:2000]

print('--- input_position (S, 3) ---')
a = f['input_position'][:][idx]
print('mean', a.mean(0))
print('std ', a.std(0))
print('min ', a.min(0), 'max', a.max(0))

print('--- input_velocity ---')
a = f['input_velocity'][:][idx]
print('mean', a.mean(0))
print('std ', a.std(0))
print('min ', a.min(0), 'max', a.max(0))

print('--- input_angular ---')
a = f['input_angular'][:][idx]
print('mean', a.mean(0))
print('std ', a.std(0))

print('--- input_radius ---')
a = f['input_radius'][:][idx]
print(f'mean={a.mean():.4f} std={a.std():.4f} range=[{a.min():.3f}, {a.max():.3f}]')

print('--- input_mass ---')
a = f['input_mass'][:][idx]
print(f'mean={a.mean():.4f} std={a.std():.4f} range=[{a.min():.3f}, {a.max():.3f}]')

print('--- ball_position over all frames (200 samples) ---')
a = f['ball_position'][idx[:200]].reshape(-1, 3)
print('mean', a.mean(0))
print('std ', a.std(0))
print('min ', a.min(0), 'max', a.max(0))

print('--- ball_velocity over all frames (200 samples) ---')
a = f['ball_velocity'][idx[:200]].reshape(-1, 3)
print('mean', a.mean(0))
print('std ', a.std(0))
print('min ', a.min(0), 'max', a.max(0))

print('--- particle_position over all frames (50 samples) ---')
a = f['particle_position'][idx[:50]]
print('shape', a.shape)
print(f'global mean={a.mean():.4f} std={a.std():.4f} '
      f'range=[{a.min():.3f}, {a.max():.3f}]')
print('per-axis mean', a.reshape(-1, 3).mean(0))
print('per-axis std ', a.reshape(-1, 3).std(0))

print('--- particle rest pos (channel-mean over the topology) ---')
import json
topo = json.loads(f.attrs['topology_json'])
rest = np.asarray(topo['rest_positions'], dtype=np.float32)
print('rest shape', rest.shape, 'mean', rest.mean(0), 'std', rest.std(0))

print('--- net displacement (particle_pos - rest) ---')
disp = a - rest[None, None, :, :]
print(f'disp std={disp.std():.4f}, max abs={np.abs(disp).max():.3f}')
