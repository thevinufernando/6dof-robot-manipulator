"""Minimal binary/ASCII STL helpers (numpy only, no trimesh needed)."""
import struct
import numpy as np


def load_stl(path):
    """Return triangles as an (N, 3, 3) float array [triangle, vertex, xyz]."""
    with open(path, 'rb') as f:
        data = f.read()
    n = struct.unpack('<I', data[80:84])[0]
    if 84 + 50 * n == len(data):  # binary STL
        rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype(
            [('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        return rec['v'].astype(float)
    # ASCII fallback
    verts = [list(map(float, ln.split()[1:4])) for ln in data.decode().splitlines()
             if ln.strip().startswith('vertex')]
    return np.array(verts, dtype=float).reshape(-1, 3, 3)


def save_stl(path, tris, header='gp7_description'):
    """Write (N, 3, 3) triangles as a binary STL with computed facet normals."""
    tris = np.asarray(tris, dtype=np.float32)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-20
    rec = np.zeros(len(tris), dtype=np.dtype(
        [('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = nrm, tris
    with open(path, 'wb') as f:
        f.write(header.encode()[:80].ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        f.write(rec.tobytes())


def unique_vertices(tris, decimals=6):
    return np.unique(tris.reshape(-1, 3).round(decimals), axis=0)
