"""Frozen source-compatible terrain collider path for the radial five-pose smoke."""

def terrain_mesh_path(import_root: str) -> str:
    """IsaacLab create_prim_from_mesh creates Xform at root and CollisionAPI Mesh child."""
    if not import_root.startswith('/World/') or import_root.endswith('/'):
        raise ValueError('invalid terrain import root')
    return import_root + '/mesh'
