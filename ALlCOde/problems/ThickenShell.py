import os.path
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import pyvista as pv
try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass
import torch

from .problemBase import problemBase


class ThickenShell(problemBase):
    problemName = 'ThickenShell'

    def __init__(
        self,
        thickness,
        BC_dir,
        Load_magnitude,
        voxel_size,
        extra_layers=1,
        tensors=None,
        tangential_tol=None,
        load_case="tensile_compression",
        load_dir=None,
        load_surface_dir=None,
        load_surface_side="max",
        fixed_side="min",
        force_side=None,
    ):
        super().__init__()
        self.name = self.problemName

        self.brep_bbox = None
        self.thickness = float(thickness)
        self.face_bboxes = None
        self.samples_by_face = None
        self.voxel_size = float(voxel_size)
        self.extra_layers = int(extra_layers)
        self.tangential_tol = None if tangential_tol is None else float(tangential_tol)
        self.BC_dir = str(BC_dir).lower()
        self.Load_magnitude = float(Load_magnitude)
        self.load_case = str(load_case).lower()
        self.load_dir = None if load_dir is None else str(load_dir).lower()
        self.load_surface_dir = None if load_surface_dir is None else str(load_surface_dir).lower()
        self.load_surface_side = str(load_surface_side).lower()
        self.fixed_side = str(fixed_side).lower()
        self.force_side = None if force_side is None else str(force_side).lower()

        self.grid_geom = None
        self.elem_centers = None
        self.node_coords = None

        self.elem_sample_idx = None
        self.sample_elem_idx = None
        self.elem_sample_count = None
        self.elem_fiber = None
        self.elem_phi = None
        self.elem_theta = None

        self.uv = None
        self.points_xyz = None
        self.face_areas = None
        self.Xu = None
        self.Xv = None
        self.faces_ijk = None
        self.pv_faces = None
        self.face_id = None
        self.boundary_idx_ring1 = None
        self.min_vol_frac = None
        self.sample_normals = None

        self.elem_occupancy = None
        self.elem_density = None
        self._occupied_voxel_mesh_cache = None
        self._surface_cloud_cache = None

        if tensors is None:
            raise ValueError("tensors must be provided at this stage")

        # Get CAD info about the shell geometry and samples from the tensors. Sample: points_xyz, Xu, Xv, face_areas, etc. Also parse the bounding box of the BREP geometry.
        self.set_cad_samples(tensors)
   

        # Build a full structured voxel grid that covers the padded bounding box of the shell.
        # This creates a rectangular grid (mesh), initializes empty boundary conditions,
        # and defines material properties. At this stage NO voxels are trimmed or filtered
        # based on the shell geometry — the grid is still a complete box. The actual shell
        # shape is imposed later by voxelize_shell_from_samples(), which marks which voxels
        # belong to the shell thickness via elem_occupancy.
        self.mesh, self.boundaryCondition, self.materialProperty = self.shellSettings()



        # Voxelize the shell midsurface samples into the structured voxel grid.
        # For each voxel center, we find the nearest sampled point on the CAD surface
        # The distance from the voxel center to the surface sample is decomposed into:
        #   - normal distance (dn): distance along the surface normal
        #   - tangential distance (dt): distance in the surface tangent plane
        # A voxel is marked as part of the shell if:
        #   dn <= thickness/2                (within shell thickness)
        #   dt <= tangential_tol             (close enough along the surface)
        #   de <= max_euclid                 (overall distance safety bound)
        # Outputs:
        #   occ        : voxel occupancy grid (1 = shell voxel, 0 = empty)
        #   sample_idx : index of the nearest surface sample for each voxel
        self.elem_occupancy,self.elem_sample_idx = self.voxelize_shell_from_samples(self.thickness,tangential_tol=self.tangential_tol)
        self.sample_elem_idx = self.surface_samples_to_element_indices()
        self.ensure_surface_sample_elements_are_occupied()
        self.elem_sample_count = self.count_surface_samples_per_element()


        # Create a simple binary density field based on voxel occupancy.
        # Occupied voxels (shell) get density ≈ 1, empty voxels get a small void density rho_min.
        # This is only an initialization useful for debugging / visualization.
        # The actual density and fiber fields used in the FEM solve will later come
        # from the neural decoder and can be assigned via assign_decoder_fields().
        rho_min = 1e-3
        self.elem_density = rho_min + (1.0 - rho_min) * self.elem_occupancy.reshape(-1).astype(np.float32)


        self.apply_load_case_boundary_conditions()

    def apply_load_case_boundary_conditions(self):
        """
        Dispatch boundary-condition construction by load-case category.

        Current implemented case:
        - tensile_compression: fixed slab on the negative side of BC_dir and
          loaded slab on the positive side of BC_dir. The sign of
          Load_magnitude decides tension/compression.
        """
        if self.load_case in ("tensile_compression", "tensile", "compression"):
            self.apply_tensile_compression_boundary_conditions()
            return

        if self.load_case in ("three_point_bending", "threepoint_bending", "3_point_bending"):
            self.apply_three_point_bending_boundary_conditions()
            return

        if self.load_case in ("torsion", "twist", "torque"):
            self.apply_torsion_boundary_conditions()
            return

        raise ValueError(
            f"Unsupported load_case: {self.load_case}. "
            "Currently supported: tensile_compression, three_point_bending, torsion"
        )

    def _axis_bounds_keys(self, axis):
        axis = str(axis).lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Unsupported BC_dir: {axis}")
        return f"{axis}min", f"{axis}max"

    def select_axis_end_slab_nodes(self, axis, side, bbox, tol):
        lo_key, hi_key = self._axis_bounds_keys(axis)
        side = str(side).lower()

        if side == "min":
            lo = bbox[lo_key]
            hi = bbox[lo_key] + tol
        elif side == "max":
            lo = bbox[hi_key] - tol
            hi = bbox[hi_key]
        else:
            raise ValueError(f"Unsupported slab side: {side}")

        kwargs = {lo_key: lo, hi_key: hi}
        return self.select_nodes_in_box(**kwargs)

    def select_axis_middle_slab_nodes(self, axis, bbox, tol):
        lo_key, hi_key = self._axis_bounds_keys(axis)
        center = 0.5 * (bbox[lo_key] + bbox[hi_key])
        half_width = 0.5 * tol
        kwargs = {
            lo_key: center - half_width,
            hi_key: center + half_width,
        }
        return self.select_nodes_in_box(**kwargs)

    def select_shell_axis_slab_nodes(self, shell_nodes, axis, side, tol, max_expand=8):
        axis = str(axis).lower()
        comp_map = {"x": 0, "y": 1, "z": 2}
        if axis not in comp_map:
            raise ValueError(f"Unsupported axis: {axis}")

        shell_nodes = np.asarray(shell_nodes, dtype=np.int64).reshape(-1)
        if shell_nodes.size == 0:
            return shell_nodes

        _node_ids, coords = self.get_flat_node_coords()
        values = coords[shell_nodes, comp_map[axis]]
        side = str(side).lower()
        lo = float(values.min())
        hi = float(values.max())

        for scale in range(1, int(max_expand) + 1):
            width = float(tol) * float(scale)
            if side == "min":
                mask = values <= lo + width
            elif side == "max":
                mask = values >= hi - width
            elif side == "middle":
                center = 0.5 * (lo + hi)
                mask = np.abs(values - center) <= 0.5 * width
            else:
                raise ValueError(f"Unsupported slab side: {side}")

            selected = shell_nodes[mask]
            if selected.size > 0:
                return selected

        if side == "min":
            target = lo
        elif side == "max":
            target = hi
        else:
            target = 0.5 * (lo + hi)

        nearest = np.argmin(np.abs(values - target))
        return shell_nodes[[nearest]]

    def filter_shell_nodes_by_surface(self, shell_nodes, axis, side, tol, max_expand=8):
        axis = str(axis).lower()
        comp_map = {"x": 0, "y": 1, "z": 2}
        if axis not in comp_map:
            raise ValueError(f"Unsupported surface axis: {axis}")

        shell_nodes = np.asarray(shell_nodes, dtype=np.int64).reshape(-1)
        if shell_nodes.size == 0:
            return shell_nodes

        _node_ids, coords = self.get_flat_node_coords()
        values = coords[shell_nodes, comp_map[axis]]
        side = str(side).lower()
        lo = float(values.min())
        hi = float(values.max())

        for scale in range(1, int(max_expand) + 1):
            width = float(tol) * float(scale)
            if side == "min":
                mask = values <= lo + width
            elif side == "max":
                mask = values >= hi - width
            else:
                raise ValueError(f"Unsupported surface side: {side}")

            selected = shell_nodes[mask]
            if selected.size > 0:
                return selected

        target = lo if side == "min" else hi
        nearest = np.argmin(np.abs(values - target))
        return shell_nodes[[nearest]]

    @staticmethod
    def _opposite_side(side):
        side = str(side).lower()
        if side == "min":
            return "max"
        if side == "max":
            return "min"
        raise ValueError(f"Unsupported side: {side}. Expected 'min' or 'max'.")

    def apply_tensile_compression_boundary_conditions(self):
        tol = 0.5* self.voxel_size
        shell_nodes = self.occupied_node_ids()

        fixed_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "min", tol)
        force_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "max", tol)

        if fixed_nodes.size == 0:
            raise ValueError(
                f"No fixed shell nodes selected for load_case={self.load_case}, BC_dir={self.BC_dir}"
            )
        if force_nodes.size == 0:
            raise ValueError(
                f"No force shell nodes selected for load_case={self.load_case}, BC_dir={self.BC_dir}"
            )

        self.set_boundary_conditions_from_regions(
            fixed_nodes=fixed_nodes,
            force_nodes=force_nodes,
            force_direction=self.BC_dir,
            force_value=self.Load_magnitude,
        )

    def apply_three_point_bending_boundary_conditions(self):
        load_dir = self.load_dir
        if load_dir is None:
            raise ValueError("load_dir must be provided for load_case='three_point_bending'")
        self._axis_bounds_keys(load_dir)

        tol = 0.5 * self.voxel_size
        shell_nodes = self.occupied_node_ids()

        min_support_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "min", tol)
        max_support_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "max", tol)
        force_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, "middle", tol)
        if self.load_surface_dir is not None:
            force_nodes = self.filter_shell_nodes_by_surface(
                force_nodes,
                self.load_surface_dir,
                self.load_surface_side,
                tol,
            )
        fixed_nodes = np.union1d(min_support_nodes, max_support_nodes)

        if fixed_nodes.size == 0:
            raise ValueError(
                f"No support shell nodes selected for load_case={self.load_case}, span BC_dir={self.BC_dir}"
            )
        if force_nodes.size == 0:
            raise ValueError(
                f"No middle load shell nodes selected for load_case={self.load_case}, span BC_dir={self.BC_dir}"
            )

        self.set_boundary_conditions_from_regions(
            fixed_nodes=fixed_nodes,
            force_nodes=force_nodes,
            force_direction=load_dir,
            force_value=self.Load_magnitude,
        )

    def apply_torsion_boundary_conditions(self):
        self._axis_bounds_keys(self.BC_dir)
        fixed_side = self.fixed_side
        force_side = self.force_side if self.force_side is not None else self._opposite_side(fixed_side)
        if force_side == fixed_side:
            raise ValueError("force_side must be opposite to fixed_side for load_case='torsion'")

        tol = 0.5 * self.voxel_size
        shell_nodes = self.occupied_node_ids()

        fixed_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, fixed_side, tol)
        torque_nodes = self.select_shell_axis_slab_nodes(shell_nodes, self.BC_dir, force_side, tol)

        if fixed_nodes.size == 0:
            raise ValueError(
                f"No fixed shell nodes selected for load_case={self.load_case}, "
                f"BC_dir={self.BC_dir}, fixed_side={fixed_side}"
            )
        if torque_nodes.size == 0:
            raise ValueError(
                f"No torque shell nodes selected for load_case={self.load_case}, "
                f"BC_dir={self.BC_dir}, force_side={force_side}"
            )

        self.set_torsion_boundary_conditions(
            fixed_nodes=fixed_nodes,
            torque_nodes=torque_nodes,
            torque_axis=self.BC_dir,
            total_torque=self.Load_magnitude,
        )

    def shellSettings(self):
        mesh, grid_geom, elem_centers, node_coords = self.build_voxel_grid_for_shell(
            self.brep_bbox,
            self.thickness,
            self.voxel_size,
            self.extra_layers
        )

        self.grid_geom = grid_geom
        self.elem_centers = elem_centers
        self.node_coords = node_coords

        # matProp = {
        #     'E': 1.0,
        #     'nu': 0.3,
        #     'Ef': 1.0,
        #     'Et': 1.0,
        #     'nuf': 0.3,
        #     'nut': 0.3,
        #     'penal': 3
        # }
        matProp = {
            'E': 1.0,
            'nu': 0.3,
            'Ef': 10.0,
            'Et': 1.0,
            'nuf': 0.25,
            'nut': 0.3,
            'penal': 3
        }

        ndof = 3 * (mesh['nelx'] + 1) * (mesh['nely'] + 1) * (mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)
        fixed = np.array([], dtype=np.int64)

        bc = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

        return mesh, bc, matProp

    def to_numpy(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _normalize_single_bbox(self, bbox):
        required = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if not isinstance(bbox, dict) or not all(k in bbox for k in required):
            raise ValueError(f"Unsupported single-face BBX format: {bbox}")
        return {
            'xmin': float(bbox['xmin']),
            'xmax': float(bbox['xmax']),
            'ymin': float(bbox['ymin']),
            'ymax': float(bbox['ymax']),
            'zmin': float(bbox['zmin']),
            'zmax': float(bbox['zmax']),
        }
    def parse_bbox(self, bbox_raw):
        """
        Accept either:
        - {'xmin':..., 'xmax':..., 'ymin':..., 'ymax':..., 'zmin':..., 'zmax':...}
        - {0: {...}, 1: {...}, ...}

        Returns
        -------
        union_bbox : dict
            Global union bounding box across all faces.
        face_bboxes : dict[int, dict]
            Per-face bounding boxes. For a single-face input, face id 0 is used.
        """
        if not isinstance(bbox_raw, dict):
            raise ValueError(f"Unsupported BBX format: {bbox_raw}")

        required = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if all(k in bbox_raw for k in required):
            b = self._normalize_single_bbox(bbox_raw)
            return b, {0: b.copy()}

        face_bboxes = {}
        for fid, bbox in bbox_raw.items():
            if not isinstance(bbox, dict):
                raise ValueError(f"Unsupported BBX entry for face {fid}: {bbox}")
            face_bboxes[int(fid)] = self._normalize_single_bbox(bbox)

        if len(face_bboxes) == 0:
            raise ValueError(f"Unsupported empty BBX format: {bbox_raw}")

        union_bbox = {
            'xmin': min(b['xmin'] for b in face_bboxes.values()),
            'xmax': max(b['xmax'] for b in face_bboxes.values()),
            'ymin': min(b['ymin'] for b in face_bboxes.values()),
            'ymax': max(b['ymax'] for b in face_bboxes.values()),
            'zmin': min(b['zmin'] for b in face_bboxes.values()),
            'zmax': max(b['zmax'] for b in face_bboxes.values()),
        }
        return union_bbox, face_bboxes

    def set_cad_samples(self, tensors):
        self.uv = self.to_numpy(tensors["uv"])
        self.points_xyz = self.to_numpy(tensors["points_xyz"]).reshape(-1, 3)
        self.face_areas = self.to_numpy(tensors["face_areas"])
        self.Xu = self.to_numpy(tensors["Xu"]).reshape(-1, 3)
        self.Xv = self.to_numpy(tensors["Xv"]).reshape(-1, 3)
        self.faces_ijk = self.to_numpy(tensors["faces_ijk"])
        self.pv_faces = self.to_numpy(tensors["pv_faces"])
        self.face_id = self.to_numpy(tensors["face_id"]).reshape(-1).astype(np.int64)
        self.boundary_idx_ring1 = self.to_numpy(tensors["boundary_idx_ring1"])
        self.min_vol_frac = self.to_numpy(tensors["min_vol_frac"])

        self.sample_normals = self.compute_sample_normals(self.Xu, self.Xv)

        bbox_raw = tensors["BBX"]
        self.brep_bbox, self.face_bboxes = self.parse_bbox(bbox_raw)

        self.samples_by_face = {}
        if self.face_id.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("face_id must have same length as points_xyz")
        for fid in np.unique(self.face_id):
            self.samples_by_face[int(fid)] = np.flatnonzero(self.face_id == fid).astype(np.int64)

    def occupied_node_ids(self):
        occ = self.elem_occupancy.astype(bool)   # shape (nelz, nelx, nely)
        nelz, nelx, nely = occ.shape

        node_mask = np.zeros((nelz + 1, nelx + 1, nely + 1), dtype=bool)

        for k in range(nelz):
            for i in range(nelx):
                for j in range(nely):
                    if occ[k, i, j]:
                        node_mask[k:k+2, i:i+2, j:j+2] = True

        node_ids = np.arange(node_mask.size, dtype=np.int64).reshape(node_mask.shape)
        return node_ids[node_mask]    
    def intersect_node_sets(self, a, b):
        return np.intersect1d(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))

    def compute_sample_normals(self, Xu, Xv, eps=1e-12):
        normals = np.cross(Xu, Xv)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(norm, eps, None)
        return normals

    def voxel_center_is_near_bbox(self, center, bbox, margin):
        return (
            (bbox['xmin'] - margin <= center[0] <= bbox['xmax'] + margin) and
            (bbox['ymin'] - margin <= center[1] <= bbox['ymax'] + margin) and
            (bbox['zmin'] - margin <= center[2] <= bbox['zmax'] + margin)
        )

    def candidate_face_ids_for_center(self, center, margin):
        if not self.face_bboxes:
            return []
        return [
            fid for fid, bbox in self.face_bboxes.items()
            if self.voxel_center_is_near_bbox(center, bbox, margin)
        ]
    def voxelize_shell_from_samples(self, thickness, tangential_tol=None):
        centers = self.elem_centers.reshape(-1, 3)
        points = self.points_xyz
        normals = self.sample_normals

        if tangential_tol is None:
            tangential_tol = 0.35 * self.voxel_size

        half_t = 0.5 * thickness
        max_euclid = np.sqrt(half_t * half_t + tangential_tol * tangential_tol)
        bbox_margin = half_t + tangential_tol + self.voxel_size

        occ = np.zeros((centers.shape[0],), dtype=np.uint8)
        sample_idx = -np.ones((centers.shape[0],), dtype=np.int64)

        for i, x in enumerate(centers):
            candidate_face_ids = self.candidate_face_ids_for_center(x, bbox_margin)

            if candidate_face_ids:
                candidate_idx = np.concatenate([
                    self.samples_by_face[fid] for fid in candidate_face_ids if fid in self.samples_by_face
                ])
            else:
                candidate_idx = np.arange(points.shape[0], dtype=np.int64)

            if candidate_idx.size == 0:
                continue

            candidate_points = points[candidate_idx]
            diff = candidate_points - x[None, :]
            dist2 = np.einsum('ij,ij->i', diff, diff)
            local_j = np.argmin(dist2)
            j = candidate_idx[local_j]

            p = points[j]
            n = normals[j]

            r = x - p
            de = np.linalg.norm(r)
            dn = abs(np.dot(r, n))
            rt = r - np.dot(r, n) * n
            dt = np.linalg.norm(rt)

            if dn <= half_t and dt <= tangential_tol and de <= max_euclid:
                occ[i] = 1
                sample_idx[i] = j

        occ = occ.reshape(self.elem_centers.shape[:3])
        sample_idx = sample_idx.reshape(self.elem_centers.shape[:3])

        return occ, sample_idx

    def padded_bbox_from_midsurface(self, bbox, thickness, voxel_size, extra_layers=1):
        pad = thickness / 2.0 + extra_layers * voxel_size

        return {
            'xmin': bbox['xmin'] - pad,
            'xmax': bbox['xmax'] + pad,
            'ymin': bbox['ymin'] - pad,
            'ymax': bbox['ymax'] + pad,
            'zmin': bbox['zmin'] - pad,
            'zmax': bbox['zmax'] + pad,
        }

    def structured_grid_from_bbox(self, bbox, voxel_size):
        hx = hy = hz = float(voxel_size)

        lx = bbox['xmax'] - bbox['xmin']
        ly = bbox['ymax'] - bbox['ymin']
        lz = bbox['zmax'] - bbox['zmin']

        nelx = int(math.ceil(lx / hx))
        nely = int(math.ceil(ly / hy))
        nelz = int(math.ceil(lz / hz))

        mesh = {
            'nelx': nelx,
            'nely': nely,
            'nelz': nelz,
            'elemSize': np.array([hx, hy, hz], dtype=float),
            'type': 'grid'
        }

        grid_geom = {
            'xmin': bbox['xmin'],
            'ymin': bbox['ymin'],
            'zmin': bbox['zmin'],
            'hx': hx,
            'hy': hy,
            'hz': hz
        }

        return mesh, grid_geom

    def element_centers(self, mesh, grid_geom):
        nelx, nely, nelz = mesh['nelx'], mesh['nely'], mesh['nelz']
        hx, hy, hz = grid_geom['hx'], grid_geom['hy'], grid_geom['hz']
        xmin, ymin, zmin = grid_geom['xmin'], grid_geom['ymin'], grid_geom['zmin']

        xs = xmin + (np.arange(nelx) + 0.5) * hx
        ys = ymin + (np.arange(nely) + 0.5) * hy
        zs = zmin + (np.arange(nelz) + 0.5) * hz

        Z, X, Y = np.meshgrid(zs, xs, ys, indexing='ij')
        centers = np.stack([X, Y, Z], axis=-1)
        return centers

    def node_coordinates(self, mesh, grid_geom):
        nelx, nely, nelz = mesh['nelx'], mesh['nely'], mesh['nelz']
        hx, hy, hz = grid_geom['hx'], grid_geom['hy'], grid_geom['hz']
        xmin, ymin, zmin = grid_geom['xmin'], grid_geom['ymin'], grid_geom['zmin']

        xs = xmin + np.arange(nelx + 1) * hx
        ys = ymin + np.arange(nely + 1) * hy
        zs = zmin + np.arange(nelz + 1) * hz

        Z, X, Y = np.meshgrid(zs, xs, ys, indexing='ij')
        coords = np.stack([X, Y, Z], axis=-1)
        return coords

    def build_voxel_grid_for_shell(self, brep_bbox, thickness, voxel_size, extra_layers=1):
        padded = self.padded_bbox_from_midsurface(
            brep_bbox,
            thickness=thickness,
            voxel_size=voxel_size,
            extra_layers=extra_layers
        )

        mesh, grid_geom = self.structured_grid_from_bbox(padded, voxel_size)
        elem_centers = self.element_centers(mesh, grid_geom)
        node_coords = self.node_coordinates(mesh, grid_geom)

        return mesh, grid_geom, elem_centers, node_coords

    def surface_samples_to_element_indices(self):
        """
        Assign each midsurface sample to exactly one structured FEA element.

        Exact ties on voxel faces/edges/corners are broken toward the element
        with the larger grid index on the tied axis, so every sample still has
        exactly one owner.
        """
        points = self.points_xyz
        nelx, nely, nelz = self.mesh['nelx'], self.mesh['nely'], self.mesh['nelz']
        hx, hy, hz = self.grid_geom['hx'], self.grid_geom['hy'], self.grid_geom['hz']
        xmin, ymin, zmin = self.grid_geom['xmin'], self.grid_geom['ymin'], self.grid_geom['zmin']

        center_x0 = xmin + 0.5 * hx
        center_y0 = ymin + 0.5 * hy
        center_z0 = zmin + 0.5 * hz

        ix = np.floor((points[:, 0] - center_x0) / hx + 0.5).astype(np.int64)
        iy = np.floor((points[:, 1] - center_y0) / hy + 0.5).astype(np.int64)
        iz = np.floor((points[:, 2] - center_z0) / hz + 0.5).astype(np.int64)

        valid = (
            (0 <= ix) & (ix < nelx) &
            (0 <= iy) & (iy < nely) &
            (0 <= iz) & (iz < nelz)
        )

        elem_idx = -np.ones(points.shape[0], dtype=np.int64)
        elem_idx[valid] = iz[valid] * (nelx * nely) + ix[valid] * nely + iy[valid]
        return elem_idx

    def ensure_surface_sample_elements_are_occupied(self):
        """
        The midsurface/core layer is defined by the elements that own surface
        samples. Force those elements into the shell and keep one representative
        sample index for fallback/visualization.
        """
        sample_elem_idx = np.asarray(self.sample_elem_idx, dtype=np.int64).reshape(-1)
        valid = sample_elem_idx >= 0
        if not np.any(valid):
            return

        occ_flat = self.elem_occupancy.reshape(-1)
        elem_sample_flat = self.elem_sample_idx.reshape(-1)
        sample_ids = np.arange(sample_elem_idx.shape[0], dtype=np.int64)

        occ_flat[sample_elem_idx[valid]] = 1
        missing_rep = elem_sample_flat < 0
        for elem_idx, sample_id in zip(sample_elem_idx[valid], sample_ids[valid]):
            if missing_rep[elem_idx]:
                elem_sample_flat[elem_idx] = sample_id
                missing_rep[elem_idx] = False

    def count_surface_samples_per_element(self):
        num_elems = int(np.prod(self.elem_centers.shape[:3]))
        sample_elem_idx = np.asarray(self.sample_elem_idx, dtype=np.int64).reshape(-1)
        valid = sample_elem_idx >= 0
        return np.bincount(sample_elem_idx[valid], minlength=num_elems).astype(np.int64)

    def shell_layer_core_element_indices(self):
        """
        Map each occupied shell element to the midsurface/core element whose
        density and fiber it should inherit.

        Core elements map to themselves. Offset inner/outer layer elements map
        through their nearest surface sample to that sample's core element.
        """
        sample_idx = self.elem_sample_idx.reshape(-1)
        sample_elem_idx = self.sample_elem_idx.reshape(-1)
        occ = self.elem_occupancy.reshape(-1).astype(bool)

        num_elems = sample_idx.shape[0]
        counts = self.elem_sample_count
        if counts is None:
            counts = self.count_surface_samples_per_element()

        core_elem_idx = -np.ones((num_elems,), dtype=np.int64)
        has_core_samples = counts > 0
        core_elem_idx[occ & has_core_samples] = np.flatnonzero(occ & has_core_samples)

        layer = occ & (~has_core_samples) & (sample_idx >= 0)
        layer_sample_idx = sample_idx[layer]
        valid_layer_sample = (
            (layer_sample_idx >= 0)
            & (layer_sample_idx < sample_elem_idx.shape[0])
            & (sample_elem_idx[layer_sample_idx] >= 0)
        )
        layer_elem_ids = np.flatnonzero(layer)
        core_elem_idx[layer_elem_ids[valid_layer_sample]] = sample_elem_idx[layer_sample_idx[valid_layer_sample]]

        return core_elem_idx
    
    def assign_surface_fields_to_voxels(self, rho_surface, fiber_surface, rho_void=1e-3):
        rho_surface = self.to_numpy(rho_surface).reshape(-1)
        fiber_surface = self.to_numpy(fiber_surface).reshape(-1, 3)

        if rho_surface.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("rho_surface must have same length as points_xyz")

        if fiber_surface.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("fiber_surface must have same length as points_xyz")

        sample_elem_idx = self.sample_elem_idx.reshape(-1)
        occ = self.elem_occupancy.reshape(-1)

        num_elems = occ.shape[0]

        elem_density = np.full((num_elems,), rho_void, dtype=np.float32)
        elem_fiber = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (num_elems, 1))

        valid_samples = sample_elem_idx >= 0
        counts = np.bincount(sample_elem_idx[valid_samples], minlength=num_elems).astype(np.float32)
        rho_sum = np.bincount(
            sample_elem_idx[valid_samples],
            weights=rho_surface[valid_samples],
            minlength=num_elems,
        ).astype(np.float32)

        fiber_sum = np.zeros((num_elems, 3), dtype=np.float32)
        np.add.at(fiber_sum, sample_elem_idx[valid_samples], fiber_surface[valid_samples].astype(np.float32))

        has_samples = counts > 0

        core_density = np.full((num_elems,), rho_void, dtype=np.float32)
        core_fiber = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (num_elems, 1))
        core_density[has_samples] = rho_sum[has_samples] / counts[has_samples]
        core_fiber[has_samples] = fiber_sum[has_samples] / counts[has_samples, None]

        core_norms = np.linalg.norm(core_fiber[has_samples], axis=1, keepdims=True)
        core_fiber[has_samples] /= np.clip(core_norms, 1e-12, None)

        source_core_elem_idx = self.shell_layer_core_element_indices()
        filled = (occ > 0) & (source_core_elem_idx >= 0)
        filled_ids = np.flatnonzero(filled)
        filled[filled_ids] = has_samples[source_core_elem_idx[filled_ids]]

        elem_density[filled] = core_density[source_core_elem_idx[filled]]
        elem_fiber[filled] = core_fiber[source_core_elem_idx[filled]]

        self.elem_density = elem_density
        self.elem_fiber = elem_fiber
        self.elem_phi, self.elem_theta = self.fiber_vectors_to_angles(elem_fiber)

        return elem_density, elem_fiber
    
    def fiber_vectors_to_angles(self, fiber_vec):
        fiber_vec = self.to_numpy(fiber_vec).reshape(-1, 3)

        norms = np.linalg.norm(fiber_vec, axis=1, keepdims=True)
        v = fiber_vec / np.clip(norms, 1e-12, None)

        ax = v[:, 0]
        ay = v[:, 1]
        az = v[:, 2]

        phi = np.arctan2(ay, ax).astype(np.float32)
        theta = np.arccos(np.clip(az, -1.0, 1.0)).astype(np.float32)

        return phi, theta
    def assign_decoder_fields(self, rho_surface, fiber_surface, rho_void=1e-3):
        elem_density, elem_fiber = self.assign_surface_fields_to_voxels(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            rho_void=rho_void
        )
        elem_phi, elem_theta = self.fiber_vectors_to_angles(elem_fiber)

        self.elem_density = elem_density
        self.elem_fiber = elem_fiber
        self.elem_phi = elem_phi
        self.elem_theta = elem_theta

        return elem_density, elem_phi, elem_theta
    def show_voxels_and_surface(self):
        surface = self.points_xyz

        plotter = pv.Plotter()

        mesh = self.occupied_voxel_mesh()
        if mesh is not None:
            plotter.add_mesh(mesh, color="lightblue", opacity=0.6)
        else:
            print("No occupied voxels to display")

        cloud = self.surface_cloud()
        if cloud is not None:
            plotter.add_mesh(cloud, color="red", point_size=10, render_points_as_spheres=True)

        plotter.show()

    def surface_cloud(self):
        if self.points_xyz is None or self.points_xyz.shape[0] == 0:
            return None

        if self._surface_cloud_cache is None:
            self._surface_cloud_cache = pv.PolyData(self.points_xyz)

        return self._surface_cloud_cache

    def occupied_voxel_mesh(self, use_cache=True):
        """
        Build a PyVista mesh for all occupied voxels in one shot.

        This is much faster than creating and merging one pv.Cube per voxel.
        The result is cached because occupancy is fixed after shell voxelization.
        """
        if use_cache and self._occupied_voxel_mesh_cache is not None:
            return self._occupied_voxel_mesh_cache

        if self.elem_occupancy is None or not np.any(self.elem_occupancy):
            return None

        nelx, nely, nelz = self.mesh['nelx'], self.mesh['nely'], self.mesh['nelz']
        hx, hy, hz = self.grid_geom['hx'], self.grid_geom['hy'], self.grid_geom['hz']
        xmin, ymin, zmin = self.grid_geom['xmin'], self.grid_geom['ymin'], self.grid_geom['zmin']

        grid_cls = getattr(pv, "ImageData", None)
        if grid_cls is None:
            grid_cls = pv.UniformGrid

        grid = grid_cls(
            dimensions=(nelx + 1, nely + 1, nelz + 1),
            spacing=(hx, hy, hz),
            origin=(xmin, ymin, zmin),
        )

        occ_xyz = np.transpose(self.elem_occupancy.astype(np.uint8), (1, 2, 0))
        grid.cell_data["occupied"] = occ_xyz.ravel(order="F")
        mesh = grid.threshold(value=0.5, scalars="occupied")

        if use_cache:
            self._occupied_voxel_mesh_cache = mesh

        return mesh

    def get_flat_node_coords(self):
        """
        Returns
        -------
        node_ids : ndarray, shape (num_nodes,)
            Flat global node ids: 0, 1, 2, ..., num_nodes-1

        coords : ndarray, shape (num_nodes, 3)
            Flat node coordinates [x, y, z] for each node id.
        """
        coords = self.node_coords.reshape(-1, 3)
        node_ids = np.arange(coords.shape[0], dtype=np.int64)
        return node_ids, coords    
    def node_ids_to_dofs(self, node_ids, components=(0, 1, 2)):
        """
        Convert node ids to global DOF ids.

        Parameters
        ----------
        node_ids : array-like
            Global node ids.
        components : tuple
            Which displacement components to include:
            0 -> ux, 1 -> uy, 2 -> uz

        Returns
        -------
        dofs : ndarray
            Flat array of global DOF ids.
        """
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)

        dofs = []
        for c in components:
            dofs.append(3 * node_ids + int(c))

        if len(dofs) == 0:
            return np.array([], dtype=np.int64)

        return np.concatenate(dofs).astype(np.int64)
    
    def make_empty_force(self):
        """
        Create an empty global force vector of shape (ndof, 1).
        """
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        return np.zeros((ndof, 1), dtype=float)
    
    def apply_nodal_force(self, force, node_ids, direction, total_value):
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if node_ids.size == 0:
            raise ValueError("No nodes selected for force application")

        comp_map = {'x': 0, 'y': 1, 'z': 2}
        c = comp_map[direction]

        val_per_node = total_value / node_ids.size
        dofs = 3 * node_ids + c
        force[dofs, 0] += val_per_node
        return force

    def apply_nodal_torque(self, force, node_ids, axis, total_torque):
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if node_ids.size == 0:
            raise ValueError("No nodes selected for torque application")

        axis_map = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        axis = str(axis).lower()
        if axis not in axis_map:
            raise ValueError(f"Unsupported torque axis: {axis}")

        _all_node_ids, coords = self.get_flat_node_coords()
        pts = coords[node_ids].astype(float, copy=False)
        axis_vec = axis_map[axis]

        center = pts.mean(axis=0)
        r = pts - center[None, :]
        r -= np.outer(r @ axis_vec, axis_vec)

        tangent = np.cross(axis_vec[None, :], r)
        radius_sq = np.sum(r * r, axis=1)
        denom = float(np.sum(radius_sq))
        if denom <= 1e-20:
            raise ValueError(
                f"Cannot apply torsion around axis={axis}: selected torque nodes have near-zero radius"
            )

        nodal_forces = (float(total_torque) / denom) * tangent
        for comp in range(3):
            dofs = 3 * node_ids + comp
            force[dofs, 0] += nodal_forces[:, comp]

        return force

    def set_boundary_conditions_from_regions(self, fixed_nodes, force_nodes, force_direction='z', force_value=-1.0):
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)

        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        force = self.apply_nodal_force(force, force_nodes, force_direction, force_value)

        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

    def set_torsion_boundary_conditions(self, fixed_nodes, torque_nodes, torque_axis='z', total_torque=1.0):
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)

        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        force = self.apply_nodal_torque(force, torque_nodes, torque_axis, total_torque)

        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

    def show_voxels_surface_and_bc(
        self,
        show_load_arrows=True,
        return_img=False,
        off_screen=False,
        window_size=None,
        show=False,
        show_window_size=(1040, 560),
        max_load_arrows=48,
    ):
        plotter_kwargs = {"off_screen": off_screen if not show else False}
        if window_size is not None:
            plotter_kwargs["window_size"] = window_size
        plotter = pv.Plotter(**plotter_kwargs)

        mesh = self.occupied_voxel_mesh()
        if mesh is not None:
            plotter.add_mesh(mesh, color="#94a3b8", opacity=0.22)

        cloud = self.surface_cloud()
        if cloud is not None:
            plotter.add_mesh(
                cloud,
                color="#ef4444",
                opacity=0.16,
                point_size=2,
                render_points_as_spheres=False,
            )

        node_ids, coords = self.get_flat_node_coords()

        fixed_dofs = self.boundaryCondition['fixed']
        fixed_node_ids = np.unique(fixed_dofs // 3)

        force = self.boundaryCondition['force'].reshape(-1)
        force_node_ids = np.unique(np.where(np.abs(force) > 0)[0] // 3)

        if fixed_node_ids.size > 0:
            fixed_pts = coords[fixed_node_ids]
            plotter.add_mesh(
                pv.PolyData(fixed_pts),
                color="#00b894",
                point_size=7,
                render_points_as_spheres=True
            )

        if force_node_ids.size > 0:
            force_pts = coords[force_node_ids]
            plotter.add_mesh(
                pv.PolyData(force_pts),
                color="#ffb000",
                point_size=7,
                render_points_as_spheres=True
            )
            if show_load_arrows:
                force_components = force.reshape(-1, 3)
                force_vecs = force_components[force_node_ids]
                force_norms = np.linalg.norm(force_vecs, axis=1)
                arrow_mask = force_norms > 0.0
                if np.any(arrow_mask):
                    arrow_pts = force_pts[arrow_mask]
                    arrow_vecs = force_vecs[arrow_mask] / force_norms[arrow_mask, None]
                    if arrow_pts.shape[0] > int(max_load_arrows):
                        keep_idx = np.linspace(
                            0,
                            arrow_pts.shape[0] - 1,
                            int(max_load_arrows),
                            dtype=np.int64,
                        )
                        arrow_pts = arrow_pts[keep_idx]
                        arrow_vecs = arrow_vecs[keep_idx]
                    arrow_cloud = pv.PolyData(arrow_pts)
                    arrow_cloud["vectors"] = arrow_vecs
                    arrows = arrow_cloud.glyph(
                        orient="vectors",
                        scale=False,
                        factor=4.2 * self.voxel_size,
                    )
                    plotter.add_mesh(arrows, color="#ff5a1f", opacity=1.0)

        plotter.show_axes()

        img = None
        if return_img:
            plotter.set_background("white")
            try:
                plotter.view_isometric()
                plotter.reset_camera()
                plotter.camera.zoom(1.12)
            except Exception:
                pass
            img = plotter.screenshot(return_img=True, transparent_background=False)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            legend_items = [
                ("Shell voxels", (71, 85, 105)),
                ("Surface samples", (239, 68, 68)),
                ("Fixed nodes", (0, 184, 148)),
                ("Loaded nodes", (255, 176, 0)),
                ("Load arrows", (255, 90, 31)),
            ]
            box_w = 132
            box_h = 18 + 16 * len(legend_items)
            margin = 10
            x0 = margin
            y0 = margin
            overlay = img.copy()
            cv2.rectangle(
                overlay,
                (x0, y0),
                (x0 + box_w, y0 + box_h),
                (255, 255, 255),
                thickness=-1,
            )
            img = cv2.addWeighted(overlay, 0.86, img, 0.14, 0)
            cv2.rectangle(
                img,
                (x0, y0),
                (x0 + box_w, y0 + box_h),
                (31, 41, 55),
                thickness=1,
            )
            cv2.putText(
                img,
                "Legend",
                (x0 + 8, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (17, 24, 39),
                1,
                cv2.LINE_AA,
            )
            for idx, (label, color) in enumerate(legend_items):
                y = y0 + 31 + idx * 16
                cv2.rectangle(img, (x0 + 8, y - 8), (x0 + 17, y + 1), color, thickness=-1)
                cv2.putText(
                    img,
                    label,
                    (x0 + 23, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.31,
                    (17, 24, 39),
                    1,
                    cv2.LINE_AA,
                )

        if show and show_window_size is not None:
            try:
                plotter.window_size = show_window_size
                plotter.ren_win.SetSize(int(show_window_size[0]), int(show_window_size[1]))
                plotter.reset_camera()
            except Exception:
                pass

        if show or not return_img:
            plotter.show()
        else:
            plotter.close()

        if return_img:
            return img

    def occupied_axis_bounds(self):
        occ = self.elem_occupancy.astype(bool)
        if not np.any(occ):
            return self.padded_bbox_from_midsurface(self.brep_bbox, self.thickness, self.voxel_size, self.extra_layers)

        occ_centers = self.elem_centers[occ]
        half = 0.5 * self.voxel_size
        return {
            'xmin': float(np.min(occ_centers[:, 0]) - half),
            'xmax': float(np.max(occ_centers[:, 0]) + half),
            'ymin': float(np.min(occ_centers[:, 1]) - half),
            'ymax': float(np.max(occ_centers[:, 1]) + half),
            'zmin': float(np.min(occ_centers[:, 2]) - half),
            'zmax': float(np.max(occ_centers[:, 2]) + half),
        }
    
    def select_nodes_in_box(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
        """
        Select node ids whose coordinates lie inside a rectangular box.

        Any bound can be left as None to mean 'no restriction' in that direction.

        Returns
        -------
        node_ids : ndarray
            Flat array of selected global node ids.
        """
        node_ids, coords = self.get_flat_node_coords()

        mask = np.ones(coords.shape[0], dtype=bool)

        if xmin is not None:
            mask &= coords[:, 0] >= xmin
        if xmax is not None:
            mask &= coords[:, 0] <= xmax

        if ymin is not None:
            mask &= coords[:, 1] >= ymin
        if ymax is not None:
            mask &= coords[:, 1] <= ymax

        if zmin is not None:
            mask &= coords[:, 2] >= zmin
        if zmax is not None:
            mask &= coords[:, 2] <= zmax

        return node_ids[mask]
    def debug_voxel_stats(self):
        if self.elem_occupancy is None:
            print("elem_occupancy is None")
            return

        occ = self.elem_occupancy
        num_occ = int(occ.sum())
        num_total = int(occ.size)

        hx, hy, hz = self.mesh['elemSize']
        voxel_vol = hx * hy * hz
        vox_vol = num_occ * voxel_vol

        try:
            target_vol = float(np.sum(self.face_areas)) * self.thickness
        except Exception:
            target_vol = None

        print("=== Voxel Stats ===")
        print("brep_bbox:", self.brep_bbox)
        print("mesh:", self.mesh)
        print("elem_centers shape:", self.elem_centers.shape)
        print("node_coords shape:", self.node_coords.shape)
        print("occupied voxels:", num_occ)
        print("total voxels:", num_total)
        print("occupancy ratio:", num_occ / max(num_total, 1))
        if self.elem_sample_count is not None:
            num_with_samples = int(np.count_nonzero(self.elem_sample_count))
            occ_flat = self.elem_occupancy.reshape(-1).astype(bool)
            num_occ_with_samples = int(np.count_nonzero(self.elem_sample_count[occ_flat]))
            print("voxels with assigned surface samples:", num_with_samples)
            print("occupied voxels with assigned surface samples:", num_occ_with_samples)
        print("voxelized volume:", vox_vol)
        print("thickness:", self.thickness)
        print("voxel_size:", self.voxel_size)

        if target_vol is not None:
            print("target approx volume (sum(face_areas)*thickness):", target_vol)
            if target_vol > 0:
                print("volume ratio voxel/target:", vox_vol / target_vol)
    def build_fem_fields_from_decoder(self, rho_surface, fiber_surface, rho_void=1e-3):
        elem_density, elem_phi, elem_theta = self.assign_decoder_fields(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            rho_void=rho_void
        )

        return {
            'density': elem_density,
            'phi': elem_phi,
            'theta': elem_theta,
            'fixed': self.boundaryCondition['fixed'],
            'force': self.boundaryCondition['force'],
            'mesh': self.mesh,
            'materialProperty': self.materialProperty,
        }
    def build_fem_fields_from_decoder_torch(self, rho_surface, fiber_surface, rho_void=1e-3):
        device = rho_surface.device

        sample_idx = torch.as_tensor(self.elem_sample_idx.reshape(-1), device=device, dtype=torch.long)
        sample_elem_idx = torch.as_tensor(self.sample_elem_idx.reshape(-1), device=device, dtype=torch.long)
        occ = torch.as_tensor(self.elem_occupancy.reshape(-1), device=device, dtype=torch.bool)
        source_core_elem_idx = torch.as_tensor(
            self.shell_layer_core_element_indices(),
            device=device,
            dtype=torch.long,
        )

        num_elems = sample_idx.numel()

        density = torch.full((num_elems,), rho_void, dtype=rho_surface.dtype, device=device)
        fiber = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        fiber[:, 0] = 1.0

        valid_samples = sample_elem_idx >= 0
        valid_sample_elem_idx = sample_elem_idx[valid_samples]

        counts = torch.zeros((num_elems,), dtype=rho_surface.dtype, device=device)
        counts.index_add_(0, valid_sample_elem_idx, torch.ones_like(rho_surface[valid_samples]))

        rho_sum = torch.zeros((num_elems,), dtype=rho_surface.dtype, device=device)
        rho_sum.index_add_(0, valid_sample_elem_idx, rho_surface[valid_samples])

        fiber_sum = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        fiber_sum.index_add_(0, valid_sample_elem_idx, fiber_surface[valid_samples])

        has_samples = counts > 0

        core_density = torch.full((num_elems,), rho_void, dtype=rho_surface.dtype, device=device)
        core_fiber = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        core_fiber[:, 0] = 1.0
        core_density[has_samples] = rho_sum[has_samples] / counts[has_samples]
        core_fiber[has_samples] = fiber_sum[has_samples] / counts[has_samples, None]

        core_fiber_norm = torch.linalg.norm(core_fiber, dim=1, keepdim=True).clamp_min(1e-12)
        core_fiber = core_fiber / core_fiber_norm

        filled = occ & (source_core_elem_idx >= 0)
        filled_ids = torch.nonzero(filled, as_tuple=False).reshape(-1)
        if filled_ids.numel() > 0:
            valid_filled_ids = filled_ids[has_samples[source_core_elem_idx[filled_ids]]]
            density[valid_filled_ids] = core_density[source_core_elem_idx[valid_filled_ids]]
            fiber[valid_filled_ids] = core_fiber[source_core_elem_idx[valid_filled_ids]]

        angle_eps = 1e-6
        fiber_norm = torch.linalg.norm(fiber, dim=1, keepdim=True).clamp_min(angle_eps)
        fiber = fiber / fiber_norm

        ax = fiber[:, 0]
        ay = fiber[:, 1]
        az = fiber[:, 2]

        xy_norm = torch.linalg.norm(fiber[:, :2], dim=1)
        ax_safe = torch.where(xy_norm > angle_eps, ax, torch.ones_like(ax))
        ay_safe = torch.where(xy_norm > angle_eps, ay, torch.zeros_like(ay))
        phi = torch.atan2(ay_safe, ax_safe)
        theta = torch.acos(torch.clamp(az, -1.0 + angle_eps, 1.0 - angle_eps))

        return {
            "density": density,
            "phi": phi,
            "theta": theta,
            "fixed": self.boundaryCondition["fixed"],
            "force": self.boundaryCondition["force"],
            "mesh": self.mesh,
            "materialProperty": self.materialProperty,
        }
    def show_voxels_surface_and_bc_NEW(self):
        occ = self.elem_occupancy
        centers = self.elem_centers
        surface = self.points_xyz

        vox_pts = centers[occ.astype(bool)]

        plotter = pv.Plotter()

        if vox_pts.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(vox_pts),
                color="lightblue",
                point_size=6,
                render_points_as_spheres=True,
            )

        if surface is not None and surface.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(surface),
                color="red",
                point_size=4,
                render_points_as_spheres=True,
            )

        node_ids, coords = self.get_flat_node_coords()

        fixed_dofs = self.boundaryCondition['fixed']
        fixed_node_ids = np.unique(fixed_dofs // 3)

        force = self.boundaryCondition['force'].reshape(-1)
        force_node_ids = np.unique(np.where(np.abs(force) > 0)[0] // 3)

        if fixed_node_ids.size > 0:
            fixed_pts = coords[fixed_node_ids]
            plotter.add_mesh(
                pv.PolyData(fixed_pts),
                color="green",
                point_size=12,
                render_points_as_spheres=True
            )

        if force_node_ids.size > 0:
            force_pts = coords[force_node_ids]
            plotter.add_mesh(
                pv.PolyData(force_pts),
                color="yellow",
                point_size=12,
                render_points_as_spheres=True
            )

    # Better text placement
        plotter.add_text("Yellow: Applied load", position="upper_right", font_size=12, color="yellow")
        plotter.add_text("Green: Fixed nodes", position="upper_left", font_size=12, color="green")
        plotter.add_text("Blue: Occupied voxels", position="lower_left", font_size=12, color="lightblue")
        plotter.add_text("Red: Surface points", position="lower_right", font_size=12, color="red")

   
        plotter.show_axes()
        plotter.show()

#if __name__ == '__main__':
    # expects `tensors` to already exist in the current scope
    # shell_problem = ThickenShell(
    #     thickness=2.0,
    #     voxel_size=1.0,
    #     extra_layers=1,
    #     tensors=tensors
    # )

    # shell_problem.debug_voxel_stats()

    # savePath = os.path.join('data', 'settings', '{}.npy'.format(shell_problem.name))
    # shell_problem.serialize(savePath)
    # print("saved to:", savePath)
