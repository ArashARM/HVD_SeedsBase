import os

import numpy as np
import torch
import torch.nn.functional as F

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None


class CADTensorGenerator:
    """
    Single-face CAD UV-domain provider for decoder optimization.

    The loaded CAD file must contain exactly one TopoDS_Face. The decoder owns
    seed variables in normalized UV [0, 1]^2 and calls this class for trim SDF
    access plus continuous UV-to-XYZ and metric evaluation.
    """

    def __init__(
        self,
        metric_tol: float = 1e-9,
        device: str = "cpu",
        seed_domain_mask_res: int = 128,
        seed_domain_trim_tol: float = 1e-7,
    ):
        self.metric_tol = float(metric_tol)
        self.device = device
        self.seed_domain_mask_res = int(seed_domain_mask_res)
        self.seed_domain_trim_tol = float(seed_domain_trim_tol)

        self._active_shape = None
        self._active_face = None
        self._active_surface = None
        self._active_u_raw_bounds = None
        self._active_v_raw_bounds = None
        self._seed_domain_mask_grid = None
        self._seed_domain_sdf_grid = None
        self._boundary_parameter_loops = None
        self._u_periodic = False
        self._v_periodic = False
        self._u_period = None
        self._v_period = None

    # =========================================================================
    # 1) CAD loading + single-face helpers
    # =========================================================================

    @staticmethod
    def load_shape(path: str):
        """Load STEP/IGES into a TopoDS_Shape."""
        p = os.fspath(path).lower()

        if p.endswith((".step", ".stp")):
            reader = STEPControl_Reader()
            if reader.ReadFile(os.fspath(path)) != IFSelect_RetDone:
                raise RuntimeError("STEP read failed")
            reader.TransferRoots()
            return reader.OneShape()

        if p.endswith((".iges", ".igs")):
            reader = IGESControl_Reader()
            if reader.ReadFile(os.fspath(path)) != IFSelect_RetDone:
                raise RuntimeError("IGES read failed")
            reader.TransferRoots()
            return reader.OneShape()

        raise ValueError("Unsupported file type (need .step/.stp/.iges/.igs)")

    @staticmethod
    def iter_faces(shape):
        """Yield TopoDS_Face items in OpenCascade traversal order."""
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            yield topods.Face(exp.Current())
            exp.Next()

    @classmethod
    def get_single_face(cls, shape):
        faces = list(cls.iter_faces(shape))
        if len(faces) != 1:
            raise ValueError(f"Expected CAD file with exactly one face, got {len(faces)}.")
        return faces[0]

    @staticmethod
    def face_uv_bounds_and_surface(face):
        """Return (umin, umax, vmin, vmax) and the underlying OCC surface."""
        umin, umax, vmin, vmax = breptools.UVBounds(face)
        surf = BRep_Tool.Surface(face)
        return (float(umin), float(umax), float(vmin), float(vmax)), surf

    @staticmethod
    def face_uv_periodicity(face):
        """Return periodicity for the underlying OCC surface."""
        surf_ad = BRepAdaptor_Surface(face, False)
        u_per = bool(surf_ad.IsUPeriodic())
        v_per = bool(surf_ad.IsVPeriodic())
        u_period = float(surf_ad.UPeriod()) if u_per else None
        v_period = float(surf_ad.VPeriod()) if v_per else None
        return u_per, v_per, u_period, v_period

    def _require_active_face(self):
        if self._active_face is None:
            raise RuntimeError("Call generate_from_file(shape_path) before evaluating UV points.")
        return self._active_face

    # =========================================================================
    # 2) UV conversion + periodic helpers
    # =========================================================================

    @staticmethod
    def uv_raw_to_norm(u, v, umin, umax, vmin, vmax):
        Lu = float(umax - umin)
        Lv = float(vmax - vmin)
        if abs(Lu) < 1e-30:
            Lu = 1.0
        if abs(Lv) < 1e-30:
            Lv = 1.0
        return (float(u) - float(umin)) / Lu, (float(v) - float(vmin)) / Lv

    @staticmethod
    def uv_norm_to_raw_from_bounds(
        uv_norm,
        u_raw_bounds: tuple[float, float],
        v_raw_bounds: tuple[float, float],
    ):
        """Convert normalized UV in [0,1]^2 to raw CAD face UV."""
        umin, umax = map(float, u_raw_bounds)
        vmin, vmax = map(float, v_raw_bounds)

        if isinstance(uv_norm, torch.Tensor):
            u = umin + uv_norm[..., 0] * (umax - umin)
            v = vmin + uv_norm[..., 1] * (vmax - vmin)
            return torch.stack((u, v), dim=-1)

        uv_norm = np.asarray(uv_norm, dtype=float)
        u = umin + uv_norm[..., 0] * (umax - umin)
        v = vmin + uv_norm[..., 1] * (vmax - vmin)
        return np.stack((u, v), axis=-1)

    @staticmethod
    def uv_raw_to_norm_from_bounds(
        uv_raw,
        u_raw_bounds: tuple[float, float],
        v_raw_bounds: tuple[float, float],
    ):
        """Convert raw CAD UV to normalized UV in [0,1]^2."""
        umin, umax = map(float, u_raw_bounds)
        vmin, vmax = map(float, v_raw_bounds)
        Lu = (umax - umin) if abs(umax - umin) > 1e-30 else 1.0
        Lv = (vmax - vmin) if abs(vmax - vmin) > 1e-30 else 1.0

        if isinstance(uv_raw, torch.Tensor):
            u = (uv_raw[..., 0] - umin) / Lu
            v = (uv_raw[..., 1] - vmin) / Lv
            return torch.stack((u, v), dim=-1)

        uv_raw = np.asarray(uv_raw, dtype=float)
        u = (uv_raw[..., 0] - umin) / Lu
        v = (uv_raw[..., 1] - vmin) / Lv
        return np.stack((u, v), axis=-1)

    @staticmethod
    def periodic_uv_difference(
        uv_a,
        uv_b,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ):
        """Difference in normalized UV, wrapping periodic dimensions."""
        diff = uv_a - uv_b
        round_fn = torch.round if isinstance(diff, torch.Tensor) else np.round
        if u_periodic:
            diff = diff.clone() if isinstance(diff, torch.Tensor) else diff.copy()
            diff[..., 0] = diff[..., 0] - round_fn(diff[..., 0])
        if v_periodic:
            diff = diff.clone() if isinstance(diff, torch.Tensor) else diff.copy()
            diff[..., 1] = diff[..., 1] - round_fn(diff[..., 1])
        return diff

    @staticmethod
    def periodic_uv_distance(
        uv_a,
        uv_b,
        u_periodic: bool = False,
        v_periodic: bool = False,
        eps: float = 1e-12,
    ):
        """Euclidean distance in normalized UV, wrapping periodic dimensions."""
        diff = CADTensorGenerator.periodic_uv_difference(
            uv_a,
            uv_b,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        if isinstance(diff, torch.Tensor):
            return torch.sqrt((diff * diff).sum(dim=-1) + float(eps))
        return np.sqrt((diff * diff).sum(axis=-1) + float(eps))

    # =========================================================================
    # 3) Trim classification, mask, and SDF
    # =========================================================================

    @staticmethod
    def _classify_inside(face, u, v, classifier, tol):
        classifier.Perform(face, gp_Pnt2d(float(u), float(v)), float(tol))
        st = classifier.State()
        return (st == TopAbs_IN) or (st == TopAbs_ON)

    @classmethod
    def classify_uv_points_on_face(
        cls,
        face,
        uv_raw,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Trim-aware inside mask for raw UV points on a CAD face."""
        if isinstance(uv_raw, torch.Tensor):
            uv_raw = uv_raw.detach().cpu().numpy()
        uv_raw = np.asarray(uv_raw, dtype=float)
        if uv_raw.shape[-1] != 2:
            raise ValueError(f"uv_raw must end with dimension 2, got {uv_raw.shape}")

        flat = uv_raw.reshape(-1, 2)
        classifier = BRepClass_FaceClassifier()
        inside = np.zeros((flat.shape[0],), dtype=bool)
        for i, (u, v) in enumerate(flat):
            inside[i] = cls._classify_inside(face, float(u), float(v), classifier, tol)
        return inside.reshape(uv_raw.shape[:-1])

    @classmethod
    def build_seed_domain_mask_grid(
        cls,
        face,
        u_raw_bounds: tuple[float, float],
        v_raw_bounds: tuple[float, float],
        res: int = 128,
        trim_tol: float = 1e-7,
    ) -> np.ndarray:
        """Build a binary trim-validity mask on a normalized [0,1]^2 grid."""
        res = int(res)
        if res < 2:
            raise ValueError(f"seed domain mask resolution must be >= 2, got {res}")

        u_lin = np.linspace(0.0, 1.0, res, dtype=np.float32)
        v_lin = np.linspace(0.0, 1.0, res, dtype=np.float32)
        uu, vv = np.meshgrid(u_lin, v_lin, indexing="xy")
        uv_norm = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
        uv_raw = cls.uv_norm_to_raw_from_bounds(
            uv_norm=uv_norm,
            u_raw_bounds=u_raw_bounds,
            v_raw_bounds=v_raw_bounds,
        )
        inside = cls.classify_uv_points_on_face(face, uv_raw, tol=trim_tol)
        return inside.astype(np.float32).reshape(res, res)
    
    @staticmethod
    def build_seed_domain_sdf_grid(mask_grid_np: np.ndarray) -> np.ndarray:
        """Signed distance to nearest invalid/valid boundary in normalized UV units."""
        if distance_transform_edt is None:
            raise ImportError("scipy.ndimage.distance_transform_edt is required for trim SDF grids.")

        mask_grid_np = np.asarray(mask_grid_np, dtype=np.float32)
        if mask_grid_np.ndim != 2 or mask_grid_np.shape[0] != mask_grid_np.shape[1]:
            raise ValueError(f"mask_grid_np must be square [res,res], got {mask_grid_np.shape}")

        res = int(mask_grid_np.shape[0])
        dx = 1.0 / float(res - 1)
        inside = mask_grid_np > 0.5

        yy, xx = np.mgrid[0:res, 0:res]
        u = xx / float(res - 1)
        v = yy / float(res - 1)

        box_dist = np.minimum.reduce([
            u,
            1.0 - u,
            v,
            1.0 - v,
        ])

        if np.all(inside):
            return box_dist.astype(np.float32)

        if not np.any(inside):
            return (-box_dist).astype(np.float32)

        dist_to_outside = distance_transform_edt(inside) * dx
        dist_to_inside = distance_transform_edt(~inside) * dx

        sdf = dist_to_outside - dist_to_inside

        sdf = np.minimum(sdf, box_dist)

        return sdf.astype(np.float32)

    def _build_boundary_parameter_loops(self) -> list[np.ndarray]:
        """Extract ordered normalized-UV trim loops from the cached SDF grid."""
        if self._seed_domain_sdf_grid is None:
            raise RuntimeError("Call generate_from_file(shape_path) before extracting trim loops.")

        # Matplotlib's contour engine gives ordered polylines and correctly
        # separates the outer trim wire from any inner-hole wires.
        from matplotlib.figure import Figure

        sdf = self._seed_domain_sdf_grid.detach().cpu().numpy()
        height, width = sdf.shape
        u = np.linspace(0.0, 1.0, width)
        v = np.linspace(0.0, 1.0, height)
        figure = Figure()
        axis = figure.subplots()
        contour = axis.contour(u, v, sdf, levels=[0.0])
        loops = [
            np.asarray(segment, dtype=np.float64)
            for segment in contour.allsegs[0]
            if np.asarray(segment).shape[0] >= 2
        ]
        figure.clear()

        if not loops:
            raise RuntimeError(
                "No trim-boundary loops could be extracted from the CAD SDF grid."
            )
        self._boundary_parameter_loops = loops
        return loops

    def boundary_parameter(self, uv_norm):
        """Return nearest trim-loop id and cyclic arclength for UV boundary points.

        This is a topology/debugging operation: loop assignment is intentionally
        detached while the graph's node coordinates remain differentiable.
        """
        self._require_active_face()
        device = uv_norm.device if isinstance(uv_norm, torch.Tensor) else self.device
        dtype = uv_norm.dtype if isinstance(uv_norm, torch.Tensor) else torch.float32
        points = (
            uv_norm.detach().cpu().numpy()
            if isinstance(uv_norm, torch.Tensor)
            else np.asarray(uv_norm, dtype=np.float64)
        )
        original_shape = points.shape[:-1]
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        loops = self._boundary_parameter_loops
        if loops is None:
            loops = self._build_boundary_parameter_loops()

        loop_ids = np.full((points.shape[0],), -1, dtype=np.int64)
        parameters = np.zeros((points.shape[0],), dtype=np.float64)
        best_distance2 = np.full((points.shape[0],), np.inf, dtype=np.float64)

        for loop_id, loop in enumerate(loops):
            if np.linalg.norm(loop[0] - loop[-1]) > 1e-10:
                loop = np.concatenate((loop, loop[:1]), axis=0)
            starts = loop[:-1]
            deltas = loop[1:] - starts
            lengths = np.linalg.norm(deltas, axis=1)
            valid = lengths > 1e-12
            starts = starts[valid]
            deltas = deltas[valid]
            lengths = lengths[valid]
            if lengths.size == 0:
                continue
            cumulative = np.concatenate(([0.0], np.cumsum(lengths[:-1])))

            rel = points[:, None, :] - starts[None, :, :]
            fraction = np.sum(rel * deltas[None, :, :], axis=-1) / (
                lengths[None, :] ** 2
            )
            fraction = np.clip(fraction, 0.0, 1.0)
            projected = starts[None, :, :] + fraction[..., None] * deltas[None, :, :]
            distance2 = np.sum((points[:, None, :] - projected) ** 2, axis=-1)
            segment_id = np.argmin(distance2, axis=1)
            row = np.arange(points.shape[0])
            nearest_distance2 = distance2[row, segment_id]
            better = nearest_distance2 < best_distance2
            loop_ids[better] = loop_id
            parameters[better] = (
                cumulative[segment_id[better]]
                + fraction[row[better], segment_id[better]] * lengths[segment_id[better]]
            )
            best_distance2[better] = nearest_distance2[better]

        if np.any(loop_ids < 0):
            raise RuntimeError("Could not assign all UV points to a CAD trim loop.")
        return {
            "loop_id": torch.as_tensor(loop_ids.reshape(original_shape), dtype=torch.long, device=device),
            "parameter": torch.as_tensor(parameters.reshape(original_shape), dtype=dtype, device=device),
        }

    def uv_norm_inside_mask(self, uv_norm, tol: float | None = None):
        """
        Debug/final-filter hard trim mask. Optimization should prefer the SDF.
        """
        face = self._require_active_face()
        tol = self.seed_domain_trim_tol if tol is None else float(tol)
        device = uv_norm.device if isinstance(uv_norm, torch.Tensor) else self.device
        uv_norm_np = (
            uv_norm.detach().cpu().numpy()
            if isinstance(uv_norm, torch.Tensor)
            else np.asarray(uv_norm, dtype=float)
        )
        if uv_norm_np.shape[-1] != 2:
            raise ValueError(f"uv_norm must end with dimension 2, got {uv_norm_np.shape}")

        uv_raw = self.uv_norm_to_raw_from_bounds(
            uv_norm_np,
            u_raw_bounds=self._active_u_raw_bounds,
            v_raw_bounds=self._active_v_raw_bounds,
        )
        inside = self.classify_uv_points_on_face(face, uv_raw, tol=tol)
        return torch.tensor(
            inside.reshape(-1).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )

    def sample_trim_sdf(self, uv_norm):
        """Differentiably sample the active trim SDF grid at normalized UVs."""
        self._require_active_face()
        if self._seed_domain_sdf_grid is None:
            raise RuntimeError("Call generate_from_file(shape_path) before sampling the trim SDF.")

        device = uv_norm.device if isinstance(uv_norm, torch.Tensor) else self.device
        uv_norm = torch.as_tensor(uv_norm, dtype=torch.float32, device=device)

        if uv_norm.shape[-1] != 2:
            raise ValueError(f"uv_norm must end with dimension 2, got {uv_norm.shape}")

        original_shape = uv_norm.shape[:-1]
        uv_flat = uv_norm.reshape(-1, 2)
        grid = torch.empty((1, uv_flat.shape[0], 1, 2), dtype=uv_flat.dtype, device=uv_flat.device)
        grid[0, :, 0, 0] = 2.0 * uv_flat[:, 0] - 1.0
        grid[0, :, 0, 1] = 2.0 * uv_flat[:, 1] - 1.0

        sdf_image = self._seed_domain_sdf_grid.to(device=device, dtype=uv_norm.dtype)
        sdf_image = sdf_image.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            sdf_image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sdf = sampled.reshape(-1).reshape(original_shape)

        uv_min = uv_norm.amin(dim=-1)
        uv_max = uv_norm.amax(dim=-1)
        outside_box = (uv_min < 0.0) | (uv_max > 1.0)

        return torch.where(
            outside_box,
            -torch.ones_like(sdf),
            sdf,
        )

    def smooth_inside_activity(self, uv_norm, tau: float = 0.01):
        sdf = self.sample_trim_sdf(uv_norm)
        return torch.sigmoid(sdf / float(tau))

    # =========================================================================
    # 4) Continuous surface evaluation
    # =========================================================================

    def eval_uv_norm(
        self,
        uv_norm,
        metric_tol: float | None = None,
        trim_tol: float | None = None,
        return_inside_mask: bool = True,
    ):
        """
        Evaluate the active CAD face at normalized UV query points.

        Returns xyz, raw UV, first derivatives, first fundamental form terms,
        area density J, and optionally a hard inside mask.
        """
        self._require_active_face()
        metric_tol = self.metric_tol if metric_tol is None else float(metric_tol)
        trim_tol = self.seed_domain_trim_tol if trim_tol is None else float(trim_tol)

        input_was_tensor = isinstance(uv_norm, torch.Tensor)
        device = uv_norm.device if input_was_tensor else self.device
        uv_norm_np = uv_norm.detach().cpu().numpy() if input_was_tensor else np.asarray(uv_norm, dtype=float)
        if uv_norm_np.shape[-1] != 2:
            raise ValueError(f"uv_norm must end with dimension 2, got {uv_norm_np.shape}")

        query_shape = uv_norm_np.shape[:-1]
        uv_norm_flat = uv_norm_np.reshape(-1, 2)
        uv_raw = self.uv_norm_to_raw_from_bounds(
            uv_norm_flat,
            u_raw_bounds=self._active_u_raw_bounds,
            v_raw_bounds=self._active_v_raw_bounds,
        )

        xyz = np.empty((uv_raw.shape[0], 3), dtype=np.float32)
        Xu = np.empty((uv_raw.shape[0], 3), dtype=np.float32)
        Xv = np.empty((uv_raw.shape[0], 3), dtype=np.float32)

        for i, (u, v) in enumerate(uv_raw):
            p = self._active_surface.Value(float(u), float(v))
            props = GeomLProp_SLProps(self._active_surface, float(u), float(v), 1, metric_tol)
            du = props.D1U()
            dv = props.D1V()
            xyz[i] = [p.X(), p.Y(), p.Z()]
            Xu[i] = [du.X(), du.Y(), du.Z()]
            Xv[i] = [dv.X(), dv.Y(), dv.Z()]

        out_shape_3 = (*query_shape, 3)
        out_shape_2 = (*query_shape, 2)
        xyz_t = torch.tensor(xyz.reshape(out_shape_3), dtype=torch.float32, device=device)
        Xu_t = torch.tensor(Xu.reshape(out_shape_3), dtype=torch.float32, device=device)
        Xv_t = torch.tensor(Xv.reshape(out_shape_3), dtype=torch.float32, device=device)
        uv_norm_t = torch.tensor(uv_norm_np.reshape(out_shape_2), dtype=torch.float32, device=device)
        uv_raw_t = torch.tensor(uv_raw.reshape(out_shape_2), dtype=torch.float32, device=device)

        E = (Xu_t * Xu_t).sum(dim=-1)
        Fm = (Xu_t * Xv_t).sum(dim=-1)
        G = (Xv_t * Xv_t).sum(dim=-1)
        J = torch.linalg.norm(torch.cross(Xu_t, Xv_t, dim=-1), dim=-1)

        out = {
            "uv_norm": uv_norm_t,
            "uv_raw": uv_raw_t,
            "xyz": xyz_t,
            "Xu": Xu_t,
            "Xv": Xv_t,
            "E": E,
            "F": Fm,
            "G": G,
            "J": J,
        }
        if return_inside_mask:
            out["inside_mask"] = self.uv_norm_inside_mask(uv_norm_t, tol=trim_tol).reshape(query_shape)
        return out

    def eval_uv_norm_batch(self, uv_norm, *args, **kwargs):
        return self.eval_uv_norm(uv_norm, *args, **kwargs)

    # =========================================================================
    # 5) Decoder contract
    # =========================================================================

    def generate_from_file(self, shape_path: str):
        """
        Load a single-face CAD file and return continuous trimmed UV metadata.

        No mesh vertices, mesh faces, nearest-neighbor projections, or
        selected-face logic appear in this decoder path.
        """
        shape = self.load_shape(shape_path)
        face = self.get_single_face(shape)
        (umin, umax, vmin, vmax), surf = self.face_uv_bounds_and_surface(face)
        surface_u_periodic, surface_v_periodic, u_period, v_period = self.face_uv_periodicity(face)

        u_span = abs(float(umax) - float(umin))
        v_span = abs(float(vmax) - float(vmin))

        self._u_periodic = (
        bool(surface_u_periodic)
        and u_period is not None
        and abs(u_span - float(u_period)) <= max(1e-6, 1e-4 * abs(float(u_period)))
        )

        self._v_periodic = (
        bool(surface_v_periodic)
        and v_period is not None
        and abs(v_span - float(v_period)) <= max(1e-6, 1e-4 * abs(float(v_period)))
        )

        self._u_period = None if not self._u_periodic else float(u_period)
        self._v_period = None if not self._v_periodic else float(v_period)

        mask_grid_np = self.build_seed_domain_mask_grid(
            face=face,
            u_raw_bounds=(umin, umax),
            v_raw_bounds=(vmin, vmax),
            res=self.seed_domain_mask_res,
            trim_tol=self.seed_domain_trim_tol,
        )
        sdf_grid_np = self.build_seed_domain_sdf_grid(mask_grid_np)

        self._active_shape = shape
        self._active_face = face
        self._active_surface = surf
        self._active_u_raw_bounds = (float(umin), float(umax))
        self._active_v_raw_bounds = (float(vmin), float(vmax))
        self._seed_domain_mask_grid = torch.tensor(
            mask_grid_np,
            dtype=torch.float32,
            device=self.device,
        )
        self._seed_domain_sdf_grid = torch.tensor(
            sdf_grid_np,
            dtype=torch.float32,
            device=self.device,
        )
        self._boundary_parameter_loops = None

        return {
            "u_raw_bounds": self._active_u_raw_bounds,
            "v_raw_bounds": self._active_v_raw_bounds,
            "seed_domain_mask_grid": self._seed_domain_mask_grid,
            "seed_domain_sdf_grid": self._seed_domain_sdf_grid,
            "seed_domain_mask_kind": "cad_trim_grid",
            "device": self.device,
            "u_periodic": self._u_periodic,
            "v_periodic": self._v_periodic,
            "u_period": self._u_period,
            "v_period": self._v_period,
        }
    def print_face_info(self):
        self._require_active_face()

        box = Bnd_Box()
        brepbndlib.Add(self._active_face, box)

        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        print("\n=== Active Face Info ===")

        print(
            f"XYZ bounds:\n"
            f"  X: [{xmin:.6f}, {xmax:.6f}]  span={dx:.6f}\n"
            f"  Y: [{ymin:.6f}, {ymax:.6f}]  span={dy:.6f}\n"
            f"  Z: [{zmin:.6f}, {zmax:.6f}]  span={dz:.6f}"
        )

        print(
            f"\nUV bounds:\n"
            f"  U: [{self._active_u_raw_bounds[0]:.6f}, "
            f"{self._active_u_raw_bounds[1]:.6f}]"
        )

        print(
            f"  V: [{self._active_v_raw_bounds[0]:.6f}, "
            f"{self._active_v_raw_bounds[1]:.6f}]"
        )

        print(
            f"\nPeriodic:\n"
            f"  U periodic: {self._u_periodic}\n"
            f"  V periodic: {self._v_periodic}"
        )

        print("========================\n")
