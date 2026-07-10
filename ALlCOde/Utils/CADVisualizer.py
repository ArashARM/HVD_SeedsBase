import numpy as np
import torch
import pyvista as pv
try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass
class CADVisualizer:
    """
    Visualization utilities for shell meshes, boundary conditions,
    densities, and seed points.
    """

    @staticmethod
    def to_numpy(x):
        """Convert torch / list / numpy input to numpy array."""
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # =========================================================================
    # 1) BC + load visualization
    # =========================================================================

    @classmethod
    def plot_bc_and_load_pyvista(
        cls,
        points_xyz,              # torch (N,3) or np (N,3)
        faces_ijk,               # torch (F,3) or np (F,3)
        fixed_nodes,             # torch (M,) or np/list
        load_nodes,              # int or list/np/torch
        load_direction=(0.0, 1.0, 0.0),
        arrow_scale=0.15,
        mesh_opacity=0.6,
        show_ids=False,
        title="BCs and Loads",
    ):
        """
        Visualize shell mesh with:
          - fixed nodes in red
          - loaded node(s) in blue
          - load direction arrows in green
        """
        points_np = cls.to_numpy(points_xyz)
        faces_np = cls.to_numpy(faces_ijk).astype(np.int64)

        fixed_np = cls.to_numpy(fixed_nodes).astype(np.int64).reshape(-1)

        if np.isscalar(load_nodes):
            load_np = np.array([int(load_nodes)], dtype=np.int64)
        else:
            load_np = cls.to_numpy(load_nodes).astype(np.int64).reshape(-1)

        faces_pv = np.hstack([
            np.full((faces_np.shape[0], 1), 3, dtype=np.int64),
            faces_np
        ])

        mesh = pv.PolyData(points_np, faces_pv)

        fixed_points = points_np[fixed_np] if fixed_np.size else np.zeros((0, 3))
        load_points = points_np[load_np] if load_np.size else np.zeros((0, 3))

        load_dir = np.asarray(load_direction, dtype=float).reshape(3)
        nrm = np.linalg.norm(load_dir)
        if nrm < 1e-12:
            raise ValueError("load_direction must be a non-zero 3D vector.")
        load_dir = load_dir / nrm

        bbox_size = float(np.linalg.norm(np.ptp(points_np, axis=0)))
        length = float(arrow_scale * bbox_size)

        plotter = pv.Plotter()
        plotter.add_title(title, font_size=14)

        plotter.add_mesh(mesh, color="lightgray", opacity=mesh_opacity)

        if fixed_points.shape[0] > 0:
            plotter.add_points(
                fixed_points,
                color="red",
                point_size=10,
                render_points_as_spheres=True,
            )
            fixed_centroid = fixed_points.mean(axis=0)
            plotter.add_point_labels(
                fixed_centroid.reshape(1, 3),
                ["Fixed (clamped)"],
                font_size=16,
                text_color="red",
            )

        if load_points.shape[0] > 0:
            plotter.add_points(
                load_points,
                color="blue",
                point_size=15,
                render_points_as_spheres=True,
            )

            plotter.add_point_labels(
                load_points[:1],
                [f"Loaded node(s): {load_np.size}"],
                font_size=16,
                text_color="blue",
            )

            for p in load_points:
                arrow = pv.Arrow(start=p, direction=load_dir, scale=length)
                plotter.add_mesh(arrow, color="green")

            tip = load_points[0] + load_dir * length
            plotter.add_point_labels(
                tip.reshape(1, 3),
                [f"Load direction ({load_dir[0]:.2f}, {load_dir[1]:.2f}, {load_dir[2]:.2f})"],
                font_size=16,
                text_color="green",
            )

        if show_ids:
            mesh["ids"] = np.arange(points_np.shape[0])
            plotter.add_point_labels(mesh, "ids", font_size=10)

        plotter.show()
        return plotter

    # =========================================================================
    # 2) Seed collection
    # =========================================================================

    @staticmethod
    @torch.no_grad()
    def collect_seed_points_3d(models, datasets):
        """
        Returns seed_points_3d: (S_total, 3) numpy array

        Seed UV -> nearest vertex UV -> corresponding XYZ.
        Expected:
            datasets[f]["uv"]  : (Nf,2) torch
            datasets[f]["xyz"] : (Nf,3) torch
            models[f].seeds_uv() -> (Sf,2)
        """
        seed_pts = []

        for f, m in models.items():
            data = datasets[f]
            uv_f = data["uv"]
            xyz_f = data["xyz"]

            seeds_uv = m.seeds_uv()
            nn = torch.cdist(seeds_uv, uv_f).argmin(dim=1)
            seed_xyz = xyz_f[nn]

            seed_pts.append(seed_xyz.detach().cpu().numpy())

        if len(seed_pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        return np.vstack(seed_pts)

    # =========================================================================
    # 3) Density normalization helper
    # =========================================================================

    @staticmethod
    def viz_normalize(d, lo_q=0.02, hi_q=0.98):
        lo = torch.quantile(d, lo_q)
        hi = torch.quantile(d, hi_q)
        return (d.clamp(lo, hi) - lo) / (hi - lo + 1e-8)

    # =========================================================================
    # 4) 3-stage density + seed visualization
    # =========================================================================

    @classmethod
    def plot_density_and_seedpoints_3stage(
        cls,
        mesh_points,
        pv_faces,
        density_init,
        density_mid,
        density_final,
        seed_points_init,
        seed_points_mid,
        seed_points_final,
        window_size=(1600, 900),
        cmap_density="viridis",
        point_size=5,
        link_views=True,
        shared_clim=True,
        q_lo=0.02,
        q_hi=0.98,
    ):
        mesh_points = cls.to_numpy(mesh_points)
        pv_faces = cls.to_numpy(pv_faces)

        m0 = pv.PolyData(mesh_points, pv_faces)
        m1 = pv.PolyData(mesh_points, pv_faces)
        m2 = pv.PolyData(mesh_points, pv_faces)

        m0["density"] = np.asarray(density_init)
        m1["density"] = np.asarray(density_mid)
        m2["density"] = np.asarray(density_final)

        if shared_clim:
            all_d = np.concatenate([m0["density"], m1["density"], m2["density"]])
            vmin = np.quantile(all_d, q_lo)
            vmax = np.quantile(all_d, q_hi)
            clim0 = clim1 = clim2 = [vmin, vmax]
        else:
            clim0 = [np.quantile(m0["density"], q_lo), np.quantile(m0["density"], q_hi)]
            clim1 = [np.quantile(m1["density"], q_lo), np.quantile(m1["density"], q_hi)]
            clim2 = [np.quantile(m2["density"], q_lo), np.quantile(m2["density"], q_hi)]

        pl = pv.Plotter(shape=(2, 3), window_size=window_size)

        pl.subplot(0, 0)
        pl.add_text("Initial Density", font_size=12)
        pl.add_mesh(m0, scalars="density", cmap=cmap_density, clim=clim0)
        pl.show_axes()

        pl.subplot(0, 1)
        pl.add_text("Middle Density", font_size=12)
        pl.add_mesh(m1, scalars="density", cmap=cmap_density, clim=clim1)
        pl.show_axes()

        pl.subplot(0, 2)
        pl.add_text("Final Density", font_size=12)
        pl.add_mesh(m2, scalars="density", cmap=cmap_density, clim=clim2)
        pl.show_axes()

        pl.subplot(1, 0)
        pl.add_text("Initial Seeds", font_size=12)
        pl.add_mesh(m0, color="lightgray", opacity=1.0)
        pl.add_mesh(seed_points_init, render_points_as_spheres=True, point_size=point_size, color="red")
        pl.show_axes()

        pl.subplot(1, 1)
        pl.add_text("Middle Seeds", font_size=12)
        pl.add_mesh(m1, color="lightgray", opacity=1.0)
        pl.add_mesh(seed_points_mid, render_points_as_spheres=True, point_size=point_size, color="red")
        pl.show_axes()

        pl.subplot(1, 2)
        pl.add_text("Final Seeds", font_size=12)
        pl.add_mesh(m2, color="lightgray", opacity=1.0)
        pl.add_mesh(seed_points_final, render_points_as_spheres=True, point_size=point_size, color="red")
        pl.show_axes()

        if link_views:
            pl.link_views()

        pl.show()
        return pl
    
    @classmethod
    def plot_density_and_seedpoints_3stage_2(
        cls,
        mesh_points,
        pv_faces,
        density_init,
        density_mid,
        density_final,
        seed_points_init,
        seed_points_mid,
        seed_points_final,
        window_size=(1600, 900),
        point_size=5,
        link_views=True,
        show_shell_background=True,
    ):
        import numpy as np
        import pyvista as pv

        mesh_points = cls.to_numpy(mesh_points)
        pv_faces = cls.to_numpy(pv_faces)

        m0 = pv.PolyData(mesh_points, pv_faces)
        m1 = pv.PolyData(mesh_points, pv_faces)
        m2 = pv.PolyData(mesh_points, pv_faces)

        d0 = np.asarray(density_init).reshape(-1)
        d1 = np.asarray(density_mid).reshape(-1)
        d2 = np.asarray(density_final).reshape(-1)

        m0["density"] = d0
        m1["density"] = d1
        m2["density"] = d2

        pl = pv.Plotter(shape=(2, 3), window_size=window_size)

        # --- top row: real density field ---
        pl.subplot(0, 0)
        pl.add_text("Initial Density", font_size=12)
        if show_shell_background:
            pl.add_mesh(m0, color="lightgray", opacity=0.15)
        pl.add_mesh(
            m0,
            scalars="density",
            cmap="viridis",
            clim=[0.0, 1.0],
            show_edges=False,
            scalar_bar_args={"title": "Density"},
        )
        pl.show_axes()

        pl.subplot(0, 1)
        pl.add_text("Middle Density", font_size=12)
        if show_shell_background:
            pl.add_mesh(m1, color="lightgray", opacity=0.15)
        pl.add_mesh(
            m1,
            scalars="density",
            cmap="viridis",
            clim=[0.0, 1.0],
            show_edges=False,
            scalar_bar_args={"title": "Density"},
        )
        pl.show_axes()

        pl.subplot(0, 2)
        pl.add_text("Final Density", font_size=12)
        if show_shell_background:
            pl.add_mesh(m2, color="lightgray", opacity=0.15)
        pl.add_mesh(
            m2,
            scalars="density",
            cmap="viridis",
            clim=[0.0, 1.0],
            show_edges=False,
            scalar_bar_args={"title": "Density"},
        )
        pl.show_axes()

        # --- bottom row: seeds on shell ---
        pl.subplot(1, 0)
        pl.add_text("Initial Seeds", font_size=12)
        pl.add_mesh(m0, color="lightgray", opacity=1.0)
        pl.add_mesh(
            seed_points_init,
            render_points_as_spheres=True,
            point_size=point_size,
            color="red",
        )
        pl.show_axes()

        pl.subplot(1, 1)
        pl.add_text("Middle Seeds", font_size=12)
        pl.add_mesh(m1, color="lightgray", opacity=1.0)
        pl.add_mesh(
            seed_points_mid,
            render_points_as_spheres=True,
            point_size=point_size,
            color="red",
        )
        pl.show_axes()

        pl.subplot(1, 2)
        pl.add_text("Final Seeds", font_size=12)
        pl.add_mesh(m2, color="lightgray", opacity=1.0)
        pl.add_mesh(
            seed_points_final,
            render_points_as_spheres=True,
            point_size=point_size,
            color="red",
        )
        pl.show_axes()

        if link_views:
            pl.link_views()

        pl.show()
        return pl

    @classmethod
    def visualize_density_thresholded(
        cls,
        points,
        pv_faces,
        density_total,
        edge_ratio=None,
        thr: float | None = None,
        clip_max: float = 0.5,
        ratio_thresh: float = 0.20,
        edge_density_percentile: float = 20.0,
        min_threshold: float = 0.0,
        tube_radius: float | None = None,
        show_solid: bool = True,
        show_base_mesh: bool = True,
        base_opacity: float = 0.05,
        export_stl: str | None = None,
        verbose: bool = True,
    ):
        P = cls.to_numpy(points).astype(np.float32)
        rho = cls.to_numpy(density_total).astype(np.float64).reshape(-1)

        F_in = cls.to_numpy(pv_faces)
        if F_in.ndim == 2 and F_in.shape[1] == 3:
            F = F_in.astype(np.int64)
            faces_pv = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
        else:
            faces_pv = F_in.astype(np.int64).ravel()

        mesh = pv.PolyData(P, faces_pv)

        rho_clip = np.clip(rho, 0.0, float(clip_max)).astype(np.float32)
        mesh["rho_clip"] = rho_clip
        mesh["rho"] = rho.astype(np.float32)

        reason = "manual"
        if thr is None:
            if edge_ratio is not None:
                ratio = cls.to_numpy(edge_ratio).astype(np.float32).reshape(-1)
                mesh["edge_ratio"] = ratio
                edge_mask = ratio > float(ratio_thresh)
                edge_vals = rho[edge_mask]

                if edge_vals.size:
                    thr = float(np.percentile(edge_vals, float(edge_density_percentile)))
                    reason = f"auto: {edge_density_percentile}% of rho on edges (ratio>{ratio_thresh})"
                else:
                    thr = float(np.percentile(rho[rho > 0], 99)) if np.any(rho > 0) else 0.0
                    reason = "auto: no edge pts -> fallback p99 of rho"
            else:
                thr = float(np.percentile(rho[rho > 0], 99)) if np.any(rho > 0) else 0.0
                reason = "auto: no edge_ratio -> fallback p99 of rho"

            thr = max(float(thr), float(min_threshold))

        rho_bin = (rho > float(thr)).astype(np.uint8)
        mesh["rho_bin"] = rho_bin

        if verbose:
            solid_pct = 100.0 * rho_bin.mean()
            msg = f"threshold={thr:.6g} ({reason}) | solid%={solid_pct:.3f}%"
            if edge_ratio is not None and "edge_ratio" in mesh.array_names:
                edge_mask = mesh["edge_ratio"] > float(ratio_thresh)
                msg = f"edge-like%={100.0 * edge_mask.mean():.3f}% | " + msg
            print(msg)

        solid = mesh.threshold(value=0.5, scalars="rho_bin")

        pl = pv.Plotter()

        if show_base_mesh:
            pl.add_mesh(mesh, opacity=float(base_opacity), show_edges=False)

        if tube_radius is None:
            if show_solid:
                pl.add_mesh(solid, scalars="rho_bin", clim=[0, 1], categories=True, cmap=["white", "gray"])
            else:
                pl.add_mesh(solid, scalars="rho_clip")
        else:
            edges = solid.extract_all_edges()
            tubes = edges.tube(radius=float(tube_radius))
            pl.add_mesh(tubes, scalars="rho_clip")
            if export_stl is not None:
                tubes.save(export_stl)
                print("Saved STL to:", export_stl)

        pl.show_axes()
        pl.show()
        return solid, float(thr), pl
    
    def visualize_show_Model(
        cls,
        points,
        pv_faces,
    ):
        P = cls.to_numpy(points).astype(np.float32)

        F_in = cls.to_numpy(pv_faces)
        if F_in.ndim == 2 and F_in.shape[1] == 3:
            F = F_in.astype(np.int64)
            faces_pv = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
        else:
            faces_pv = F_in.astype(np.int64).ravel()

        mesh = pv.PolyData(P, faces_pv)
        pl = pv.Plotter()
        pl.add_mesh(mesh, opacity=1.0, show_edges=True)
        pl.show_axes()
        pl.show()