import numpy as np
import torch

import matplotlib.pyplot as plt
import pyvista as pv


try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass


class CADDomainVisualizer:
    def __init__(self, cad_generator):
        self.cad = cad_generator

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _require_domain_grids(self):
        mask_grid = getattr(self.cad, "_seed_domain_mask_grid", None)
        sdf_grid = getattr(self.cad, "_seed_domain_sdf_grid", None)
        if mask_grid is None or sdf_grid is None:
            raise RuntimeError("Call cad.generate_from_file(shape_path) before visualizing the CAD domain.")
        return mask_grid, sdf_grid

    @staticmethod
    def _resize_grid_for_display(grid_np, res):
        res = int(res)
        if grid_np.shape == (res, res):
            return grid_np

        grid_t = torch.tensor(grid_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        grid_t = torch.nn.functional.interpolate(
            grid_t,
            size=(res, res),
            mode="bilinear",
            align_corners=True,
        )
        return grid_t.squeeze(0).squeeze(0).numpy()

    @staticmethod
    def _show_figure(fig):
        try:
            from IPython.display import display

            display(fig)
            return
        except Exception:
            pass
        backend = str(plt.get_backend()).lower()
        if "agg" not in backend:
            try:
                plt.show()
            except Exception:
                pass

    def plot_uv_domain(
        self,
        res: int = 256,
        show_sdf: bool = True,
        show_mask: bool = True,
        show_boundary_curves: bool = True,
        show_curve_points: bool = False,
        show: bool = True,
        save_path: str | None = None,
    ):
        mask_grid, sdf_grid = self._require_domain_grids()
        mask_np = self._resize_grid_for_display(self._to_numpy(mask_grid), res)
        sdf_np = self._resize_grid_for_display(self._to_numpy(sdf_grid), res)

        fig, ax = plt.subplots(figsize=(7, 6))
        extent = [0.0, 1.0, 0.0, 1.0]

        if show_sdf:
            im = ax.imshow(
                sdf_np,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                interpolation="bilinear",
                aspect="equal",
            )
            fig.colorbar(im, ax=ax, label="trim SDF")
        elif show_mask:
            im = ax.imshow(
                mask_np,
                origin="lower",
                extent=extent,
                cmap="gray",
                interpolation="nearest",
                aspect="equal",
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(im, ax=ax, label="trim mask")

        if show_mask and show_sdf:
            ax.contour(
                mask_np,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="black",
                linewidths=1.0,
            )

        ax.contour(
            sdf_np,
            levels=[0.0],
            origin="lower",
            extent=extent,
            colors="black",
            linewidths=1.5,
        )
        if show_boundary_curves:
            self._draw_boundary_curve_pieces(
                ax,
                show_curve_points=show_curve_points,
                linewidth=2.0,
            )
        ax.set_xlabel("normalized u")
        ax.set_ylabel("normalized v")
        ax.set_title("CAD Face UV Domain")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            self._show_figure(fig)
        return fig, ax

    def _boundary_curve_data(self):
        if hasattr(self.cad, "boundary_curve_tensors"):
            return self.cad.boundary_curve_tensors(as_torch=False)
        raise RuntimeError("CAD generator does not expose boundary_curve_tensors().")

    def _draw_boundary_curve_pieces(
        self,
        ax,
        show_curve_points: bool = False,
        linewidth: float = 2.0,
    ):
        data = self._boundary_curve_data()
        uv = np.asarray(data["boundary_curve_uv"], dtype=float)
        offsets = np.asarray(data["boundary_curve_offsets"], dtype=np.int64)
        loop_kind = np.asarray(data["boundary_curve_loop_kind"], dtype=np.int64)
        if offsets.size <= 1:
            return
        cmap = plt.get_cmap("tab20")
        for piece_id in range(offsets.size - 1):
            start = int(offsets[piece_id])
            end = int(offsets[piece_id + 1])
            points = uv[start:end]
            if points.shape[0] < 2:
                continue
            color = cmap(piece_id % cmap.N)
            linestyle = "-" if int(loop_kind[piece_id]) == 0 else "--"
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"piece {piece_id}" if piece_id < 20 else None,
            )
            ax.scatter(
                points[0, 0],
                points[0, 1],
                color=color,
                s=24,
                marker="o",
                zorder=4,
            )
            ax.scatter(
                points[-1, 0],
                points[-1, 1],
                color=color,
                s=24,
                marker="x",
                zorder=4,
            )
            if show_curve_points:
                ax.scatter(points[:, 0], points[:, 1], color=color, s=6, alpha=0.5)

    def plot_uv_boundary_curves(
        self,
        show_sdf_background: bool = True,
        show_curve_points: bool = True,
        annotate: bool = True,
        show: bool = True,
        save_path: str | None = None,
    ):
        mask_grid, sdf_grid = self._require_domain_grids()
        sdf_np = self._to_numpy(sdf_grid)
        fig, ax = plt.subplots(figsize=(7, 6))
        if show_sdf_background:
            ax.imshow(
                sdf_np,
                origin="lower",
                extent=[0.0, 1.0, 0.0, 1.0],
                cmap="Greys",
                alpha=0.25,
                interpolation="bilinear",
                aspect="equal",
            )
        self._draw_boundary_curve_pieces(
            ax,
            show_curve_points=show_curve_points,
            linewidth=2.2,
        )
        data = self._boundary_curve_data()
        uv = np.asarray(data["boundary_curve_uv"], dtype=float)
        offsets = np.asarray(data["boundary_curve_offsets"], dtype=np.int64)
        if annotate:
            for piece_id in range(offsets.size - 1):
                start = int(offsets[piece_id])
                end = int(offsets[piece_id + 1])
                points = uv[start:end]
                if points.shape[0] == 0:
                    continue
                mid = points[points.shape[0] // 2]
                ax.text(mid[0], mid[1], str(piece_id), fontsize=8, ha="center", va="center")
        num_pieces = int(np.asarray(data["boundary_curve_num_pieces"]).reshape(()))
        ax.set_title(f"C1-Split UV Boundary Curves ({num_pieces} pieces)")
        ax.set_xlabel("normalized u")
        ax.set_ylabel("normalized v")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            self._show_figure(fig)
        return fig, ax

    def plot_uv_voronoi_graph(
        self,
        graph_or_out,
        show_domain: bool = True,
        show_trim_nodes: bool = True,
        show: bool = True,
        save_path: str | None = None,
    ):
        edge_curves_uv = None
        if isinstance(graph_or_out, dict) and "edge_curves_uv" in graph_or_out:
            edge_curves_uv = self._to_numpy(graph_or_out["edge_curves_uv"])
        graph = graph_or_out.get("graph", graph_or_out) if isinstance(graph_or_out, dict) else graph_or_out
        if show_domain:
            fig, ax = self.plot_uv_domain(show=False, show_curve_points=False)
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            self._draw_boundary_curve_pieces(ax, show_curve_points=False, linewidth=2.0)

        nodes = self._to_numpy(graph["nodes_uv"])
        edges = self._to_numpy(graph["edge_index"]).astype(np.int64)
        edge_type = self._to_numpy(graph.get("edge_type", np.zeros((edges.shape[0],), dtype=np.int64))).astype(np.int64)
        source_type = self._to_numpy(graph.get("boundary_source_type", graph.get("node_type", np.zeros((nodes.shape[0],), dtype=np.int64)))).astype(np.int64)

        colors = {
            0: "tab:blue",
            1: "tab:orange",
            3: "tab:red",
            4: "black",
        }
        for edge_id, (a, b) in enumerate(edges):
            if a < 0 or b < 0 or a >= nodes.shape[0] or b >= nodes.shape[0]:
                continue
            etype = int(edge_type[edge_id]) if edge_id < edge_type.shape[0] else 0
            color = colors.get(etype, "tab:gray")
            linewidth = 1.6 if etype != 4 else 1.0
            alpha = 0.9 if etype != 4 else 0.55
            if edge_curves_uv is not None and edge_id < edge_curves_uv.shape[0]:
                curve = edge_curves_uv[edge_id]
                ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=linewidth, alpha=alpha)
            else:
                ax.plot(
                    [nodes[a, 0], nodes[b, 0]],
                    [nodes[a, 1], nodes[b, 1]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )

        interior = source_type == 0
        boundary = source_type != 0
        if np.any(interior):
            ax.scatter(nodes[interior, 0], nodes[interior, 1], s=18, color="tab:blue", zorder=5)
        if np.any(boundary):
            ax.scatter(nodes[boundary, 0], nodes[boundary, 1], s=22, color="tab:orange", zorder=6)
        if show_trim_nodes and np.any(source_type == 5):
            trim = source_type == 5
            ax.scatter(nodes[trim, 0], nodes[trim, 1], s=42, facecolors="none", edgecolors="tab:red", linewidths=1.6, zorder=7)

        ax.set_title("Trim-Aware Voronoi Graph in UV")
        ax.set_xlabel("normalized u")
        ax.set_ylabel("normalized v")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            self._show_figure(fig)
        return fig, ax

    @torch.no_grad()
    def sample_surface_grid(
        self,
        res_u: int = 160,
        res_v: int = 160,
        inside_only: bool = True,
    ):
        res_u = int(res_u)
        res_v = int(res_v)
        if res_u < 2 or res_v < 2:
            raise ValueError("res_u and res_v must both be >= 2.")

        u = torch.linspace(0.0, 1.0, res_u, dtype=torch.float32, device=self.cad.device)
        v = torch.linspace(0.0, 1.0, res_v, dtype=torch.float32, device=self.cad.device)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack((uu.reshape(-1), vv.reshape(-1)), dim=-1)

        out = self.cad.eval_uv_norm_batch(uv, return_inside_mask=True)
        xyz = out["xyz"]
        inside_mask = out["inside_mask"].reshape(-1).to(dtype=torch.bool)

        return {
            "uv": uv,
            "xyz": xyz,
            "inside_mask": inside_mask,
            "res_u": res_u,
            "res_v": res_v,
        }

    def build_pyvista_surface(
        self,
        res_u: int = 160,
        res_v: int = 160,
    ):
        samples = self.sample_surface_grid(res_u=res_u, res_v=res_v, inside_only=True)
        xyz = self._to_numpy(samples["xyz"]).reshape(-1, 3)
        inside = self._to_numpy(samples["inside_mask"]).reshape(-1).astype(bool)
        res_u = int(samples["res_u"])
        res_v = int(samples["res_v"])

        faces = []
        for j in range(res_v - 1):
            for i in range(res_u - 1):
                a = j * res_u + i
                b = j * res_u + i + 1
                c = (j + 1) * res_u + i + 1
                d = (j + 1) * res_u + i

                if inside[a] and inside[b] and inside[c]:
                    faces.extend([3, a, b, c])
                if inside[a] and inside[c] and inside[d]:
                    faces.extend([3, a, c, d])

        faces_np = np.asarray(faces, dtype=np.int64)
        return pv.PolyData(xyz, faces_np)

    def show_3d(
        self,
        res_u: int = 160,
        res_v: int = 160,
        show_edges: bool = False,
        color: str = "lightgray",
    ):
        mesh = self.build_pyvista_surface(res_u=res_u, res_v=res_v)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color=color, show_edges=show_edges)
        plotter.add_axes()
        plotter.show()
        return plotter

    def show_all(
        self,
        res: int = 256,
        res_u: int = 160,
        res_v: int = 160,
        show_sdf: bool = True,
        show_mask: bool = True,
        show_edges: bool = False,
        color: str = "lightgray",
        save_path: str | None = None,
    ):
        fig, ax = self.plot_uv_domain(
            res=res,
            show_sdf=show_sdf,
            show_mask=show_mask,
            save_path=save_path,
        )
        plotter = self.show_3d(
            res_u=res_u,
            res_v=res_v,
            show_edges=show_edges,
            color=color,
        )
        return fig, ax, plotter
    
