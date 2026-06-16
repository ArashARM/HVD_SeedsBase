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

    def plot_uv_domain(
        self,
        res: int = 256,
        show_sdf: bool = True,
        show_mask: bool = True,
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
        ax.set_xlabel("normalized u")
        ax.set_ylabel("normalized v")
        ax.set_title("CAD Face UV Domain")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    

