import os
import shutil   
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
import re


class TimelapseRecorder:
    def __init__(
        self,
        out_dir="timelapse_frames",
        video_path="timelapse.mp4",
        fps=10,
        header_title="",
        header_subtitle="",
    ):
        self.out_dir = out_dir
        self.video_path = video_path
        self.fps = fps
        self.header_title = str(header_title or "")
        self.header_subtitle = str(header_subtitle or "")
        self.frame_width = 2560
        self.frame_height = 1440
        os.makedirs(out_dir, exist_ok=True)
        self.frame_paths = []

    @staticmethod
    def _add_panel_border(img, pad=12, border=2, bg_color=(247, 247, 245), border_color=(180, 186, 195)):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        inner = cv2.copyMakeBorder(
            img,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color,
        )
        return cv2.copyMakeBorder(
            inner,
            border,
            border,
            border,
            border,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )

    @staticmethod
    def _select_video_writer(video_path, fps, frame_size):
        root, ext = os.path.splitext(video_path)
        ext = ext.lower()

        codec_candidates = []
        if ext == ".avi":
            codec_candidates = [
                ("MJPG", video_path),
            ]
        else:
            codec_candidates = [
                ("avc1", video_path),
                ("mp4v", video_path),
                ("MJPG", root + ".avi"),
            ]

        for fourcc_name, out_path in codec_candidates:
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*fourcc_name),
                fps,
                frame_size,
            )
            if writer.isOpened():
                return writer, out_path, fourcc_name
            writer.release()

        raise RuntimeError("Could not open any supported video writer codec.")

    @staticmethod
    def _parse_summary_metrics(title_text):
        metrics = []
        for chunk in title_text.split("|"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                key, value = chunk.split("=", 1)
                metrics.append((key.strip(), value.strip()))
            else:
                metrics.append(("stage", chunk))
        return metrics

    @staticmethod
    def _format_metric_label(key):
        label_map = {
            "iter": "Iteration",
            "vol": "Vol. Fraction",
            "vol_total": "VF_total",
            "vol_internal": "VF_int",
            "vol_eff": "VF_eff_int",
            "total_volume_fraction": "VF_total",
            "efficient_total_volume_fraction": "VF_eff_total",
            "interior_voronoi_edges_only_volume_fraction": "VF_int",
            "efficient_interior_voronoi_edges_only_volume_fraction": "VF_eff_int",
            "Vol_total": "VF_total",
            "Vol_internal": "VF_int",
            "Vol_eff": "VF_eff_int",
            "VF_total": "VF_total",
            "VF_eff_total": "VF_eff_total",
            "VF_int": "VF_int",
            "VF_eff_int": "VF_eff_int",
            "W": "Mean Width",
            "bw": "Bw",
            "compute_time": "Com. Time",
            "fem_elems": "FEM Elems",
            "mesh_pts": "SurfacePts",
            "bbox": "BBox",
            "load_F": "F",
            "Δrho": "Delta Rho",
            "Δseed": "Delta Seed",
            "grad_mean": "Mean Grad.",
            "stage": "Status",
        }
        return label_map.get(key, key.replace("_", " ").title())

    @staticmethod
    def _composite_to_white(img):
        if img.ndim != 3:
            return img
        if img.shape[2] == 3:
            return img
        if img.shape[2] != 4:
            return img[..., :3]

        rgb = img[..., :3].astype(np.float32)
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        white = np.full_like(rgb, 255.0)
        out = rgb * alpha + white * (1.0 - alpha)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _wrap_cv2_text(text, max_width, font, font_scale, thickness):
        words = str(text or "").split()
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            text_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
            if current and text_width > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [""]

    def _draw_header(self, canvas, highlight_best=False):
        if not self.header_title and not self.header_subtitle:
            return 0

        header_h = 132
        bg = (230, 247, 234) if highlight_best else (244, 247, 251)
        accent = (34, 139, 34) if highlight_best else (37, 99, 235)
        text = (17, 24, 39)
        muted = (75, 85, 99)

        cv2.rectangle(canvas, (0, 0), (self.frame_width, header_h), bg, thickness=-1)
        cv2.rectangle(canvas, (0, header_h - 4), (self.frame_width, header_h), accent, thickness=-1)

        x = 44
        title_y = 48
        subtitle_y = 88
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.header_title:
            max_title_width = self.frame_width - (2 * x)
            title_scale = 1.0
            title_thickness = 2
            title_width = cv2.getTextSize(
                self.header_title,
                font,
                title_scale,
                title_thickness,
            )[0][0]
            if title_width > max_title_width:
                title_scale = max(0.68, title_scale * max_title_width / max(title_width, 1))
            cv2.putText(
                canvas,
                self.header_title,
                (x, title_y),
                font,
                title_scale,
                text,
                title_thickness,
                cv2.LINE_AA,
            )

        if self.header_subtitle:
            max_width = self.frame_width - (2 * x)
            lines = self._wrap_cv2_text(
                self.header_subtitle,
                max_width=max_width,
                font=font,
                font_scale=0.62,
                thickness=1,
            )
            for line_idx, line in enumerate(lines[:2]):
                cv2.putText(
                    canvas,
                    line,
                    (x, subtitle_y + line_idx * 26),
                    font,
                    0.62,
                    muted,
                    1,
                    cv2.LINE_AA,
                )

        return header_h

    def _make_loss_chart(
        self,
        loss_dict,
        title_text="",
        height=700,
        width=700,
        highlight_best=False,
        chart_title=None,
        summary_title=None,
        results_title=None,
        results_text="",
    ):
        keys = list(loss_dict.keys())
        vals = [float(loss_dict[k]) for k in keys]
        fig_bg = "#eef7ef" if highlight_best else "#f7f7f5"
        card_bg = "#f6fff7" if highlight_best else "#ffffff"
        card_edge = "#16a34a" if highlight_best else "#d1d5db"
        bar_color = "#16a34a" if highlight_best else "#2563eb"
        edge_color = "#15803d" if highlight_best else "#1d4ed8"
        header_bg = "#14532d" if highlight_best else None
        header_fg = "#f0fdf4" if highlight_best else "#111827"
        summary_title = (
            summary_title
            if summary_title is not None
            else ("Tuned Parameters" if highlight_best else "Run Summary")
        )

        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=fig_bg)
        if highlight_best:
            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[1.42, 1.24],
                left=0.18,
                right=0.96,
                top=0.94,
                bottom=0.07,
                hspace=0.20,
            )
            ax_results = None
            ax = fig.add_subplot(gs[0, 0])
            ax_text = fig.add_subplot(gs[1, 0])
        else:
            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[1.8, 1.0],
                left=0.16,
                right=0.96,
                top=0.94,
                bottom=0.07,
                hspace=0.22,
            )
            ax_results = None
            ax = fig.add_subplot(gs[0, 0])
            ax_text = fig.add_subplot(gs[1, 0])

        y = np.arange(len(keys))
        ax.barh(y, vals, color=bar_color, edgecolor=edge_color, alpha=0.88, height=0.62)
        ax.set_yticks(y, labels=keys, fontsize=12)
        ax.invert_yaxis()
        ax.set_title(
            chart_title if chart_title is not None else ("Tuned Parameters" if highlight_best else "Optimization Losses"),
            fontsize=18,
            pad=12,
            weight="bold",
        )
        ax.set_facecolor(card_bg)
        ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        xmax = max(vals) if len(vals) else 1.0
        xmax = max(xmax * 1.18, 1e-8)
        ax.set_xlim(0.0, xmax)

        for yi, v in zip(y, vals):
            ax.text(
                min(v + 0.02 * xmax, 0.98 * xmax),
                yi,
                f"{v:.3g}",
                va="center",
                ha="left",
                fontsize=11,
                color="#0f172a",
                weight="semibold",
            )

        ax_text.axis("off")
        metrics = self._parse_summary_metrics(title_text)
        if highlight_best and results_text:
            metrics = self._parse_summary_metrics(results_text) + metrics

        ax_text.text(
            0.0,
            1.0,
            "Best Result Summary" if highlight_best else summary_title,
            fontsize=18 if highlight_best else 16,
            weight="bold",
            va="top",
            ha="left",
            color="#111827",
            transform=ax_text.transAxes,
        )

        card_x = 0.0
        card_y = 0.08
        card_w = 0.94
        card_h = 0.82 if highlight_best else 0.76
        card = plt.Rectangle(
            (card_x, card_y),
            card_w,
            card_h,
            transform=ax_text.transAxes,
            facecolor=card_bg,
            edgecolor=card_edge,
            linewidth=2.0 if highlight_best else 1.0,
        )
        ax_text.add_patch(card)

        rows = metrics
        if rows:
            n_cols = 2 if len(rows) > 7 else 1
            n_rows = int(np.ceil(len(rows) / n_cols))
            start_y = card_y + card_h - (0.12 if highlight_best else 0.12)
            row_step = min(0.16 if highlight_best else 0.14, (card_h - 0.13) / max(n_rows, 1))

            for idx, (key, value) in enumerate(rows):
                col_idx = idx // n_rows
                row_idx = idx % n_rows
                left_x = card_x + 0.04 + col_idx * 0.46
                label = self._format_metric_label(key)
                long_label = len(label) > 32
                right_x = left_x + (
                    0.36 if long_label else (0.20 if highlight_best else 0.18)
                )
                y = start_y - row_idx * row_step
                if long_label:
                    ax_text.text(
                        left_x,
                        y + 0.022,
                        label,
                        fontsize=7.4,
                        weight="bold" if highlight_best else "semibold",
                        va="center",
                        ha="left",
                        color="#374151",
                        transform=ax_text.transAxes,
                    )
                    ax_text.text(
                        left_x,
                        y - 0.024,
                        value,
                        fontsize=11 if highlight_best else 10,
                        va="center",
                        ha="left",
                        color="#111827",
                        weight="bold" if highlight_best else "normal",
                        family="monospace",
                        transform=ax_text.transAxes,
                    )
                else:
                    ax_text.text(
                        left_x,
                        y,
                        label,
                        fontsize=12 if highlight_best else 11,
                        weight="bold" if highlight_best else "semibold",
                        va="center",
                        ha="left",
                        color="#374151",
                        transform=ax_text.transAxes,
                    )
                    ax_text.text(
                        right_x,
                        y,
                        value,
                        fontsize=12 if highlight_best else 11,
                        va="center",
                        ha="left",
                        color="#111827",
                        weight="bold" if highlight_best else "normal",
                        family="monospace",
                        transform=ax_text.transAxes,
                    )

                if row_idx != n_rows - 1 and idx + 1 < len(rows):
                    ax_text.plot(
                        [left_x, min(left_x + 0.39, card_x + card_w - 0.03)],
                        [y - 0.055, y - 0.055],
                        color="#e5e7eb",
                        linewidth=0.8,
                        transform=ax_text.transAxes,
                    )


        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3].copy()   # RGB
        plt.close(fig)
        return img

    def add_frame(
        self,
        step,
        cad_img,
        loss_dict,
        title_text="",
        highlight_best=False,
        chart_title=None,
        summary_title=None,
        prefix_step_in_summary=True,
        results_title=None,
        results_text="",
    ):
        if cad_img is None:
            raise ValueError("cad_img is None")

        if cad_img.ndim != 3 or cad_img.shape[2] not in (3, 4):
            raise ValueError(f"cad_img must be HxWx3 or HxWx4, got shape {cad_img.shape}")

        cad_img = self._composite_to_white(cad_img)
        # PyVista screenshots arrive as RGB; convert once before mixing with OpenCV BGR images.
        cad_img = cv2.cvtColor(cad_img, cv2.COLOR_RGB2BGR)

        h_left, w_left = cad_img.shape[:2]
        summary_text = title_text
        if prefix_step_in_summary:
            summary_text = f"iter {step} | {title_text}"

        chart_img = self._make_loss_chart(
            loss_dict=loss_dict,
            title_text=summary_text,
            height=h_left,
            width=max(980, int(0.42 * w_left)),
            highlight_best=highlight_best,
            chart_title=chart_title,
            summary_title=summary_title,
            results_title=results_title,
            results_text=results_text,
        )

        h_right, w_right = chart_img.shape[:2]

        target_h = max(h_left, h_right)

        if h_left != target_h:
            cad_img = cv2.resize(cad_img, (w_left, target_h))
        if h_right != target_h:
            chart_img = cv2.resize(chart_img, (w_right, target_h))

        # matplotlib gives RGB, cv2 prefers BGR when writing
        chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)
        chart_img = self._add_panel_border(chart_img)

        gap = 24
        target_h = max(cad_img.shape[0], chart_img.shape[0])
        if cad_img.shape[0] != target_h:
            cad_img = cv2.resize(cad_img, (cad_img.shape[1], target_h))
        if chart_img.shape[0] != target_h:
            chart_img = cv2.resize(chart_img, (chart_img.shape[1], target_h))

        gap_tile = np.full((target_h, gap, 3), 255, dtype=np.uint8)
        combined = np.hstack([cad_img, gap_tile, chart_img])

        canvas_color = (235, 251, 238) if highlight_best else (255, 255, 255)
        canvas = np.full((self.frame_height, self.frame_width, 3), canvas_color, dtype=np.uint8)

        if highlight_best:
            cv2.rectangle(
                canvas,
                (24, 24),
                (self.frame_width - 24, self.frame_height - 24),
                color=(34, 139, 34),
                thickness=8,
            )

        header_h = self._draw_header(canvas, highlight_best=highlight_best)
        bottom_pad = 28
        content_top = header_h + (24 if header_h else 20)
        content_h = max(self.frame_height - content_top - bottom_pad, 1)

        scale = min(
            (self.frame_width - 40) / combined.shape[1],
            content_h / combined.shape[0],
        )
        scale = max(scale, 1e-6)
        new_w = max(1, int(round(combined.shape[1] * scale)))
        new_h = max(1, int(round(combined.shape[0] * scale)))
        combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y0 = content_top + max((content_h - new_h) // 2, 0)
        x0 = (self.frame_width - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = combined

        frame_path = os.path.join(self.out_dir, f"frame_{step:06d}.png")
        cv2.imwrite(frame_path, canvas)
        self.frame_paths.append(frame_path)
        return frame_path

    def build_video(self, delete_frames=True, hold_last_seconds=0.0):
        if not self.frame_paths:
            raise RuntimeError("No frames recorded.")

        first = cv2.imread(self.frame_paths[0])
        if first is None:
            raise RuntimeError(f"Could not read first frame: {self.frame_paths[0]}")

        h, w = first.shape[:2]

        writer, actual_video_path, codec_name = self._select_video_writer(
            self.video_path,
            self.fps,
            (w, h),
        )

        for fp in self.frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)

        hold_frames = max(int(round(float(hold_last_seconds) * float(self.fps))), 0)
        if hold_frames > 0:
            last = cv2.imread(self.frame_paths[-1])
            if last is not None:
                if last.shape[:2] != (h, w):
                    last = cv2.resize(last, (w, h))
                for _ in range(hold_frames):
                    writer.write(last)

        writer.release()
        self.video_path = actual_video_path
        print(f"Saved video to: {self.video_path} (codec={codec_name})")
        if delete_frames:
            try:
                shutil.rmtree(self.out_dir)
                #print(f"Deleted frames directory: {self.out_dir}")
            except Exception as e:
                print(f"Warning: could not delete frames directory: {e}")
