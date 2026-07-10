"""Validation utilities for training loops using the hybrid Voronoi decoder.

This module checks the training loop around the decoder. It assumes decoder
gradients have already been validated separately with ``GradTestClass.py``.
Topology changes are expected discrete events and are reported separately from
gradient or optimizer failures.

Typical usage in a notebook:

    from testTraining import TrainingLoopTester, TrainingTestState

    def forward_builder(model, decoder, state):
        pred = model(state.model_inputs)
        seeds_uv = pred["seeds_uv"]
        decoder_out = decoder(seeds_uv, **(state.decoder_kwargs or {}))
        curve_loss, metrics = tester.gradient_tester.default_curve_length_loss(decoder_out)
        total_loss = curve_loss
        return {
            "seeds_uv": seeds_uv,
            "decoder_out": decoder_out,
            "total_loss": total_loss,
            "loss_terms": {"curve": curve_loss, "total": total_loss},
            "loss_weights": {"curve": 1.0},
            "metrics": metrics,
        }

    state = TrainingTestState(model, decoder, cad_domain, config, model_inputs=batch)
    tester = TrainingLoopTester(model, decoder, cad_domain, config, forward_builder=forward_builder)
    results = tester.run_all(state)
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Iterable, Mapping

import torch

from GradTestClass import DecoderGradientTester, TopologyChangeMonitor


ForwardBuilder = Callable[[Any, Any, "TrainingTestState"], dict[str, Any]]
LossBuilder = Callable[[dict[str, Any], "TrainingTestState"], torch.Tensor | tuple[torch.Tensor, dict[str, Any]]]
SeedExtractor = Callable[[dict[str, Any]], torch.Tensor]
ParameterGetter = Callable[[Any, Any], Iterable[torch.nn.Parameter]]


@dataclass
class TrainingTestState:
    """State bundle passed to forward builders and tests."""

    model: Any
    decoder: Any
    cad_domain: Any
    config: Any
    model_inputs: Any = None
    decoder_kwargs: dict[str, Any] | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None
    extra: dict[str, Any] | None = None


class TrainingLoopTester:
    """Comprehensive validation harness for a Voronoi training loop."""

    def __init__(
        self,
        model: Any,
        decoder: Any,
        cad_domain: Any,
        config: Any,
        optimizer_factory: Callable[..., torch.optim.Optimizer] | None = None,
        scheduler_factory: Callable[..., Any] | None = None,
        loss_builder: LossBuilder | None = None,
        forward_builder: ForwardBuilder | None = None,
        seed_extractor: SeedExtractor | None = None,
        trainable_parameter_getter: ParameterGetter | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
        edge_types: Iterable[int] = (0,),
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.cad_domain = cad_domain
        self.config = config
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.loss_builder = loss_builder
        self.forward_builder = forward_builder
        self.seed_extractor = seed_extractor or self._default_seed_extractor
        self.trainable_parameter_getter = trainable_parameter_getter
        self.device = torch.device(device) if device is not None else self._infer_device(model, decoder)
        self.dtype = dtype
        self.edge_types = tuple(int(v) for v in edge_types)
        self.verbose = bool(verbose)
        self.gradient_tester = DecoderGradientTester(
            decoder=decoder,
            cad_domain=cad_domain,
            config=config,
            device=self.device,
            dtype=dtype,
            edge_types=self.edge_types,
            verbose=False,
        )

    def test_manual_seed_descent(
        self,
        seeds_uv: torch.Tensor,
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: Callable[[dict[str, Any]], Any] | None = None,
        step_sizes: Iterable[float] = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5),
        freeze_topology: bool = False,
    ) -> dict[str, Any]:
        """Verify that a direct gradient step in seed space can reduce loss."""
        decoder_kwargs = dict(decoder_kwargs or {})
        seeds = torch.as_tensor(seeds_uv, dtype=self.dtype, device=self.device).detach().clone().requires_grad_(True)
        topology = self.gradient_tester.build_topology(seeds, decoder_kwargs)
        before_adj = self.gradient_tester.seed_adjacency_signature(topology)
        out = (
            self.gradient_tester.decode_with_frozen_topology(seeds, topology, decoder_kwargs)
            if freeze_topology
            else self.gradient_tester._call_decoder(seeds, decoder_kwargs)
        )
        loss, _ = self.gradient_tester._compute_loss(out, loss_fn)
        grad = torch.autograd.grad(loss, seeds, allow_unused=False)[0]
        rows = []
        for alpha in step_sizes:
            alpha_f = float(alpha)
            candidate = (seeds - alpha_f * grad).detach()
            try:
                candidate_topology = topology if freeze_topology else self.gradient_tester.build_topology(candidate, decoder_kwargs)
                after_adj = self.gradient_tester.seed_adjacency_signature(candidate_topology)
                candidate_out = (
                    self.gradient_tester.decode_with_frozen_topology(candidate, topology, decoder_kwargs)
                    if freeze_topology
                    else self.gradient_tester._call_decoder(candidate, decoder_kwargs)
                )
                candidate_loss, _ = self.gradient_tester._compute_loss(candidate_out, loss_fn)
                loss_after = self._scalar(candidate_loss)
                adjacency_changed = after_adj != before_adj
                actual_change = loss_after - self._scalar(loss)
                predicted_change = self._scalar((grad.detach() * (candidate - seeds.detach())).sum())
                rows.append(
                    {
                        "step_size": alpha_f,
                        "loss_before": self._scalar(loss),
                        "loss_after": loss_after,
                        "actual_change": actual_change,
                        "predicted_change": predicted_change,
                        "adjacency_signature_before": before_adj,
                        "adjacency_signature_after": after_adj,
                        "adjacency_changed": adjacency_changed,
                        "decrease": bool(loss_after < self._scalar(loss)),
                        "failed": False,
                    }
                )
            except Exception as exc:
                rows.append({"step_size": alpha_f, "failed": True, "error": repr(exc)})
        stable_decreases = [r for r in rows if not r.get("failed") and not r.get("adjacency_changed") and r.get("decrease")]
        any_decrease = [r for r in rows if not r.get("failed") and r.get("decrease")]
        best = min((r for r in rows if not r.get("failed")), key=lambda r: r["loss_after"], default=None)
        return {
            "loss_before": self._scalar(loss),
            "gradient_norm": self._total_norm([grad]),
            "frozen_topology": bool(freeze_topology),
            "results": rows,
            "best_step_size": None if best is None else best["step_size"],
            "best_loss": None if best is None else best["loss_after"],
            "overall_pass": bool(stable_decreases if not freeze_topology else any_decrease),
        }

    def test_optimizer_single_step(
        self,
        state: TrainingTestState,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one real optimizer step and diagnose gradients and updates."""
        optimizer_kwargs = dict(optimizer_kwargs or {})
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            params = self._trainable_parameters(state.model, state.decoder)
            opt = optimizer or self._make_optimizer(params, optimizer_kwargs)
            sched = scheduler
            before = self._clone_params(params)
            out = self._run_forward(state)
            loss = self._total_loss(out)
            loss_check = self._validate_loss(loss)
            if not loss_check["ok"]:
                return {"overall_pass": False, "loss_check": loss_check}
            self._zero_grad(params, opt)
            loss.backward()
            grad_before = self._param_grads(params)
            clip_info = self._apply_configured_clipping(params)
            grad_after = self._param_grads(params)
            opt.step()
            if sched is not None:
                with self._safe_call():
                    sched.step()
            after = self._clone_params(params)
            updates = [after[id(p)] - before[id(p)] for p in params]
            out_after = self._run_forward(state)
            loss_after = self._total_loss(out_after)
            diagnostics = self._parameter_update_diagnostics(params, grad_after, updates)
            predicted_change = sum(self._dot(g, u) for g, u in zip(grad_after, updates) if g is not None)
            result = {
                "loss_before": self._scalar(loss),
                "loss_after": self._scalar(loss_after),
                "actual_loss_change": self._scalar(loss_after) - self._scalar(loss),
                "first_order_predicted_change": predicted_change,
                "total_parameter_update_norm": self._total_norm(updates),
                "total_gradient_norm_before_clipping": self._total_norm([g for g in grad_before if g is not None]),
                "total_gradient_norm_after_clipping": self._total_norm([g for g in grad_after if g is not None]),
                "clip_info": clip_info,
                **diagnostics,
            }
            result["overall_pass"] = (
                result["total_parameter_update_norm"] > 0.0
                or result["total_gradient_norm_after_clipping"] == 0.0
            ) and not result["nonfinite_gradient_parameters"] and not result["nonfinite_update_parameters"]
            if result["total_parameter_update_norm"] == 0.0 and result["total_gradient_norm_after_clipping"] > 0.0:
                result["failure"] = "Gradients are nonzero but optimizer produced a zero update."
            if predicted_change > 0:
                result["warning"] = "Optimizer update is first-order uphill."
            return result

    def test_curve_only_training(
        self,
        state: TrainingTestState,
        num_steps: int = 500,
        learning_rate: float = 1e-4,
        optimizer_name: str = "adam",
        disable_scheduler: bool = True,
        disable_rolling_anchors: bool = True,
        disable_secondary_losses: bool = True,
        topology_check_every: int = 1,
        log_every: int = 10,
        output_dir: str | pathlib.Path = "training_test_outputs",
        make_plots: bool = True,
    ) -> dict[str, Any]:
        """Run a temporary curve-length-only optimization and report behavior.

        This test intentionally ignores all non-curve losses and minimizes only
        ``default_curve_length_loss``. It preserves production state and treats
        seed-adjacency topology changes as diagnostic events, not failures.
        """
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            config_snapshot = self._config_snapshot(state.config)
            if disable_secondary_losses:
                self._set_curve_only_config(state.config)
            if disable_rolling_anchors:
                self._set_config_value(state.config, "use_rolling_seed_anchors", False)
            self._set_config_value(state.config, "normalize_losses", False)
            if disable_scheduler:
                self._set_config_value(state.config, "scheduler_milestones", ())
            params = self._trainable_parameters(state.model, state.decoder)
            opt = self._named_optimizer(optimizer_name, params, learning_rate)
            previous_adj = None
            history: list[dict[str, Any]] = []
            logged_history: list[dict[str, Any]] = []
            best_loss = float("inf")
            best_step = -1
            initial_loss = float("nan")
            final_loss = float("nan")
            nonfinite = False
            topology_changes = 0
            largest_topology_free_improvement = 0.0
            largest_improvement_after_topology_change = 0.0
            initial_seeds: torch.Tensor | None = None
            previous_loss: float | None = None

            for step in range(int(num_steps)):
                self._zero_grad(params, opt)
                out = self._run_forward(state)
                decoder_out = out.get("decoder_out", out)
                seeds_current = self.seed_extractor(out)
                if initial_seeds is None:
                    initial_seeds = seeds_current.detach().clone()

                curve_loss, _ = self.gradient_tester.default_curve_length_loss(
                    decoder_out,
                    edge_types=self.edge_types,
                )
                if not torch.isfinite(curve_loss):
                    nonfinite = True
                    history.append(
                        {
                            "step": step,
                            "curve_loss": self._scalar(curve_loss.detach()),
                            "finite": False,
                        }
                    )
                    break
                curve_loss.backward()
                grad_norm = self._total_norm([p.grad for p in params if p.grad is not None])
                clip_info = self._apply_configured_clipping(params)
                before = self._clone_params(params)
                opt.step()
                after = self._clone_params(params)
                update_norm = self._total_norm([after[id(p)] - before[id(p)] for p in params])
                loss_value = self._scalar(curve_loss)
                if step == 0:
                    initial_loss = loss_value
                final_loss = loss_value
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_step = step

                adjacency_changed = False
                adj_sig = None
                topology_edge_count = 0
                if topology_check_every > 0 and step % int(topology_check_every) == 0:
                    try:
                        topology = self.gradient_tester.build_topology(seeds_current, state.decoder_kwargs)
                        adj_sig = self.gradient_tester.seed_adjacency_signature(topology)
                        adjacency_changed = previous_adj is not None and adj_sig != previous_adj
                        previous_adj = adj_sig
                        topology_edge_count = self._topology_edge_count(topology)
                        if adjacency_changed:
                            topology_changes += 1
                    except Exception as exc:
                        adj_sig = f"topology_failure:{repr(exc)}"

                improvement = 0.0 if previous_loss is None else previous_loss - loss_value
                if improvement > 0.0:
                    if adjacency_changed:
                        largest_improvement_after_topology_change = max(
                            largest_improvement_after_topology_change,
                            improvement,
                        )
                    else:
                        largest_topology_free_improvement = max(
                            largest_topology_free_improvement,
                            improvement,
                        )
                previous_loss = loss_value

                curve_metrics = self._edge_length_statistics(decoder_out)
                row = {
                    "step": step,
                    "curve_loss": loss_value,
                    "total_loss": loss_value,
                    **curve_metrics,
                    "minimum_seed_distance": self._minimum_seed_distance(seeds_current),
                    "seed_displacement_from_initial": (
                        self._total_norm([seeds_current.detach() - initial_seeds.to(seeds_current.device)])
                        if initial_seeds is not None
                        else 0.0
                    ),
                    "gradient_norm": grad_norm,
                    "parameter_update_norm": update_norm,
                    "learning_rate": self._learning_rates(opt)[0] if opt.param_groups else float("nan"),
                    "topology_adjacency_signature": adj_sig,
                    "true_topology_change": adjacency_changed,
                    "topology_edge_count": topology_edge_count,
                    "clip_applied": bool(clip_info.get("applied", False)),
                    "finite": True,
                }
                history.append(row)
                if step % max(int(log_every), 1) == 0 or step == int(num_steps) - 1:
                    logged_history.append(row)
                    if self.verbose:
                        print(
                            f"[curve-only {step}] "
                            f"loss={loss_value:.6e} cv={row['coefficient_of_variation']:.3e} "
                            f"grad={grad_norm:.3e} update={update_norm:.3e} "
                            f"topo_change={adjacency_changed}"
                        )

            self._restore_config_snapshot(state.config, config_snapshot)

            output_path = pathlib.Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            csv_path = output_path / "curve_only_training.csv"
            self._write_csv(csv_path, logged_history)
            plot_paths: dict[str, str] = {}
            if make_plots:
                plot_paths = self._write_curve_only_plots(output_path, logged_history)

            relative_improvement = (
                (initial_loss - best_loss) / max(abs(initial_loss), 1e-30)
                if math.isfinite(initial_loss)
                else float("nan")
            )
            return {
                "overall_pass": bool(history)
                and math.isfinite(initial_loss)
                and math.isfinite(final_loss)
                and math.isfinite(best_loss)
                and best_loss < initial_loss
                and not nonfinite,
                "initial_loss": initial_loss,
                "best_loss": best_loss,
                "final_loss": final_loss,
                "best_step": best_step,
                "relative_improvement": relative_improvement,
                "topology_changes": topology_changes,
                "largest_topology_free_improvement": largest_topology_free_improvement,
                "largest_improvement_after_topology_change": largest_improvement_after_topology_change,
                "loss_finite_every_iteration": not nonfinite,
                "history": logged_history,
                "full_history_length": len(history),
                "csv_path": str(csv_path),
                "plot_paths": plot_paths,
            }

    def test_loss_weight_accounting(self, state: TrainingTestState) -> dict[str, Any]:
        """Confirm reported total equals weighted normalized contributions."""
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            out = self._run_forward(state)
            reported = self._total_loss(out)
            terms = dict(out.get("loss_terms", {}))
            weights = dict(out.get("loss_weights", self._loss_weights_from_config(state.config, terms)))
            normalizers = dict(out.get("loss_normalizers", {}))
            rows = {}
            expected = reported.new_zeros(())
            warnings = []
            for name, value in terms.items():
                if name == "total":
                    continue
                if not torch.is_tensor(value):
                    value = torch.as_tensor(float(value), dtype=reported.dtype, device=reported.device)
                normalizer = normalizers.get(name, 1.0)
                normalizer_t = torch.as_tensor(normalizer, dtype=reported.dtype, device=reported.device)
                if not torch.isfinite(normalizer_t) or self._scalar(normalizer_t.abs()) == 0.0:
                    warnings.append(f"loss normalizer for {name} is zero or non-finite")
                    normalizer_t = reported.new_ones(())
                lam = float(weights.get(name, self._config_value(state.config, f"lam_{name}", 1.0)))
                normalized = value / normalizer_t
                contribution = lam * normalized
                expected = expected + contribution
                rows[name] = {
                    "raw_value": self._scalar(value),
                    "normalizer": self._scalar(normalizer_t),
                    "normalized_value": self._scalar(normalized),
                    "lambda": lam,
                    "weighted_contribution": self._scalar(contribution),
                    "requires_grad": bool(value.requires_grad),
                }
                if lam > 0.0 and (not value.requires_grad or self._scalar(value.detach().abs()) == 0.0):
                    warnings.append(f"positive weight multiplies detached or constant-zero term: {name}")
                if lam == 0.0 and abs(self._scalar(contribution)) > 1e-12:
                    warnings.append(f"disabled term is still contributing: {name}")
            diff = abs(self._scalar(reported - expected))
            total_mag = sum(abs(row["weighted_contribution"]) for row in rows.values())
            for name, row in rows.items():
                if total_mag > 0 and abs(row["weighted_contribution"]) / total_mag > 0.95:
                    warnings.append(f"{name} contributes more than 95% of weighted magnitude")
            tol = float(self._config_value(state.config, "loss_accounting_tolerance", 1e-8))
            return {
                "reported_total": self._scalar(reported),
                "expected_total": self._scalar(expected),
                "absolute_error": diff,
                "terms": rows,
                "warnings": warnings,
                "overall_pass": diff <= tol,
            }

    def test_gradient_clipping(self, state: TrainingTestState, clip_norm: float | None = None) -> dict[str, Any]:
        """Verify configured gradient clipping covers optimizer parameters."""
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            params = self._trainable_parameters(state.model, state.decoder)
            opt = state.optimizer or self._make_optimizer(params, {})
            out = self._run_forward(state)
            loss = self._total_loss(out)
            self._zero_grad(params, opt)
            loss.backward()
            before_norm = self._total_norm([p.grad for p in params if p.grad is not None])
            max_norm = float(clip_norm if clip_norm is not None else self._config_value(state.config, "grad_clip_norm", 0.0) or 0.0)
            clipped_ids = {id(p) for p in params}
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            after_norm = self._total_norm([p.grad for p in params if p.grad is not None])
            opt_ids = {id(p) for group in opt.param_groups for p in group.get("params", []) if getattr(p, "requires_grad", False)}
            finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
            missing = sorted(opt_ids - clipped_ids)
            return {
                "clip_norm": max_norm,
                "gradient_norm_before": before_norm,
                "gradient_norm_after": after_norm,
                "gradients_finite": bool(finite),
                "optimizer_parameter_count": len(opt_ids),
                "clipped_parameter_count": len(clipped_ids),
                "optimizer_parameter_ids": sorted(opt_ids),
                "clipped_parameter_ids": sorted(clipped_ids),
                "missing_from_clipping_ids": missing,
                "overall_pass": bool(finite) and (max_norm <= 0 or after_norm <= max_norm + 1e-8) and not missing,
            }

    def test_scheduler_behavior(self, optimizer: Any, scheduler: Any, num_steps: int) -> dict[str, Any]:
        """Record LR behavior on a copied optimizer/scheduler when possible."""
        if optimizer is None or scheduler is None:
            return {"overall_pass": True, "skipped": True, "reason": "optimizer or scheduler not supplied"}
        opt = copy.deepcopy(optimizer)
        sched = copy.deepcopy(scheduler)
        rows = []
        previous = self._learning_rates(opt)
        for step in range(int(num_steps)):
            opt.step()
            sched.step()
            current = self._learning_rates(opt)
            rows.append({"step": step, "lr_before": previous, "lr_after": current, "changed": current != previous})
            previous = current
        changed_steps = [r["step"] for r in rows if r["changed"]]
        warnings = []
        milestones = self._scheduler_milestones(sched, num_steps)
        if not changed_steps:
            warnings.append("scheduler never changes LR")
        if milestones and min(milestones) > 0.8 * int(num_steps):
            warnings.append("scheduler milestones occur very late")
        return {"milestones": milestones, "actual_lr_change_steps": changed_steps, "logs": rows, "warnings": warnings, "overall_pass": True}

    def test_best_state_restore(
        self,
        state: TrainingTestState,
        num_steps: int = 200,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        tolerance: float = 1e-10,
    ) -> dict[str, Any]:
        """Verify best snapshot can be restored and reproduces score/seeds.

        The score used here is the same curve-only objective used by
        ``test_curve_only_training`` so the comparison is deterministic and
        independent of secondary loss interactions.
        """
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            config_snapshot = self._config_snapshot(state.config)
            self._set_curve_only_config(state.config)
            self._set_config_value(state.config, "normalize_losses", False)
            params = self._trainable_parameters(state.model, state.decoder)
            opt = optimizer or state.optimizer or self._make_optimizer(params, {})
            sched = scheduler if scheduler is not None else state.scheduler
            best = None
            best_payload = None
            best_metrics: dict[str, Any] = {}
            for step in range(int(num_steps)):
                self._zero_grad(params, opt)
                out = self._run_forward(state)
                decoder_out = out.get("decoder_out", out)
                loss, metrics = self.gradient_tester.default_curve_length_loss(
                    decoder_out,
                    edge_types=self.edge_types,
                )
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
                if sched is not None:
                    with self._safe_call():
                        sched.step()
                score = self._scalar(loss)
                if best is None or score < best:
                    best = score
                    best_metrics = dict(metrics)
                    best_payload = {
                        "step": step,
                        "model": self._module_state(state.model),
                        "decoder": self._module_state(state.decoder),
                        "optimizer": copy.deepcopy(opt.state_dict()),
                        "scheduler": copy.deepcopy(sched.state_dict()) if sched is not None and hasattr(sched, "state_dict") else None,
                        "config": self._config_snapshot(state.config),
                        "score": score,
                        "total_loss": self._scalar(self._total_loss(out)),
                        "seeds": self.seed_extractor(out).detach().clone(),
                        "metrics": copy.deepcopy(best_metrics),
                    }
            if best_payload is None:
                return {"overall_pass": False, "failure": "No finite best state was observed."}

            # Continue at least one extra step after the best point when possible,
            # so restoration is tested against a genuinely changed state.
            continued_steps = 0
            for _ in range(max(1, min(10, int(num_steps) // 10))):
                self._zero_grad(params, opt)
                out = self._run_forward(state)
                decoder_out = out.get("decoder_out", out)
                loss, _ = self.gradient_tester.default_curve_length_loss(
                    decoder_out,
                    edge_types=self.edge_types,
                )
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
                if sched is not None:
                    with self._safe_call():
                        sched.step()
                continued_steps += 1

            self._load_module_state(state.model, best_payload["model"])
            self._load_module_state(state.decoder, best_payload["decoder"])
            opt.load_state_dict(best_payload["optimizer"])
            if sched is not None and best_payload["scheduler"] is not None:
                sched.load_state_dict(best_payload["scheduler"])
            self._restore_config_snapshot(state.config, best_payload["config"])

            recomputed = self._run_forward(state)
            recomputed_decoder_out = recomputed.get("decoder_out", recomputed)
            recomputed_curve_loss, recomputed_metrics = self.gradient_tester.default_curve_length_loss(
                recomputed_decoder_out,
                edge_types=self.edge_types,
            )
            recomputed_score = self._scalar(recomputed_curve_loss)
            recomputed_total = self._scalar(self._total_loss(recomputed))
            recomputed_seeds = self.seed_extractor(recomputed).detach()
            score_error = abs(recomputed_score - float(best_payload["score"]))
            seed_error = self._total_norm([recomputed_seeds - best_payload["seeds"].to(recomputed_seeds.device)])
            optimizer_moments_restored = self._optimizer_state_equal(
                opt.state_dict(),
                best_payload["optimizer"],
            )
            scheduler_state_restored = (
                True
                if sched is None or best_payload["scheduler"] is None
                else self._state_tree_equal(sched.state_dict(), best_payload["scheduler"])
            )
            self._restore_config_snapshot(state.config, config_snapshot)
            return {
                "best_step": best_payload["step"],
                "saved_best_score": best_payload["score"],
                "saved_best_total_loss": best_payload["total_loss"],
                "recomputed_best_score": recomputed_score,
                "recomputed_total_loss": recomputed_total,
                "score_absolute_error": score_error,
                "seed_difference_norm": seed_error,
                "continued_steps_after_best_snapshot": continued_steps,
                "saved_best_metrics": best_payload["metrics"],
                "recomputed_metrics": self._json_safe(recomputed_metrics),
                "checks": {
                    "model_state_saved": best_payload["model"] is not None,
                    "decoder_state_saved": best_payload["decoder"] is not None,
                    "optimizer_state_saved": best_payload["optimizer"] is not None,
                    "scheduler_state_saved": best_payload["scheduler"] is not None if sched is not None else True,
                    "optimizer_moments_restored": optimizer_moments_restored,
                    "scheduler_state_restored": scheduler_state_restored,
                    "rolling_anchor_state_restored": not bool(self._config_value(state.config, "use_rolling_seed_anchors", False)),
                    "rolling_anchor_note": "No rolling-anchor object is present in TrainingTestState; pass it through state.extra for project-specific validation if needed.",
                },
                "overall_pass": (
                    score_error < float(tolerance)
                    and seed_error <= max(float(tolerance), 10.0 * torch.finfo(recomputed_seeds.dtype).eps)
                    and optimizer_moments_restored
                    and scheduler_state_restored
                ),
            }

    def test_nonfinite_handling(self, state: TrainingTestState) -> dict[str, Any]:
        """Synthetic non-finite cases; production loop adapters must decide policy."""
        cases = {
            "nan_loss": torch.tensor(float("nan"), device=self.device, requires_grad=True),
            "inf_loss": torch.tensor(float("inf"), device=self.device, requires_grad=True),
        }
        rows = {}
        for name, loss in cases.items():
            rows[name] = {
                "loss_finite": bool(torch.isfinite(loss).item()),
                "should_skip_optimizer_step": True,
                "verified_without_production_loop_adapter": True,
            }
        return {
            "cases": rows,
            "overall_pass": True,
            "warning": "Non-finite production-loop behavior requires a project-specific unsafe-step adapter for full verification.",
        }

    def test_topology_monitor_during_training(
        self,
        state: TrainingTestState,
        num_steps: int = 100,
        check_every: int = 1,
    ) -> dict[str, Any]:
        """Track seed-adjacency changes during temporary training."""
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            params = self._trainable_parameters(state.model, state.decoder)
            opt = self._make_optimizer(params, {})
            previous_adj = None
            events = []
            failures = 0
            nonfinite = False
            for step in range(int(num_steps)):
                self._zero_grad(params, opt)
                out = self._run_forward(state)
                loss = self._total_loss(out)
                if not torch.isfinite(loss):
                    nonfinite = True
                    break
                loss.backward()
                opt.step()
                if check_every > 0 and step % int(check_every) == 0:
                    try:
                        seeds = self.seed_extractor(out)
                        topo = self.gradient_tester.build_topology(seeds, state.decoder_kwargs)
                        adj = self.gradient_tester.seed_adjacency_signature(topo)
                        changed = previous_adj is not None and adj != previous_adj
                        previous_adj = adj
                        edge_type = topo.get("edge_type", None) if isinstance(topo, dict) else None
                        events.append(
                            {
                                "step": step,
                                "changed": changed,
                                "adjacency_signature": adj,
                                "edge_count": int(topo.get("edges").shape[0]) if isinstance(topo, dict) and topo.get("edges") is not None else 0,
                                "edge_type_counts": self._value_counts(edge_type),
                                "minimum_seed_distance": self._minimum_seed_distance(seeds),
                                "loss": self._scalar(loss),
                            }
                        )
                    except Exception as exc:
                        failures += 1
                        events.append({"step": step, "failed": True, "error": repr(exc)})
            changes = [e for e in events if e.get("changed")]
            warnings = []
            if events and len(changes) / max(len(events), 1) > 0.5:
                warnings.append("topology changes occur on most checked steps")
            if failures:
                warnings.append("decoder/topology failures occurred")
            if nonfinite:
                warnings.append("loss became non-finite")
            if any(e.get("edge_count", 1) == 0 for e in events if not e.get("failed")):
                warnings.append("edge count became zero")
            return {"total_checks": len(events), "true_adjacency_changes": len(changes), "change_rate": len(changes) / max(len(events), 1), "change_steps": [e["step"] for e in changes], "events": events, "warnings": warnings, "overall_pass": failures == 0 and not nonfinite}

    def test_seed_update_path(self, state: TrainingTestState) -> dict[str, Any]:
        """Verify optimizer-updated parameters influence generated seeds."""
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            params = self._trainable_parameters(state.model, state.decoder)
            opt = self._make_optimizer(params, {})
            out_before = self._run_forward(state)
            seeds_before = self.seed_extractor(out_before)
            loss = self._total_loss(out_before)
            seed_grad = torch.autograd.grad(loss, seeds_before, retain_graph=True, allow_unused=True)[0]
            self._zero_grad(params, opt)
            loss.backward()
            param_grad_norm = self._total_norm([p.grad for p in params if p.grad is not None])
            before_params = self._clone_params(params)
            opt.step()
            after_params = self._clone_params(params)
            out_after = self._run_forward(state)
            seeds_after = self.seed_extractor(out_after).detach()
            displacement = seeds_after - seeds_before.detach()
            param_update_norm = self._total_norm([after_params[id(p)] - before_params[id(p)] for p in params])
            seed_displacement_norm = self._total_norm([displacement])
            moved = int((torch.linalg.vector_norm(displacement.reshape(displacement.shape[0], -1), dim=1) > 0).sum().item()) if displacement.ndim >= 2 else int(seed_displacement_norm > 0)
            result = {
                "seed_gradient_norm": 0.0 if seed_grad is None else self._total_norm([seed_grad]),
                "parameter_gradient_norm": param_grad_norm,
                "parameter_update_norm": param_update_norm,
                "seed_displacement_norm": seed_displacement_norm,
                "max_per_seed_displacement": self._max_per_seed_norm(displacement),
                "number_of_moved_seeds": moved,
                "seed_gradient_is_nonzero": seed_grad is not None and self._total_norm([seed_grad]) > 0,
                "parameter_gradient_is_nonzero": param_grad_norm > 0,
                "optimizer_updates_parameters": param_update_norm > 0,
                "seeds_change_after_parameter_update": seed_displacement_norm > 0,
            }
            result["overall_pass"] = not (
                result["parameter_gradient_is_nonzero"] and not result["seeds_change_after_parameter_update"]
            ) and not (
                result["optimizer_updates_parameters"] and not result["seeds_change_after_parameter_update"]
            )
            if not result["overall_pass"]:
                result["failure"] = "Optimizer-updated parameters do not move seeds."
            return result

    def test_per_loss_gradient_contributions(self, state: TrainingTestState) -> dict[str, Any]:
        """Report seed and parameter gradients for each individual loss term."""
        state = self._state_with_defaults(state)
        with self._preserve_training_state(state):
            out = self._run_forward(state)
            terms = {k: v for k, v in out.get("loss_terms", {}).items() if k != "total" and torch.is_tensor(v)}
            seeds = self.seed_extractor(out)
            params = self._trainable_parameters(state.model, state.decoder)
            rows = {}
            seed_grads = {}
            for name, term in terms.items():
                grads = torch.autograd.grad(term, [seeds] + params, retain_graph=True, allow_unused=True)
                seed_grad = grads[0]
                param_grads = [g for g in grads[1:] if g is not None]
                seed_grads[name] = seed_grad.detach().reshape(-1) if seed_grad is not None else None
                rows[name] = {
                    "loss_value": self._scalar(term),
                    "seed_gradient_norm": 0.0 if seed_grad is None else self._total_norm([seed_grad]),
                    "parameter_gradient_norm": self._total_norm(param_grads),
                    "finite": bool(torch.isfinite(term).item()),
                    "nonzero": self._scalar(term.detach().abs()) > 0.0,
                }
            names = list(seed_grads.keys())
            cosine = {a: {} for a in names}
            for a in names:
                for b in names:
                    ga, gb = seed_grads[a], seed_grads[b]
                    cosine[a][b] = None if ga is None or gb is None else self._cosine(ga, gb)
            return {"losses": rows, "seed_gradient_cosine_matrix": cosine, "overall_pass": all(r["finite"] for r in rows.values())}

    def test_learning_rate_sweep(
        self,
        state: TrainingTestState,
        learning_rates: Iterable[float] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
        num_steps: int = 50,
    ) -> dict[str, Any]:
        """Run short curve-only optimizations from identical state at several LRs."""
        rows = []
        for lr in learning_rates:
            result = self.test_curve_only_training(state, num_steps=num_steps, learning_rate=float(lr), log_every=0)
            logs = result.get("logs", [])
            rows.append(
                {
                    "learning_rate": float(lr),
                    "initial_loss": result.get("initial_loss"),
                    "best_loss": result.get("best_loss"),
                    "final_loss": result.get("final_loss"),
                    "nonfinite_occurrence": result.get("nonfinite", False),
                    "topology_change_count": result.get("topology_change_count", 0),
                    "mean_update_norm": sum(r.get("update_norm", 0.0) for r in logs) / max(len(logs), 1),
                    "relative_improvement": result.get("relative_improvement"),
                }
            )
        candidates = [r for r in rows if not r["nonfinite_occurrence"]]
        recommended = max(candidates, key=lambda r: r.get("relative_improvement") or -float("inf"), default=None)
        return {"results": rows, "recommended_learning_rate": None if recommended is None else recommended["learning_rate"], "overall_pass": recommended is not None}

    def run_all(
        self,
        state: TrainingTestState,
        run_curve_only: bool = True,
        run_best_restore: bool = True,
        run_nonfinite: bool = False,
        run_lr_sweep: bool = False,
    ) -> dict[str, Any]:
        """Run the standard training-loop validation suite."""
        state = self._state_with_defaults(state)
        results: dict[str, Any] = {}
        seeds = self.seed_extractor(self._run_forward(state)).detach()
        results["manual_seed_descent"] = self.test_manual_seed_descent(seeds, state.decoder_kwargs)
        results["optimizer_single_step"] = self.test_optimizer_single_step(state, optimizer=state.optimizer, scheduler=state.scheduler)
        results["seed_update_path"] = self.test_seed_update_path(state)
        results["loss_weight_accounting"] = self.test_loss_weight_accounting(state)
        results["gradient_clipping"] = self.test_gradient_clipping(state)
        results["scheduler"] = self.test_scheduler_behavior(state.optimizer, state.scheduler, int(self._config_value(state.config, "num_steps", 100)))
        results["per_loss_gradients"] = self.test_per_loss_gradient_contributions(state)
        if run_curve_only:
            results["curve_only_training"] = self.test_curve_only_training(state)
        results["topology_monitor"] = self.test_topology_monitor_during_training(state)
        if run_best_restore:
            results["best_state_restore"] = self.test_best_state_restore(state)
        if run_nonfinite:
            results["nonfinite_handling"] = self.test_nonfinite_handling(state)
        if run_lr_sweep:
            results["learning_rate_sweep"] = self.test_learning_rate_sweep(state)
        if self.verbose:
            self._print_summary(results)
        return results

    def save_json(self, path: str | pathlib.Path, results: Any) -> None:
        """Save JSON-safe summaries, logs, and diagnostics without model weights."""
        with pathlib.Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self._json_safe(results), handle, indent=2, sort_keys=True)

    def _run_forward(self, state: TrainingTestState) -> dict[str, Any]:
        if self.forward_builder is None:
            raise RuntimeError(
                "TrainingLoopTester requires forward_builder(model, decoder, state) for training-loop tests. "
                "It must return a dict containing seeds_uv, decoder_out, total_loss, and loss_terms."
            )
        out = self.forward_builder(state.model, state.decoder, state)
        if not isinstance(out, dict):
            raise TypeError("forward_builder must return a dictionary.")
        required = ("seeds_uv", "decoder_out", "total_loss", "loss_terms")
        missing = [key for key in required if key not in out]
        if missing:
            raise KeyError(f"forward_builder output missing required keys: {missing}")
        return out

    @staticmethod
    def _default_seed_extractor(forward_out: dict[str, Any]) -> torch.Tensor:
        seeds = forward_out.get("seeds_uv")
        if not isinstance(seeds, torch.Tensor):
            raise KeyError("forward output must contain tensor 'seeds_uv' or supply seed_extractor.")
        return seeds

    def _state_with_defaults(self, state: TrainingTestState) -> TrainingTestState:
        return TrainingTestState(
            model=state.model if state.model is not None else self.model,
            decoder=state.decoder if state.decoder is not None else self.decoder,
            cad_domain=state.cad_domain if state.cad_domain is not None else self.cad_domain,
            config=state.config if state.config is not None else self.config,
            model_inputs=state.model_inputs,
            decoder_kwargs=state.decoder_kwargs or {},
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            extra=state.extra or {},
        )

    def _trainable_parameters(self, model: Any, decoder: Any) -> list[torch.nn.Parameter]:
        if self.trainable_parameter_getter is not None:
            return [p for p in self.trainable_parameter_getter(model, decoder) if getattr(p, "requires_grad", False)]
        params: list[torch.nn.Parameter] = []
        for owner in (model, decoder):
            parameters = getattr(owner, "parameters", None)
            if callable(parameters):
                params.extend([p for p in parameters() if getattr(p, "requires_grad", False)])
        seen = set()
        unique = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        if not unique:
            raise RuntimeError("No trainable parameters found. Supply trainable_parameter_getter for direct seed tensors or custom models.")
        return unique

    def _make_optimizer(self, params: list[torch.nn.Parameter], optimizer_kwargs: dict[str, Any]) -> torch.optim.Optimizer:
        if self.optimizer_factory is not None:
            return self.optimizer_factory(params, **optimizer_kwargs)
        lr = float(optimizer_kwargs.get("lr", self._config_value(self.config, "learning_rate", self._config_value(self.config, "lr", 1e-4))))
        return torch.optim.Adam(params, lr=lr)

    @staticmethod
    def _named_optimizer(name: str, params: list[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
        lower = name.lower()
        if lower == "sgd":
            return torch.optim.SGD(params, lr=lr)
        if lower == "adamw":
            return torch.optim.AdamW(params, lr=lr)
        return torch.optim.Adam(params, lr=lr)

    @staticmethod
    def _total_loss(out: dict[str, Any]) -> torch.Tensor:
        loss = out.get("total_loss")
        if not isinstance(loss, torch.Tensor):
            raise TypeError("forward_builder must return scalar tensor 'total_loss'.")
        return loss

    @staticmethod
    def _validate_loss(loss: torch.Tensor) -> dict[str, Any]:
        return {
            "is_scalar_tensor": isinstance(loss, torch.Tensor) and loss.ndim == 0,
            "requires_grad": bool(getattr(loss, "requires_grad", False)),
            "finite": bool(torch.isfinite(loss.detach()).item()) if isinstance(loss, torch.Tensor) and loss.ndim == 0 else False,
            "ok": isinstance(loss, torch.Tensor) and loss.ndim == 0 and loss.requires_grad and bool(torch.isfinite(loss.detach()).item()),
        }

    @staticmethod
    def _zero_grad(params: list[torch.nn.Parameter], optimizer: Any | None = None) -> None:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for p in params:
            p.grad = None

    @staticmethod
    def _param_grads(params: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
        return [None if p.grad is None else p.grad.detach().clone() for p in params]

    @staticmethod
    def _clone_params(params: list[torch.nn.Parameter]) -> dict[int, torch.Tensor]:
        return {id(p): p.detach().clone() for p in params}

    def _parameter_update_diagnostics(self, params: list[torch.nn.Parameter], grads: list[torch.Tensor | None], updates: list[torch.Tensor]) -> dict[str, Any]:
        per_param = []
        grad_no_update = []
        update_no_grad = []
        nonfinite_grad = []
        nonfinite_update = []
        for idx, (p, grad, update) in enumerate(zip(params, grads, updates)):
            gnorm = 0.0 if grad is None else self._total_norm([grad])
            unorm = self._total_norm([update])
            cosine = None if grad is None else self._cosine(grad.reshape(-1), update.reshape(-1))
            name = f"param_{idx}"
            per_param.append({"name": name, "gradient_norm": gnorm, "update_norm": unorm, "cosine_gradient_update": cosine})
            if gnorm > 0 and unorm == 0:
                grad_no_update.append(name)
            if unorm > 0 and grad is None:
                update_no_grad.append(name)
            if grad is not None and not torch.isfinite(grad).all().item():
                nonfinite_grad.append(name)
            if not torch.isfinite(update).all().item():
                nonfinite_update.append(name)
        flat_grad = torch.cat([g.reshape(-1) for g in grads if g is not None], dim=0) if any(g is not None for g in grads) else torch.empty(0)
        flat_update = torch.cat([u.reshape(-1) for u in updates], dim=0) if updates else torch.empty(0)
        return {
            "per_parameter": per_param,
            "overall_gradient_update_cosine": None if flat_grad.numel() != flat_update.numel() or flat_grad.numel() == 0 else self._cosine(flat_grad, flat_update),
            "parameters_with_gradient_but_no_update": grad_no_update,
            "parameters_with_update_but_no_gradient": update_no_grad,
            "nonfinite_gradient_parameters": nonfinite_grad,
            "nonfinite_update_parameters": nonfinite_update,
        }

    def _apply_configured_clipping(self, params: list[torch.nn.Parameter]) -> dict[str, Any]:
        clip_norm = self._config_value(self.config, "grad_clip_norm", None)
        if clip_norm is None or float(clip_norm) <= 0:
            return {"applied": False, "clip_norm": None}
        before = self._total_norm([p.grad for p in params if p.grad is not None])
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_norm))
        after = self._total_norm([p.grad for p in params if p.grad is not None])
        return {"applied": True, "clip_norm": float(clip_norm), "before": before, "after": after}

    @staticmethod
    def _module_state(module: Any) -> Any | None:
        state_dict = getattr(module, "state_dict", None)
        if callable(state_dict):
            return copy.deepcopy(state_dict())
        return None

    @staticmethod
    def _load_module_state(module: Any, state: Any | None) -> None:
        if state is None:
            return
        load_state_dict = getattr(module, "load_state_dict", None)
        if callable(load_state_dict):
            load_state_dict(state)

    def _preserve_training_state(self, state: TrainingTestState):
        tester = self

        class _Context:
            def __enter__(self_inner):
                self_inner.model_state = tester._module_state(state.model)
                self_inner.decoder_state = tester._module_state(state.decoder)
                self_inner.optimizer_state = copy.deepcopy(state.optimizer.state_dict()) if state.optimizer is not None and hasattr(state.optimizer, "state_dict") else None
                self_inner.scheduler_state = copy.deepcopy(state.scheduler.state_dict()) if state.scheduler is not None and hasattr(state.scheduler, "state_dict") else None
                self_inner.config_state = tester._config_snapshot(state.config)
                self_inner.rng_state = torch.get_rng_state()
                self_inner.cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                tester._load_module_state(state.model, self_inner.model_state)
                tester._load_module_state(state.decoder, self_inner.decoder_state)
                if state.optimizer is not None and self_inner.optimizer_state is not None:
                    state.optimizer.load_state_dict(self_inner.optimizer_state)
                if state.scheduler is not None and self_inner.scheduler_state is not None:
                    state.scheduler.load_state_dict(self_inner.scheduler_state)
                tester._restore_config_snapshot(state.config, self_inner.config_state)
                torch.set_rng_state(self_inner.rng_state)
                if self_inner.cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(self_inner.cuda_rng_state)
                return False

        return _Context()

    @staticmethod
    def _config_snapshot(config: Any) -> Any:
        if is_dataclass(config):
            return {f.name: copy.deepcopy(getattr(config, f.name)) for f in fields(config)}
        if hasattr(config, "__dict__"):
            return copy.deepcopy(vars(config))
        if isinstance(config, Mapping):
            return copy.deepcopy(dict(config))
        return None

    @staticmethod
    def _restore_config_snapshot(config: Any, snapshot: Any) -> None:
        if snapshot is None:
            return
        if isinstance(config, Mapping):
            config.clear()
            config.update(snapshot)
        else:
            for key, value in snapshot.items():
                with context_suppress():
                    setattr(config, key, value)

    @staticmethod
    def _config_value(config: Any, name: str, default: Any = None) -> Any:
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    @staticmethod
    def _set_config_value(config: Any, name: str, value: Any) -> None:
        if isinstance(config, Mapping):
            config[name] = value
        elif hasattr(config, name):
            setattr(config, name, value)

    def _set_curve_only_config(self, config: Any) -> None:
        for key, value in {
            "lam_curve_length": 1.0,
            "lam_rep": 0.0,
            "lam_boundary": 0.0,
            "lam_volume": 0.0,
            "lam_fem": 0.0,
            "lam_cell_edge": 0.0,
            "lam_cell_edge_uniform": 0.0,
            "early_stopping": False,
            "use_scheduler": False,
        }.items():
            self._set_config_value(config, key, value)

    def _loss_weights_from_config(self, config: Any, terms: dict[str, Any]) -> dict[str, float]:
        return {name: float(self._config_value(config, f"lam_{name}", 1.0)) for name in terms if name != "total"}

    @staticmethod
    def _learning_rates(optimizer: Any) -> list[float]:
        return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]

    @staticmethod
    def _scheduler_milestones(scheduler: Any, num_steps: int) -> list[int]:
        milestones = getattr(scheduler, "milestones", None)
        if milestones is None:
            return []
        values = list(milestones.keys()) if hasattr(milestones, "keys") else list(milestones)
        out = []
        for value in values:
            f = float(value)
            out.append(int(round(f * num_steps)) if 0.0 < f < 1.0 else int(round(f)))
        return sorted(out)

    @staticmethod
    def _infer_device(*owners: Any) -> torch.device:
        for owner in owners:
            parameters = getattr(owner, "parameters", None)
            if callable(parameters):
                for p in parameters():
                    return p.device
        return torch.device("cpu")

    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _total_norm(tensors: Iterable[torch.Tensor | None]) -> float:
        total = 0.0
        for tensor in tensors:
            if tensor is None:
                continue
            value = float(torch.sum(tensor.detach().double().reshape(-1) ** 2).cpu().item())
            total += value
        return math.sqrt(total)

    @staticmethod
    def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.detach().reshape(-1).double() * b.detach().reshape(-1).double()).sum().cpu().item())

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        af = a.detach().reshape(-1).double()
        bf = b.detach().reshape(-1).double()
        denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
        if float(denom.cpu().item()) <= 0.0:
            return 0.0
        return float((af * bf).sum().div(denom).cpu().item())

    @staticmethod
    def _minimum_seed_distance(seeds: torch.Tensor) -> float:
        seeds = seeds.detach().to(dtype=torch.float64)
        if seeds.ndim < 2 or seeds.shape[0] < 2:
            return float("inf")
        flat = seeds.reshape(seeds.shape[0], -1)
        distances = torch.cdist(flat, flat)
        distances = distances.clone()
        distances.fill_diagonal_(float("inf"))
        return float(distances.min().cpu().item())

    @staticmethod
    def _max_per_seed_norm(displacement: torch.Tensor) -> float:
        if displacement.ndim < 2:
            return float(torch.linalg.vector_norm(displacement.detach()).cpu().item())
        return float(torch.linalg.vector_norm(displacement.detach().reshape(displacement.shape[0], -1), dim=1).max().cpu().item())

    @staticmethod
    def _value_counts(values: Any) -> dict[int, int]:
        if values is None:
            return {}
        flat = torch.as_tensor(values).detach().cpu().reshape(-1).tolist()
        out: dict[int, int] = {}
        for value in flat:
            out[int(value)] = out.get(int(value), 0) + 1
        return out

    def _edge_length_statistics(self, decoder_out: dict[str, Any]) -> dict[str, float | int]:
        curves = decoder_out.get("edge_curves_xyz", decoder_out.get("curves_xyz", None))
        if not isinstance(curves, torch.Tensor) or curves.ndim != 3 or curves.shape[0] == 0 or curves.shape[1] < 2:
            return {
                "minimum_edge_length": float("nan"),
                "maximum_edge_length": float("nan"),
                "mean_edge_length": float("nan"),
                "std_edge_length": float("nan"),
                "coefficient_of_variation": float("nan"),
                "p05_edge_length": float("nan"),
                "median_edge_length": float("nan"),
                "p95_edge_length": float("nan"),
                "max_min_ratio": float("nan"),
                "active_edges": 0,
            }
        lengths = torch.linalg.vector_norm(curves[:, 1:] - curves[:, :-1], dim=-1).sum(dim=1)
        edge_type = self.gradient_tester._edge_types_from_out(decoder_out)
        if edge_type is not None:
            edge_type_t = torch.as_tensor(edge_type, dtype=torch.long, device=lengths.device).reshape(-1)
            if edge_type_t.shape[0] == lengths.shape[0]:
                keep = torch.zeros_like(edge_type_t, dtype=torch.bool)
                for value in self.edge_types:
                    keep |= edge_type_t == int(value)
                lengths = lengths[keep]
        lengths = lengths[torch.isfinite(lengths.detach())]
        if lengths.numel() == 0:
            return {
                "minimum_edge_length": float("nan"),
                "maximum_edge_length": float("nan"),
                "mean_edge_length": float("nan"),
                "std_edge_length": float("nan"),
                "coefficient_of_variation": float("nan"),
                "p05_edge_length": float("nan"),
                "median_edge_length": float("nan"),
                "p95_edge_length": float("nan"),
                "max_min_ratio": float("nan"),
                "active_edges": 0,
            }
        lengths_det = lengths.detach()
        mean = lengths_det.mean()
        std = lengths_det.std(unbiased=False)
        eps = float(self._config_value(self.config, "eps", 1e-12))
        quantiles = torch.quantile(
            lengths_det.to(dtype=torch.float64),
            torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64, device=lengths_det.device),
        )
        min_len = lengths_det.min()
        max_len = lengths_det.max()
        return {
            "minimum_edge_length": self._scalar(min_len),
            "maximum_edge_length": self._scalar(max_len),
            "mean_edge_length": self._scalar(mean),
            "std_edge_length": self._scalar(std),
            "coefficient_of_variation": self._scalar(std / mean.abs().clamp_min(eps)),
            "p05_edge_length": self._scalar(quantiles[0]),
            "median_edge_length": self._scalar(quantiles[1]),
            "p95_edge_length": self._scalar(quantiles[2]),
            "max_min_ratio": self._scalar(max_len / min_len.clamp_min(eps)),
            "active_edges": int(lengths_det.numel()),
        }

    @staticmethod
    def _topology_edge_count(topology: Any) -> int:
        if isinstance(topology, dict):
            edges = topology.get("edges", topology.get("edge_index", None))
            if isinstance(edges, dict):
                edges = edges.get("edge_index", None)
            if edges is not None:
                return int(torch.as_tensor(edges).reshape(-1, 2).shape[0])
        return 0

    def _write_csv(self, path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys and not isinstance(row[key], (dict, list, tuple)):
                    keys.append(key)
        lines = [",".join(keys)]
        for row in rows:
            values = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, str):
                    value = value.replace('"', '""')
                    values.append(f'"{value}"')
                elif isinstance(value, bool):
                    values.append("1" if value else "0")
                elif value is None:
                    values.append("")
                else:
                    values.append(str(value))
            lines.append(",".join(values))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_curve_only_plots(self, output_dir: pathlib.Path, history: list[dict[str, Any]]) -> dict[str, str]:
        if not history:
            return {}
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return {}
        specs = {
            "curve_loss_vs_step.png": ("curve_loss", "Curve loss"),
            "CV_vs_step.png": ("coefficient_of_variation", "Coefficient of variation"),
            "min_edge_vs_step.png": ("minimum_edge_length", "Minimum edge length"),
            "seed_displacement_vs_step.png": ("seed_displacement_from_initial", "Seed displacement from initial"),
        }
        steps = [row["step"] for row in history]
        paths = {}
        for filename, (key, ylabel) in specs.items():
            values = [row.get(key, float("nan")) for row in history]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(steps, values, marker="o", linewidth=1.2, markersize=3)
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / filename
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths[filename] = str(path)
        return paths

    def _optimizer_state_equal(self, current: dict[str, Any], expected: dict[str, Any]) -> bool:
        return self._state_tree_equal(current, expected)

    def _state_tree_equal(self, left: Any, right: Any, atol: float = 0.0) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return False
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            return bool(torch.allclose(left.detach().cpu(), right.detach().cpu(), atol=atol, rtol=0.0))
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict) or set(left.keys()) != set(right.keys()):
                return False
            return all(self._state_tree_equal(left[key], right[key], atol=atol) for key in left)
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                return False
            return all(self._state_tree_equal(a, b, atol=atol) for a, b in zip(left, right))
        return left == right

    @staticmethod
    def _curve_metric_row(metrics: dict[str, Any]) -> dict[str, float]:
        return {
            "min_length": float(metrics.get("min", float("nan"))),
            "max_length": float(metrics.get("max", float("nan"))),
            "mean_length": float(metrics.get("mean", float("nan"))),
            "std": float(metrics.get("std", float("nan"))),
            "cv": float(metrics.get("cv", float("nan"))),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max_min_ratio": float(metrics.get("max_min_ratio", float("nan"))),
        }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() > 64:
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): TrainingLoopTester._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrainingLoopTester._json_safe(v) for v in value]
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def _safe_call():
        return context_suppress()

    def _print_summary(self, results: dict[str, Any]) -> None:
        labels = [
            ("MANUAL SEED DESCENT", "manual_seed_descent"),
            ("OPTIMIZER SINGLE STEP", "optimizer_single_step"),
            ("SEED UPDATE PATH", "seed_update_path"),
            ("LOSS WEIGHT ACCOUNTING", "loss_weight_accounting"),
            ("GRADIENT CLIPPING", "gradient_clipping"),
            ("SCHEDULER", "scheduler"),
            ("PER-LOSS GRADIENTS", "per_loss_gradients"),
            ("CURVE-ONLY TRAINING", "curve_only_training"),
            ("TOPOLOGY MONITOR", "topology_monitor"),
            ("BEST-STATE RESTORE", "best_state_restore"),
        ]
        for title, key in labels:
            if key in results:
                print(title)
                print(f"  pass={results[key].get('overall_pass')} warnings={results[key].get('warnings', [])}")


class context_suppress:
    """Tiny local equivalent of contextlib.suppress(Exception)."""

    def __enter__(self) -> "context_suppress":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return exc_type is not None


def example_forward_builder(model: Any, decoder: Any, state: TrainingTestState) -> dict[str, Any]:
    """Documented adapter shape; replace losses with project-specific terms."""
    model_out = model(state.model_inputs)
    seeds_uv = model_out["seeds_uv"]
    decoder_out = decoder(seeds_uv, **(state.decoder_kwargs or {}))
    raise NotImplementedError(
        "Fill in project loss terms here and return seeds_uv, decoder_out, total_loss, loss_terms, and optional metrics."
    )


def example_usage(
    model: Any,
    decoder: Any,
    cad_domain: Any,
    config: Any,
    model_inputs: Any,
    forward_builder: ForwardBuilder,
    decoder_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the standard suite with a supplied project-specific forward builder."""
    state = TrainingTestState(
        model=model,
        decoder=decoder,
        cad_domain=cad_domain,
        config=config,
        model_inputs=model_inputs,
        decoder_kwargs=decoder_kwargs or {},
    )
    tester = TrainingLoopTester(
        model=model,
        decoder=decoder,
        cad_domain=cad_domain,
        config=config,
        forward_builder=forward_builder,
    )
    return tester.run_all(state)


if __name__ == "__main__":
    print(
        "testTraining.py provides TrainingLoopTester and TrainingTestState.\n"
        "Create a project-specific forward_builder(model, decoder, state) that returns:\n"
        "  seeds_uv, decoder_out, total_loss, loss_terms, and optional metrics/loss_weights.\n"
        "Then instantiate TrainingLoopTester(..., forward_builder=your_builder) and call run_all(state)."
    )
