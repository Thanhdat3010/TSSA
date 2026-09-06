"""
LogTracker: Comprehensive Training Dynamics and Gate Logger ("Log Lại Hết")
Records:
1. Step-level training losses (loss_mt, loss_struct, loss_prime, loss_route, loss_total, lr).
2. Epoch-level gate activation maps [n_layers, n_heads] for Head Specialization Heatmaps.
3. Checkpoint evaluation trajectories for publication curves.
All data is stored in structured JSON and PyTorch tensor formats in checkpoints/ablation_logs/.
"""

import os
import json
import torch
import numpy as np

class LogTracker:
    def __init__(self, exp_name: str = "default_exp", log_dir: str = "checkpoints/ablation_logs"):
        self.exp_name = exp_name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.step_history = []
        self.epoch_gate_history = []
        self.eval_trajectory = []

    def log_step(self, step: int, epoch: float, loss_dict: dict, lr: float = None):
        """Logs step-level losses."""
        entry = {
            "step": step,
            "epoch": round(float(epoch), 4),
            "losses": {k: round(float(v), 6) for k, v in loss_dict.items() if isinstance(v, (int, float, torch.Tensor))},
        }
        if lr is not None:
            entry["lr"] = float(lr)
        self.step_history.append(entry)

    def log_gate_activations(self, epoch: int, router_gates: torch.Tensor):
        """
        Logs gate activation tensor [B, L, H, T, 1] or [B, L, H, T].
        Computes mean activation per head across layer: [L, H].
        """
        if router_gates is None:
            return
        with torch.no_grad():
            if router_gates.dim() == 5 and router_gates.size(-1) == 1:
                router_gates = router_gates.squeeze(-1) # [B, L, H, T]
            
            # Mean over batch and token dimensions -> [L, H]
            mean_head_matrix = router_gates.mean(dim=(0, -1)).detach().cpu()
            active_heads = (mean_head_matrix > 0.5).sum().item()
            total_heads = mean_head_matrix.numel()

            self.epoch_gate_history.append({
                "epoch": epoch,
                "head_matrix": mean_head_matrix.tolist(),
                "active_heads": active_heads,
                "total_heads": total_heads,
                "mean_activation": round(float(mean_head_matrix.mean().item()), 4)
            })

    def log_eval_metrics(self, epoch: int, metrics: dict):
        """Logs validation evaluation scores."""
        self.eval_trajectory.append({
            "epoch": epoch,
            "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()}
        })

    def save(self):
        """Persists all logged trajectories to JSON and PT files."""
        json_path = os.path.join(self.log_dir, f"{self.exp_name}_training_dynamics.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "exp_name": self.exp_name,
                    "step_history_sample": self.step_history[::max(1, len(self.step_history) // 200)], # 200 subsamples for fast plotting
                    "epoch_gates": self.epoch_gate_history,
                    "eval_trajectory": self.eval_trajectory
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Lỗi ghi log dynamics: {e}")

        # Save gate tensors for 600 DPI publication heatmap plotting
        if self.epoch_gate_history:
            pt_path = os.path.join(self.log_dir, f"{self.exp_name}_gate_heatmap.pt")
            try:
                final_matrix = torch.tensor(self.epoch_gate_history[-1]["head_matrix"])
                torch.save(final_matrix, pt_path)
            except Exception:
                pass

DynamicLogTracker = LogTracker
