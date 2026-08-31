"""
TSSALossScheduler Module (TSSA 2.0)
Implements a 3-phase progressive loss schedule for TSSA 2.0:
- Phase 1 (0 <= t < T1): Base NMT Warmup (lambda_1=0, lambda_2=0, lambda_3=0)
- Phase 2 (T1 <= t < T2): Target Cross-Attention Anchoring & Sentence Priming ramp up linearly
- Phase 3 (t >= T2): Dynamic Head-Wise Routing activates and specializes
"""

class TSSALossScheduler:
    def __init__(self, total_steps: int, max_l1: float = 0.10, max_l2: float = 0.05, max_l3: float = 0.10,
                 warmup_ratio: float = 0.10, rampup_ratio: float = 0.30, routing_ratio: float = 0.40):
        self.total_steps = max(1, total_steps)
        self.max_l1 = max_l1
        self.max_l2 = max_l2
        self.max_l3 = max_l3

        self.t1 = int(warmup_ratio * self.total_steps)
        self.t2 = int(routing_ratio * self.total_steps)
        self.rampup_steps = max(1, int(rampup_ratio * self.total_steps))

    def get_lambdas(self, current_step: int) -> tuple:
        """
        Returns (lambda_1, lambda_2, lambda_3) at current_step.
        """
        if current_step < self.t1:
            # Phase 1: Pure Translation warmup
            return 0.0, 0.0, 0.0
            
        elif current_step < self.t2:
            # Phase 2: Linear ramp-up for Cross-Attention Anchoring and Sentence Priming
            progress = min(1.0, (current_step - self.t1) / self.rampup_steps)
            l1 = self.max_l1 * progress
            l2 = self.max_l2 * progress
            return l1, l2, 0.0
            
        else:
            # Phase 3: Anchoring & Priming at full strength, Router activates
            progress_route = min(1.0, (current_step - self.t2) / self.rampup_steps)
            l3 = self.max_l3 * progress_route
            return self.max_l1, self.max_l2, l3
