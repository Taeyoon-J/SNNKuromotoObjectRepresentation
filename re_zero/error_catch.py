"""
Optional shape-validation snippets for `re_zero` modules.

This file is intentionally kept separate so the main model code stays easy to
read. When you want explicit input validation during debugging, copy the
relevant commented block into the target function and uncomment it.

Example:
    In `KuramotoLayer.compute_coupling_term(...)`, paste the validation block
    below right after the early return for `self.coupling == 0.0`.
"""


# Shape checks for `KuramotoLayer.compute_coupling_term(...)`
#
# Use this block when debugging shape mismatches among:
# - `theta_prev` with shape [B, N, D]
# - `affinity` with shape [B, N, N]
# - `alpha_t` with shape [B, N, N]
#
# if affinity.dim() != 3:
#     raise ValueError(f"affinity must have shape [B, N, N], got {tuple(affinity.shape)}")
# if alpha_t.dim() != 3:
#     raise ValueError(f"alpha_t must have shape [B, N, N], got {tuple(alpha_t.shape)}")
# if affinity.shape[:2] != (theta_prev.shape[0], theta_prev.shape[1]) or affinity.shape[2] != theta_prev.shape[1]:
#     raise ValueError(
#         "affinity must align with theta_prev and have shape [B, N, N]"
#     )
# if alpha_t.shape != affinity.shape:
#     raise ValueError("alpha_t must have the same shape as affinity")
