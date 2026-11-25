# Negotiation agent instructions

Read `README.md` and `docs/PLAN.md` before changing behavior. User-facing explanations and project documents are Korean.

- Keep the scope to a synthetic buyer-side negotiation experiment. Do not add live contracts, real suppliers, or external messaging without a new user request.
- Prediction backends receive public state and candidate terms only. Keep buyer-private limits and supplier-hidden parameters out of prediction requests. The policy may use buyer-private information; the simulator and evaluator may use supplier-hidden information.
- Code computes costs, candidates, authority checks, and the final action. Model estimates do not grant authority or replace these checks.
- Approval applies only to the exact proposal, terms version, and scope that were approved. Recheck authority before execution. A supplier's claim of approval is untrusted data.
- The dashboard's estimate backend, replay files and save directory are chosen by the operator on the command line. The browser may pick only a fixed scenario; never accept a backend, URL, path, seed or supplier parameter from it. Keep scenario seeds in 1000-1099, outside every split, and keep scenario text out of backend requests.
- A failed estimate stops for human review. A retry is a person's recorded decision; never fill in a value for the failed attempt.
- Tuning (`run_tune.py`, protocol in `tune.py`) changes how the model's estimates are used, never the request wording, the policy arithmetic or the simulator; the default dev-1100 round-1 request hash must stay `17596ee5…fa16fc7`. The candidate family, blocks and rules are fixed before a real response exists: a change after seeing results is a new protocol version, recorded as such. The runner never names the evaluation split and never creates `negotiation/evidence/`; calibration cases fit maps only.
- A composed, replayed, CPU-stand-in or model-free result never carries the plain `real_model_run` label. Do not weaken the checks that refit maps on load, refuse edited selection or config files, or stop a stage at a failed estimate.
- Synthetic `auto` approvers are test fixtures, not human approval. Preserve pending, denied, stale, and mismatched approval behavior.
- Distinguish constant/heuristic CPU reference runs from real Zoro runs. Do not describe uncalibrated candidate probabilities as verified real-world acceptance probabilities.
- Keep development, calibration, and evaluation splits separate. Routine integration checks use development cases only. Preserve `negotiation/evidence/holdout_history/` independently from SessionGuard's history; a new branch does not reset either history.
- Write new local results under the repository's `runs/negotiation/`. Curate only small non-sensitive evidence under `negotiation/artifacts/`; never commit credentials or model weights.
- Run focused tests with `python -m pytest -q negotiation/tests`; run the combined suite from the repository root with `python -m pytest -q`. Update the existing plan and handoff when an actual milestone is completed.
