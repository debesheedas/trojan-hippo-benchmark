# Aggregated test results (multi-seed)

This folder holds **per-seed CSV exports** of attack results and **mean ± std** outputs.

- **results_seed{N}.csv** – One row per result file (split, model, topic, backend, defense, test_name, user/attack/stealth rates, seed). One file per run; seed is read from `agent_config.yaml` when exporting.
- **mean_std_summary.csv** – After running `aggregate_seed_runs.py`: one row per (split, model, topic, backend, defense) with `user_rate_mean`, `user_rate_std`, `attack_rate_mean`, `attack_rate_std`, `n_seeds`.
- **asr_vs_session_defense_none_*.png** – ASR vs session index (defense=none, all topics), mean ± std across seeds.

See project root or scripts for the exact run sequence (3 seeds → export each → aggregate).
