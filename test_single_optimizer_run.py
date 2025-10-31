import json
import uuid
from pathlib import Path

from environment_state import EnvironmentState
from optimization_strategies.dspy_optimizer import DSPyOptimizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


def main():
    # Paths
    repo_root = Path(__file__).parent
    config_path = repo_root / "config.yaml"
    test_case_path = repo_root / "attack_bench" / "indirect" / "05_test.json"

    # Load config and test case
    config = load_config(str(config_path))
    with open(test_case_path, "r", encoding="utf-8") as f:
        test_case = json.load(f)

    # Prepare environment state
    env_state = EnvironmentState()
    env_state.update_from_test_data(test_case.get("initial_data", {}), config)

    # Extract original attack and first step
    attack_emails = test_case.get("initial_data", {}).get("attack_emails", [])
    if not attack_emails:
        raise RuntimeError("No attack_emails found in test case.")
    original_attack = attack_emails[0]

    steps = test_case.get("steps", [])
    if not steps:
        raise RuntimeError("No steps found in test case.")
    step = steps[0]

    # Session id for this run
    session_id = f"single_{uuid.uuid4().hex[:8]}"

    # Optimizer with max_variations=1 for a single-iteration check
    opt = DSPyOptimizer({
        "dspy": {
            "model_name": config.get("openai", {}).get("model", "gpt-4o-mini"),
            "max_tokens": 1200,
            "max_variations": 1,
        },
        "data": config.get("data", {}),
    })

    # Run optimization for a single iteration
    result = opt.optimize_attack(
        original_attack_email=original_attack,
        failed_step=step,
        step_num=1,
        session_id=session_id,
        test_config=config,
        environment_state=env_state,
        max_iterations=1,
    )

    # Print concise outcome
    print(json.dumps({
        "success": result.success,
        "iterations": result.iterations,
        "final_score": (result.final_evaluation or {}).get("score"),
        "feedback_tail": result.feedback[-10:],
        "metadata": result.metadata,
    }, indent=2))


if __name__ == "__main__":
    main()


