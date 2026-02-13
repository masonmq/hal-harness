# hal/benchmarks/replicatorbench.py

import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .base_benchmark import BaseBenchmark

# Evaluation mode:
#   - "offline": deterministic compare against pre-extracted GT JSON
#   - "llm": if we want to call GPT judge
REPLICATORBENCH_EVAL_MODE = os.getenv("REPLICATORBENCH_EVAL_MODE", "offline").strip().lower()


class ReplicatorBenchmark(BaseBenchmark):
    """
    ReplicatorBench benchmark.

    Agent output contract (inside HAL sandbox/container):
      - setup.sh downloads capsules to:
          /root/environment/workspace/capsules/<task_id>/
      - agent must write:
          /root/environment/workspace/<task_id>/execution_results.json
      - agent returns to harness:
          { "<task_id>": { "execution_results_path": "<absolute path>" } }

    Ground truth contract (private to benchmark, not visible to agent): :
      - stored in the private benchmark repo, e.g.:
          cos/llm-benchmarks/replicatorbench/ground_truth/execute_ground_truth.json

    These are the scaffold for evaluation:
      (1) load ground truth into self._ground_truth
      (2) compare execution_results.json to ground truth in _score_offline()
      (3) or, we want to use _score_llm_hook() with gpt real validator
    """

    TARGET_ROOT = "/root/environment/workspace"

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        if not hasattr(self, "benchmark_name"):
            self.benchmark_name = "replicatorbench"

        self.requires_sandbox = True
        self.setup_script = "hal/benchmarks/replicatorbench/setup.sh"
        self.config = config or {}

        # benchmark dataset: {task_id: {"prompt": ..., "files": ..., "gpu": ...}}
        self.benchmark: Dict[str, Dict[str, Any]] = {}

        # ground truth: {task_id: {...}}
        self._ground_truth: Dict[str, Dict[str, Any]] = {}

        self._bench_dir = os.path.join(os.path.dirname(__file__), "replicatorbench")
        self._tasks_path = os.path.join(self._bench_dir, "tasks.json")

        # Ground truth lives in cos private repo
        self._gt_path = os.path.join(self._bench_dir, "ground_truth", "execute_ground_truth.json")

        self._load_tasks()
        self._load_ground_truth()

        try:
            super().__init__(
                agent_dir,
                config,
                requires_sandbox=self.requires_sandbox,
                setup_script=self.setup_script,
            )
        except TypeError:
            super().__init__(agent_dir, config)

    # Load tasks
    def _load_tasks(self) -> None:
        if not os.path.exists(self._tasks_path):
            raise FileNotFoundError(f"[replicatorbench] tasks.json not found at: {self._tasks_path}")

        with open(self._tasks_path, "r") as f:
            payload = json.load(f)

        if not (isinstance(payload, dict) and isinstance(payload.get("tasks"), list)):
            raise ValueError(
                "[replicatorbench] tasks.json must be a dict with key 'tasks' holding a list, "
                'e.g. {"tasks": [ ... ]}'
            )

        tasks: List[Dict[str, Any]] = payload["tasks"]
        seen_ids = set()

        for t in tasks:
            if not isinstance(t, dict):
                raise ValueError("[replicatorbench] Each entry in tasks must be an object/dict.")

            task_id = str(t.get("task_id") or "").strip()
            if not task_id:
                raise ValueError("[replicatorbench] Task entry missing required field: task_id")
            if task_id in seen_ids:
                raise ValueError(f"[replicatorbench] Duplicate task_id in tasks.json: {task_id}")
            seen_ids.add(task_id)

            capsule_type = str(t.get("capsule_type") or "").strip()
            capsule_url = str(t.get("capsule_url") or "").strip()
            prompt = str(t.get("prompt") or "").strip()
            gpu = bool(t.get("gpu", False))

            if not capsule_type:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: capsule_type")
            if not capsule_url:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: capsule_url")
            if not prompt:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: prompt")

            # Capsules only exist after setup.sh runs inside the sandbox.
            self.benchmark[task_id] = {
                "prompt": prompt,
                "files": {},  # agents read directly from /root/environment/workspace/capsules/<task_id>/
                "gpu": gpu,
            }

    # Load ground truth (private)
    def _load_ground_truth(self) -> None:
        """
        Expected file format (execute_ground_truth.json):
        {
          "replicatorbench_01": {
            "label": "met",
            "criterion": {"alpha": 0.05, "expected_direction": "positive"},
            "expected_primary_effect": {"value": 0.12, "p_value": 0.03, "direction": "positive"},
            "tolerances": {"value_abs": 0.05, "p_value_abs": 0.05}
          },
          ...
        }
        """
        if not os.path.exists(self._gt_path):
            # missing GT marks tasks unevaluated.
            self._ground_truth = {}
            return

        with open(self._gt_path, "r") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise ValueError(f"[replicatorbench] Ground truth must be a JSON object/dict: {self._gt_path}")

        gt: Dict[str, Dict[str, Any]] = {}
        for task_id, entry in obj.items():
            if not isinstance(entry, dict):
                continue
            gt[str(task_id)] = entry
        self._ground_truth = gt

    # Parsing ......
    def _parse_agent_obj(self, solution: Any) -> Optional[Dict[str, Any]]:
        """Accept dict, JSON string, or Python-literal dict string."""
        if solution is None:
            return None
        if isinstance(solution, dict):
            return solution
        if not isinstance(solution, str):
            solution = str(solution)
        s = solution.strip()
        if not s:
            return None

        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        return None

    def _read_json_file(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _to_float(self, v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("<"):
                s = s[1:].strip()
            if s.endswith("%"):
                s = s[:-1].strip()
            try:
                return float(s)
            except Exception:
                return None
        return None

    def _extract_primary_effect(self, execution_results: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        """
        Extract (estimate, p_value, direction) from results.findings_summary[0].
        """
        results = execution_results.get("results", {}) if isinstance(execution_results, dict) else {}
        findings = results.get("findings_summary", [])
        if not (isinstance(findings, list) and findings and isinstance(findings[0], dict)):
            return None, None, ""
        primary = findings[0]
        est = self._to_float(primary.get("value"))
        pval = self._to_float(primary.get("p_value"))
        direction = str(primary.get("direction", "")).strip().lower()

        if not direction and est is not None:
            if est > 0:
                direction = "positive"
            elif est < 0:
                direction = "negative"
            else:
                direction = "null"

        return est, pval, direction

    # Evaluation, no llm
    def _score_offline(self, execution_results: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic scorer (no LLM). Two supported GT modes:

        Mode A: criterion-based
          - met iff sign matches expected_direction AND p < alpha

        Mode B: value-based
          - compare (estimate, p_value, direction) to expected_primary_effect within tolerances
        """
        est, pval, direction = self._extract_primary_effect(execution_results)
        if est is None or pval is None:
            return {
                "label": "unmet",
                "reason": "Missing numeric primary effect value and/or p_value in results.findings_summary[0].",
                "extracted": {"estimate": est, "p_value": pval, "direction": direction},
            }

        # Mode B (if expected_primary_effect exists)
        exp = gt.get("expected_primary_effect")
        if isinstance(exp, dict):
            exp_est = self._to_float(exp.get("value"))
            exp_p = self._to_float(exp.get("p_value"))
            exp_dir = str(exp.get("direction", "")).strip().lower()

            tol = gt.get("tolerances", {}) if isinstance(gt.get("tolerances"), dict) else {}
            tol_est = self._to_float(tol.get("value_abs")) or 0.0
            tol_p = self._to_float(tol.get("p_value_abs")) or 0.0

            est_ok = True if exp_est is None else (abs(est - exp_est) <= tol_est)
            p_ok = True if exp_p is None else (abs(pval - exp_p) <= tol_p)
            dir_ok = True if not exp_dir else (direction == exp_dir)

            label = "met" if (est_ok and p_ok and dir_ok) else "unmet"
            return {
                "label": label,
                "mode": "expected_primary_effect",
                "extracted": {"estimate": est, "p_value": pval, "direction": direction},
                "expected": {"estimate": exp_est, "p_value": exp_p, "direction": exp_dir},
                "tolerances": {"value_abs": tol_est, "p_value_abs": tol_p},
                "checks": {"estimate_ok": est_ok, "p_ok": p_ok, "direction_ok": dir_ok},
            }

        # Mode A (criterion-only)
        criterion = gt.get("criterion", {}) if isinstance(gt.get("criterion"), dict) else {}
        alpha = self._to_float(criterion.get("alpha", 0.05)) or 0.05
        expected_direction = str(criterion.get("expected_direction", "positive")).strip().lower()

        sign_ok = True
        if expected_direction == "positive":
            sign_ok = est > 0
        elif expected_direction == "negative":
            sign_ok = est < 0

        p_ok = pval < alpha
        label = "met" if (sign_ok and p_ok) else "unmet"
        return {
            "label": label,
            "mode": "criterion",
            "extracted": {"estimate": est, "p_value": pval, "direction": direction},
            "criterion_used": {"alpha": alpha, "expected_direction": expected_direction},
            "checks": {"sign_ok": sign_ok, "p_ok": p_ok},
        }

    def _score_llm_hook(self, study_path: str) -> Dict[str, Any]:
        """
        for LLM scoring.
        e.g., from validator.cli.evaluate_execute_cli import main(...)
        """
        # to do, implement real call.
        return {"label": "unmet", "reason": "...."}

    # HAL API
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for task_id, solution in (agent_output or {}).items():
            task_id = str(task_id)
            parsed = self._parse_agent_obj(solution)

            if not parsed or "execution_results_path" not in parsed:
                results[task_id] = {
                    "correct": False,
                    "error": "Agent output must be a dict with key 'execution_results_path'.",
                    "received": solution,
                }
                continue

            path = str(parsed["execution_results_path"]).strip()
            execution_results = self._read_json_file(path)
            if execution_results is None:
                results[task_id] = {
                    "correct": False,
                    "error": f"Could not read execution_results.json at: {path}",
                    "received_pointer": parsed,
                }
                continue

            gt = self._ground_truth.get(task_id)
            if gt is None:
                results[task_id] = {
                    "correct": False,
                    "unevaluated": True,
                    "error": f"No ground truth found for task_id={task_id}. Expected file: {self._gt_path}",
                }
                continue

            gt_label = str(gt.get("label", "")).strip().lower()
            if gt_label not in {"met", "unmet"}:
                results[task_id] = {
                    "correct": False,
                    "error": "Ground truth missing required field: label ('met' or 'unmet')",
                    "expected_raw": gt,
                }
                continue

            if REPLICATORBENCH_EVAL_MODE == "llm":
                study_path = os.path.join(self.TARGET_ROOT, "capsules", task_id)
                score = self._score_llm_hook(study_path)
            else:
                score = self._score_offline(execution_results, gt)

            pred_label = str(score.get("label", "")).strip().lower()
            correct = (gt_label == pred_label)

            results[task_id] = {
                "correct": correct,
                "ground_truth_label": gt_label,
                "predicted_label": pred_label,
                "score_details": score,
                "received_pointer": parsed,
            }

        # Any task not returned by the agent is an automatic failure
        for tid in self.get_dataset().keys():
            tid = str(tid)
            if tid not in results:
                results[tid] = {
                    "correct": False,
                    "error": "Agent did not return an output for this task_id.",
                }

        return results

    # HAL API
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        task_ids = list(eval_results.keys())
        successful = [tid for tid in task_ids if eval_results[tid].get("correct", False)]
        failed = [tid for tid in task_ids if not eval_results[tid].get("correct", False)]

        total = len(task_ids)
        acc = (len(successful) / total) if total > 0 else 0.0

        return {
            "accuracy": acc,
            "n_tasks": total,
            "n_success": len(successful),
            "n_failed": len(failed),
            "successful_tasks": successful,
            "failed_tasks": failed,
            "eval_mode": REPLICATORBENCH_EVAL_MODE,
            "ground_truth_path": self._gt_path,
        }


class ReplicatorBenchEasy(ReplicatorBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "replicatorbench_easy"
        super().__init__(agent_dir, config)


class ReplicatorBenchHard(ReplicatorBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "replicatorbench_hard"
        super().__init__(agent_dir, config)

    def _filter_files_dict(self, files_dict: Dict[str, str], task: Dict[str, Any]) -> Dict[str, str]:
        """
        Hard tier: do not provide original paper code.
        """
        filtered: Dict[str, str] = {}
        for target_path, source_path in files_dict.items():
            p = target_path.replace("\\", "/")
            if "/original_code/" in p or p.endswith("/original_code"):
                continue
            filtered[target_path] = source_path
        return filtered