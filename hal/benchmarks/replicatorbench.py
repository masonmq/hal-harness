# COS
# hal/benchmarks/replicatorbench.py

import ast
import json
import os
from typing import Any, Dict, List, Optional

from .base_benchmark import BaseBenchmark

# Use a placeholder scorer to enable end-to-end tests.
USE_PLACEHOLDER_SCORER = True


class ReplicatorBenchmark(BaseBenchmark):
    """
    ReplicatorBench benchmark

    tasks.json is the source of task prompts and ground truth.

    Runtime contract (inside the HAL sandbox/container):
      - setup.sh downloads each capsule into:
          /root/environment/workspace/capsules/<task_id>/
      - the agent writes its structured output to:
          /root/environment/workspace/<task_id>/execution_results.json
      - the agent returns a small dict in chat:
          {"execution_results_path": ".../execution_results.json"}

    Note: The benchmark must NOT require capsule paths to exist at import/init time,
    because benchmark initialization happens on the host before the sandbox is prepared.
    """

    TARGET_ROOT = "/root/environment/workspace"

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        if not hasattr(self, "benchmark_name"):
            self.benchmark_name = "replicatorbench"

        self.requires_sandbox = True
        self.setup_script = "hal/benchmarks/replicatorbench/setup.sh"

        self.config = config or {}
        self.benchmark: Dict[str, Dict[str, Any]] = {}
        self._answers: Dict[str, Any] = {}

        self._bench_dir = os.path.join(os.path.dirname(__file__), "replicatorbench")
        self._tasks_path = os.path.join(self._bench_dir, "tasks.json")

        # Load tasks without capsule directories, capsules are created later by setup.sh.
        self._load_tasks()

        try:
            super().__init__(
                agent_dir,
                config,
                requires_sandbox=self.requires_sandbox,
                setup_script=self.setup_script,
            )
        except TypeError:
            super().__init__(agent_dir, config)

    def _load_tasks(self) -> None:
        if not os.path.exists(self._tasks_path):
            raise FileNotFoundError(f"[replicatorbench] tasks.json not found at: {self._tasks_path}")

        with open(self._tasks_path, "r") as f:
            payload = json.load(f)

        if not (isinstance(payload, dict) and isinstance(payload.get("tasks"), list)):
            raise ValueError(
                "[replicatorbench] tasks.json must be a dict with key 'tasks' holding a list, "
                'for example: {"tasks": [ ... ]}'
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

            if not capsule_type:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: capsule_type")
            if not capsule_url:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: capsule_url")
            if not prompt:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: prompt")

            gpu = bool(t.get("gpu", False))

            # Capsules only exist after setup.sh runs inside the sandbox.
            self.benchmark[task_id] = {
                "prompt": prompt,
                "files": {},  # agents read directly from /root/environment/workspace/capsules/<task_id>/
                "gpu": gpu,
            }

            # Ground truth for evaluation.
            #self._answers[task_id] = t.get("answer", None)

    def _parse_agent_obj(self, solution: Any) -> Optional[Dict[str, Any]]:
        """
        Accept:
          - dict directly
          - JSON string
          - Python-literal dict string
        """
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

    def _score_placeholder(self, execution_results: Dict[str, Any], gt_answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use findings_summary[0] as the primary effect.
        "met" iff:
          - sign matches expected_direction
          - p_value < alpha
        """
        criterion = (gt_answer or {}).get("criterion", {}) if isinstance(gt_answer, dict) else {}
        alpha = self._to_float(criterion.get("alpha", 0.05)) or 0.05
        expected_direction = str(criterion.get("expected_direction", "positive")).strip().lower()

        results = execution_results.get("results", {}) if isinstance(execution_results, dict) else {}
        findings = results.get("findings_summary", [])
        if not (isinstance(findings, list) and len(findings) > 0 and isinstance(findings[0], dict)):
            return {
                "label": "unmet",
                "reason": "Missing results.findings_summary[0] for primary effect.",
            }

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

        if est is None or pval is None:
            return {
                "label": "unmet",
                "reason": "Primary effect missing numeric value and/or p_value.",
                "extracted": {"value": primary.get("value"), "p_value": primary.get("p_value")},
            }

        sign_ok = True
        if expected_direction == "positive":
            sign_ok = est > 0
        elif expected_direction == "negative":
            sign_ok = est < 0

        p_ok = pval < alpha

        label = "met" if (sign_ok and p_ok) else "unmet"
        return {
            "label": label,
            "extracted": {"estimate": est, "p_value": pval, "direction": direction},
            "criterion_used": {"alpha": alpha, "expected_direction": expected_direction},
            "checks": {"sign_ok": sign_ok, "p_ok": p_ok},
        }

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

            expected = self._answers.get(task_id, None)
            if expected is None:
                results[task_id] = {
                    "correct": False,
                    "unevaluated": True,
                    "error": "No ground-truth answer provided for this task yet.",
                    "received_pointer": parsed,
                }
                continue

            if not isinstance(expected, dict) or "label" not in expected:
                results[task_id] = {
                    "correct": False,
                    "error": "Ground-truth answer missing required field: answer.label",
                    "expected_raw": expected,
                }
                continue

            if USE_PLACEHOLDER_SCORER:
                score = self._score_placeholder(execution_results, expected)
            else:
                # replace with our real evaluation scorer.
                score = self._score_placeholder(execution_results, expected)

            gt_label = str(expected.get("label", "")).strip().lower()
            pred_label = str(score.get("label", "")).strip().lower()
            correct = (gt_label == pred_label)

            results[task_id] = {
                "correct": correct,
                "ground_truth_label": gt_label,
                "predicted_label": pred_label,
                "score_details": score,
                "received_pointer": parsed,
            }

        for tid in self.get_dataset().keys():
            tid = str(tid)
            if tid not in results:
                results[tid] = {
                    "correct": False,
                    "error": "Agent did not return an output for this task_id.",
                }

        return results

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
        Assumes original_code/ as the code folder.
        """
        filtered: Dict[str, str] = {}
        for target_path, source_path in files_dict.items():
            p = target_path.replace("\\", "/")
            if "/original_code/" in p or p.endswith("/original_code"):
                continue
            filtered[target_path] = source_path
        return filtered
