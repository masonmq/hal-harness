# hal/benchmarks/replicatorbench.py

import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_benchmark import BaseBenchmark

# Evaluation mode:
#   - "offline": pre-extracted GT JSON (execute stage only)
#   - "llm": LLM judge
REPLICATORBENCH_EVAL_MODE = os.getenv("REPLICATORBENCH_EVAL_MODE", "offline").strip().lower()


JsonObj = Union[Dict[str, Any], List[Any]]


class ReplicatorBenchmark(BaseBenchmark):
    """
    ReplicatorBench benchmark.

    Inside HAL sandbox/container:
      - setup.sh downloads capsules to:
          /root/environment/workspace/capsules/<task_id>/

    Agent output:
      - agent writes stage outputs under:
          /root/environment/workspace/<task_id>/
        Required outputs:
          Extract stage:
            - post_registration.json
            - merged-urls.json
          Design stage:
            - replication_info.json
          Execute stage:
            - execution_results.json
          Interpret stage:
            - interpret_results.json

    Agent return to harness:
      - we accept either:
          { "<task_id>": { "execution_results_path": "<abs path>", ... } }
        or returning nothing but still writing files to the required locations.
        This evaluator will fall back to the canonical paths under /root/environment/workspace/<task_id>/.

    Ground truth contract (private to benchmark, not visible to agent):
      - execute-stage GT lives in:
          hal/benchmarks/replicatorbench/ground_truth/execute_ground_truth.json

    Current scoring behavior:
      - Placeholder "eval_summary-like" scoring for all stages:
          stage score = 1.0 if required file exists and parses as JSON (dict/list as appropriate), else 0.0
      - Execute stage can additionally use offline GT compare (met/unmet) if GT exists.
    """

    TARGET_ROOT = "/root/environment/workspace"

    # Required outputs per stage
    REQUIRED_OUTPUTS = {
        "extract": ["post_registration.json", "merged-urls.json"],
        "design": ["replication_info.json"],
        "execute": ["execution_results.json"],
        "interpret": ["interpret_results.json"],
    }

    # Optional: agent-return pointer keys we understand
    POINTER_KEYS = {
        "post_registration.json": ["post_registration_path"],
        "merged-urls.json": ["merged_urls_path", "merged-urls_path"],
        "replication_info.json": ["replication_info_path"],
        "execution_results.json": ["execution_results_path"],
        "interpret_results.json": ["interpret_results_path"],
    }

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        if not hasattr(self, "benchmark_name"):
            self.benchmark_name = "replicatorbench"

        self.requires_sandbox = True
        self.setup_script = "hal/benchmarks/replicatorbench/setup.sh"
        self.config = config or {}

        # dataset: {task_id: {"prompt": ..., "files": ..., "gpu": ...}}
        self.benchmark: Dict[str, Dict[str, Any]] = {}

        # ground truth: {task_id: {...}}
        self._ground_truth: Dict[str, Dict[str, Any]] = {}

        self._bench_dir = os.path.join(os.path.dirname(__file__), "replicatorbench")
        self._tasks_path = os.path.join(self._bench_dir, "tasks.json")

        # Execute-stage GT lives in the benchmark repo (private to benchmark, not in capsules)
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

    def _load_ground_truth(self) -> None:
        """
        execute_ground_truth.json:
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
            self._ground_truth = {}
            return

        with open(self._gt_path, "r") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise ValueError(f"[replicatorbench] Ground truth must be a JSON object/dict: {self._gt_path}")

        gt: Dict[str, Dict[str, Any]] = {}
        for task_id, entry in obj.items():
            if isinstance(entry, dict):
                gt[str(task_id)] = entry
        self._ground_truth = gt

    #  accept dict, JSON string, or Python dict string
    def _parse_agent_obj(self, solution: Any) -> Optional[Dict[str, Any]]:

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

    def _read_json_any(self, path: str) -> Optional[JsonObj]:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
            if isinstance(obj, (dict, list)):
                return obj
            return None
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

    def _canonical_task_dir(self, task_id: str) -> str:
        return os.path.join(self.TARGET_ROOT, task_id)

    def _canonical_output_path(self, task_id: str, filename: str) -> str:
        return os.path.join(self._canonical_task_dir(task_id), filename)

    def _resolve_path_from_pointer(
        self, parsed_ptr: Optional[Dict[str, Any]], task_id: str, filename: str
    ) -> str:
        if parsed_ptr and isinstance(parsed_ptr, dict):
            for k in self.POINTER_KEYS.get(filename, []):
                v = parsed_ptr.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return self._canonical_output_path(task_id, filename)
    
    # scoring without llm
    def _score_offline_execute(self, execution_results: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic execute scorer. Two supported GT modes:

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

    # scring with llm 
    def _score_llm_hook(self, study_path: str) -> Dict[str, Any]:

        return {"label": "unmet", "reason": "LLM scoring scaffold."}

    # Stage checks 
    def _stage_check_json(
        self,
        task_id: str,
        stage: str,
        filename: str,
        parsed_ptr: Optional[Dict[str, Any]],
        allow_list: bool = False,
    ) -> Dict[str, Any]:
        """
        want to check:
          - file exists
          - JSON parses
          - type is dict
        """
        path = self._resolve_path_from_pointer(parsed_ptr, task_id, filename)
        info: Dict[str, Any] = {"file": filename, "path": path, "ok": False, "error": None}

        if not os.path.exists(path):
            info["error"] = "missing_file"
            return info

        obj = self._read_json_any(path)
        if obj is None:
            info["error"] = "invalid_json"
            return info

        if isinstance(obj, dict):
            info["ok"] = True
            return info
        if allow_list and isinstance(obj, list):
            info["ok"] = True
            return info

        info["error"] = "wrong_json_type"
        return info

    def _evaluate_task_placeholder(
        self, task_id: str, parsed_ptr: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:

        # Extract stage
        extract_checks = [
            self._stage_check_json(task_id, "extract", "post_registration.json", parsed_ptr, allow_list=False),
            self._stage_check_json(task_id, "extract", "merged-urls.json", parsed_ptr, allow_list=True),
        ]
        extract_score = 1.0 if all(c["ok"] for c in extract_checks) else 0.0

        # Design stage
        design_checks = [
            self._stage_check_json(task_id, "design", "replication_info.json", parsed_ptr, allow_list=False),
        ]
        design_score = 1.0 if all(c["ok"] for c in design_checks) else 0.0

        # Execute stage (structural + optional GT compare)
        execute_check = self._stage_check_json(task_id, "execute", "execution_results.json", parsed_ptr, allow_list=False)
        execute_score = 1.0 if execute_check["ok"] else 0.0

        execute_details: Dict[str, Any] = {"structural_ok": execute_check["ok"], "gt_compare": None}
        gt_label = None
        pred_label = None
        execute_correct_vs_gt = None

        if execute_check["ok"]:
            exec_path = execute_check["path"]
            execution_results_obj = self._read_json_any(exec_path)
            execution_results = execution_results_obj if isinstance(execution_results_obj, dict) else None

            gt = self._ground_truth.get(task_id)
            if isinstance(gt, dict) and execution_results is not None:
                gt_label = str(gt.get("label", "")).strip().lower()
                if gt_label in {"met", "unmet"}:
                    if REPLICATORBENCH_EVAL_MODE == "llm":
                        study_path = os.path.join(self.TARGET_ROOT, "capsules", task_id)
                        score = self._score_llm_hook(study_path)
                    else:
                        score = self._score_offline_execute(execution_results, gt)

                    pred_label = str(score.get("label", "")).strip().lower()
                    execute_details["gt_compare"] = {
                        "ground_truth_label": gt_label,
                        "predicted_label": pred_label,
                        "score_details": score,
                    }
                    execute_correct_vs_gt = (gt_label == pred_label)
                    # If we have GT, the execute stage score is whether it matched GT label.
                    execute_score = 1.0 if execute_correct_vs_gt else 0.0

        # Interpret stage
        interpret_checks = [
            self._stage_check_json(task_id, "interpret", "interpret_results.json", parsed_ptr, allow_list=False),
        ]
        interpret_score = 1.0 if all(c["ok"] for c in interpret_checks) else 0.0

        stage_scores = {
            "extract": extract_score,
            "design": design_score,
            "execute": execute_score,
            "interpret": interpret_score,
        }

        overall_score = sum(stage_scores.values()) / 4.0

        eval_summary = {
            "task_id": task_id,
            "stage_scores": stage_scores,
            "overall_score": overall_score,
            "stage_checks": {
                "extract": extract_checks,
                "design": design_checks,
                "execute": [execute_check],
                "interpret": interpret_checks,
            },
            "execute_details": execute_details,
            "eval_mode": REPLICATORBENCH_EVAL_MODE,
        }

        # write eval_summary.json 
        try:
            out_dir = self._canonical_task_dir(task_id)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "eval_summary.json")
            with open(out_path, "w") as f:
                json.dump(eval_summary, f, indent=2)
            eval_summary["eval_summary_path"] = out_path
        except Exception:
            pass

        # treat as pass only if all stages score 1.0
        all_stages_ok = all(v >= 1.0 for v in stage_scores.values())

        return {
            "correct": bool(all_stages_ok),
            "overall_score": overall_score,
            "eval_summary": eval_summary,
            "ground_truth_label": gt_label,
            "predicted_label": pred_label,
            "execute_correct_vs_gt": execute_correct_vs_gt,
            "received_pointer": parsed_ptr,
        }

    # HAL API
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """
        Evaluate each task. We accept that some agents may only return one pointer (or none),
        but we always check for all required files at default paths.
        """
        results: Dict[str, Any] = {}

        for task_id, solution in (agent_output or {}).items():
            task_id = str(task_id)
            parsed_ptr = self._parse_agent_obj(solution)

            # Evaluate using default paths
            results[task_id] = self._evaluate_task_placeholder(task_id, parsed_ptr)

        # Any task not returned by the agent is an automatic failure, but we still evaluate
        # using default paths.
        for tid in self.get_dataset().keys():
            tid = str(tid)
            if tid not in results:
                results[tid] = self._evaluate_task_placeholder(tid, parsed_ptr=None)
                results[tid]["error"] = "Agent did not return an output for this task_id."

        return results

    # HAL API
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate:
          - pass rate (all 4 stages ok)
          - average overall_score
          - per-stage average score
        """
        task_ids = list(eval_results.keys())
        total = len(task_ids)

        passed = [tid for tid in task_ids if eval_results[tid].get("correct", False)]
        failed = [tid for tid in task_ids if not eval_results[tid].get("correct", False)]

        avg_overall = 0.0
        stage_sums = {"extract": 0.0, "design": 0.0, "execute": 0.0, "interpret": 0.0}

        for tid in task_ids:
            ev = eval_results.get(tid, {}) or {}
            avg_overall += float(ev.get("overall_score") or 0.0)
            summary = (ev.get("eval_summary") or {})
            stage_scores = (summary.get("stage_scores") or {})
            for k in stage_sums.keys():
                stage_sums[k] += float(stage_scores.get(k) or 0.0)

        avg_overall = (avg_overall / total) if total > 0 else 0.0
        stage_avgs = {k: (v / total if total > 0 else 0.0) for k, v in stage_sums.items()}
        pass_rate = (len(passed) / total) if total > 0 else 0.0

        return {
            "pass_rate_all_stages": pass_rate,
            "avg_overall_score": avg_overall,
            "avg_stage_scores": stage_avgs,
            "n_tasks": total,
            "n_passed": len(passed),
            "n_failed": len(failed),
            "passed_tasks": passed,
            "failed_tasks": failed,
            "eval_mode": REPLICATORBENCH_EVAL_MODE,
            "execute_ground_truth_path": self._gt_path,
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