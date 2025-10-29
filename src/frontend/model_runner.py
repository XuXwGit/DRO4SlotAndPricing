"""Utility helpers for running optimization models from the web front-end."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import importlib
import logging

from src.utils.model_params import generate_feasible_test_case, construct_model_params
from src.utils.data_manager import DataManager
from src.utils.read_data import DataReader

logger = logging.getLogger(__name__)


class ModelRunError(RuntimeError):
    """Raised when the model execution workflow fails."""


@dataclass
class ModelRunRequest:
    """Structured configuration describing a model execution request."""

    model_type: str
    data_source: str
    case_id: Optional[str]
    num_paths: int
    num_periods: int
    num_prices: int
    uncertainty_dim: int
    price_range: Tuple[float, float]
    demand_sensitivity: float
    base_mean_demand: float
    base_capacity: int
    uncertainty_std_ratio: float
    seed: Optional[int]
    use_lp_relaxation: bool = True


@dataclass
class ModelRunResult:
    """Payload returned to the front-end with aggregated information."""

    success: bool
    model_type: str
    message: str
    objective: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a serialisable dictionary."""

        return {
            "success": self.success,
            "model_type": self.model_type,
            "message": self.message,
            "objective": self.objective,
            "metrics": self.metrics,
            "details": self.details,
        }


SUPPORTED_MODELS = {
    "deterministic": "src.models.dm.determine_model.DeterministicModel",
    "dro_gurobi": "src.models.dro.ldr.SOCP4LDR_GRB.SOCP4LDR_GRB",
    "dro_mosek": "src.models.dro.ldr.SOCP4LDR_Mosek.SOCP4LDR_Mosek",
}


def run_model(request: ModelRunRequest) -> ModelRunResult:
    """Generate parameters, execute the chosen model and summarise the output."""

    try:
        model_params = _build_model_params(request)
        metrics = _summarise_parameters(model_params)
    except ModelRunError as exc:
        return ModelRunResult(
            success=False,
            model_type=request.model_type,
            message=str(exc),
        )

    try:
        result = _execute_model(request, model_params)
        if not result.success:
            result.metrics.update(metrics)
        else:
            result.metrics = {**metrics, **result.metrics}
        return result
    except ModelRunError as exc:
        return ModelRunResult(
            success=False,
            model_type=request.model_type,
            message=str(exc),
            metrics=metrics,
        )


def _build_model_params(request: ModelRunRequest) -> Dict[str, Any]:
    """Construct a valid ``model_params`` dictionary for optimisation models."""

    if request.data_source == "case":
        logger.info("Loading case data via DataManager (case=%s)", request.case_id)
        data_manager = DataManager()
        reader = DataReader(input_data=data_manager, time_horizon=request.num_periods)
        try:
            reader.read(case=request.case_id or "2")
            data_manager.generate_demand_and_price()
        except Exception as exc:  # pragma: no cover - protective fallback
            logger.exception("Failed to load case data: %s", exc)
            raise ModelRunError(
                "案例数据读取失败，请确认数据文件存在且格式正确。"
            ) from exc

        try:
            return construct_model_params(
                data_manager=data_manager,
                price_sensitivity=request.demand_sensitivity,
                uncertainty_dim=request.uncertainty_dim,
                uncertainty_std_ratio=request.uncertainty_std_ratio,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to construct parameters from case data: %s", exc)
            raise ModelRunError("案例数据构造模型参数失败，请检查输入配置。") from exc

    logger.info("Generating synthetic scenario (num_paths=%s)", request.num_paths)
    try:
        return generate_feasible_test_case(
            num_paths=request.num_paths,
            num_periods=request.num_periods,
            num_prices=request.num_prices,
            uncertainty_dim=request.uncertainty_dim,
            price_range=request.price_range,
            demand_sensitivity=request.demand_sensitivity,
            base_mean_demand=request.base_mean_demand,
            base_capacity=request.base_capacity,
            uncertainty_std_ratio=request.uncertainty_std_ratio,
            seed=request.seed,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Synthetic parameter generation failed: %s", exc)
        raise ModelRunError("随机算例生成失败，请调整参数后重试。") from exc


def _execute_model(request: ModelRunRequest, model_params: Dict[str, Any]) -> ModelRunResult:
    """Dispatch to the requested model solver and capture its solution."""

    model_path = SUPPORTED_MODELS.get(request.model_type)
    if not model_path:
        return ModelRunResult(
            success=False,
            model_type=request.model_type,
            message="不支持的模型类型。",
            metrics=_summarise_parameters(model_params),
        )

    module_name, class_name = model_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        model_cls = getattr(module, class_name)
    except Exception as exc:  # pragma: no cover - depends on optional solvers
        logger.exception("Model import failed: %s", exc)
        raise ModelRunError(
            "模型依赖未安装或求解器不可用，请检查运行环境。"
        ) from exc

    model = None
    try:
        if request.model_type == "deterministic":
            model = model_cls(model_params=model_params, use_lp_relaxation=request.use_lp_relaxation)
        else:
            model = model_cls(model_params=model_params)
        model.build_model()
        model.solve()
        status = getattr(model, "get_status", lambda: "OPTIMAL")()
        solution = getattr(model, "get_solution", lambda: {})()
    except Exception as exc:  # pragma: no cover - solver interaction
        logger.exception("Model execution failed: %s", exc)
        raise ModelRunError("模型求解失败，请查看日志或调整参数。") from exc

    if status != "OPTIMAL":
        message = f"求解完成，但状态为 {status}。"
        return ModelRunResult(
            success=False,
            model_type=request.model_type,
            message=message,
            details=_format_solution_preview(solution),
            metrics={"status": status},
        )

    message = "求解成功，以下为关键结果摘要。"
    details = _format_solution_preview(solution)
    metrics = {"status": status, "objective": solution.get("obj_val")}
    return ModelRunResult(
        success=True,
        model_type=request.model_type,
        message=message,
        objective=solution.get("obj_val"),
        metrics=metrics,
        details=details,
    )


def _summarise_parameters(model_params: Dict[str, Any]) -> Dict[str, Any]:
    """Collect human-friendly statistics from ``model_params`` for display."""

    price_values = model_params.get("p_list", [])
    capacities = list(model_params.get("A_prime", {}).values())
    sigma_sq = model_params.get("sigma_sq")
    return {
        "路径数量": model_params.get("num_paths"),
        "时间段数": model_params.get("num_periods"),
        "价格点数量": len(price_values),
        "价格范围": _format_range(price_values),
        "容量范围": _format_range(capacities),
        "不确定性维度": model_params.get("I1"),
        "方差估计": list(sigma_sq) if sigma_sq is not None else None,
    }


def _format_range(values: Iterable[Any]) -> Optional[str]:
    """Return a formatted ``min - max`` string for a numeric collection."""

    values = list(values)
    if not values:
        return None
    try:
        return f"{min(values):.2f} - {max(values):.2f}"
    except TypeError:
        return None


def _format_solution_preview(solution: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact view of potentially large solution dictionaries."""

    preview = {}
    if not solution:
        return preview

    def head(items: Iterable[Tuple[Any, Any]], limit: int = 5) -> List[Tuple[Any, Any]]:
        collected: List[Tuple[Any, Any]] = []
        for key, value in items:
            collected.append((key, value))
            if len(collected) >= limit:
                break
        return collected

    if "X" in solution:
        preview["X"] = head(sorted(solution["X"].items(), key=lambda item: str(item[0])))
    if "Y" in solution:
        preview["Y"] = head(sorted(solution["Y"].items(), key=lambda item: str(item[0])))
    if "R" in solution:
        preview["R"] = head(sorted(solution["R"].items(), key=lambda item: (str(item[0]), str(item[1]))))
    if "G" in solution:
        preview["G"] = head(sorted(solution["G"].items(), key=lambda item: str(item[0])), limit=3)

    return preview
