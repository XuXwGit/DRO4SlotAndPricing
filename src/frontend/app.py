"""Flask application exposing an interactive dashboard for model execution."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace

from flask import Flask, render_template, request

from src.frontend.model_runner import ModelRunRequest, ModelRunResult, run_model


def create_app() -> Flask:
    """Application factory used by the WSGI server entry point."""

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        """Render the main dashboard and handle form submissions."""

        errors: List[str] = []
        result: Optional[ModelRunResult] = None
        form_values = _default_form_values()

        if request.method == "POST":
            form_values.update(request.form.to_dict())
            run_request, parse_errors = _parse_request(form_values)
            errors.extend(parse_errors)
            if run_request and not errors:
                result = run_model(run_request)
                if not result.success:
                    errors.append(result.message)
            elif errors:
                result = None

        form_namespace = SimpleNamespace(**form_values)

        return render_template(
            "index.html",
            form_values=form_namespace,
            errors=errors,
            result=result,
            model_types=_model_type_options(),
            data_sources=_data_source_options(),
        )

    return app


def _default_form_values() -> Dict[str, str]:
    """Return default values for the dashboard form controls."""

    return {
        "model_type": "deterministic",
        "data_source": "synthetic",
        "case_id": "2",
        "num_paths": "12",
        "num_periods": "12",
        "num_prices": "6",
        "uncertainty_dim": "3",
        "price_min": "100",
        "price_max": "500",
        "demand_sensitivity": "0.5",
        "base_mean_demand": "80",
        "base_capacity": "5000",
        "uncertainty_std_ratio": "0.2",
        "seed": "42",
        "use_lp_relaxation": "on",
    }


def _parse_request(form_values: Dict[str, str]) -> Tuple[Optional[ModelRunRequest], List[str]]:
    """Translate form data into a :class:`ModelRunRequest`."""

    errors: List[str] = []

    def parse_int(name: str, minimum: Optional[int] = None) -> int:
        raw = (form_values.get(name) or "").strip()
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"字段“{name}”需要整数。")
            return 0
        if minimum is not None and value < minimum:
            errors.append(f"字段“{name}”必须不小于 {minimum}。")
        return value

    def parse_float(name: str, minimum: Optional[float] = None) -> float:
        raw = (form_values.get(name) or "").strip()
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"字段“{name}”需要数字。")
            return 0.0
        if minimum is not None and value < minimum:
            errors.append(f"字段“{name}”必须不小于 {minimum}。")
        return value

    model_type = form_values.get("model_type", "deterministic")
    data_source = form_values.get("data_source", "synthetic")
    case_id = form_values.get("case_id") or None

    num_paths = parse_int("num_paths", minimum=1)
    num_periods = parse_int("num_periods", minimum=1)
    num_prices = parse_int("num_prices", minimum=1)
    uncertainty_dim = parse_int("uncertainty_dim", minimum=1)

    price_min = parse_float("price_min", minimum=0)
    price_max = parse_float("price_max", minimum=price_min)

    demand_sensitivity = parse_float("demand_sensitivity", minimum=0)
    base_mean_demand = parse_float("base_mean_demand", minimum=0)
    base_capacity = parse_int("base_capacity", minimum=1)
    uncertainty_std_ratio = parse_float("uncertainty_std_ratio", minimum=0)

    seed_raw = (form_values.get("seed") or "").strip()
    seed = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError:
            errors.append("字段“seed”需要整数。")

    use_lp_relaxation = form_values.get("use_lp_relaxation") == "on"

    if price_max < price_min:
        errors.append("价格上限必须大于或等于价格下限。")

    if errors:
        return None, errors

    request = ModelRunRequest(
        model_type=model_type,
        data_source=data_source,
        case_id=case_id,
        num_paths=num_paths,
        num_periods=num_periods,
        num_prices=num_prices,
        uncertainty_dim=uncertainty_dim,
        price_range=(price_min, price_max),
        demand_sensitivity=demand_sensitivity,
        base_mean_demand=base_mean_demand,
        base_capacity=base_capacity,
        uncertainty_std_ratio=uncertainty_std_ratio,
        seed=seed,
        use_lp_relaxation=use_lp_relaxation,
    )
    return request, errors


def _model_type_options() -> List[Tuple[str, str]]:
    """Return model type select box labels."""

    return [
        ("deterministic", "确定性模型"),
        ("dro_gurobi", "DRO (Gurobi)"),
        ("dro_mosek", "DRO (Mosek)"),
    ]


def _data_source_options() -> List[Tuple[str, str]]:
    """Return data source select box labels."""

    return [
        ("synthetic", "随机生成数据"),
        ("case", "案例数据"),
    ]


if __name__ == "__main__":  # pragma: no cover - manual launch
    create_app().run(host="0.0.0.0", port=5000, debug=True)
