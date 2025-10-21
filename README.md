# DRO4SlotAndPricing

Distributionally Robust Optimization (DRO) toolkit for slot allocation and freight pricing in liner shipping networks. The code base combines data ingestion utilities, network entities, and optimization models (deterministic and DRO variants) to evaluate routing, capacity assignment, and price decisions under demand uncertainty.

> **Last updated / 最后更新：** 2025-10-21

## Features

- **Comprehensive data pipeline** – `src/utils/read_data.py` ingests ports, routes, vessel paths, container paths, and demand ranges either from the flat files under `data/` or from a relational database with consistent column naming.
- **Rich domain model** – Entity objects in `src/entity/` represent ports, vessels, OD ranges, laden/empty paths, and requests, enabling transparent manipulation of maritime logistics data.
- **Network abstraction** – The `src/network/` package defines time-space nodes and arcs that are shared by vessel planning and container routing models.
- **Optimization models** –
  - Deterministic baseline (`src/models/dm`) with optional LP relaxation.
  - Distributionally robust models (`src/models/dro`) including linear decision rule (LDR) formulations solved via second-order cone programming (SOCP) with Gurobi or Mosek back ends.
  - Shared model builder utilities in `src/models/model_builder.py` to configure objectives, constraints, and solver parameters.
- **Scenario tools** – Utilities in `src/utils/model_params.py` construct model inputs from data or generate synthetic test cases for validation.
- **Example experiment** – `src/test/test_LDR_SOCP.py` demonstrates the full pipeline from data loading through solving deterministic and DRO models, including feasibility checks.

## Repository Layout

```
DRO4SlotAndPricing/
├── data/                # Sample input instances and schema documentation
├── src/
│   ├── config/          # Default configuration values and file/table mappings
│   ├── entity/          # Domain classes (ports, vessels, requests, paths, …)
│   ├── models/          # Deterministic & DRO formulations plus model builder
│   ├── network/         # Time-space network primitives (nodes, arcs)
│   ├── test/            # End-to-end example runner for LDR SOCP models
│   └── utils/           # Data ingestion, parameter assembly, helper classes
└── README.md
```

Consult `data/README.md` for a detailed description of every input file and the expected column names.

## Requirements

- Python 3.9+
- [Gurobi Optimizer](https://www.gurobi.com/) with the `gurobipy` Python package and a valid license (required for the default solver).
- Optional: [Mosek](https://www.mosek.com/) if you want to run the SOCP solver variant in `SOCP4LDR_Mosek`.
- Python packages listed below (installable via `pip`):
  - `numpy`
  - `pandas`
  - `gurobipy`
  - `mosek` (optional)
  - `scipy` (for linear algebra utilities used in some models)

> **Note**: Some commercial solvers require manual installation. Ensure the solver binaries and licenses are available before running optimization models.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas gurobipy mosek scipy
```

If you do not plan to use Mosek, omit `mosek` from the installation command.

## Preparing Input Data

1. Review `data/README.md` for schema requirements.
2. Place the raw text/CSV/Excel files for ports, routes, nodes, arcs, vessel paths, container paths, and demands inside `data/<case>/` folders as expected by `Config.DATA_PATH` and `Config.CASE_PATH`.
3. Adjust settings in `src/config/config.py` if your directory structure or filenames differ. You can also enable database reading by supplying a SQLite file and setting `use_db=True` when constructing a `DataReader`.
4. After loading data via `DataReader`, call `DataManager.generate_demand_and_price()` to create stochastic demand and pricing attributes for each request when needed.

## Running the Example Workflow

The repository includes an end-to-end example that builds both deterministic and DRO models:

```bash
python -m src.test.test_LDR_SOCP
```

The script will attempt to load the default dataset defined in the configuration. If the data is unavailable, it automatically creates a synthetic, feasible test instance. The example then:

1. Solves an LP-relaxed deterministic model (`DeterministicModel`).
2. Solves the DRO LDR model (`SOCP4LDR_Mosek` by default, switch to `SOCP4LDR` for Gurobi-based SOCP).
3. Writes solution artifacts and validates feasibility with `run_all_validations`.

Enable detailed logging and debug timing by toggling flags in `src/config/config.py` (e.g., `Config.debug_mode`, `Config.WHETHER_PRINT_DATA_STATUS`).

## Extending the Project

- Implement new solver strategies by subclassing `ModelBuilder` and following the structure in `src/models/dro`.
- Add alternative demand scenarios by extending `DataManager.sample_scenes` and updating `construct_model_params`.
- Integrate additional data sources by expanding the column mappings in `src/config/data_config.py`.

## Support & Citation

If you use this repository in academic work, please cite the associated paper on slot allocation and pricing with distributionally robust optimization. For questions or contributions, open an issue or submit a pull request.
