import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="data ingestion")

with app.setup:
    import pandas as pd

    from config import load, path

    COLUMNS = [
        "simulated_flight_id",
        "flight_path",
        "tag_type",
        "pose_angle",
        "pose_orientation",
        "column",
        "antenna_port",
        "timestamp",
        "cycle_no",
        "rssi_dbm",
    ]


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Data ingestion

    Loads the raw synth reads and maps each antenna port to its zone (`outside`, `middle`,
    `inside`). Writes `data/interim/reads.parquet`, one row per read.
    """)
    return


@app.function
def ingest(cfg):
    reads = pd.read_parquet(path(cfg, "raw"), columns=COLUMNS)
    reads["zone"] = reads["antenna_port"].map(cfg["zone_of_port"])
    return reads


@app.cell
def _():
    cfg = load("configs/synth.yaml")
    reads = ingest(cfg)
    _out = path(cfg, "reads")
    _out.parent.mkdir(parents=True, exist_ok=True)
    reads.to_parquet(_out)
    reads.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
