import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="feature engineering")

with app.setup:
    import pandas as pd

    from config import load, path


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Feature engineering

    Per-flight features for the labelled flights (`in` or `none`): read counts and 0/1 heard flags
    per zone, cycle timing per zone, tag strength, pose angle, three RSSI features and the number of
    wall antennas that heard the tag. Removed straight passes are left out. Writes
    `data/processed/features.parquet`. Why each feature is kept is in `feature_selection.py`.
    """)
    return


@app.function
def build(reads, flights, cfg):
    zones = cfg["zones"]
    low, high = cfg["rssi_quantiles"]

    zone_reads = reads.groupby(["simulated_flight_id", "zone"]).size().unstack(fill_value=0)[zones]

    # Cycles count from the flight's first cycle, since `cycle_no` does not restart per flight.
    r = reads.assign(cycle=reads["cycle_no"] - reads.groupby("simulated_flight_id")["cycle_no"].transform("min"))
    g = r.groupby(["simulated_flight_id", "zone"])
    first = g["cycle"].min().unstack()[zones].add_prefix("first_cycle_")
    last = g["cycle"].max().unstack()[zones].add_prefix("last_cycle_")
    peak = (
        r.loc[g["rssi_dbm"].idxmax()].set_index(["simulated_flight_id", "zone"])["cycle"]
        .unstack()[zones].add_prefix("peak_cycle_")
    )

    zone_max = (
        reads.groupby(["simulated_flight_id", "zone"])["rssi_dbm"].max().unstack()[zones].fillna(cfg["rssi_floor_dbm"])
    )
    rssi = reads.groupby("simulated_flight_id")["rssi_dbm"]
    mid_ports = reads[reads["zone"] == "middle"].groupby("simulated_flight_id")["antenna_port"].nunique()

    return (
        flights.dropna(subset=["transition"])
        .drop(columns="removed")
        .assign(tag_strength=lambda f: f["tag_type"].map(cfg["tag_strength"]))
        .join(zone_reads.add_prefix("reads_"))
        .join((zone_reads > 0).astype(int).add_prefix("read_"))
        .assign(reads_total=zone_reads.sum(axis=1))
        .assign(n_cycles=r.groupby("simulated_flight_id")["cycle"].max() + 1)
        .join(first)
        .join(last)
        .join(peak)
        .join(reads.groupby("simulated_flight_id")["pose_angle"].first())
        .assign(
            max_mid_minus_edge=zone_max["middle"] - zone_max[["outside", "inside"]].max(axis=1),
            p90_p10=rssi.quantile(high) - rssi.quantile(low),
            max_in_out_gap=(zone_max["inside"] - zone_max["outside"]).abs(),
            n_mid_ports=lambda f: mid_ports.reindex(f.index, fill_value=0),
        )
    )


@app.cell
def _():
    cfg = load("configs/synth.yaml")
    features = build(pd.read_parquet(path(cfg, "reads")), pd.read_parquet(path(cfg, "flights")), cfg)
    _out = path(cfg, "features")
    _out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(_out)
    features.head()
    return


@app.function
def normalize(features):
    # Shares and positions keep the flight's shape (which zone heard the tag, and in what order)
    # without its length, speed or tag brightness, which differ between the synth and real readers.
    f = features
    return f.assign(
        share_outside=f["reads_outside"] / f["reads_total"],
        share_middle=f["reads_middle"] / f["reads_total"],
        share_inside=f["reads_inside"] / f["reads_total"],
        first_pos_outside=f["first_cycle_outside"] / f["n_cycles"],
        first_pos_middle=f["first_cycle_middle"] / f["n_cycles"],
        first_pos_inside=f["first_cycle_inside"] / f["n_cycles"],
        last_pos_outside=f["last_cycle_outside"] / f["n_cycles"],
        last_pos_middle=f["last_cycle_middle"] / f["n_cycles"],
        last_pos_inside=f["last_cycle_inside"] / f["n_cycles"],
        peak_pos_outside=f["peak_cycle_outside"] / f["n_cycles"],
        peak_pos_middle=f["peak_cycle_middle"] / f["n_cycles"],
        peak_pos_inside=f["peak_cycle_inside"] / f["n_cycles"],
    )


@app.cell
def _():
    # Adds the normalized columns next to the raw ones in the saved features, so feature selection
    # can pick raw, normalized or both.
    _cfg = load("configs/synth.yaml")
    _out = path(_cfg, "features")
    _features = normalize(pd.read_parquet(_out))
    _features.to_parquet(_out)
    _features.filter(regex="^share_|_pos_").head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
