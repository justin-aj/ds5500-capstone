import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="data labeling")

with app.setup:
    import pandas as pd

    from config import load, path


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Data labeling

    One row per flight with its label. Writes `data/interim/flights.parquet`.

    - **`in`:** a straight pass (path 0) where the tag moves forward. It is heard at the walls and
      at one end or both. The entrance never hears it in a later cycle than the walls, and the exit
      never in an earlier cycle. Reads in the same cycle don't count as before or after: the reader
      checks the walls (ports 9-12) before the entrance and exit in every cycle.
    - **`none`:** turn-backs (paths 1, 2) and strays (path -1).
    - **No label:** straight passes that don't show the forward move. They get a `removed` reason.
    """)
    return (mo,)


@app.function
def label(reads, zones):
    flights = reads.groupby("simulated_flight_id").agg(
        flight_path=("flight_path", "first"),
        tag_type=("tag_type", "first"),
        zones_read=("zone", lambda z: ", ".join(n for n in zones if n in set(z))),
    )
    # Compare cycles, not timestamps: the reader sweeps ports 9-16 in order within a cycle, so a
    # same-cycle entrance read always lands after a wall read without the tag moving back.
    span = reads.groupby(["simulated_flight_id", "zone"])["cycle_no"].agg(["min", "max"]).unstack()
    looks_back = (span[("max", "outside")] > span[("max", "middle")]) | (
        span[("min", "inside")] < span[("min", "middle")]
    )
    passes = flights["flight_path"] == 0
    middle_only = flights["zones_read"] == "middle"
    flights.loc[passes & middle_only, "removed"] = "middle only"
    flights.loc[passes & ~middle_only & looks_back, "removed"] = "looks backward"
    flights.loc[passes & flights["removed"].isna(), "transition"] = "in"
    flights.loc[flights["flight_path"] != 0, "transition"] = "none"
    return flights


@app.cell
def _():
    cfg = load("configs/synth.yaml")
    flights = label(pd.read_parquet(path(cfg, "reads")), cfg["zones"])
    flights.to_parquet(path(cfg, "flights"))
    flights["transition"].value_counts(dropna=False).rename("flights").reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What counts as forward

    Reads in time order: `o` outside (entrance), `m` middle (walls), `i` inside (exit).

    | reads | label | why |
    |---|---|---|
    | `o o m m m i i` | `in` | entrance, walls, exit, in that order |
    | `o o m m m` | `in` | the exit missed the tag, but it never went back |
    | `m m i i` | `in` | the entrance missed the tag, but it moved on to the exit |
    | `m m m` | no label (`middle only`) | walls only, no direction |
    | `o om om m` | `in` | entrance and walls in the same cycle is not a step back |
    | `o m m o o` | no label (`looks backward`) | the entrance hears it again in a later cycle |
    | `i m m i` | no label (`looks backward`) | the exit hears it a cycle before the walls |

    Below, one real flight per case. Each group is one cycle and the zones that heard the tag in it.
    """)
    return


@app.cell
def _():
    # One real flight per case, picked among flights with a typical read count (20th-40th percentile).
    _cfg = load("configs/synth.yaml")
    _reads = pd.read_parquet(path(_cfg, "reads"))
    _flights = pd.read_parquet(path(_cfg, "flights"))
    _letter = {"outside": "o", "middle": "m", "inside": "i"}
    _n = _reads.groupby("simulated_flight_id").size()
    _cases = {
        "in, all zones": (_flights["transition"] == "in") & (_flights["zones_read"] == "outside, middle, inside"),
        "in, exit not heard": (_flights["transition"] == "in") & (_flights["zones_read"] == "outside, middle"),
        "no label: middle only": _flights["removed"] == "middle only",
        "no label: looks backward": _flights["removed"] == "looks backward",
        "none: turn-back from outside": _flights["flight_path"] == 1,
    }

    def _cycles(_fid):
        _f = _reads[_reads["simulated_flight_id"] == _fid]
        return " ".join(
            _f.groupby(_f["cycle_no"] - _f["cycle_no"].min())["zone"].agg(
                lambda _z: "".join(_c for _c in "omi" if _c in {_letter[_v] for _v in _z})
            )
        )

    _rows = []
    for _case, _mask in _cases.items():
        _c = _n[_mask]
        _fid = _c[_c.between(_c.quantile(0.2), _c.quantile(0.4))].index[0]
        _rows.append({"case": _case, "flight": _fid, "tag_type": _flights.loc[_fid, "tag_type"], "cycles": _cycles(_fid)})
    pd.DataFrame(_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why a forward rule

    Every straight pass crossed, but `in` should mean the reads show it. Which zones heard the tag
    is not enough: bright turn-backs are heard at all three zones, and dim crossings often miss one.
    So the rule looks at the order, cycle by cycle.

    **Removed** (straight passes only, path 0). They did cross, so they are not `none`; they keep a
    `removed` reason and form a hard test set:

    - `middle` only: the pass was heard only at the walls, so its reads show no direction.
    - looks backward: the pass crossed, but its reads look like a turn-back, because the far
      antennas keep hearing the tag (the entrance in a later cycle, or the exit in an earlier one).
    """)
    return


@app.cell
def _():
    # Straight passes kept as `in` or removed, by tag type.
    _flights = pd.read_parquet(path(load("configs/synth.yaml"), "flights"))
    _flights[_flights["flight_path"] == 0].fillna({"removed": "kept as in"}).groupby(
        ["removed", "tag_type"]
    ).size().unstack().reset_index()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
