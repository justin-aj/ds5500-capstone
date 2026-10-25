import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="eda")

with app.setup:
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle

    from config import load, path

    # Categorical slots 1-3 of the validated palette (blue, orange, aqua), one per zone.
    # Aqua is below 3:1 on the light surface, so every zone is also labelled directly and
    # has its own marker shape.
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    INK_2 = "#52514e"
    GRID = "#e4e3df"
    ZONE_COLOR = {"outside": "#2a78d6", "middle": "#eb6834", "inside": "#1baf7a"}
    ZONE_MARKER = {"outside": "o", "middle": "s", "inside": "^"}

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "font.size": 10,
        }
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Synth setup and example flights

    **Setup.** The antennas are fixed on the tunnel; the tags move along the belt. Position
    is in cm from the tunnel `center`, negative on the entrance side. Ports 13 and 15 sit at
    the tunnel entrance (-40 cm), ports 9-12 at `center` (0 cm), ports 14 and 16 at the
    tunnel exit (+40 cm). The three zones are inputs describing where each antenna sits,
    not separate gates; the tunnel is one read zone around one boundary.

    **Flights.** Each simulated flight is synthetic data built from static reads, so its
    RSSI shows what the model sees, not the real motion of a tag.
    """)
    return (mo,)


@app.cell
def _(mo):
    args = mo.cli_args()
    cfg = load(args.get("config", "configs/synth.yaml"))
    return (cfg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Antenna layout along the belt
    """)
    return


@app.cell
def _(cfg):
    _ports_at = {}
    for _port, _cm in cfg["position_cm"].items():
        _ports_at.setdefault(_cm, []).append(_port)

    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.add_patch(Rectangle((-120, -1), 240, 2, facecolor="none", edgecolor=INK_2, linewidth=1.2))
    _ax.add_patch(
        Rectangle((-40, -1.4), 80, 2.8, facecolor="none", edgecolor=INK_2, linestyle=":", linewidth=1.2)
    )
    _ax.text(-38, -1.9, "tunnel", color=INK_2, fontsize=9)
    _ax.axvline(0, color=INK_2, linewidth=0.8, linestyle="--")
    _ax.text(2, -1.9, "center", color=INK_2, fontsize=9)

    for _cm, _ports in sorted(_ports_at.items()):
        _zone = cfg["zone_of_port"][_ports[0]]
        _ax.plot(
            [_cm], [1.7], ZONE_MARKER[_zone], color=ZONE_COLOR[_zone],
            markersize=11, markeredgecolor=SURFACE, markeredgewidth=2,
        )
        _ax.plot([_cm, _cm], [1.7, 0], color=ZONE_COLOR[_zone], linewidth=1.5)
        _ax.text(
            _cm, 2.25 if _cm == 0 else 2.15, f"{_zone}\nports {', '.join(map(str, sorted(_ports)))}",
            color=ZONE_COLOR[_zone], ha="center", va="bottom", fontsize=9,
        )
    _ax.text(-118, -0.2, "tag path along the belt  →", color=INK_2, fontsize=9)

    _ax.set_xlim(-125, 125)
    _ax.set_ylim(-2.3, 3.8)
    _ax.set_yticks([])
    _ax.set_xticks([-40, 0, 40])
    _ax.set_xlabel("position along the belt (cm from center)")
    _ax.grid(False)
    _ax.spines["left"].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read it.** Left to right is the position along the belt, in cm from the tunnel
    `center`. Each marker is one antenna group, drawn where it is mounted: `outside` (ports 13
    and 15) at the tunnel entrance, `middle` (ports 9-12) at `center`, `inside` (ports 14 and 16)
    at the tunnel exit. The dotted box is the tunnel. The tags travel along the belt through it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Ports | Zone | Where the antennas sit |
    |---|---|---|
    | 13, 15 | `outside` | in front of the tunnel entrance, -40 cm |
    | 9-12 | `middle` | tunnel walls at center, 0 cm (9 bottom, 10 top, 11 left, 12 right) |
    | 14, 16 | `inside` | behind the tunnel exit, +40 cm |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example flights

    One flight per path type, the one whose read count is closest to the median of its
    type, among the flights that match the filters below. A path type with no matching
    flight has no panel. RSSI per read, coloured by zone, against time since the flight's first read. The
    `in` label is a straight pass; turn-backs and static strays are `none`. A static stray is
    not a flight: the tag never moves, and each of the 16,824 stray windows is a capture of about
    300 s. Real flights (paths 0, 1, 2) last at most about 22 s.
    """)
    return


@app.cell
def _(cfg):
    # Reads from the data ingestion stage, with each antenna port mapped to its zone.
    reads = pd.read_parquet(path(cfg, "reads"))
    return (reads,)


@app.cell
def _(reads):
    reads.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Straight passes (`flight_path` 0) that never read `inside`, by how the tag sits. The tag
    still crossed; the exit antennas just did not hear it.
    """)
    return


@app.cell
def _(reads):
    _passes = (
        reads[reads["flight_path"] == 0]
        .groupby("simulated_flight_id")
        .agg(
            tag_type=("tag_type", "first"),
            pose_orientation=("pose_orientation", "first"),
            pose_angle=("pose_angle", "first"),
            lane=("column", "first"),
            inside_read=("zone", lambda _z: (_z == "inside").any()),
        )
    )
    pd.concat(
        [
            _passes.groupby(_c)
            .agg(passes=("inside_read", "size"), no_inside_share=("inside_read", lambda _s: 1 - _s.mean()))
            .round({"no_inside_share": 2})
            .rename_axis("value")
            .reset_index()
            .astype({"value": str})
            .assign(condition=_c)
            for _c in ["tag_type", "pose_orientation", "pose_angle", "lane"]
        ]
    )[["condition", "value", "passes", "no_inside_share"]]
    return


@app.cell
def _(mo, reads):
    flight_id = mo.ui.number(
        start=int(reads["simulated_flight_id"].min()),
        stop=int(reads["simulated_flight_id"].max()),
        value=69,
        label="flight id",
    )
    flight_path = mo.ui.dropdown(
        ["all"] + [str(_p) for _p in sorted(reads["flight_path"].unique())],
        value="all",
        label="flight path",
    )
    mo.hstack([flight_id, flight_path], justify="start")
    return flight_id, flight_path


@app.cell
def _(flight_id, flight_path, reads):
    _r = reads[reads["simulated_flight_id"] == flight_id.value]
    if flight_path.value != "all":
        _r = _r[_r["flight_path"] == int(flight_path.value)]
    _r.sort_values("timestamp")
    return


@app.cell
def _(reads):
    reads.groupby("flight_path")["simulated_flight_id"].nunique().reset_index()
    return


@app.cell
def _(reads):
    single_flight = reads[(reads["simulated_flight_id"] == 0) & (reads["flight_path"] == 0)].sort_values("timestamp")
    return (single_flight,)


@app.cell
def _(single_flight):
    single_flight["zone"].unique()
    return


@app.cell
def _(cfg, reads):
    _zones_read = reads.groupby("simulated_flight_id").agg(
        flight_path=("flight_path", "first"),
        zones_read=("zone", lambda _z: ", ".join(_n for _n in cfg["zones"] if _n in set(_z))),
    )
    _zone_reads = reads.groupby(["simulated_flight_id", "zone"]).size().unstack(fill_value=0)
    for _zone in cfg["zones"]:
        _zones_read[f"{_zone}_reads"] = _zone_reads[_zone]
    _zones_read.groupby(["flight_path", "zones_read"]).agg(
        flights=("flight_path", "size"),
        **{f"{_zone}_reads_mean": (f"{_zone}_reads", "mean") for _zone in cfg["zones"]},
    ).round(1).reset_index()
    return


@app.cell
def _(cfg, reads):
    reads.groupby(["flight_path", "zone"])["simulated_flight_id"].nunique().unstack()[cfg["zones"]].reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where a flight starts and ends

    Two made-up flights show why the cells below vote over a window of reads and count each
    zone per antenna. `o` = outside, `m` = middle, `i` = inside, reads in time order.

    **1. One stray read can mislead.** A turn-back from outside, 30 reads:

    `o o o o m o m m m o m m o m m o o m o o m o o o m o o o o i`

    The tag never went inside, but its very last read leaked from an inside antenna. The last
    read alone says the flight ended `inside`, which looks like a crossing. A vote over the
    last 10% of reads (`o o i`) says `outside`, which is right.

    **2. Middle has more antennas.** The last 10% of a straight pass: `m m i m i`.

    - Plain vote: middle 3, inside 2, so it ends in `middle`.
    - Per antenna: middle 3 / 4 antennas = 0.75, inside 2 / 2 antennas = 1.0, so it ends in
      `inside`, which is right: the tag left through the exit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Where each flight starts and ends: the zone heard strongest in its first cycle and in its last
    cycle. Cycles, not single reads, because the reader checks ports in a fixed order within a cycle.
    """)
    return


@app.cell
def _(reads):
    # By cycle, not timestamp: within a cycle the reader sweeps ports 9-16 in order, so read order
    # inside a cycle says nothing about where the tag is. A cycle's zone is the one heard strongest.
    _r = reads.assign(cycle=reads["cycle_no"] - reads.groupby("simulated_flight_id")["cycle_no"].transform("min"))
    _strongest = _r.loc[_r.groupby(["simulated_flight_id", "cycle"])["rssi_dbm"].idxmax()].sort_values(
        ["simulated_flight_id", "cycle"]
    )
    (
        _strongest.groupby("simulated_flight_id")
        .agg(flight_path=("flight_path", "first"), first_zone=("zone", "first"), last_zone=("zone", "last"))
        .groupby(["flight_path", "first_zone", "last_zone"])
        .size()
        .rename("flights")
        .reset_index()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same, but one stray read cannot decide it. The start zone is the most common zone in
    the reads of the flight's first 25% of cycles, the end zone the most common in its last 25%
    (at least one cycle each). On a tie, the zone heard in the cycle nearer that end wins.
    """)
    return


@app.cell
def _(reads):
    # Position is the cycle's rank among the cycles that heard the tag, so reads in one cycle share it.
    _r = reads.copy()
    _g = _r.groupby("simulated_flight_id")
    _r["position"] = _g["cycle_no"].rank(method="dense").astype(int) - 1
    _n = _g["cycle_no"].transform("nunique")
    _k = (_n * 0.25).round().clip(lower=1)

    def _majority(_window, _edge):
        _counts = (
            _window.groupby(["simulated_flight_id", "zone"])["position"]
            .agg(reads="size", nearest=_edge)
            .reset_index()
            .sort_values(["simulated_flight_id", "reads", "nearest"], ascending=[True, False, _edge == "min"])
        )
        return _counts.groupby("simulated_flight_id")["zone"].first()

    _ends = _g.agg(flight_path=("flight_path", "first"))
    _ends["start_zone"] = _majority(_r[_r["position"] < _k], "min")
    _ends["end_zone"] = _majority(_r[_r["position"] >= _n - _k], "max")
    _ends.groupby(["flight_path", "start_zone", "end_zone"]).size().rename("flights").reset_index()
    return


@app.cell
def _(reads):
    # Same as above with a 10% window: first 10% and last 10% of each flight's cycles.
    # Position is the cycle's rank among the cycles that heard the tag, so reads in one cycle share it.
    _r = reads.copy()
    _g = _r.groupby("simulated_flight_id")
    _r["position"] = _g["cycle_no"].rank(method="dense").astype(int) - 1
    _n = _g["cycle_no"].transform("nunique")
    _k = (_n * 0.10).round().clip(lower=1)

    def _majority(_window, _edge):
        _counts = (
            _window.groupby(["simulated_flight_id", "zone"])["position"]
            .agg(reads="size", nearest=_edge)
            .reset_index()
            .sort_values(["simulated_flight_id", "reads", "nearest"], ascending=[True, False, _edge == "min"])
        )
        return _counts.groupby("simulated_flight_id")["zone"].first()

    _ends = _g.agg(flight_path=("flight_path", "first"))
    _ends["start_zone"] = _majority(_r[_r["position"] < _k], "min")
    _ends["end_zone"] = _majority(_r[_r["position"] >= _n - _k], "max")
    _ends.groupby(["flight_path", "start_zone", "end_zone"]).size().rename("flights").reset_index()
    return


@app.cell
def _(cfg, reads):
    # 10% window, but each zone's reads are divided by its number of antennas (middle has 4,
    # outside and inside 2 each), so middle does not win just because it has more antennas.
    _antennas = pd.Series(cfg["zone_of_port"]).value_counts()
    # Position is the cycle's rank among the cycles that heard the tag, so reads in one cycle share it.
    _r = reads.copy()
    _g = _r.groupby("simulated_flight_id")
    _r["position"] = _g["cycle_no"].rank(method="dense").astype(int) - 1
    _n = _g["cycle_no"].transform("nunique")
    _k = (_n * 0.10).round().clip(lower=1)

    def _majority(_window, _edge):
        _counts = (
            _window.groupby(["simulated_flight_id", "zone"])["position"]
            .agg(reads="size", nearest=_edge)
            .reset_index()
        )
        _counts["vote"] = _counts["reads"] / _counts["zone"].map(_antennas)
        _counts = _counts.sort_values(
            ["simulated_flight_id", "vote", "nearest"], ascending=[True, False, _edge == "min"]
        )
        return _counts.groupby("simulated_flight_id")["zone"].first()

    _ends = _g.agg(flight_path=("flight_path", "first"))
    _ends["start_zone"] = _majority(_r[_r["position"] < _k], "min")
    _ends["end_zone"] = _majority(_r[_r["position"] >= _n - _k], "max")
    _ends.groupby(["flight_path", "start_zone", "end_zone"]).size().rename("flights").reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Movement inside one flight

    One row per cycle of the flight picked in the box below, counted from the
    flight's first cycle. Each cell is the strongest RSSI (dBm) that zone heard in that cycle;
    empty means the zone did not hear the tag. Movement shows as the zone that hears the tag
    changing over the cycles, and as RSSI rising when the tag gets closer and falling as it leaves.
    """)
    return


@app.cell
def _(mo, reads):
    cycle_flight_id = mo.ui.number(
        start=int(reads["simulated_flight_id"].min()),
        stop=int(reads["simulated_flight_id"].max()),
        value=70,
        label="flight id",
    )
    cycle_flight_path = mo.ui.dropdown(
        ["all"] + [str(_p) for _p in sorted(reads["flight_path"].unique())],
        value="all",
        label="flight path",
    )
    mo.hstack([cycle_flight_id, cycle_flight_path], justify="start")
    return cycle_flight_id, cycle_flight_path


@app.cell
def _(cfg, cycle_flight_id, cycle_flight_path, reads):
    _f = reads[reads["simulated_flight_id"] == cycle_flight_id.value]
    if cycle_flight_path.value != "all":
        _f = _f[_f["flight_path"] == int(cycle_flight_path.value)]
    _f.assign(cycle=_f["cycle_no"] - _f["cycle_no"].min()).pivot_table(
        index="cycle", columns="zone", values="rssi_dbm", aggfunc="max"
    ).reindex(columns=cfg["zones"]).reset_index()
    return


@app.cell
def _(cfg, cycle_flight_id, cycle_flight_path, reads):
    # Same flight, but each cell is the number of reads that zone had in that cycle.
    _f = reads[reads["simulated_flight_id"] == cycle_flight_id.value]
    if cycle_flight_path.value != "all":
        _f = _f[_f["flight_path"] == int(cycle_flight_path.value)]
    _f.assign(cycle=_f["cycle_no"] - _f["cycle_no"].min()).pivot_table(
        index="cycle", columns="zone", values="rssi_dbm", aggfunc="size", fill_value=0
    ).reindex(columns=cfg["zones"], fill_value=0).reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Do all zones hear a pass?

    No. Some passes miss a zone, in real data too. Share of passes by zones read:

    | zones read | synth path 0 | real 008 |
    |---|---|---|
    | outside, middle, inside | 61.9% | 69% |
    | outside, middle | 22.8% | 12% |
    | middle only | 11.5% | 10% |
    | middle, inside | 3.7% | 7% |

    A pass that misses a zone still crossed, so the label comes from `flight_path`, not from
    the zones read. Open question: in the synth, does a chosen window of cycles end before the
    tag reaches the exit antennas?
    """)
    return


@app.cell
def _(mo, reads):
    zones_tag_type = mo.ui.dropdown(
        ["all"] + sorted(reads["tag_type"].unique()),
        value="all",
        label="tag type",
    )
    zones_tag_type
    return (zones_tag_type,)


@app.cell
def _(cfg, reads, zones_tag_type):
    # Synth path 0 share of passes by zones read, for the chosen tag type. Real 008 has no tag type.
    _p = reads[reads["flight_path"] == 0]
    if zones_tag_type.value != "all":
        _p = _p[_p["tag_type"] == zones_tag_type.value]
    _zones = _p.groupby("simulated_flight_id")["zone"].agg(
        lambda _z: ", ".join(_n for _n in cfg["zones"] if _n in set(_z))
    )
    _zones.value_counts().rename("passes").to_frame().assign(
        share_pct=lambda _t: (_t["passes"] / _t["passes"].sum() * 100).round(1)
    ).rename_axis("zones read").reset_index()
    return


if __name__ == "__main__":
    app.run()
