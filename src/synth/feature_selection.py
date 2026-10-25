import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="feature selection")

with app.setup:
    import pandas as pd

    from config import load, path


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Feature selection

    Why each feature built by the feature engineering stage is kept, and why the other candidates
    were skipped. Features are one row per flight in `features.parquet`; the model learns from the
    labelled flights (`in` or `none`).
    """)
    return (mo,)


@app.cell
def _(mo):
    args = mo.cli_args()
    cfg = load(args.get("config", "configs/synth.yaml"))
    return (cfg,)


@app.cell
def _(cfg):
    # Labelled flights with features from the feature engineering stage, and reads from data
    # ingestion for the checks below.
    labelled = pd.read_parquet(path(cfg, "features"))
    reads = pd.read_parquet(path(cfg, "reads"))
    return labelled, reads


@app.cell
def _(labelled):
    # Flights per label and tag strength (dim 0, medium 1, bright 2).
    labelled.groupby(["transition", "tag_strength"]).size().unstack().reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What `tag_strength` teaches the model

    Each strength has both `in` and `none`, so `tag_strength` alone predicts nothing. It changes
    what the reads mean. The same zone pattern points opposite ways:

    | zones read | bright | dim |
    |---|---|---|
    | outside, middle | 562 `in` vs 3,992 turn-backs: mostly `none` | 2,944 `in` vs 178 turn-backs: mostly `in` |
    | middle, inside | 31 `in` vs 3,286 turn-backs: mostly `none` | 606 `in` vs 440 turn-backs: mixed |

    - **Bright:** reaches far, so a missing end zone is suspicious and points to a turn-back.
    - **Dim:** fades early, so a missing end zone is normal and still a crossing.

    The model learns strength together with the pattern, so it needs to model interactions: a
    tree model or a neural net, or logistic regression with interaction terms. RSSI already hints
    at strength (dim reads are about 7 dB lower); `tag_strength` makes it explicit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Labelled flights with read features

    One row per flight the model can learn from (`in` or `none`). Read counts per zone show how
    clearly each zone heard the tag; the 0/1 `read_` flags mark whether it heard it at all, a jump
    a linear model cannot get from counts. `tag_strength` tells the model how to read a missing zone.
    """)
    return


@app.cell
def _(labelled):
    labelled[
        ["transition", "tag_strength", "reads_outside", "reads_middle", "reads_inside",
         "read_outside", "read_middle", "read_inside", "reads_total"]
    ].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cycle features

    Counts and flags carry no order, but order is what separates a crossing from a turn-back. The
    cycle of each zone's first read, last read and strongest RSSI gives that order. Cycles count
    from the flight's first cycle, because `cycle_no` does not restart per flight.

    Watch `n_cycles`: strays are captures of about 300 s (up to 1,472 cycles) against a median
    flight of 42, so it alone can single out strays.
    """)
    return


@app.cell
def _(labelled):
    labelled[["transition", "n_cycles"] + [c for c in labelled if c.endswith(("_outside", "_middle", "_inside")) and "cycle" in c]].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pose angle

    Like `tag_strength`, `pose_angle` changes how well the zones hear the tag: 70.1% of straight
    passes at 90° read all three zones, against 54.2% at 0°. It is balanced across all paths, so
    it is no shortcut, and it is ordered, so it stays a number (0, 45, 90). No records are removed
    for it.

    `pose_orientation` is left out: the synth has no Bottom turn-backs, so a model would learn
    "Bottom means no turn-back" from the synth design, not from the reads.
    """)
    return


@app.cell
def _(labelled):
    labelled[["transition", "pose_angle"]].head()
    return


@app.cell
def _(mo, reads):
    # Flight duration in seconds against `n_cycles`. They move together, so duration is skipped.
    _g = reads.groupby("simulated_flight_id")
    _duration_s = (_g["timestamp"].max() - _g["timestamp"].min()) / 1e6
    _n_cycles = _g["cycle_no"].max() - _g["cycle_no"].min() + 1
    mo.md(
        f"Correlation of flight duration (s) with `n_cycles`: **{_duration_s.corr(_n_cycles):.3f}**. "
        "Duration adds nothing `n_cycles` does not already carry, so it is skipped as a feature."
    )
    return


@app.cell
def _(mo, reads):
    # `column` (belt lane) against the flight path. Lane -1 is only strays, so it leaks the label.
    _f = reads.groupby("simulated_flight_id")[["column", "flight_path"]].first()
    mo.vstack(
        [
            mo.md(
                f"Correlation of `column == -1` with stray (`flight_path == -1`): "
                f"**{(_f['column'] == -1).corr(_f['flight_path'] == -1):.3f}**. Lane -1 gives the stray "
                "label away, and lanes 0-2 are balanced across paths with little effect, so `column` "
                "is skipped as a feature."
            ),
            pd.crosstab(_f["column"], _f["flight_path"]).reset_index(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RSSI features

    18 per-flight RSSI features were considered and each was scored alone by AUC (pass vs
    turn-back), then rescored with a small decision tree to catch two-sided patterns, where passes
    sit in the middle and turn-backs on both sides. Three are kept. `w` = 10^(rssi / 10) is linear power; `pos` is the antenna's belt
    position (-40, 0, +40 cm); `t` is time in the flight from 0 to 1.

    | # | feature | formula | kept | reason |
    |---|---|---|---|---|
    | 1 | `max_all` | max(rssi) | no | mostly says which tag it is (AUC 0.867, about 0.998 within a tag type) |
    | 2 | `max_out` | max(rssi), outside | no | raw level, repeats `tag_strength` |
    | 3 | `max_mid` | max(rssi), middle | no | raw level, already inside #6 |
    | 4 | `max_in` | max(rssi), inside | no | raw level, repeats `tag_strength` |
    | 5 | `max_in_out_gap` | \|`max_in` − `max_out`\| | **yes** | passes are heard equally at both ends (median -1), turn-backs at one end (-14.5, +15.5); one-direction AUC 0.51 hid it, the gap scores 0.877; ends balance, a different mechanism from #6 |
    | 6 | `max_mid_minus_edge` | `max_mid` − max(`max_out`, `max_in`) | **yes** | strongest (AUC 0.990), relative so it holds across tag types |
    | 7 | `mid_power_frac` | Σw (middle) / Σw (all) | no | same signal as #6 (correlation 0.92) |
    | 8 | `range_all` | max(rssi) − min(rssi) | no | same as #9 (0.96), sensitive to single outlier reads |
    | 9 | `p90_p10` | p90(rssi) − p10(rssi) | **yes** | how much the signal swings: a moving tag swings, a turn-back less |
    | 10 | `max_minus_med` | max(rssi) − median(rssi) | no | same as #9 (0.92) |
    | 11 | `pos_first` | Σ(w·pos) / Σw, t ≤ 1/3 | no | two-sided (tree AUC 0.881), same "stays on one side" idea as #5, and relies on the assumed cm positions |
    | 12 | `pos_last` | Σ(w·pos) / Σw, t ≥ 2/3 | no | two-sided (tree AUC 0.873), same idea as #5, relies on the assumed cm positions |
    | 13 | `pos_shift` | `pos_last` − `pos_first` | no | adds nothing on top of the cycle features |
    | 14 | `pos_slope` | weighted slope of pos against t | no | same idea as #13 |
    | 15 | `tcent_in_minus_out` | Σ(w·t)/Σw inside − outside | no | needs both ends heard (34% of flights), cycle features carry the timing |
    | 16 | `diff_corr_t` | corr(cycle, max inside − max outside) | no | same coverage problem, weak |
    | 17 | `t_peak` | t of the max(rssi) read | no | weak (AUC 0.63), `peak_cycle_` already has peak timing |
    | 18 | `peak_zone_mid` | 1 if max(rssi) is in middle | no | yes/no version of #6 (0.87) |

    **Why only a hedge.** A quick tree model (5-fold, pass vs turn-back, 46,934 flights) gets 99.94%
    with counts, cycles, `tag_strength` and `pose_angle` alone, and 99.99% with #6 and #9: errors
    drop from about 28 to about 5. The base is already near perfect because the `in` label comes from
    read order, which the cycle features carry. #6 and #9 measure the RF physics directly, so they
    should matter more on noisier reads. #6 may look better than it is: in the synth, turn-backs
    never reach a wall peak as strong as passes do. #5 was added after that check, as a backup to
    6 that does not depend on the wall peak.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What AUC says

    Each feature is one number per flight. AUC (Area Under the ROC Curve) checks whether that number
    tells straight passes (path 0) apart from turn-backs (paths 1 and 2): pick one random pass and one
    random turn-back, and AUC is the chance the pass has the higher value.

    - **Path 0 values look like turn-back values:** the feature cannot tell them apart. AUC near 0.5,
      so it is not added.
    - **Path 0 values differ from turn-back values:** the feature tells them apart. AUC near 1, so it
      is added, unless a feature already kept does the same job (checked by correlation).

    AUC only sees one direction, "are passes higher". For #5 the passes sit between path 1 and
    path 2, so they differ on both sides and AUC reads 0.51. The absolute value fixes it (0.877).
    """)
    return


@app.cell
def _(labelled):
    labelled[["transition", "max_mid_minus_edge", "p90_p10", "max_in_out_gap"]].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RF phase features

    **From the records.** Phase is an angle on a circle. angle = `rf_phase_raw` / 4096 × 360° (synth
    and 008; 003 `phase_angle` is already in degrees). The step is the change between consecutive
    reads of the same tag by the same antenna, taken the short way round:
    step = |((Δ + 180) mod 360) − 180|, between 0° and 180°. Example, synth flight 70, port 9: raw
    836, 198, 106, 4072 gives 73.5°, 17.4°, 9.3°, 357.9° and steps 56.1°, 8.1°, 11.4°.

    **No AUC check for phase.** The synth phase does not behave like a real moving tag, so scoring
    phase features on it would say nothing about real crossings. A real moving tag in 008 jumps
    about at random between reads (median step 1.546 rad); a simulated synth pass barely changes
    (0.061 rad), like a static one. The table below shows the gap.

    **Real captures vs synth.** Median step and share of steps above 90°:

    | data | tags | median step | steps > 90° |
    |---|---|---|---|
    | 008 | target tags, moving passes | 1.546 rad | 49% |
    | 008 | shelf tags, static | 0.402 rad | 24% |
    | 003 | shelf tags, static | 0.341 rad | 23% |
    | synth | path 0, simulated pass | 0.061 rad | 20% |
    | synth | strays, static | 0.052 rad | 0.4% |

    - Real moving phase is about random between reads (random gives π/2 = 1.571 rad and 50%). The
      tag moves about 11 cm per read of an antenna, a third of the ~33 cm wavelength, so the phase
      wraps and cannot be followed. Phase cannot show direction at this read rate.
    - A simulated pass never physically moved: it is stitched from static captures at spots along
      the belt. Within a capture the phase barely changes (median 0.061 rad, close to strays); where
      the generator switches to the next spot it jumps (20% of steps above 90°, against 0.4% for
      strays). So phase measures the stitching, not the movement. Synth static phase is also cleaner
      than real static phase, likely because real readers hop frequency.

    **Decision: no RF phase features.** Synth phase does not change with movement and does not
    behave like real phase, so a phase feature learned here would not carry over.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-antenna features: skipped

    Every feature is per zone: the 4 wall antennas (9 bottom, 10 top, 11 left, 12 right) count as one
    `middle`, the 2 entrance and the 2 exit antennas as one zone each. Per-antenna features would
    compute the same values for each of the 8 ports, for example `reads_9` to `reads_12` in place of
    `reads_middle`.

    They are skipped. Which wall antenna hears the tag mostly reflects how the tag is posed, which
    `pose_angle` already gives (and `pose_orientation` was left out on purpose). The crossing signal
    comes from zone order and from walls against ends, which the zone features carry. Per-antenna
    features would add 4 columns per feature for little new information.
    """)
    return


@app.cell
def _(labelled, mo):
    mo.vstack(
        [
            mo.md(
                "`n_mid_ports`: a crossing passes all 4 walls (median 4), a turn-back sees fewer (median 1). "
                "AUC 0.955; lifts middle-only hard passes called `in` from 83.6% to 94.9%. Risk: a real "
                "turn-back that reaches the walls would also hear 4."
            ),
            labelled[["transition", "n_mid_ports"]].head(),
        ]
    )
    return


@app.cell
def _(cfg, labelled):
    # The model's 19 features. Left out: `tag_strength` and `pose_angle`, which a real reader does
    # not report; `n_cycles`, which follows the synth capture length and belt speed; and the raw
    # counts and cycles, which the shares and positions replace without their length and speed.
    selected = labelled[
        [
            "transition",
            "read_outside",
            "read_middle",
            "read_inside",
            "share_outside",
            "share_middle",
            "share_inside",
            "first_pos_outside",
            "first_pos_middle",
            "first_pos_inside",
            "last_pos_outside",
            "last_pos_middle",
            "last_pos_inside",
            "peak_pos_outside",
            "peak_pos_middle",
            "peak_pos_inside",
            "max_mid_minus_edge",
            "p90_p10",
            "max_in_out_gap",
            "n_mid_ports",
        ]
    ]
    selected.to_parquet(path(cfg, "selected_features"))
    selected.head()
    return


@app.cell
def _(cfg):
    # Pairs of selected features that move together. A high-AUC feature set can repeat one signal
    # several times, which gives the model nothing new for the hard cases, so near-copies (above
    # about 0.8) are candidates to drop. Spearman, because it ranks values like AUC and ignores scale.
    _corr = pd.read_parquet(path(cfg, "selected_features")).drop(columns="transition").corr(method="spearman").abs()
    _pairs = _corr.stack()
    _pairs = _pairs[_pairs.index.get_level_values(0) < _pairs.index.get_level_values(1)]
    _pairs.sort_values(ascending=False).rename_axis(["feature_a", "feature_b"]).rename("abs_spearman").round(
        2
    ).reset_index()
    return


@app.cell
def _(cfg, mo):
    # Drops one feature from each pair above 0.8 Spearman and keeps the one with the higher AUC
    # (`in` against `none`). It goes from the strongest pair down and skips pairs whose feature is
    # already dropped, since one drop can clear several pairs. AUC comes from ranks (Mann-Whitney):
    # the share of (`in`, `none`) pairs where `in` ranks higher, flipped when under 0.5. A zone that
    # never heard the tag counts as -1, below every heard value.
    _selected = pd.read_parquet(path(cfg, "selected_features"))
    _x = _selected.drop(columns="transition")
    _is_in = _selected["transition"] == "in"
    _n_in, _n_none = _is_in.sum(), (~_is_in).sum()
    _auc = (_x.fillna(-1).rank()[_is_in].sum() - _n_in * (_n_in + 1) / 2) / (_n_in * _n_none)
    _auc = _auc.where(_auc >= 0.5, 1 - _auc)

    _pairs = _x.corr(method="spearman").abs().stack()
    _pairs = _pairs[_pairs.index.get_level_values(0) < _pairs.index.get_level_values(1)]
    _pairs = _pairs[_pairs > 0.8].sort_values(ascending=False)

    _dropped = []
    _rows = []
    for (_a, _b), _r in _pairs.items():
        if _a in _dropped or _b in _dropped:
            continue
        _drop = _a if _auc[_a] < _auc[_b] else _b
        _dropped.append(_drop)
        _rows.append({"feature_a": _a, "feature_b": _b, "abs_spearman": round(_r, 2),
                      "auc_a": round(_auc[_a], 3), "auc_b": round(_auc[_b], 3), "dropped": _drop})

    mo.vstack(
        [
            pd.DataFrame(_rows),
            mo.md(f"Kept {len(_x.columns) - len(_dropped)} of {len(_x.columns)}: "
                  + ", ".join(f"`{c}`" for c in _x.columns if c not in _dropped)),
        ]
    )
    return


@app.cell
def _(cfg):
    # The 13 features left after the correlation drop above, with the label, saved for training.
    # `selected_features` keeps all 19, so the two sets can be compared once a model exists.
    _model = pd.read_parquet(path(cfg, "selected_features"))[
        [
            "transition",
            "read_outside",
            "first_pos_outside",
            "first_pos_middle",
            "first_pos_inside",
            "last_pos_outside",
            "last_pos_middle",
            "last_pos_inside",
            "peak_pos_outside",
            "peak_pos_middle",
            "peak_pos_inside",
            "max_mid_minus_edge",
            "max_in_out_gap",
            "n_mid_ports",
        ]
    ]
    _model.to_parquet(path(cfg, "model_features"))
    _model.head()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
