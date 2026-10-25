import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="synth config")

with app.setup:
    from pathlib import Path

    import yaml

    ROOT = Path(__file__).resolve().parents[2]


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Config helpers

    Shared by every stage notebook: `load` reads `configs/synth.yaml`, `path` resolves a
    path from its `paths` block against the project root.
    """)
    return (mo,)


@app.function
def load(config_path):
    return yaml.safe_load((ROOT / config_path).read_text())


@app.function
def path(cfg, key):
    return ROOT / cfg["paths"][key]


@app.cell
def _(mo):
    mo.ui.code_editor(
        (ROOT / "configs/synth.yaml").read_text(), language="yaml", disabled=True
    )
    return


if __name__ == "__main__":
    app.run()
