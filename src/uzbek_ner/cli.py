"""CLI entrypoint for DVC stages and ad-hoc runs."""

from __future__ import annotations

import typer
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig
from rich.console import Console

from uzbek_ner.pipeline import run_evaluate, run_prepare, run_train
from uzbek_ner.settings import REPO_ROOT

app = typer.Typer(
    name="ner",
    help="Uzbek NER pipeline CLI (prepare / train / evaluate).",
    no_args_is_help=True,
)
console = Console()


def _load_config(config_name: str = "default") -> DictConfig:
    config_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name)


@app.command()
def prepare(
    config_name: str = typer.Option("default", help="Hydra config name under configs/"),
) -> None:
    """Validate raw data layout and write processed manifest."""
    cfg = _load_config(config_name)
    path = run_prepare(cfg)
    console.print(f"[green]✓[/green] prepare → {path}")


@app.command()
def train(
    config_name: str = typer.Option("default", help="Hydra config name under configs/"),
) -> None:
    """Train NER model (stub until implementation lands)."""
    cfg = _load_config(config_name)
    path = run_train(cfg)
    console.print(f"[green]✓[/green] train → {path}")


@app.command()
def evaluate(
    config_name: str = typer.Option("default", help="Hydra config name under configs/"),
) -> None:
    """Evaluate model and write metrics.json for DVC."""
    cfg = _load_config(config_name)
    path = run_evaluate(cfg)
    console.print(f"[green]✓[/green] evaluate → {path}")


@app.command()
def pipeline(
    config_name: str = typer.Option("default", help="Hydra config name under configs/"),
) -> None:
    """Run prepare → train → evaluate sequentially."""
    cfg = _load_config(config_name)
    run_prepare(cfg)
    run_train(cfg)
    run_evaluate(cfg)
    console.print("[green]✓[/green] full pipeline finished")


@app.command("mlflow-ui")
def mlflow_ui(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(5000, help="Bind port"),
) -> None:
    """Print the MLflow UI command (use `make mlflow-ui` to launch)."""
    from uzbek_ner.settings import get_settings

    uri = get_settings().mlflow_tracking_uri
    cmd = f"uv run mlflow ui --backend-store-uri {uri} --host {host} --port {port}"
    console.print(f"Run:\n  [bold cyan]{cmd}[/bold cyan]")


def main() -> None:
    app()


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: console.print(msg, end=""), colorize=True)
    main()
