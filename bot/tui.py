"""Rich TUI monitor for deepiri-sorge reviews"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from bot.config import Config
from bot.decision_engine import Action, DecisionEngine
from bot.diff_parser import DiffParser

console = Console()

SEVERITY_COLORS = {
    "critical": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "cyan",
}

SEVERITY_ICONS = {
    "critical": "✖",
    "high": "⚠",
    "medium": "◆",
    "low": "◎",
}

SCORE_COLOR = {
    (0, 4): "red",
    (4, 6): "yellow",
    (6, 8): "green",
    (8, 11): "bold green",
}


def score_color(score: float) -> str:
    for (lo, hi), color in SCORE_COLOR.items():
        if lo <= score < hi:
            return color
    return "white"


def render_header(config: Config) -> Panel:
    backends = []
    if config.github_models.enabled:
        backends.append(f"[cyan]GitHub Models[/cyan] ({config.github_models.model})")
    if config.gemini.enabled:
        backends.append(f"[magenta]Gemini[/magenta] ({config.gemini.model})")
    if config.gpu.enabled:
        backends.append("[yellow]GPU[/yellow]")

    routing = (
        f"Small ≤{config.routing.small_pr_threshold:,} tokens → GitHub Models  |  "
        f"Large >{config.routing.large_pr_threshold:,} tokens → Gemini"
    )

    content = Text.assemble(
        ("deepiri-sorge", "bold white"),
        "  •  ",
        ("Backends: ", "dim"),
        ", ".join(backends) if backends else "[red]none configured[/red]",
        "\n",
        ("Routing:  ", "dim"),
        (routing, "dim white"),
    )
    return Panel(content, title="[bold blue]Sorge Monitor[/bold blue]", box=box.DOUBLE_EDGE)


def render_decision_table(decision, diff, estimated_tokens: int) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("key", style="dim", width=18)
    table.add_column("value")

    action_colors = {
        Action.GITHUB_MODELS: "cyan",
        Action.GEMINI: "magenta",
        Action.CPU_REVIEW: "yellow",
        Action.GPU_REVIEW: "green",
        Action.SKIP: "dim",
    }
    color = action_colors.get(decision.action, "white")
    table.add_row("Action", f"[{color}]{decision.action.value}[/{color}]")
    table.add_row("Reason", decision.reason)
    table.add_row("Confidence", f"{decision.confidence * 100:.0f}%")
    table.add_row("Est. tokens", f"{estimated_tokens:,}")
    table.add_row("Files changed", str(diff.files_changed))
    table.add_row("Lines +/-", f"[green]+{diff.lines_added}[/green] [red]-{diff.lines_deleted}[/red]")

    return Panel(table, title="[bold]Routing Decision[/bold]", box=box.ROUNDED)


def render_result_panel(result) -> Panel:
    color = score_color(result.score)
    latency = getattr(result, "latency_ms", None)
    tokens = getattr(result, "tokens_used", None)

    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    meta.add_column("k", style="dim", width=14)
    meta.add_column("v")
    meta.add_row("Model", f"[bold]{result.model}[/bold]")
    meta.add_row("Review type", getattr(result, "review_type", "unknown"))
    meta.add_row("Score", f"[{color}]{result.score:.1f}/10[/{color}]")
    if latency is not None:
        meta.add_row("Latency", f"{latency / 1000:.2f}s")
    if tokens is not None:
        meta.add_row("Tokens used", f"{tokens:,}")

    summary_panel = Panel(
        Text(result.summary, overflow="fold"),
        title="Summary",
        box=box.SIMPLE,
    )

    from rich.columns import Columns
    from rich import print as rprint
    from io import StringIO
    from rich.console import Console as _Console

    buf = _Console(file=StringIO(), width=120)
    buf.print(meta)
    buf.print(summary_panel)

    return Panel(
        "\n".join([
            _render_text(meta),
            "",
            _render_text(summary_panel),
        ]),
        title=f"[bold green]Review Result[/bold green]",
        box=box.ROUNDED,
    )


def _render_text(renderable) -> str:
    from io import StringIO
    from rich.console import Console as _Console
    buf = StringIO()
    c = _Console(file=buf, width=120, highlight=False)
    c.print(renderable)
    return buf.getvalue().rstrip()


def render_issues_table(issues: list) -> Panel:
    if not issues:
        return Panel("[dim]No issues found[/dim]", title="[bold]Issues[/bold]", box=box.ROUNDED)

    table = Table(box=box.SIMPLE_HEAD, show_lines=False, padding=(0, 1))
    table.add_column("Sev", width=8)
    table.add_column("File", style="cyan", max_width=30, overflow="ellipsis")
    table.add_column("Line", width=6, justify="right")
    table.add_column("Message", ratio=3)
    table.add_column("Suggestion", ratio=2, style="dim")

    for issue in issues:
        sev = issue.severity.lower()
        color = SEVERITY_COLORS.get(sev, "white")
        icon = SEVERITY_ICONS.get(sev, "•")
        table.add_row(
            f"[{color}]{icon} {sev}[/{color}]",
            issue.file or "[dim]general[/dim]",
            str(issue.line) if issue.line else "—",
            issue.message,
            issue.suggestion or "",
        )

    return Panel(table, title=f"[bold]Issues ({len(issues)})[/bold]", box=box.ROUNDED)


def render_recommendations(recs: list) -> Panel:
    if not recs:
        return Panel("[dim]None[/dim]", title="[bold]Recommendations[/bold]", box=box.ROUNDED)

    text = Text()
    for i, rec in enumerate(recs, 1):
        text.append(f"  {i}. ", style="dim")
        text.append(rec + "\n")
    return Panel(text, title="[bold]Recommendations[/bold]", box=box.ROUNDED)


def run_tui(diff_content: str, config: Config, dry_run: bool = False) -> None:
    console.print(render_header(config))

    diff_parser = DiffParser()
    parsed_diff = diff_parser.parse(diff_content)

    engine = DecisionEngine(config)
    decision = engine.decide(parsed_diff)
    estimated_tokens = engine._estimate_tokens(parsed_diff)

    console.print(render_decision_table(decision, parsed_diff, estimated_tokens))

    if decision.action == Action.SKIP:
        console.print(Panel(
            f"[yellow]{decision.reason}[/yellow]",
            title="[yellow]Skipped[/yellow]",
            box=box.ROUNDED,
        ))
        return

    # Run the review with a live spinner
    result = None
    start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        action_label = {
            Action.GITHUB_MODELS: f"Calling GitHub Models ({config.github_models.model})",
            Action.GEMINI: f"Calling Gemini ({config.gemini.model})",
            Action.CPU_REVIEW: "Running CPU review",
            Action.GPU_REVIEW: "Running GPU review",
        }.get(decision.action, "Reviewing…")

        task = progress.add_task(action_label, total=None)

        error_messages: list[str] = []

        def _try_github_models() -> None:
            from bot.runners import GitHubModelsRunner
            from loguru import logger as _log
            import sys as _sys
            _log.remove()
            _log.add(lambda m: error_messages.append(m.rstrip()), level="ERROR")
            nonlocal result
            runner = GitHubModelsRunner(api_key=config.github_models.api_key, model=config.github_models.model)
            result = runner.review(parsed_diff)

        def _try_gemini() -> None:
            from bot.runners import GeminiRunner
            from loguru import logger as _log
            _log.remove()
            _log.add(lambda m: error_messages.append(m.rstrip()), level="ERROR")
            nonlocal result
            runner = GeminiRunner(api_key=config.gemini.api_key, model=config.gemini.model)
            result = runner.review(parsed_diff)

        if decision.action == Action.GITHUB_MODELS:
            _try_github_models()
            if result is None and config.gemini.enabled:
                progress.update(task, description=f"GitHub Models failed, falling back to Gemini ({config.gemini.model})")
                _try_gemini()

        elif decision.action == Action.GEMINI:
            _try_gemini()

        elif decision.action == Action.CPU_REVIEW:
            from bot.cpu_reviewer import CPUReviewer
            reviewer = CPUReviewer(config)
            result = reviewer.review(parsed_diff)

        elif decision.action == Action.GPU_REVIEW:
            from bot.gpu_runner import GPURunner
            runner = GPURunner(config)
            result = runner.review(parsed_diff)

        progress.remove_task(task)

    if result is None:
        err_detail = "\n".join(error_messages) if error_messages else "No additional details."
        console.print(Panel(
            f"[red]Review failed — check API keys and connectivity[/red]\n\n[dim]{err_detail}[/dim]",
            box=box.ROUNDED,
        ))
        return

    # Render result sections
    console.print(render_result_panel(result))
    console.print(render_issues_table(result.issues))
    console.print(render_recommendations(result.recommendations))

    console.print(Rule(f"[dim]Completed in {time.time() - start:.1f}s  •  {datetime.now().strftime('%H:%M:%S')}[/dim]"))

    if not dry_run:
        print(json.dumps(result.to_dict(), indent=2), file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="deepiri-sorge TUI monitor")
    parser.add_argument("--diff", required=True, help="Path to diff file or raw diff content")
    parser.add_argument("--config", default="sorge.toml", help="Path to config file")
    parser.add_argument("--mode", choices=["auto", "github", "gemini", "cpu", "gpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Skip JSON stdout output")
    return parser.parse_args()


def main() -> None:
    logger.remove()  # suppress loguru output; rich handles display

    args = parse_args()

    config = Config.from_file(args.config) if Path(args.config).exists() else Config()

    # Apply --mode override to config routing
    if args.mode == "github":
        config.gemini.enabled = False
    elif args.mode == "gemini":
        config.github_models.enabled = False
    elif args.mode == "cpu":
        config.github_models.enabled = False
        config.gemini.enabled = False
    elif args.mode == "gpu":
        config.github_models.enabled = False
        config.gemini.enabled = False
        config.gpu.enabled = True

    diff_arg = args.diff
    path = Path(diff_arg)
    diff_content = path.read_text() if path.exists() and path.is_file() else diff_arg

    run_tui(diff_content, config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
