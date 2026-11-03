"""Text CLI for the Meta-backed agent; defaults to zero live/paid requests."""

from __future__ import annotations

import argparse

from petzyme.agent import run_investigation
from petzyme.agent_reporting import format_research_report
from petzyme.meta import COMPLETION_ALLOWANCES, MAX_COMPLETION_TOKENS, MetaModelClient
from petzyme.runtime_config import RuntimeLimits


def main(argv: list[str] | None = None) -> int:
    """Flow: Receives external CLI arguments list[str] | None; returns exit status int after run_investigation() and privacy-safe format_research_report()."""
    parser = argparse.ArgumentParser(
        description="PETzyme Meta research agent (live calls disabled by default)"
    )
    parser.add_argument("question")
    parser.add_argument("--csv", default="data/nrel/label_data.csv")
    parser.add_argument(
        "--authorize-paid-meta",
        action="store_true",
        help="Use only after explicit authorization for paid Meta inference",
    )
    defaults = RuntimeLimits()
    parser.add_argument("--model-requests", type=int, default=defaults.turn_models)
    parser.add_argument("--tool-attempts", type=int, default=defaults.turn_tools)
    parser.add_argument("--session-model-requests", type=int, default=defaults.session_models)
    parser.add_argument("--session-tool-attempts", type=int, default=defaults.session_tools)
    parser.add_argument("--spending-ceiling-usd", type=float, default=defaults.session_cost_usd)
    parser.add_argument(
        "--completion-tokens",
        type=int,
        choices=(MAX_COMPLETION_TOKENS, *COMPLETION_ALLOWANCES),
        default=10000,
    )
    args = parser.parse_args(argv)
    try:
        limits = RuntimeLimits(
            turn_models=args.model_requests,
            turn_tools=args.tool_attempts,
            session_models=args.session_model_requests,
            session_tools=args.session_tool_attempts,
            session_cost_usd=args.spending_ceiling_usd,
        )
    except ValueError as error:
        parser.error(str(error))
    result = run_investigation(
        args.question,
        MetaModelClient(
            live_authorized=args.authorize_paid_meta,
            completion_tokens=args.completion_tokens,
            max_estimated_cost_usd=limits.session_cost_usd,
        ),
        csv_path=args.csv,
        runtime_limits=limits,
    )
    if result["result"] is None:
        print(result["message"])
        return 1
    print(format_research_report(result["result"]))
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
