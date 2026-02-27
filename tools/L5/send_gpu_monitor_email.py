#!/usr/bin/env python3
"""
Send the nightly GPU memory monitor report by email.

- Reads GPU_MONITOR_BUNDLE_DIR from CLI or env.
- Embeds the Statistics table in the email body.
- Attaches report.html (and optionally gpu_metrics.csv).
- Reuses the same SMTP/recipient env vars as send_nightly_perf_email.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import smtplib
import sys
from collections import defaultdict
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)

ENV_SMTP_HOST = "SMTP_HOST"
ENV_SMTP_PORT = "SMTP_PORT"
ENV_SMTP_USERNAME = "SMTP_USERNAME"
ENV_SMTP_PASSWORD = "SMTP_PASSWORD"
ENV_DAILY_EMAIL_LIST = "DAILY_EMAIL_LIST"
ENV_EMAIL_SENDER = "EMAIL_SENDER"
ENV_EMAIL_SUBJECT_PREFIX = "EMAIL_SUBJECT_PREFIX"
ENV_BUILD_URL = "BUILDKITE_BUILD_URL"
ENV_COMMIT = "BUILDKITE_COMMIT"

SMTP_RETRIES = 3
SMTP_RETRY_DELAY_SEC = 5


def load_csv(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["timestamp_epoch"] = int(float(r["timestamp_epoch"]))
                r["gpu_index"] = int(r["gpu_index"])
                r["memory_used_mb"] = int(r["memory_used_mb"])
                r["memory_total_mb"] = int(r["memory_total_mb"])
                r["memory_util_pct"] = int(r["memory_util_pct"])
                rows.append(r)
            except (KeyError, ValueError):
                continue
    return rows


def compute_stats(rows: list[dict]) -> dict:
    by_gpu: Dict[int, List[int]] = defaultdict(list)
    for r in rows:
        by_gpu[r["gpu_index"]].append(r["memory_util_pct"])
    stats: Dict[int, dict] = {}
    for gpu, pcts in by_gpu.items():
        if not pcts:
            continue
        pcts_sorted = sorted(pcts)
        n = len(pcts_sorted)
        stats[gpu] = {
            "min": min(pcts),
            "max": max(pcts),
            "avg": round(sum(pcts) / n, 1),
            "p50": pcts_sorted[n // 2] if n else 0,
            "p95": pcts_sorted[int(n * 0.95)] if n > 1 else pcts_sorted[0],
            "samples": n,
        }
    return stats


def _get_required_env() -> dict[str, str]:
    required = {
        ENV_SMTP_HOST: os.environ.get(ENV_SMTP_HOST),
        ENV_SMTP_PORT: os.environ.get(ENV_SMTP_PORT),
        ENV_SMTP_USERNAME: os.environ.get(ENV_SMTP_USERNAME),
        ENV_SMTP_PASSWORD: os.environ.get(ENV_SMTP_PASSWORD),
        ENV_DAILY_EMAIL_LIST: os.environ.get(ENV_DAILY_EMAIL_LIST),
    }
    missing = [k for k, v in required.items() if not (v and str(v).strip())]
    if missing:
        raise SystemExit(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Set them (e.g. in Buildkite secrets)."
        )
    return {k: str(v).strip() for k, v in required.items()}


def _recipients_list(comma_separated: str) -> list[str]:
    return [a.strip() for a in comma_separated.split(",") if a.strip()]


def _build_subject(prefix: str | None, run_id: str, date_str: str | None) -> str:
    base = f"L5 GPU monitor report {date_str or run_id}"
    if prefix and prefix.strip():
        return f"{prefix.strip()} {base}"
    return base


def _build_body_html(
    stats: dict,
    run_id: str,
    csv_path: str,
    commit_sha: str | None,
    build_url: str | None,
) -> str:
    rows_html: list[str] = []
    for gpu in sorted(stats.keys()):
        s = stats[gpu]
        rows_html.append(
            f"<tr><td>GPU {gpu}</td>"
            f"<td>{s['min']}%</td><td>{s['max']}%</td>"
            f"<td>{s['avg']}%</td><td>{s['p50']}</td><td>{s['p95']}</td><td>{s['samples']}</td></tr>"
        )
    stats_table = "\n".join(rows_html) or "<tr><td colspan='7'>No data</td></tr>"

    commit_str = commit_sha or "N/A"
    build_str = build_url or "N/A"

    return f"""
<p>L5 GPU memory monitor summary (run: <code>{run_id}</code>, file: <code>{os.path.basename(csv_path)}</code>)</p>
<p>
Commit: <code>{commit_str}</code><br/>
Build: <a href="{build_str}">{build_str}</a>
</p>

<table border="1" cellspacing="0" cellpadding="4">
  <tr>
    <th>GPU</th><th>Min %</th><th>Max %</th><th>Avg %</th><th>P50</th><th>P95</th><th>Samples</th>
  </tr>
  {stats_table}
</table>

<p>
Full time-series line chart and anomaly table are available in the attached
<code>report.html</code> (open in a browser).
</p>
"""


def _send_mail(
    bundle_dir: str,
    dry_run: bool,
    date_str: str | None,
) -> None:
    cfg = _get_required_env()
    recipients = _recipients_list(cfg[ENV_DAILY_EMAIL_LIST])
    if not recipients:
        raise SystemExit("DAILY_EMAIL_LIST is empty after parsing.")

    bundle = Path(bundle_dir)
    if not bundle.is_dir():
        raise SystemExit(f"GPU_MONITOR_BUNDLE_DIR does not exist or is not a directory: {bundle_dir}")

    csv_path = bundle / "gpu_metrics.csv"
    report_html = bundle / "report.html"
    if not csv_path.is_file():
        raise SystemExit(f"gpu_metrics.csv not found in bundle dir: {csv_path}")
    if not report_html.is_file():
        raise SystemExit(f"report.html not found in bundle dir: {report_html}")

    rows = load_csv(str(csv_path))
    if not rows:
        raise SystemExit("gpu_metrics.csv has no valid rows")

    stats = compute_stats(rows)
    run_id = bundle.name.replace("gpu_monitor_bundle_", "")

    commit_sha = os.environ.get(ENV_COMMIT)
    build_url = os.environ.get(ENV_BUILD_URL)
    sender = os.environ.get(ENV_EMAIL_SENDER) or cfg[ENV_SMTP_USERNAME]
    prefix = os.environ.get(ENV_EMAIL_SUBJECT_PREFIX)

    body_html = _build_body_html(
        stats=stats,
        run_id=run_id,
        csv_path=str(csv_path),
        commit_sha=commit_sha,
        build_url=build_url,
    )
    subject = _build_subject(prefix=prefix, run_id=run_id, date_str=date_str)

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    with open(report_html, "rb") as f:
        part = MIMEApplication(f.read(), _subtype="html")
    part.add_header("Content-Disposition", "attachment", filename=report_html.name)
    msg.attach(part)

    if csv_path.is_file():
        with open(csv_path, "rb") as f:
            csv_part = MIMEApplication(f.read(), _subtype="csv")
        csv_part.add_header("Content-Disposition", "attachment", filename=csv_path.name)
        msg.attach(csv_part)

    if dry_run:
        LOGGER.info("dry-run: not sending mail")
        print("To:", recipients, file=sys.stderr)
        print("Subject:", subject, file=sys.stderr)
        print("Body preview:", body_html[:300] + ("..." if len(body_html) > 300 else ""), file=sys.stderr)
        print("Attachments:", [report_html.name, csv_path.name], file=sys.stderr)
        return

    port = int(cfg[ENV_SMTP_PORT], 10)
    last_err: Exception | None = None
    for attempt in range(SMTP_RETRIES):
        try:
            with smtplib.SMTP(cfg[ENV_SMTP_HOST], port=port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(cfg[ENV_SMTP_USERNAME], cfg[ENV_SMTP_PASSWORD])
                smtp.sendmail(sender, recipients, msg.as_string())
            LOGGER.info("sent GPU monitor email to %d recipient(s)", len(recipients))
            return
        except Exception as e:
            last_err = e
            LOGGER.warning("SMTP attempt %d/%d failed: %s", attempt + 1, SMTP_RETRIES, e)
            if attempt < SMTP_RETRIES - 1:
                import time

                time.sleep(SMTP_RETRY_DELAY_SEC)
    raise SystemExit(f"Failed to send email after {SMTP_RETRIES} attempts.") from last_err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send L5 GPU monitor report by email (config from env + GPU_MONITOR_BUNDLE_DIR).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default=os.environ.get("GPU_MONITOR_BUNDLE_DIR", ""),
        help="Path to gpu_monitor_bundle_<run_id> (default: env GPU_MONITOR_BUNDLE_DIR).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional date string for subject (e.g. 2026-02-27).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recipient, subject, and body; do not send.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    if not args.bundle_dir:
        raise SystemExit("bundle-dir is required (or set GPU_MONITOR_BUNDLE_DIR).")
    _send_mail(bundle_dir=args.bundle_dir, dry_run=args.dry_run, date_str=args.date)


if __name__ == "__main__":
    main()

