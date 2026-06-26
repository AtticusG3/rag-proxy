#!/usr/bin/env python3
"""Scale the nomic-embed systemd pool to fit available GPU VRAM."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from ingest.embed_pool import (  # noqa: E402
  EmbedPoolPlan,
  load_embed_pool_config,
  plan_embed_pool,
  render_pool_env,
)

DEFAULT_POOL_ENV = "/opt/ai/config/nomic-embed-pool.env"
DEFAULT_SCALE_ENV = "/opt/ai/config/nomic-embed-scale.env"
LEGACY_UNIT = "nomic-embed.service"
TEMPLATE_UNIT = "nomic-embed@"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return _run(["systemctl", *args], check=check)


def _probe_embed(url: str, *, timeout: float = 5.0) -> bool:
  try:
    with httpx.Client(timeout=timeout) as client:
      response = client.post(
        f"{url.rstrip('/')}/v1/embeddings",
        json={"model": "nomic-embed-text-v1.5", "input": ["pool-health"]},
      )
      response.raise_for_status()
      return "embedding" in response.text
  except Exception:
    return False


def _wait_for_health(urls: list[str], *, timeout_s: float = 45.0) -> list[str]:
  deadline = time.time() + timeout_s
  healthy: list[str] = []
  while time.time() < deadline:
    healthy = [url for url in urls if _probe_embed(url)]
    if len(healthy) == len(urls):
      return healthy
    time.sleep(1.0)
  return healthy


def _stop_legacy_pool_instances(config) -> None:
  _systemctl("stop", LEGACY_UNIT, check=False)
  _systemctl("disable", LEGACY_UNIT, check=False)
  for offset in range(config.max_instances):
    port = config.port_base + offset
    unit = f"{TEMPLATE_UNIT}{port}.service"
    _systemctl("stop", unit, check=False)
    _systemctl("disable", unit, check=False)
  # Also stop ports we previously skipped in static deployments.
  for port in range(config.port_base, config.port_base + config.max_instances + 4):
    unit = f"{TEMPLATE_UNIT}{port}.service"
    _systemctl("stop", unit, check=False)


def apply_plan(plan, *, pool_env_path: str, wait_health: bool) -> int:
  config = load_embed_pool_config()
  if not plan.use_gpu_pool:
    print("no GPU detected: leaving single-port pool plan only")
    Path(pool_env_path).write_text(render_pool_env(plan), encoding="utf-8")
    return 0

  _stop_legacy_pool_instances(config)

  target_urls = [f"http://127.0.0.1:{port}" for port in plan.ports]
  for port in plan.ports:
    unit = f"{TEMPLATE_UNIT}{port}.service"
    _systemctl("enable", unit, check=False)
    _systemctl("restart", unit)

  healthy_urls = _wait_for_health(target_urls) if wait_health else target_urls
  if not healthy_urls:
    print("error: no healthy nomic-embed instances after scale", file=sys.stderr)
    return 1

  healthy_ports = tuple(int(url.rsplit(":", 1)[-1]) for url in healthy_urls)
  healthy_plan = EmbedPoolPlan(
    instance_count=len(healthy_ports),
    ports=healthy_ports,
    ingest_embed_urls=",".join(healthy_urls),
    ingest_embed_concurrency=len(healthy_ports) * config.parallel_per_instance,
    gpu_total_mib=plan.gpu_total_mib,
    gpu_used_mib=plan.gpu_used_mib,
    gpu_free_mib=plan.gpu_free_mib,
    use_gpu_pool=plan.use_gpu_pool,
  )
  Path(pool_env_path).write_text(render_pool_env(healthy_plan), encoding="utf-8")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--apply", action="store_true", help="Start/stop systemd units and write pool env.")
  parser.add_argument("--pool-env", default=DEFAULT_POOL_ENV)
  parser.add_argument("--scale-env", default=DEFAULT_SCALE_ENV)
  parser.add_argument("--no-wait-health", action="store_true")
  args = parser.parse_args()

  scale_env = Path(args.scale_env)
  if scale_env.is_file():
    for line in scale_env.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      key, value = line.split("=", 1)
      import os

      os.environ.setdefault(key.strip(), value.strip())

  plan = plan_embed_pool()
  print(f"instances={plan.instance_count} ports={','.join(str(p) for p in plan.ports)}")
  if plan.gpu_free_mib is not None:
    print(
      f"gpu_free_mib={plan.gpu_free_mib} "
      f"gpu_used_mib={plan.gpu_used_mib} gpu_total_mib={plan.gpu_total_mib}"
    )
  print(f"INGEST_EMBED_CONCURRENCY={plan.ingest_embed_concurrency}")

  if not args.apply:
    print(render_pool_env(plan), end="")
    return 0

  return apply_plan(
    plan,
    pool_env_path=args.pool_env,
    wait_health=not args.no_wait_health,
  )


if __name__ == "__main__":
  raise SystemExit(main())
