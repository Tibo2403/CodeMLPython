"""Defensive local static security scanner.

This module only scans files on disk. It does not exploit vulnerabilities,
perform network probing, brute force credentials, or modify targets.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from security_bayes_optimizer.core import Config


SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}
SCANNED_EXTENSIONS = {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".env", ".ini", ".toml", ".cfg"}


@dataclass(frozen=True)
class Finding:
    """A potential defensive security finding requiring manual validation."""

    rule_id: str
    severity: str
    path: str
    line: int
    message: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "path": self.path,
            "line": self.line,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Rule:
    """Simple regex security rule."""

    rule_id: str
    severity: str
    message: str
    pattern: re.Pattern[str]
    profiles: tuple[str, ...] = ("fast", "balanced", "deep")


RULES = (
    Rule("python_eval", "high", "Use of eval can execute untrusted code.", re.compile(r"\beval\s*\(")),
    Rule("python_exec", "high", "Use of exec can execute untrusted code.", re.compile(r"\bexec\s*\(")),
    Rule("shell_true", "high", "subprocess with shell=True can enable command injection.", re.compile(r"shell\s*=\s*True")),
    Rule("debug_true", "medium", "Debug mode enabled.", re.compile(r"\bdebug\s*=\s*True|DEBUG\s*=\s*True")),
    Rule("weak_hash", "medium", "Weak hash algorithm detected.", re.compile(r"\b(md5|sha1)\s*\("), ("balanced", "deep")),
    Rule("verify_false", "high", "TLS verification disabled.", re.compile(r"verify\s*=\s*False"), ("balanced", "deep")),
    Rule("yaml_load", "medium", "yaml.load without SafeLoader can load unsafe objects.", re.compile(r"yaml\.load\s*\("), ("balanced", "deep")),
    Rule("sql_concat", "medium", "Potential SQL string concatenation.", re.compile(r"SELECT .*(\+|%|format\()"), ("deep",)),
    Rule("js_inner_html", "medium", "innerHTML assignment may introduce XSS.", re.compile(r"\.innerHTML\s*="), ("deep",)),
)


SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|secret|password|token|private[_-]?key)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{12,})"
)


def scan_path(root: Path, config: Config) -> dict[str, Any]:
    """Scan a local project path and return objective metadata."""

    start = time.perf_counter()
    files = select_files(root, config)
    findings: list[Finding] = []
    profile = str(config["rule_profile"])
    severity_floor = str(config["severity_floor"])
    entropy_threshold = float(config["entropy_threshold"])
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        findings.extend(scan_text(text, file_path, profile, severity_floor, entropy_threshold))

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    high_findings = sum(1 for finding in findings if finding.severity == "high")
    precision_proxy = precision_estimate(findings, profile, severity_floor)
    objective = security_objective(len(findings), high_findings, precision_proxy, elapsed_ms)
    return {
        "objective": objective,
        "findings": len(findings),
        "high_findings": high_findings,
        "precision_proxy": precision_proxy,
        "elapsed_ms": elapsed_ms,
        "files_scanned": len(files),
        "findings_list": [finding.to_dict() for finding in findings],
    }


def scan_text(
    text: str,
    path: Path,
    profile: str,
    severity_floor: str,
    entropy_threshold: float,
) -> list[Finding]:
    findings: list[Finding] = []
    min_severity = SEVERITY_ORDER[severity_floor]
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        for rule in RULES:
            if profile not in rule.profiles or SEVERITY_ORDER[rule.severity] < min_severity:
                continue
            if rule.pattern.search(line):
                findings.append(
                    Finding(
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        path=str(path),
                        line=line_number,
                        message=rule.message,
                        evidence=line.strip()[:180],
                    )
                )
        secret_match = SECRET_ASSIGNMENT.search(line)
        if secret_match and SEVERITY_ORDER["high"] >= min_severity:
            candidate = secret_match.group(2)
            if shannon_entropy(candidate) >= entropy_threshold:
                findings.append(
                    Finding(
                        rule_id="hardcoded_secret",
                        severity="high",
                        path=str(path),
                        line=line_number,
                        message="Possible hardcoded secret or token.",
                        evidence=mask_secret(candidate),
                    )
                )
    return findings


def select_files(root: Path, config: Config) -> list[Path]:
    include_tests = str(config["include_tests"]) == "yes"
    max_files = int(config["max_files"])
    max_file_bytes = int(config["max_file_kb"]) * 1024
    selected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(selected) >= max_files:
            break
        if not path.is_file() or path.suffix.lower() not in SCANNED_EXTENSIONS:
            continue
        lowered = str(path).lower()
        if not include_tests and ("test" in lowered or "spec" in lowered):
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        selected.append(path)
    return selected


def security_objective(findings: int, high_findings: int, precision_proxy: float, elapsed_ms: float) -> float:
    """Lower is better: reward findings and confidence, penalize slow scans."""

    return elapsed_ms * 0.015 - high_findings * 14.0 - findings * 3.0 - precision_proxy * 8.0


def precision_estimate(findings: list[Finding], profile: str, severity_floor: str) -> float:
    if not findings:
        return 0.0
    severity_score = sum(SEVERITY_ORDER[finding.severity] for finding in findings) / (len(findings) * 3.0)
    profile_penalty = {"fast": 0.02, "balanced": 0.08, "deep": 0.15}[profile]
    floor_bonus = {"low": -0.10, "medium": 0.02, "high": 0.10}[severity_floor]
    return max(0.0, min(1.0, severity_score - profile_penalty + floor_bonus))


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    return -sum((count / len(value)) * math.log2(count / len(value)) for count in counts.values())


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


class SecurityScanObjective:
    """Objective adapter used by the Bayesian optimizer."""

    def __init__(self, root: Path):
        self.root = root

    def __call__(self, config: Config) -> tuple[float, dict[str, Any]]:
        result = scan_path(self.root, config)
        return float(result["objective"]), {
            "findings": result["findings"],
            "high_findings": result["high_findings"],
            "precision_proxy": result["precision_proxy"],
            "elapsed_ms": result["elapsed_ms"],
            "files_scanned": result["files_scanned"],
            "findings_list": result["findings_list"],
        }
