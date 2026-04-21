"""Legal and data compliance scanning for reproducibility packages.

Scans package artifacts for potential private credentials, API keys,
personally identifiable information (PII), and other sensitive data
that should not be included in published packages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Patterns that may indicate credentials or secrets
_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("AWS secret key", re.compile(r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[:=]\s*\S+")),
    ("Generic API key", re.compile(r"(?i)(api[_\-]?key|apikey)\s*[:=]\s*['\"]?\S{16,}['\"]?")),
    ("Generic secret", re.compile(r"(?i)(secret|token|password|passwd)\s*[:=]\s*['\"]?\S{8,}['\"]?")),
    ("Private key block", re.compile(r"-----BEGIN\s+(RSA|DSA|EC|OPENSSH)?\s*PRIV" + r"ATE KEY-----")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")),
    ("Slack token", re.compile(r"xox[bporas]-[A-Za-z0-9-]+")),
]

# File patterns that commonly contain secrets
_SENSITIVE_FILE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Environment file", re.compile(r"\.env(\.\w+)?$")),
    ("Private key file", re.compile(r"\.(pem|key|p12|pfx)$")),
    ("Credentials file", re.compile(r"(?i)credential")),
    ("AWS config", re.compile(r"(?i)aws.*config|\.aws")),
    ("SSH key", re.compile(r"id_(rsa|dsa|ecdsa|ed25519)$")),
]

# Patterns that may indicate PII
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Email address", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("Phone number (US)", re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("SSN pattern", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IP address", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]

# Binary/non-text extensions to skip for content scanning
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
    ".mp4", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".bin", ".dat", ".npy", ".npz", ".pkl", ".h5", ".hdf5",
    ".pdf", ".doc", ".docx",
})


@dataclass
class ComplianceFinding:
    """A single compliance issue found during scanning."""

    category: str
    pattern_name: str
    file_path: str
    line_number: int | None = None
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "pattern_name": self.pattern_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "snippet": self.snippet,
        }


@dataclass
class ComplianceReport:
    """Complete compliance scan report for a reproducibility package."""

    package_name: str
    files_scanned: int = 0
    findings: list[ComplianceFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.findings) == 0

    @property
    def credential_findings(self) -> list[ComplianceFinding]:
        return [f for f in self.findings if f.category == "credential"]

    @property
    def pii_findings(self) -> list[ComplianceFinding]:
        return [f for f in self.findings if f.category == "pii"]

    @property
    def sensitive_file_findings(self) -> list[ComplianceFinding]:
        return [f for f in self.findings if f.category == "sensitive_file"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "passed": self.passed,
            "files_scanned": self.files_scanned,
            "total_findings": len(self.findings),
            "credential_findings": len(self.credential_findings),
            "pii_findings": len(self.pii_findings),
            "sensitive_file_findings": len(self.sensitive_file_findings),
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Compliance Report: {self.package_name}",
            "",
            f"**Status**: {status} ({len(self.findings)} finding(s))",
            f"**Files scanned**: {self.files_scanned}",
            "",
        ]
        if not self.findings:
            lines.append("No compliance issues found.")
        else:
            lines.extend([
                "| Category | Pattern | File | Line | Snippet |",
                "|----------|---------|------|------|---------|",
            ])
            for f in self.findings:
                ln = str(f.line_number) if f.line_number is not None else "-"
                snippet = f.snippet[:60] + "..." if len(f.snippet) > 60 else f.snippet
                # Escape pipes in snippet for markdown table
                snippet = snippet.replace("|", "\\|")
                lines.append(
                    f"| {f.category} | {f.pattern_name} | {f.file_path} | {ln} | {snippet} |"
                )
        lines.append("")
        return "\n".join(lines)


def _redact_match(match_text: str) -> str:
    """Redact a match to show only first/last few characters."""
    if len(match_text) <= 8:
        return match_text[:2] + "***"
    return match_text[:4] + "***" + match_text[-4:]


def scan_compliance(
    package_dir: Path,
    *,
    check_pii: bool = True,
    max_file_size: int = 10 * 1024 * 1024,
) -> ComplianceReport:
    """Scan a reproducibility package for compliance issues.

    Checks for credentials, secrets, PII, and sensitive file names
    within the package directory.

    Parameters
    ----------
    package_dir:
        Root directory of the reproducibility package.
    check_pii:
        Whether to also scan for PII patterns (emails, phone numbers, etc.).
    max_file_size:
        Maximum file size in bytes to scan for content patterns.
        Files larger than this are only checked by filename.

    Returns
    -------
    ComplianceReport
        Report with all findings.
    """
    package_name = package_dir.name
    findings: list[ComplianceFinding] = []
    files_scanned = 0

    for fpath in sorted(package_dir.rglob("*")):
        if not fpath.is_file():
            continue

        rel_path = str(fpath.relative_to(package_dir))
        files_scanned += 1

        # Check filename patterns
        for pattern_name, pattern in _SENSITIVE_FILE_PATTERNS:
            if pattern.search(fpath.name):
                findings.append(
                    ComplianceFinding(
                        category="sensitive_file",
                        pattern_name=pattern_name,
                        file_path=rel_path,
                        snippet=fpath.name,
                    )
                )

        # Skip binary files for content scanning
        if fpath.suffix.lower() in _BINARY_EXTENSIONS:
            continue

        # Skip large files
        try:
            if fpath.stat().st_size > max_file_size:
                continue
        except OSError:
            continue

        # Read and scan file content
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            # Credential patterns
            for pattern_name, pattern in _CREDENTIAL_PATTERNS:
                match = pattern.search(line)
                if match:
                    findings.append(
                        ComplianceFinding(
                            category="credential",
                            pattern_name=pattern_name,
                            file_path=rel_path,
                            line_number=line_num,
                            snippet=_redact_match(match.group()),
                        )
                    )

            # PII patterns
            if check_pii:
                for pattern_name, pattern in _PII_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        findings.append(
                            ComplianceFinding(
                                category="pii",
                                pattern_name=pattern_name,
                                file_path=rel_path,
                                line_number=line_num,
                                snippet=_redact_match(match.group()),
                            )
                        )

    return ComplianceReport(
        package_name=package_name,
        files_scanned=files_scanned,
        findings=findings,
    )
