"""Tool implementations used only by the GAIA agent."""

from __future__ import annotations

import csv
import html
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


USER_AGENT = "Mozilla/5.0 (compatible; AEvolve-GAIA/1.0)"


def _request(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: int = 30) -> str:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    request = urllib.request.Request(url, data=data, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_000)
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def web_search_impl(query: str, max_results: int = 8) -> str:
    """Search through Tavily when configured, otherwise DuckDuckGo HTML."""
    max_results = max(1, min(int(max_results), 12))
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        payload = json.dumps({"api_key": tavily_key, "query": query, "max_results": max_results}).encode()
        raw = _request("https://api.tavily.com/search", data=payload, headers={"Content-Type": "application/json"})
        results = json.loads(raw).get("results", [])
        return "\n\n".join(
            f"[{i}] {item.get('title','')}\nURL: {item.get('url','')}\n{item.get('content','')}"
            for i, item in enumerate(results[:max_results], 1)
        )
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    raw = _request(url)
    matches = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</',
        raw, flags=re.DOTALL,
    )
    rows = []
    for i, (link, title, snippet) in enumerate(matches[:max_results], 1):
        parsed = urllib.parse.urlparse(html.unescape(link))
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [link])[0]
        clean = lambda value: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value))).strip()
        rows.append(f"[{i}] {clean(title)}\nURL: {target}\n{clean(snippet)}")
    return "\n\n".join(rows) or "No search results returned. Configure TAVILY_API_KEY for reliable search."


def web_fetch_impl(url: str, max_chars: int = 20000) -> str:
    raw = _request(url)
    raw = re.sub(r"(?is)<(script|style|svg).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|article|section)>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[: max(1000, min(int(max_chars), 50000))]


def inspect_file_impl(path: str, max_chars: int = 30000) -> str:
    file_path = Path(path).resolve()
    if not file_path.exists():
        return f"ERROR: file not found: {file_path}"
    suffix = file_path.suffix.lower()
    limit = max(1000, min(int(max_chars), 50000))
    if suffix in {".txt", ".md", ".json", ".jsonl", ".xml", ".html", ".csv", ".tsv", ".py"}:
        return file_path.read_text(encoding="utf-8", errors="replace")[:limit]
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
            sheets = pd.read_excel(file_path, sheet_name=None)
            return "\n\n".join(f"## {name}\n{frame.to_csv(index=False)}" for name, frame in sheets.items())[:limit]
        except Exception as exc:
            return f"ERROR reading spreadsheet: {exc}"
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)[:limit]
        except Exception as exc:
            return f"ERROR reading PDF (install pypdf): {exc}"
    if suffix in {".docx", ".pptx"}:
        try:
            with zipfile.ZipFile(file_path) as archive:
                names = [
                    name for name in archive.namelist()
                    if name.endswith(".xml")
                    and (name.startswith("word/") or name.startswith("ppt/slides/"))
                ]
                chunks = []
                for name in sorted(names):
                    xml = archive.read(name).decode("utf-8", errors="replace")
                    text = html.unescape(re.sub(r"<[^>]+>", " ", xml))
                    chunks.append(f"## {name}\n" + re.sub(r"\s+", " ", text).strip())
                return "\n\n".join(chunks)[:limit]
        except Exception as exc:
            return f"ERROR reading Office document: {exc}"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            from PIL import Image
            image = Image.open(file_path)
            detail = f"Image: {file_path}\nSize: {image.size}\nMode: {image.mode}"
            try:
                import pytesseract
                ocr = pytesseract.image_to_string(image)
                return (detail + "\nOCR:\n" + ocr)[:limit]
            except Exception:
                return detail + "\nOCR unavailable; use Python with the provided file path for image analysis."
        except Exception as exc:
            return f"ERROR reading image: {exc}"
    if suffix == ".zip":
        with zipfile.ZipFile(file_path) as archive:
            return "Archive members:\n" + "\n".join(archive.namelist())[:limit]
    return f"File: {file_path}\nSize: {file_path.stat().st_size} bytes\nType: {suffix or 'unknown'}"


def python_impl(code: str, workdir: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code], cwd=workdir, capture_output=True, text=True,
            timeout=max(1, min(int(timeout), 300)),
        )
        output = (result.stdout or "") + (result.stderr or "")
        return f"exit_code={result.returncode}\n{output[-30000:]}"
    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out."
