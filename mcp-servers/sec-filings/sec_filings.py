import json
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

SEC_BASE = "https://data.sec.gov"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "mcp-sec-filings/0.1 (contact@example.com)")

mcp = FastMCP("sec-filings")


async def fetch_json(url: str) -> dict[str, Any] | None:
    headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def normalize_cik(cik: str) -> str:
    return cik.strip().lstrip("0").zfill(10)


@mcp.tool()
async def get_company_submissions(cik: str) -> str:
    """Fetch SEC submissions metadata for a company by CIK."""
    normalized = normalize_cik(cik)
    data = await fetch_json(f"{SEC_BASE}/submissions/CIK{normalized}.json")
    if not data:
        return f"Unable to fetch submissions for CIK {normalized}."

    name = data.get("name") or "Unknown"
    tickers = ", ".join(data.get("tickers") or []) or "N/A"
    exchanges = ", ".join(data.get("exchanges") or []) or "N/A"
    sic = data.get("sicDescription") or "N/A"

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])[:5]
    dates = recent.get("filingDate", [])[:5]
    nums = recent.get("accessionNumber", [])[:5]
    recent_rows = [
        f"{form} | {date} | {acc}" for form, date, acc in zip(forms, dates, nums)
    ] or ["No recent filings found."]

    summary = [
        f"Company: {name} (CIK {normalized})",
        f"Tickers: {tickers}",
        f"Exchanges: {exchanges}",
        f"SIC: {sic}",
        "Recent filings (form | date | accession):",
        *recent_rows,
    ]
    return "\n".join(summary)


@mcp.tool()
async def get_company_facts(cik: str) -> str:
    """Fetch SEC company facts (XBRL) for a company by CIK."""
    normalized = normalize_cik(cik)
    data = await fetch_json(f"{SEC_BASE}/api/xbrl/companyfacts/CIK{normalized}.json")
    if not data:
        return f"Unable to fetch company facts for CIK {normalized}."

    facts = data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    dei = facts.get("dei", {})

    gaap_keys = list(us_gaap.keys())[:10]
    dei_keys = list(dei.keys())[:10]
    return (
        f"CIK {normalized} facts loaded.\n"
        f"us-gaap keys (first 10): {gaap_keys}\n"
        f"dei keys (first 10): {dei_keys}"
    )


@mcp.tool()
async def get_recent_filings(cik: str, count: int = 10) -> str:
    """List recent filings for a company by CIK."""
    normalized = normalize_cik(cik)
    data = await fetch_json(f"{SEC_BASE}/submissions/CIK{normalized}.json")
    if not data:
        return f"Unable to fetch filings for CIK {normalized}."

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    nums = recent.get("accessionNumber", [])
    items = list(zip(forms, dates, nums))[:max(1, min(count, 50))]
    if not items:
        return f"No filings found for CIK {normalized}."

    lines = [f"{form} | {date} | {acc}" for form, date, acc in items]
    return "Recent filings (form | date | accession):\n" + "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
