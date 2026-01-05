import json
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

FISCAL_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2"
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"

mcp = FastMCP("macro-data")


async def fetch_json(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


@mcp.tool()
async def get_debt_outstanding(limit: int = 5) -> str:
    """Get recent U.S. debt outstanding (debt_to_penny) from Treasury Fiscal Data."""
    params = {"sort": "-record_date", "page[size]": max(1, min(limit, 50))}
    urls = [
        f"{FISCAL_BASE}/accounting/od/debt_to_penny",
        f"{FISCAL_BASE}/accounting/od/debt_outstanding",  # fallback if API path changes
    ]
    rows = []
    for url in urls:
        data = await fetch_json(url, params)
        rows = data.get("data", []) if data else []
        if rows:
            break
    if not rows:
        return "No debt data returned."
    lines = [f"{row.get('record_date')}: ${row.get('tot_pub_debt_out_amt')}" for row in rows]
    return "Debt outstanding (most recent first):\n" + "\n".join(lines)


@mcp.tool()
async def get_treasury_yields(limit: int = 5) -> str:
    """Get recent Treasury yield curve rates from the Treasury XML feed (official)."""
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    params = {"data": "yield"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            xml_text = resp.text
    except Exception as exc:
        return f"Failed to fetch Treasury yields: {exc}"

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return "Could not parse Treasury yield XML."

    entries = list(root.findall(".//entry"))
    if not entries:
        return "No yield data returned."

    lines = []
    for entry in entries[: max(1, min(limit, 30))]:
        date_el = entry.find("NEW_DATE")
        date_val = date_el.text if date_el is not None else "N/A"
        y2 = (entry.findtext("BC_2YEAR") or "N/A")
        y10 = (entry.findtext("BC_10YEAR") or "N/A")
        y30 = (entry.findtext("BC_30YEAR") or "N/A")
        lines.append(f"{date_val}: 2Y={y2} 10Y={y10} 30Y={y30}")

    return "Daily Treasury yield curve (most recent first):\n" + "\n".join(lines)


@mcp.tool()
async def get_worldbank_indicator(country: str, indicator: str, start_year: str | None = None, end_year: str | None = None) -> str:
    """Fetch World Bank indicator values for a country."""
    date_param = None
    if start_year and end_year:
        date_param = f"{start_year}:{end_year}"
    params = {"format": "json", "per_page": 100}
    if date_param:
        params["date"] = date_param

    url = f"{WORLD_BANK_BASE}/country/{country}/indicator/{indicator}"
    data = await fetch_json(url, params)
    if not data or len(data) < 2 or not isinstance(data[1], list):
        return "No indicator data returned."
    series = data[1][:10]
    lines = []
    for entry in series:
        lines.append(f"{entry.get('date')}: {entry.get('value')}")
    return f"World Bank indicator {indicator} for {country}:\n" + "\n".join(lines)


@mcp.tool()
async def get_ecb_series(flow_ref: str, key: str, limit: int = 20) -> str:
    """Fetch a series from the ECB Data Portal (keyless)."""
    url = f"{ECB_BASE}/{flow_ref}/{key}"
    params = {"format": "jsondata", "size": max(1, min(limit, 100))}
    data = await fetch_json(url, params)
    if not data:
        return "No ECB data returned."

    series = data.get("data", {}).get("dataSets") or data.get("dataSets")
    if not series:
        return "No series values present."

    # ECB JSON structure carries observations; surface the first few.
    observations = data.get("data", {}).get("observations") or data.get("observations") or {}
    lines = []
    for key_idx, obs in list(observations.items())[:limit]:
        # Observation structure: [value, ...]
        value = obs[0] if isinstance(obs, list) else obs
        lines.append(f"{key_idx}: {value}")
    return f"ECB series {flow_ref}/{key} (first {len(lines)} points):\n" + "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
