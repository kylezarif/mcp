import csv
import io
from typing import Literal

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fhfa-hpi")

# FHFA publishes multiple downloadable CSVs; these two are broad and stable.
HPI_FILES = {
    "state": "HPI_AT_state.csv",
    "metro": "HPI_AT_metro.csv",
}
FHFA_BASE = "https://download.data.fhfa.gov/FileDownload"


async def fetch_csv(file_name: str) -> list[dict[str, str]]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(FHFA_BASE, params={"file": file_name})
            resp.raise_for_status()
            text = resp.text
    except Exception:
        return []
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


@mcp.tool()
async def get_hpi(geo_level: Literal["state", "metro"] = "state", limit: int = 10) -> str:
    """Fetch FHFA House Price Index data (state or metro)."""
    file_name = HPI_FILES.get(geo_level)
    if not file_name:
        return "Invalid geo_level. Use 'state' or 'metro'."

    rows = await fetch_csv(file_name)
    if not rows:
        return f"No FHFA HPI data returned for {geo_level}."

    sample = rows[:max(1, min(limit, 50))]
    lines = []
    for row in sample:
        period = row.get("yr") or row.get("yr_qtr") or row.get("yrmon") or "N/A"
        geo = row.get("state_name") or row.get("place_name") or row.get("series_id") or "N/A"
        index_val = row.get("index_nsa") or row.get("hpi_nsa") or row.get("index") or "N/A"
        lines.append(f"{period} | {geo} | {index_val}")

    return f"FHFA HPI ({geo_level}) sample rows:\n" + "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
