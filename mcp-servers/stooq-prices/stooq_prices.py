import csv
import io
from datetime import datetime

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = "https://stooq.com/q/d/l/"

mcp = FastMCP("stooq-prices")


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value).strftime("%Y%m%d")
        return dt
    except ValueError:
        return None


async def fetch_csv(url: str, params: dict[str, str]) -> list[dict[str, str]]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            text = resp.text
    except Exception:
        return []

    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


@mcp.tool()
async def get_daily_ohlcv(symbol: str, start_date: str | None = None, end_date: str | None = None, interval: str = "d") -> str:
    """Fetch daily OHLCV data from Stooq (historical/delayed). Dates use ISO format."""
    d1 = _parse_date(start_date)
    d2 = _parse_date(end_date)

    params = {
        "s": symbol.lower(),
        "i": interval,
    }
    if d1:
        params["d1"] = d1
    if d2:
        params["d2"] = d2

    rows = await fetch_csv(BASE_URL, params)
    if not rows:
        return "No data returned from Stooq."

    # Stooq returns oldest first; take the most recent 10 rows
    first_rows = rows[-10:]
    lines = [
        f"{r.get('Date')}: O={r.get('Open')} H={r.get('High')} L={r.get('Low')} C={r.get('Close')} V={r.get('Volume')}"
        for r in first_rows
    ]
    return f"Stooq OHLCV for {symbol} (first {len(lines)} rows):\n" + "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
