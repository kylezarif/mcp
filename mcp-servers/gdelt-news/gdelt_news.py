import datetime as dt
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

mcp = FastMCP("gdelt-news")


def _date_param(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt.datetime.fromisoformat(value)
        return value
    except ValueError:
        return None


async def fetch_json(params: dict[str, Any]) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(GDELT_DOC_API, params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def format_articles(articles: list[dict[str, Any]], limit: int) -> str:
    rows = []
    for art in articles[:limit]:
        title = art.get("title") or "Untitled"
        url = art.get("url") or "N/A"
        date = art.get("seendate") or "N/A"
        source = art.get("sourceCommonName") or art.get("domain") or "N/A"
        rows.append(f"{date} | {source} | {title}\n{url}")
    return "\n\n".join(rows) if rows else "No results."


@mcp.tool()
async def search_news(query: str, start_date: str | None = None, end_date: str | None = None, max_records: int = 20) -> str:
    """Search global news via GDELT DOC API (ArtList mode). Dates use ISO format."""
    start = _date_param(start_date)
    end = _date_param(end_date)
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max(1, min(max_records, 50)),
    }
    if start:
        params["startdatetime"] = start
    if end:
        params["enddatetime"] = end

    data = await fetch_json(params)
    if not data or "articles" not in data:
        return "No results or failed to fetch news."
    return format_articles(data.get("articles", []), params["maxrecords"])


@mcp.tool()
async def company_news(name_or_ticker: str, max_records: int = 20) -> str:
    """Quick shortcut to search company news by name or ticker."""
    q = (name_or_ticker or "").strip().rstrip(".")
    lowered = q.lower()
    for phrase in (" in the last week", " last week", " in the past week", " this week"):
        if lowered.endswith(phrase):
            q = q[: -len(phrase)].strip()
            break
    return await search_news(q or name_or_ticker, max_records=max_records)


@mcp.tool()
async def topic_trends(topic: str, max_records: int = 20) -> str:
    """Search recent articles for a topic to gauge activity."""
    return await search_news(topic, max_records=max_records)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
