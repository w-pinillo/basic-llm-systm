#!/usr/bin/env python3
"""
Basic Research Agent
----------------------------------
A minimal, self-contained agentic system that:
1) Searches the web for a topic
2) Fetches and cleans pages
3) Creates per-source summaries
4) Synthesizes a final brief with inline numeric citations [1], [2], ...
5) Stores a JSONL memory of runs
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
import requests
from duckduckgo_search import DDGS
import trafilatura
from openai import OpenAI


# ---------------- Utilities ---------------- #

def safe_filename(text: str) -> str:
    return "".join(c for c in text if c.isalnum() or c in (" ", "_", "-", ".")).rstrip().replace(" ", "_")[:120]

def now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------------- Tools ---------------- #

class WebSearchTool:
    """DuckDuckGo search wrapper (keyless)."""
    def __init__(self, max_results: int = 6):
        self.max_results = max_results

    def run(self, query: str):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=self.max_results):
                if r.get("href") and r.get("title"):
                    results.append({
                        "title": r["title"],
                        "url": r["href"],
                        "snippet": r.get("body", "")
                    })
        return results


class WebPageLoader:
    """Fetch & clean text using trafilatura."""
    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/0.1)"
        })
        self.timeout = timeout

    def fetch(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if not resp.ok:
                return None
            return trafilatura.extract(resp.text)
        except Exception:
            return None


# ---------------- LLM ---------------- #

class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        rsp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return rsp.choices[0].message.content.strip()


# ---------------- Agent ---------------- #

class SourceDoc(BaseModel):
    idx: int
    title: str
    url: str
    snippet: str = ""
    text: Optional[str] = None
    summary_bullets: Optional[str] = None


@dataclass
class AgentConfig:
    search_results: int = 6
    min_chars: int = 800
    per_source_bullets: int = 3
    final_bullets: int = 6
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    out_dir: str = "agent_output"
    memory_file: str = "memory.jsonl"


class ResearchAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        self.search = WebSearchTool(cfg.search_results)
        self.loader = WebPageLoader()
        self.llm = LLMClient(cfg.model, cfg.temperature)

    def step_search(self, query: str) -> List[SourceDoc]:
        print(f"Searching: {query}")
        raw = self.search.run(query)
        docs = []
        seen = set()
        idx = 1
        for r in raw:
            url = r["url"]
            domain = url.split("/")[2] if "://" in url else url
            key = (r["title"].strip(), domain)
            if key in seen:
                continue
            seen.add(key)
            docs.append(SourceDoc(idx=idx, title=r["title"].strip(), url=url, snippet=r.get("snippet", "")))
            idx += 1
        return docs

    def step_fetch(self, docs: List[SourceDoc]) -> List[SourceDoc]:
        kept = []
        for d in docs:
            print(f"Fetching: [{d.idx}] {d.title} ({d.url})")
            text = self.loader.fetch(d.url)
            if text and len(text) >= self.cfg.min_chars:
                d.text = text
                kept.append(d)
            else:
                print(f"   ⚠️ Skipped (too short or failed).")
        return kept

    def step_summarize_each(self, docs: List[SourceDoc]) -> None:
        system = "You are a careful research assistant. Output only concise bullet points with facts."
        for d in docs:
            if not d.text:
                continue
            user = (
                f"Summarize the article into {self.cfg.per_source_bullets} bullets.\n\n"
                f"TITLE: {d.title}\nURL: {d.url}\n\nARTICLE:\n{d.text[:7000]}"
            )
            print(f"Summarizing source [{d.idx}]...")
            d.summary_bullets = self.llm.chat(system, user)

    def step_synthesize_report(self, query: str, docs: List[SourceDoc]) -> str:
        system = "You are a synthesis assistant. Write a neutral summary with inline numeric citations [1], [2]."
        sources_list = "\n".join(f"[{d.idx}] {d.title} — {d.url}" for d in docs if d.summary_bullets)
        per_source_points = "\n\n".join(f"[{d.idx}] {d.summary_bullets}" for d in docs if d.summary_bullets)
        user = (
            f"QUESTION: {query}\n\n"
            f"PER-SOURCE BULLETS:\n{per_source_points}\n\n"
            f"Write the report with:\n"
            f"1) Executive summary: {self.cfg.final_bullets} bullets.\n"
            f"2) Key points with inline citations.\n"
            f"3) List of sources.\n\n"
            f"Sources:\n{sources_list}"
        )
        print("Synthesizing final report...")
        return self.llm.chat(system, user)

    def persist_run(self, query: str, docs: List[SourceDoc], report_md: str) -> str:
        ts = now_iso()
        run = {
            "timestamp": ts,
            "query": query,
            "model": self.cfg.model,
            "sources": [{"idx": d.idx, "title": d.title, "url": d.url} for d in docs if d.summary_bullets],
            "report_preview": report_md[:500]
        }
        mem_path = os.path.join(self.cfg.out_dir, self.cfg.memory_file)
        with open(mem_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
        safe = safe_filename(f"{ts}_{query}.md")
        out_path = os.path.join(self.cfg.out_dir, safe)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        return out_path

    def run(self, query: str) -> str:
        docs = self.step_search(query)
        docs = self.step_fetch(docs)
        if not docs:
            raise RuntimeError("No usable sources found.")
        self.step_summarize_each(docs)
        docs = [d for d in docs if d.summary_bullets]
        report = self.step_synthesize_report(query, docs)
        out_path = self.persist_run(query, docs, report)
        return out_path


def main():
    load_dotenv()
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python basic_research_agent.py \"your research question\"")
        sys.exit(1)

    cfg = AgentConfig()
    agent = ResearchAgent(cfg)
    out_path = agent.run(query)
    print("\n Done. Report saved to:", out_path)
    print("Memory appended to:", os.path.join(cfg.out_dir, cfg.memory_file))


if __name__ == "__main__":
    main()
