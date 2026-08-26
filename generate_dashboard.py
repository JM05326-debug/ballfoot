#!/usr/bin/env python
"""
把 predict.py 產生的 data/predictions/latest.json 轉成一個靜態網頁儀表板，
輸出到 docs/index.html——GitHub Pages 預設可以直接把 repo 的 docs/ 資料夾當網站發布，
不需要另外的伺服器或建置流程。

用法：
    python generate_dashboard.py

搭配 .github/workflows/daily.yml 的排程，每天自動重新產生一次，
GitHub Pages 就會顯示最新一輪的預測結果。
"""

from __future__ import annotations

import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src import config

REPO_URL = "https://github.com/JM05326-debug/ballfoot"

PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>英超賽事預測</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⚽</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+TC:wght@400;500;700;900&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
  :root {{
    --bg: #F5F3EC;
    --surface: #FFFFFF;
    --surface-2: #ECE7D8;
    --border: #DDD5BE;
    --text: #17251D;
    --text-muted: #5A6A5C;
    --text-faint: #8B9389;
    --accent: #1E6B3F;
    --home: #1E6B3F;
    --draw: #B8860B;
    --away: #35618C;
    --conf-high: #1E6B3F;
    --conf-med: #A9791F;
    --conf-low: #8B8478;
    --shadow: 0 1px 2px rgba(23,37,29,0.06), 0 6px 20px -8px rgba(23,37,29,0.15);
  }}

  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0E1712; --surface: #15201A; --surface-2: #1B2820; --border: #2A392E;
      --text: #ECF2EC; --text-muted: #9FB1A2; --text-faint: #6E7E70;
      --accent: #55BC80; --home: #55BC80; --draw: #E3BE55; --away: #7CACDA;
      --conf-high: #55BC80; --conf-med: #E3BE55; --conf-low: #8B9689;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 10px 28px -10px rgba(0,0,0,0.55);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0E1712; --surface: #15201A; --surface-2: #1B2820; --border: #2A392E;
    --text: #ECF2EC; --text-muted: #9FB1A2; --text-faint: #6E7E70;
    --accent: #55BC80; --home: #55BC80; --draw: #E3BE55; --away: #7CACDA;
    --conf-high: #55BC80; --conf-med: #E3BE55; --conf-low: #8B9689;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 10px 28px -10px rgba(0,0,0,0.55);
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: 'Noto Sans TC', system-ui, -apple-system, sans-serif;
    line-height: 1.6; -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 2.5rem 1.5rem 5rem; }}
  .masthead {{
    display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
    gap: 1.5rem; padding-bottom: 1.75rem; margin-bottom: 2rem; border-bottom: 2px solid var(--text);
  }}
  .masthead-title {{
    font-family: 'Noto Sans TC', sans-serif;
    font-weight: 900;
    font-size: clamp(2.2rem, 5.4vw, 3.6rem); line-height: 1.15; letter-spacing: 0.01em; margin: 0;
    text-wrap: balance;
  }}
  .masthead-title .accent {{ color: var(--accent); }}
  .masthead-meta {{
    display: flex; flex-direction: column; gap: 0.35rem; align-items: flex-start;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--text-muted);
  }}
  .masthead-meta a {{ color: var(--accent); text-decoration: none; border-bottom: 1px solid transparent; }}
  .masthead-meta a:hover, .masthead-meta a:focus-visible {{ border-bottom-color: var(--accent); }}
  .summary-strip {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 2.25rem; }}
  .summary-chip {{
    display: flex; align-items: baseline; gap: 0.4rem; background: var(--surface);
    border: 1px solid var(--border); border-radius: 999px; padding: 0.45rem 1rem; box-shadow: var(--shadow);
  }}
  .summary-chip .num {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.95rem; }}
  .summary-chip .label {{ font-size: 0.78rem; color: var(--text-muted); }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .dot.high {{ background: var(--conf-high); }} .dot.med {{ background: var(--conf-med); }} .dot.low {{ background: var(--conf-low); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.1rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.35rem 1.4rem 1.2rem; box-shadow: var(--shadow);
    display: flex; flex-direction: column; gap: 1rem;
  }}
  .card-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 0.75rem; }}
  .kickoff {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--text-faint);
  }}
  .confidence-pill {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 0.28rem 0.6rem; border-radius: 999px; white-space: nowrap;
    border: 1px solid transparent; display: inline-flex; align-items: center; gap: 0.35rem;
  }}
  .confidence-pill.High {{ color: var(--conf-high); border-color: var(--conf-high); background: color-mix(in srgb, var(--conf-high) 12%, transparent); }}
  .confidence-pill.Medium {{ color: var(--conf-med); border-color: var(--conf-med); background: color-mix(in srgb, var(--conf-med) 12%, transparent); }}
  .confidence-pill.Low {{ color: var(--conf-low); border-color: var(--conf-low); background: color-mix(in srgb, var(--conf-low) 12%, transparent); }}
  .matchup {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 0.75rem; }}
  .team-name {{ font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 0.01em; line-height: 1.05; }}
  .team-name.away-side {{ text-align: right; }}
  .score-badge {{
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.15rem;
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 0.35rem 0.7rem; white-space: nowrap;
  }}
  .prob-bar {{ display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: var(--surface-2); }}
  .prob-bar span {{ display: block; height: 100%; }}
  .seg-home {{ background: var(--home); }} .seg-draw {{ background: var(--draw); }} .seg-away {{ background: var(--away); }}
  .prob-labels {{ display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-variant-numeric: tabular-nums; }}
  .prob-labels .lab {{ display: flex; align-items: center; gap: 0.4rem; color: var(--text-muted); }}
  .prob-labels .lab strong {{ color: var(--text); font-weight: 600; }}
  .swatch {{ width: 8px; height: 8px; border-radius: 2px; }}
  .swatch.home {{ background: var(--home); }} .swatch.draw {{ background: var(--draw); }} .swatch.away {{ background: var(--away); }}
  .stat-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; }}
  .stat {{ background: var(--surface-2); border-radius: 8px; padding: 0.5rem 0.6rem; display: flex; flex-direction: column; gap: 0.15rem; }}
  .stat .stat-label {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-faint); }}
  .stat .stat-value {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.95rem; font-variant-numeric: tabular-nums; }}
  details.factors {{ border-top: 1px dashed var(--border); padding-top: 0.75rem; }}
  details.factors summary {{
    cursor: pointer; font-size: 0.8rem; color: var(--accent); font-weight: 600; list-style: none;
    display: flex; align-items: center; gap: 0.4rem; user-select: none;
  }}
  details.factors summary::-webkit-details-marker {{ display: none; }}
  details.factors summary::before {{ content: "▸"; display: inline-block; transition: transform 0.15s ease; font-size: 0.7rem; }}
  details.factors[open] summary::before {{ transform: rotate(90deg); }}
  summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 4px; }}
  .factor-list {{ list-style: none; margin: 0.7rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; }}
  .factor-list li {{ display: flex; justify-content: space-between; align-items: center; gap: 0.75rem; font-size: 0.82rem; }}
  .factor-list .f-label {{ color: var(--text-muted); }}
  .factor-list .f-side {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
    padding: 0.1rem 0.5rem; border-radius: 999px; white-space: nowrap;
  }}
  .factor-list .f-side.home {{ color: var(--home); background: color-mix(in srgb, var(--home) 14%, transparent); }}
  .factor-list .f-side.away {{ color: var(--away); background: color-mix(in srgb, var(--away) 14%, transparent); }}
  .sub-heading {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint); margin: 0.9rem 0 0.4rem; }}
  .scoreline-table, .model-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  .scoreline-table td, .model-table td, .model-table th {{ padding: 0.28rem 0.3rem; font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}
  .scoreline-table td:first-child {{ color: var(--text); font-weight: 600; width: 3.2rem; }}
  .scoreline-table td:last-child {{ color: var(--text-muted); text-align: right; }}
  .model-table {{ table-layout: fixed; }}
  .model-table th {{ text-align: right; color: var(--text-faint); font-weight: 500; font-size: 0.66rem; text-transform: uppercase; }}
  .model-table th:first-child, .model-table td:first-child {{ text-align: left; font-family: 'Noto Sans TC', sans-serif; color: var(--text-muted); }}
  .model-table td {{ text-align: right; color: var(--text); }}
  .model-table tbody tr:nth-child(odd) {{ background: var(--surface-2); }}
  .table-scroll {{ overflow-x: auto; }}
  footer {{
    margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
    font-size: 0.78rem; color: var(--text-faint); display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem;
  }}
  footer a {{ color: var(--text-muted); }}
  .empty-state {{ padding: 3rem 1rem; text-align: center; color: var(--text-muted); font-size: 1rem; }}
  @media (max-width: 520px) {{ .stat-row {{ grid-template-columns: repeat(2, 1fr); }} .team-name {{ font-size: 1.2rem; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <h1 class="masthead-title">英超<span class="accent">賽事預測</span></h1>
    <div class="masthead-meta">
      <span>英格蘭超級聯賽 {season_label} 賽季 — 第 {round_number} 輪</span>
      <span>集成模型 + Poisson 模型 · 已用驗證集校準 · 最後更新 {generated_at}</span>
      <a href="{repo_url}" target="_blank" rel="noopener">{repo_url_display}</a>
    </div>
  </header>

  <div class="summary-strip" id="summaryStrip"></div>
  <main class="grid" id="matchGrid"></main>

  <footer>
    <span>主/和/客機率：7 個模型（Logistic Regression、Random Forest、XGBoost、LightGBM、CatBoost、Dixon-Coles Poisson、模糊邏輯）依驗證集 Log Loss 加權集成，並用 Platt Scaling 校準。</span>
    <span>比分／預期進球／大小球／雙方進球：獨立的 Dixon-Coles Poisson 進球模型。</span>
    <span>由 GitHub Actions 每天自動重新產生，只使用賽前已知資料，不使用任何賽後資訊。</span>
  </footer>
</div>

<script>
const MATCHES = {matches_json};

function pct(x) {{ return (x * 100).toFixed(1) + '%'; }}

const CONF_LABEL = {{ High: '高', Medium: '中', Low: '低' }};

function renderSummary() {{
  const strip = document.getElementById('summaryStrip');
  if (MATCHES.length === 0) return;
  const counts = {{ High: 0, Medium: 0, Low: 0 }};
  MATCHES.forEach(m => counts[m.confidence]++);
  strip.innerHTML = [
    `<div class="summary-chip"><span class="num">${{MATCHES.length}}</span><span class="label">場比賽</span></div>`,
    `<div class="summary-chip"><span class="dot high"></span><span class="num">${{counts.High}}</span><span class="label">高信心</span></div>`,
    `<div class="summary-chip"><span class="dot med"></span><span class="num">${{counts.Medium}}</span><span class="label">中信心</span></div>`,
    `<div class="summary-chip"><span class="dot low"></span><span class="num">${{counts.Low}}</span><span class="label">低信心</span></div>`,
  ].join('');
}}

function formatKickoff(dateStr) {{
  const d = new Date(dateStr.replace(' ', 'T'));
  const opts = {{ month: 'numeric', day: 'numeric', weekday: 'short', hour: '2-digit', minute: '2-digit', hour12: false }};
  return d.toLocaleString('zh-TW', opts);
}}

function renderCard(m) {{
  const home = pct(m.p_home_win), draw = pct(m.p_draw), away = pct(m.p_away_win);
  const factors = m.top_influencing_factors.map(f => `
    <li><span class="f-label">${{f.label}}</span><span class="f-side ${{f.favors.toLowerCase()}}">有利 ${{f.favors === 'Home' ? m.home_team : m.away_team}}</span></li>`).join('');
  const scorelines = m.top_scorelines.map(s => `<tr><td>${{s.score}}</td><td>${{pct(s.probability)}}</td></tr>`).join('');
  const modelNames = Object.keys(m.per_model_raw_proba);
  const modelRows = modelNames.map(name => {{
    const [h, d, a] = m.per_model_raw_proba[name];
    return `<tr><td>${{name}}</td><td>${{pct(h)}}</td><td>${{pct(d)}}</td><td>${{pct(a)}}</td></tr>`;
  }}).join('');

  return `
  <article class="card">
    <div class="card-top">
      <span class="kickoff">${{formatKickoff(m.date)}}</span>
      <span class="confidence-pill ${{m.confidence}}">信心：${{CONF_LABEL[m.confidence]}}</span>
    </div>
    <div class="matchup">
      <span class="team-name home-side">${{m.home_team}}</span>
      <span class="score-badge">${{m.predicted_score}}</span>
      <span class="team-name away-side">${{m.away_team}}</span>
    </div>
    <div>
      <div class="prob-bar">
        <span class="seg-home" style="width:${{m.p_home_win * 100}}%"></span>
        <span class="seg-draw" style="width:${{m.p_draw * 100}}%"></span>
        <span class="seg-away" style="width:${{m.p_away_win * 100}}%"></span>
      </div>
      <div class="prob-labels" style="margin-top:0.4rem;">
        <span class="lab"><span class="swatch home"></span>主勝 <strong>${{home}}</strong></span>
        <span class="lab"><span class="swatch draw"></span>和局 <strong>${{draw}}</strong></span>
        <span class="lab"><span class="swatch away"></span>客勝 <strong>${{away}}</strong></span>
      </div>
    </div>
    <div class="stat-row">
      <div class="stat"><span class="stat-label">預期進球 xG</span><span class="stat-value">${{m.expected_home_goals.toFixed(2)}} – ${{m.expected_away_goals.toFixed(2)}}</span></div>
      <div class="stat"><span class="stat-label">大 2.5 球</span><span class="stat-value">${{pct(m.p_over_2_5)}}</span></div>
      <div class="stat"><span class="stat-label">雙方進球：是</span><span class="stat-value">${{pct(m.p_btts_yes)}}</span></div>
      <div class="stat"><span class="stat-label">小 2.5 球</span><span class="stat-value">${{pct(m.p_under_2_5)}}</span></div>
    </div>
    <details class="factors">
      <summary>關鍵影響因素與各模型明細</summary>
      <ul class="factor-list">${{factors}}</ul>
      <div class="sub-heading">最可能比分</div>
      <div class="table-scroll"><table class="scoreline-table"><tbody>${{scorelines}}</tbody></table></div>
      <div class="sub-heading">各模型原始機率（尚未集成、尚未校準）</div>
      <div class="table-scroll">
        <table class="model-table"><thead><tr><th>模型</th><th>主勝</th><th>和局</th><th>客勝</th></tr></thead><tbody>${{modelRows}}</tbody></table>
      </div>
    </details>
  </article>`;
}}

const grid = document.getElementById('matchGrid');
if (MATCHES.length === 0) {{
  grid.innerHTML = '<div class="empty-state">目前沒有即將開踢的賽程，等下一輪賽程公布後再回來看。</div>';
}} else {{
  grid.innerHTML = MATCHES.map(renderCard).join('');
}}
renderSummary();
</script>
</body>
</html>
"""


def main():
    latest_path = config.PREDICTIONS_DIR / "latest.json"
    meta_path = config.PREDICTIONS_DIR / "latest_meta.json"

    if not latest_path.exists() or not meta_path.exists():
        print("DATA SOURCE ERROR: 找不到 data/predictions/latest.json，請先執行 python predict.py", file=sys.stderr)
        sys.exit(1)

    matches = json.loads(latest_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    html = PAGE_TEMPLATE.format(
        season_label=meta["season_label"],
        round_number=meta["round"],
        generated_at=meta["generated_at"].replace("T", " "),
        repo_url=REPO_URL,
        repo_url_display=REPO_URL.replace("https://", ""),
        matches_json=json.dumps(matches, ensure_ascii=False),
    )

    docs_dir = config.PROJECT_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"已產生: {out_path}（{meta['n_matches']} 場比賽，{meta['season_label']} 第 {meta['round']} 輪）")


if __name__ == "__main__":
    main()
