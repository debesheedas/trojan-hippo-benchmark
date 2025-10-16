#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

ROOT = Path('/Users/ddas/Desktop/Debeshee/Thesis/memory-agent-security-benchmark')
RESULTS_DIR = ROOT / 'test_bench_results'
REPORTS_DIR = ROOT / 'html_reports'
INDEX_FILE = REPORTS_DIR / 'index.html'

REPORTS_DIR.mkdir(exist_ok=True)

items = []
for f in sorted(RESULTS_DIR.glob('*.json')):
    if f.name.startswith('summary_'):
        continue
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        test_name = data.get('test_name') or data.get('name') or f.stem
        description = data.get('description', '')
        session_id = data.get('session_id', '')
        link = f'{f.stem}.html'
        if not (REPORTS_DIR / link).exists():
            # skip if report not present
            continue
        items.append({
            'test_name': test_name,
            'description': description,
            'session_id': session_id,
            'link': link,
        })
    except Exception:
        continue

cards_html = []
for it in items:
    cards_html.append(f'''            <a href="{it['link']}" class="report-card">
                <h3>{it['test_name']}</h3>
                <div class="description">
                    {it['description']}
                </div>
                <div class="report-meta">
                    <span class="report-date">Session: {it['session_id']}</span>
                    <span class="view-report">View Report →</span>
                </div>
            </a>''')

index_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Agent Security Benchmark - Test Reports</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 50px;
        }}
        .header h1 {{
            font-size: 3em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            font-size: 1.3em;
            opacity: 0.9;
        }}
        .reports-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
        }}
        .report-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            text-decoration: none;
            color: inherit;
        }}
        .report-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }}
        .report-card h3 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        .report-card .description {{
            color: #6c757d;
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        .report-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
        }}
        .report-date {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .view-report {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 0.9em;
            font-weight: bold;
            transition: opacity 0.3s ease;
        }}
        .view-report:hover {{
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: white;
            opacity: 0.8;
        }}
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 2em; }}
            .reports-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Memory Agent Security Benchmark</h1>
            <p>Test Execution Reports</p>
        </div>
        <div class="reports-grid">
{chr(10).join(cards_html)}
        </div>
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
'''

INDEX_FILE.write_text(index_html, encoding='utf-8')
print(f'Wrote {INDEX_FILE}')
