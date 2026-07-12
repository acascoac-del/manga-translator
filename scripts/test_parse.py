import re

raw = """Here are the translations:
1. Hola
2. ¿Qué tal?
3. Adiós
"""

indexed = [(0, 'A'), (1, 'B'), (2, 'C')]
results = ['A', 'B', 'C']

lines = [l.strip() for l in raw.splitlines() if l.strip()]
parsed = {}
for line in lines:
    m = re.match(r'^(\d+)\.?\s+(.*)', line)
    if m:
        num = int(m.group(1)) - 1
        parsed[num] = m.group(2).strip()

print(f"Parsed: {parsed}")
if len(parsed) >= len(indexed):
    for n, (i, _) in enumerate(indexed):
        if n in parsed:
            results[i] = parsed[n]
else:
    clean_lines = []
    for line in lines:
        m = re.match(r'^\d+\.?\s+(.*)', line)
        clean_lines.append(m.group(1).strip() if m else line)
    print(f"Clean lines: {clean_lines}")
    for n, (i, _) in enumerate(indexed):
        if n < len(clean_lines):
            results[i] = clean_lines[n]

print(f"Results: {results}")
