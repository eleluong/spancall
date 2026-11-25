import re, glob, os

def count(path):
    s = open(path).read()
    s = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', ' ', s, flags=re.S)
    s = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', ' ', s, flags=re.S)
    s = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', ' ', s, flags=re.S)
    s = re.sub(r'\\begin\{table\}.*?\\end\{table\}', ' ', s, flags=re.S)
    s = re.sub(r'(?<!\\)%.*', ' ', s)
    s = re.sub(r'\\[a-zA-Z]+\*?', ' ', s)
    s = re.sub(r'[{}$\\]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return len(s.split())

ORIG = 'scratch/orig_sections'
NEW = 'research_paper/sections'

totals = {}
for tag, d in (('orig', ORIG), ('new', NEW)):
    totals[tag] = {os.path.basename(f): count(f) for f in sorted(glob.glob(os.path.join(d, '0*.tex')))}

names = sorted(totals['new'])
print(f"{'section':45s} {'before':>7s} {'after':>7s} {'cut':>7s}")
print('-' * 70)
b_tot = a_tot = 0
for n in names:
    b = totals['orig'][n]; a = totals['new'][n]
    b_tot += b; a_tot += a
    pct = (b - a) / b * 100 if b else 0
    print(f"{n:45s} {b:7d} {a:7d} {pct:6.1f}%")
print('-' * 70)
print(f"{'TOTAL BODY PROSE':45s} {b_tot:7d} {a_tot:7d} {(b_tot-a_tot)/b_tot*100:6.1f}%")