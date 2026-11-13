import csv, sys
path, phase = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('CS', 'total')
for r in csv.DictReader(open('attempts.csv')):
    if r['path'] == path and r['phase'] == phase:
        print(f"{r['step']:14} {r['change']:10} th={r['threads']} wall {r['wall_before_s']}->{r['wall_after_s']} ({r['d_wall_s']}, {r['d_wall_pct']}%) instr {r['instr_before_G']}->{r['instr_after_G']} ({r['d_instr_G']}, {r['d_instr_pct']}%) kept={r['kept']}")
