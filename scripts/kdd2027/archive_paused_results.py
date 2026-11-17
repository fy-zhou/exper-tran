"""CPU-only evidence export after user pause; never runs inference or alters locks."""
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path('/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction')
DEST = Path(__file__).resolve().parents[2] / 'docs/kdd2027/results/paused_2026-11-17'
DEST.mkdir(parents=True, exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(name, x):
    (DEST / name).write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')

def table(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (DEST / name).open('w') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

assert (OUT / 'extension/paused_by_user.json').exists()
legacy = []
for f in sorted((OUT / 'runs').glob('final*/results.json')):
    legacy += [dict(run=f.parent.name, **r) for r in json.loads(f.read_text())]
table('historical_full_and_partial_results.csv', legacy)
summary, inventory, audits, checks = [], [], [], []
for run in sorted((OUT / 'extension/runs').iterdir()):
    lockp = run / 'lock.json'
    if not lockp.exists():
        continue
    lock = json.loads(lockp.read_text())
    c = lock['config']
    receipts = list(run.glob('attacks/*/receipt.json'))
    failures = list(run.glob('attacks/*/failure.json'))
    inventory.append(dict(run=run.name, method=c['method'], dataset=c['dataset'],
                          variant=c['variant'], smoke=c['smoke_only'], pilot=c.get('pilot',False),
                          steps=c['steps'], frozen_images=len(receipts), failures=len(failures),
                          response_records=len(list(run.glob('evaluations/**/*.json'))),
                          lock_sha256=sha(lockp)))
    for p in receipts:
        r = json.loads(p.read_text())
        hashes_ok = all(Path(n).is_file() and sha(Path(n)) == h for n,h in r['png_hashes'].items())
        a=np.asarray(Image.open(r['adv']).convert('RGB'),dtype=np.int16)
        b=np.asarray(Image.open(r['clean']).convert('RGB'),dtype=np.int16)
        maximum=int(np.abs(a-b).max())
        checks.append(dict(run=run.name,image_id=r['image_id'],png_hashes_valid=hashes_ok,
                           linf_uint8=maximum,bounds_pass=maximum<=16,
                           no_target_access=r.get('target_access') is False,
                           steps=r['steps'],seconds=r.get('seconds'),
                           peak_reserved_bytes=r.get('peak_reserved_bytes'),
                           forward_calls=r.get('forward_calls'),
                           receipt_sha256=sha(p)))
    f=run/'task_summary.json'
    if f.exists():
        for r in json.loads(f.read_text()):
            summary.append(dict(run=run.name,**r))
    audit_files=list(run.glob('audits/*.json'))+[run/'bf16_proxies.json']
    for p in audit_files:
        if not p.exists():continue
        d=json.loads(p.read_text())
        for a in (d if isinstance(d,list) else [d]):
            q=a.get('quantization',{})
            counts=a.get('parameter_counts',{})
            audits.append(dict(run=run.name,model=a.get('model'),role=a.get('role'),
                parameter_counts=counts,parameter_bytes=a.get('parameter_bytes'),
                quantization=q,status=a.get('status'),source=str(p.relative_to(OUT)),
                bf16_only=bool(counts) and set(counts)=={'torch.bfloat16'},
                unquantized=not any(q.get(k) for k in ['is_loaded_in_4bit','is_loaded_in_8bit','is_quantized','quantized_modules','config_quantization'])))
dump('extension_all_summaries.json',summary)
table('extension_run_inventory.csv',inventory)
table('receipt_integrity.csv',checks)
dump('dtype_audits.json',audits)
assert all(r['png_hashes_valid'] and r['bounds_pass'] and r['no_target_access'] for r in checks)
assert all(a['bf16_only'] and a['unquantized'] and a['status']=='PASS' for a in audits)
# Full structured evidence preserves all versions, raw answers, losses and attempts.
suffixes={'.json','.jsonl','.csv','.log','.md','.yaml','.yml','.txt'}
sources=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in suffixes]
manifest=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(sources)]
dump('evidence_files.json',manifest)
with tarfile.open(DEST/'structured_evidence.tar.gz','w:gz') as t:
    for p in sorted(sources):t.add(p,arcname=str(p.relative_to(OUT)))
# Small honest example packet, chosen by ID/order, not by target success.
examples=DEST/'examples';examples.mkdir(exist_ok=True)
for method in ['co_attack','iau_foa']:
    candidates=[r for r in inventory if r['method']==method and not r['smoke']]
    if not candidates:continue
    run=OUT/'extension/runs'/candidates[0]['run']
    p=next(iter(sorted(run.glob('attacks/*/receipt.json'))),None)
    if p is None:continue
    r=json.loads(p.read_text())
    for kind in ['clean','adv']:shutil.copyfile(r[kind],examples/f'{method}_{kind}.png')
    (examples/f'{method}_receipt.json').write_text(json.dumps(r,indent=2)+'\n')
paused=json.loads((OUT/'extension/paused_by_user.json').read_text())
dump('snapshot.json',dict(state='PAUSED_BY_USER',pause=paused,
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    receipt_count=len(checks),audits=len(audits),all_checks_pass=True,
    formal_completed_slots=9,formal_planned_slots=63,
    note='All historical and revised source versions preserved. No new inference in this export.'))
# Full local archive: images + all metadata. Outside source tree, on persistent root.
backup=OUT.parent/'archives';backup.mkdir(exist_ok=True)
full=backup/'kdd2027_baselines_paused_2026-11-17.tar.gz'
with tarfile.open(full,'w:gz') as t:t.add(OUT,arcname=OUT.name)
dump('local_archive.json',dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)))
print(json.dumps(dict(receipts=len(checks),audits=len(audits),extension_runs=len(inventory),
                     structured_bytes=(DEST/'structured_evidence.tar.gz').stat().st_size,
                     local_archive_bytes=full.stat().st_size,latest_inventory=inventory[-1:]),indent=2))
