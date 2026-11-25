"""Write notebooks/negotiation_system1_kaggle.ipynb (Kaggle T4, real Zoro-1, System 1 + scenarios).

    python notebooks/make_system1_kaggle.py <content_sha256>

The content hash pins the source ZIP the notebook accepts (same formula as the quickstart notebook).
The ZIP holds every tracked and untracked non-ignored file except this generator and the generated
notebook, which embeds the hash.
"""
import json
import sys
from pathlib import Path

CONTENT = sys.argv[1] if len(sys.argv) > 1 else "PASTE_CONTENT_SHA256"

MD0 = f"""# Negotiation · System 1과 출품 시나리오 3종 · 실제 Zoro-1 (Kaggle T4)

설정: **Accelerator = GPU T4**, **Internet = On**. Hugging Face 인증은 **Add-ons → Secrets → `HF_TOKEN`** 또는 2번 셀의 로그인 칸(Zoro-1 접근이 승인된 본인 계정).
입력: 소스 ZIP(`negotiation-source-worktree-{CONTENT[:12]}.nszip`, 내용은 ZIP 그대로)을 Kaggle 데이터셋으로 올려 이 노트북의 Input에 추가합니다.

실행 내용: 모델 준비(FP16, 판단 온도 1.0) → System 1 질문 14개 확인 → 시나리오 2 seed 선택(1000–1099, 규칙은 코드에 고정) → 시나리오 3종 → 결과 ZIP. 각 실험은 1회 실행이며 정확도 측정이 아닙니다. 승인·검토는 합성 fixture가 처리하며 사람 승인이 아닙니다."""

C1 = f"""# 1. 소스 ZIP 확인 · 실행 폴더 준비
import hashlib, json, os, stat, subprocess, sys, uuid, zipfile, glob
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

NEG_CONTENT_HASH = '{CONTENT}'
found = sorted(glob.glob('/kaggle/input/**/negotiation-source-worktree-*.nszip', recursive=True)
               + glob.glob('/kaggle/input/**/negotiation-source-worktree-*.zip', recursive=True))
# Kaggle unpacks .zip uploads in datasets, so the archive is uploaded as .nszip (same ZIP bytes).
if len(found) != 1:
    raise RuntimeError(f'Input에 소스 ZIP이 정확히 한 개 있어야 합니다: {{found}}')
neg_archive = Path(found[0])
NEG_REPO = Path('/kaggle/temp') / ('negotiation-' + uuid.uuid4().hex[:12])
with zipfile.ZipFile(neg_archive) as bundle:
    members = [m for m in bundle.infolist() if not m.is_dir()]
    roots = [m.filename[:-len('pyproject.toml')] for m in members
             if m.filename == 'pyproject.toml' or m.filename.endswith('/pyproject.toml')]
    if len(roots) != 1 or sum(m.file_size for m in members) > 32 * 1024**2:
        raise ValueError('예상한 프로젝트 ZIP이 아닙니다.')
    prefix, contents = roots[0], {{}}
    for member in members:
        name = member.filename[len(prefix):]
        path = PurePosixPath(name)
        if (not member.filename.startswith(prefix) or not name or '\\\\' in name or ':' in name
                or path.is_absolute() or '..' in path.parts or name in contents
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError('ZIP에 허용하지 않는 경로가 있습니다.')
        contents[name] = bundle.read(member)
content_hash = hashlib.sha256(''.join(
    f'{{n}}\\0{{hashlib.sha256(contents[n]).hexdigest()}}\\n' for n in sorted(contents)).encode()).hexdigest()
if content_hash != NEG_CONTENT_HASH:
    raise ValueError(f'고정한 소스와 다릅니다: {{content_hash}}')
NEG_REPO.mkdir(parents=True)
for name, data in contents.items():
    target = NEG_REPO / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
del contents
NEG_RUN = NEG_REPO / 'runs' / 'negotiation' / ('kaggle-system1-' + uuid.uuid4().hex[:8])
NEG_RUN.mkdir(parents=True)
(NEG_RUN / 'source.json').write_text(json.dumps({{
    'content_sha256': content_hash, 'zip_sha256': hashlib.sha256(neg_archive.read_bytes()).hexdigest(),
    'zip_name': neg_archive.name, 'created_at': datetime.now(timezone.utc).isoformat(),
    'platform': 'Kaggle', 'notice': 'uncommitted worktree source; base commit is in the ZIP manifest'}}, indent=2), encoding='utf-8')
NEG_PYPATH = os.pathsep.join([str(NEG_REPO / 'src'), str(NEG_REPO / 'negotiation/src')])

def neg_run(*args):
    command = [sys.executable, '-u', *map(str, args)]
    log = NEG_RUN / ('command-' + uuid.uuid4().hex[:8] + '.log')
    print('실행:', ' '.join(command), flush=True)
    env = {{**os.environ, 'PYTHONPATH': NEG_PYPATH, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}}
    started = datetime.now(timezone.utc)
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, cwd=NEG_REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            output.write(line); output.flush(); print(line, end='', flush=True)
        exit_code = process.wait()
    seconds = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    with (NEG_RUN / 'commands.jsonl').open('a', encoding='utf-8') as history:
        history.write(json.dumps({{'argv': command, 'exit_code': exit_code, 'log': log.name, 'seconds': seconds}}) + '\\n')
    if exit_code:
        raise RuntimeError(f'실행 실패(exit={{exit_code}}). {{log.name}}을 확인하세요.')

print('소스 확인 완료:', content_hash[:12], '| 결과 폴더:', NEG_RUN)
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv'])"""

C2 = """# 2. 실행 도구 설치 · Hugging Face 인증
# Kaggle Secret HF_TOKEN이 연결되어 있으면 다음 셀의 gpu_setup.py가 직접 읽습니다.
# 없으면 아래 로그인 칸에 Zoro-1 접근이 승인된 본인 토큰을 넣으세요. 토큰을 코드에 적지 마세요.
neg_run('scripts/gpu_setup.py', '--steps', 'env,install', '--run-dir', NEG_RUN / 'setup-install')
from huggingface_hub import get_token, login
try:
    from kaggle_secrets import UserSecretsClient
    neg_has_secret = bool(UserSecretsClient().get_secret('HF_TOKEN'))
except Exception:
    neg_has_secret = False
if not neg_has_secret and not get_token():
    login(add_to_git_credential=False)
print('인증 방식:', 'Kaggle Secret HF_TOKEN' if neg_has_secret else ('저장된 로그인' if get_token() else '없음 - 로그인 후 다음 셀 실행'))"""

C3 = """# 3. 고정된 Zoro-1 다운로드 · 변환 (FP16)
NEG_CACHE = Path('/kaggle/temp/negotiation-model-cache')
neg_run('scripts/gpu_setup.py', '--steps', 'sources,model,export', '--dtype', 'fp16',
        '--work-dir', NEG_CACHE, '--run-dir', NEG_RUN / 'setup-model')
NEG_PATHS = json.loads((NEG_REPO / '.cache/gpu_paths.json').read_text(encoding='utf-8'))
NEG_ARGS = ['--backend', 'zoro', '--original-checkpoint', NEG_PATHS['original_checkpoint'],
            '--exported-checkpoint', NEG_PATHS['exported_checkpoint'],
            '--official-source-dir', NEG_PATHS['official_source_dir'],
            '--device', 'cuda:0', '--dtype', 'fp16', '--max-input-tokens', '2048', '--decision-temperature', '1.0']
print('모델 준비 완료')"""

C4 = """# 4. System 1 질문 14개 (진단·라우팅 9 + 수락 검증 5), 1회
neg_run('negotiation/run_system1_probe.py', *NEG_ARGS, '--output', NEG_RUN / 'system1_probe.json')
probe = json.loads((NEG_RUN / 'system1_probe.json').read_text(encoding='utf-8'))
print('complete:', probe['complete'], '| evidence:', probe['evidence_class'], '| build_s:', probe['backend_build_seconds'])
for r in probe['rows']:
    print(r['kind'][:5], r['label'].ljust(20), r['status'], r.get('route') or r.get('verified'),
          {k: round(v, 3) for k, v in (r.get('answers') or {}).items()}, r.get('latency_ms'), 'ms')"""

C4B = """# 4-2. 시나리오 2 seed 선택 (1000–1099 전부, 규칙: scenarios.py S2_SWEEP_RULE_VERSION), 1회
# 승인 카드 → 승인 → 그 조건 그대로 합의 → 공급사 수락 검증까지 나온 가장 작은 seed. 평가 결과가 아닙니다.
neg_run('negotiation/run_s2_sweep.py', *NEG_ARGS, '--output', NEG_RUN / 's2_sweep.json')
sweep = json.loads((NEG_RUN / 's2_sweep.json').read_text(encoding='utf-8'))
NEG_S2_SEED = sweep['chosen_seed']
print('rule:', sweep['rule_version'], '| chosen:', NEG_S2_SEED, '| qualifying:', len(sweep['qualifying_seeds']),
      '| with card:', len(sweep['seeds_with_approval_card']), '| errors:', sum(r['status'] == 'error' for r in sweep['rows']))"""

C5 = """# 5. 출품 시나리오 3종 (합성 fixture 승인·검토), 1회. 시나리오 2는 4-2에서 고른 seed (없으면 고정값 그대로)
neg_s2 = ['--s2-seed', str(NEG_S2_SEED)] if NEG_S2_SEED is not None else []
neg_run('negotiation/run_scenarios.py', *NEG_ARGS, *neg_s2, '--output-dir', NEG_RUN / 'scenarios')
for f in sorted((NEG_RUN / 'scenarios').glob('*.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    print(f.name, json.dumps(d.get('summary', {}), ensure_ascii=False)[:600])"""

C6 = """# 6. 결과 ZIP (/kaggle/working 에 저장 → 오른쪽 Output에서 내려받기)
import shutil
out = shutil.make_archive('/kaggle/working/' + NEG_RUN.name + '-results', 'zip', root_dir=NEG_RUN)
print('결과 ZIP:', out, round(Path(out).stat().st_size / 1024), 'KB')
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv'])"""


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {"cells": [cell("markdown", MD0)] + [cell("code", c) for c in (C1, C2, C3, C4, C4B, C5, C6)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "negotiation_system1_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
