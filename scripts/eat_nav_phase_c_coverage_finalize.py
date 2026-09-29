"""Detached conditional Stage 5/6 runner and immutable coverage report."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np

ROOT=Path('/tmp/eatnav-quality-identifiability-gate/phase-c-coverage')
DOC=Path('/tmp/eatnav-vnt/docs/quality-identifiability-gate/phase-c-coverage')
REPO=Path('/tmp/eatnav-vnt')
CONTAINER='isaac-lab-onix'
DROOT='/tmp/eatnav-quality-identifiability-gate/phase-c-coverage'
PRIOR='/tmp/eatnav-quality-identifiability-gate/phase-c-remediation'
PY='/isaac-sim/python.sh'
ENV='PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');t.replace(p)


def cmd(args,log):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('a') as f:
        f.write('COMMAND '+json.dumps(args)+'\n');f.flush()
        completed=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError(f'exit {completed.returncode}: {args}; log {log}')


def docker_python(script,args,log):
    cmd(['docker','exec','-e',ENV,CONTAINER,PY,'/tmp/eatnav-train/'+script,*map(str,args)],log)


def copy_from_docker():
    cmd(['docker','cp',f'{CONTAINER}:{DROOT}/.',str(ROOT)],ROOT/'runner.log')


def marked(path,outputs):
    path=Path(path)
    if not path.exists():return False
    row=json.loads(path.read_text())
    return all(Path(p).exists() and sha(p)==h for p,h in row['outputs_sha256'].items()) and set(map(str,outputs))==set(row['outputs_sha256'])


def wait_for_core():
    while not (ROOT/'structured-coverage-decision.json').exists():
        running=subprocess.run(['tmux','has-session','-t','eatnav-phase-c-coverage'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if not running:
            raise RuntimeError('core runner ended without structured coverage decision; inspect runner.log')
        log=(ROOT/'runner.log').read_text() if (ROOT/'runner.log').exists() else ''
        import re
        matches=re.findall(r'FOLLOWER_NEIGHBORHOOD (\d+) (\d+)',log)
        stage='sampling' if 'STAGE2/3 trace and sampling' in log else 'teacher_audit'
        status={'stage':stage,'formal_phase_c_allowed':False,'time_unix':time.time()}
        if matches and stage=='teacher_audit':
            status.update({'completed_neighborhoods':int(matches[-1][0]),'total_neighborhoods':int(matches[-1][1])})
        atomic(ROOT/'status.json',status)
        time.sleep(30)
    copy_from_docker()
    for name in ('teacher-audit.done.json','sampling.done.json','structured-coverage-decision.json'):
        if not (ROOT/name).exists():raise ValueError(f'incomplete core stage: {name}')


def prepare_conditional_sources():
    names=('eat_nav_phase_c_coverage_score.py','eat_nav_phase_c_coverage_sim.py',
           'eat_nav_phase_c_coverage_serial_analyze.py','eat_nav_phase_c_coverage_plots.py',
           'eat_nav_phase_c_coverage_teacher_details.py')
    hashes={name:sha(REPO/'scripts'/name) for name in names}
    frozen=json.loads((DOC/'conditional-source-provenance.json').read_text())
    if frozen['source_sha256']!=hashes or frozen['finalizer_source_sha256']!=sha(Path(__file__)):
        raise ValueError('conditional source changed after freeze')
    for name in names:cmd(['docker','cp',str(REPO/'scripts'/name),f'{CONTAINER}:/tmp/eatnav-train/{name}'],ROOT/'runner.log')


def teacher_details_stage():
    docker_python('eat_nav_phase_c_coverage_teacher_details.py',
        ['--manifest','/tmp/eatnav-extended/output/manifest.json',
         '--reference-config','/tmp/eatnav-phase-c/reference-eat.yaml','--output',DROOT],ROOT/'teacher-details.log')
    cmd(['docker','cp',f'{CONTAINER}:{DROOT}/teacher-multimodality-details.json',str(ROOT/'teacher-multimodality-details.json')],ROOT/'runner.log')


def score_stage():
    docker_python('eat_nav_phase_c_coverage_score.py',
        ['--manifest','/tmp/eatnav-extended/output/manifest.json','--reference-config','/tmp/eatnav-phase-c/reference-eat.yaml',
         '--coverage-config','/tmp/eatnav-train/coverage-config.yaml','--output',DROOT],ROOT/'stage5-score.log')
    cmd(['docker','cp',f'{CONTAINER}:{DROOT}/evaluator-score-spread.json',str(ROOT/'evaluator-score-spread.json')],ROOT/'runner.log')


def serial_stage():
    sampling=json.loads((ROOT/'sampling-coverage-decision.json').read_text())
    structured=json.loads((ROOT/'structured-coverage-decision.json').read_text())
    if sampling['decision']!='pass' and structured['decision']!='pass':
        atomic(ROOT/'serial-smoke-results.json',{'status':'not_run','reason':'offline_coverage_failed'})
        return
    if not Path('/tmp/eatnav-phase-c-coverage-bridge.log').exists():raise RuntimeError('bridge log missing')
    states=json.loads((ROOT/'diagnostic-state-ids.json').read_text())['train'][:8]
    plan=[]
    for i,state in enumerate(states):
        order=np.random.default_rng(20260929+i).permutation(8).tolist()
        plan.append({'state_index':i,'terrain_type':state['terrain_type'],'terrain_seed':state['terrain_seed'],
                     'episode_key':state['episode_key'],'candidate_order':order})
    plan_path=ROOT/'serial-order.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('serial candidate order changed')
    else:atomic(plan_path,plan)
    structured_opt=['--structured-basis',DROOT+'/residual-basis.npz'] if structured['decision']=='pass' else []
    for row in plan:
        i=row['state_index'];folder=ROOT/'serial'/f'state-{i:02d}';folder.mkdir(parents=True,exist_ok=True)
        container_folder=DROOT+f'/serial/state-{i:02d}'
        cmd(['docker','exec',CONTAINER,'mkdir','-p',container_folder],ROOT/'runner.log')
        base=['--seed',str(row['terrain_seed']),'--terrain-type',row['terrain_type'],'--difficulty','0.5',
              '--recorded',PRIOR+'/sim/duplicate.npz','--reference',PRIOR+'/reference/best.pt',
              '--reference-config','/tmp/eatnav-phase-c/reference-eat.yaml','--output',container_folder,
              '--headless','--resume',*structured_opt]
        def sim(case,repeat,candidate=None):
            opts=['--case',case,'--repeat-id',str(repeat)]
            if candidate is not None:
                opts.extend(['--candidate-index',str(candidate),'--fixed-candidate-json',container_folder+'/serial_duplicate-0.json'])
            docker_python('eat_nav_phase_c_coverage_sim.py',[*base,*opts],folder/f'{case}-{repeat}.log')
        sim('serial_duplicate',0)
        for candidate in row['candidate_order']:sim('serial_candidate',candidate,candidate)
        cmd(['docker','cp',f'{CONTAINER}:{container_folder}/.',str(folder)],ROOT/'runner.log')
        outputs=[folder/'serial_duplicate-0.json',folder/'serial_duplicate-0.npz']
        for candidate in range(8):outputs.extend([folder/f'serial_candidate-{candidate}.json',folder/f'serial_candidate-{candidate}.npz'])
        atomic(folder/'state.done.json',{'state':row,'outputs_sha256':{str(p):sha(p) for p in outputs}})
        atomic(ROOT/'status.json',{'stage':'serial_smoke','completed_states':i+1,'total_states':8,'formal_phase_c_allowed':False})
    docker_python('eat_nav_phase_c_coverage_serial_analyze.py',
        ['--output',DROOT,'--coverage-config','/tmp/eatnav-train/coverage-config.yaml'],ROOT/'serial-analysis.log')
    cmd(['docker','cp',f'{CONTAINER}:{DROOT}/serial-smoke-results.json',str(ROOT/'serial-smoke-results.json')],ROOT/'runner.log')


def finalize():
    copy_from_docker()
    sampling=json.loads((ROOT/'sampling-coverage-decision.json').read_text())
    structured=json.loads((ROOT/'structured-coverage-decision.json').read_text())
    serial=json.loads((ROOT/'serial-smoke-results.json').read_text())
    teacher=json.loads((ROOT/'teacher-multimodality.json').read_text())
    if sampling['decision']=='pass' or structured['decision']=='pass':
        serial_effect=serial.get('status')=='complete' and sum(r['rho_below_0_25'] for r in serial['rows'])>=2
        decision='A_sampling_coverage' if sampling['decision']=='pass' and serial_effect else 'B_structured_coverage' if structured['decision']=='pass' and serial_effect else 'D_offline_coverage_but_serial_effect_unconfirmed'
    else:decision='C_both_offline_coverage_fail'
    record={'decision':decision,'data_multimodality':teacher.get('classification'),
            'sampling':sampling['decision'],'structured':structured['decision'],
            'serial':serial.get('status'),'formal_phase_c_started':False,
            'heldout_used':False,'requires_separate_formal_approval':True}
    atomic(ROOT/'coverage-decision.json',record)
    if not (ROOT/'residual-basis.npz').exists():np.savez(ROOT/'residual-basis.npz',status='not_run',reason='sampling_coverage_pass')
    docker_python('eat_nav_phase_c_coverage_plots.py',[DROOT],ROOT/'plots.log')
    (ROOT/'figures').mkdir(exist_ok=True)
    cmd(['docker','cp',f'{CONTAINER}:{DROOT}/figures/.',str(ROOT/'figures')],ROOT/'runner.log')
    figures=[p for p in sorted((ROOT/'figures').glob('*.png'))]
    report=DOC/'phase-c-coverage-report.md'
    lines=['# Phase C candidate coverage diagnostic','',f"Final decision: **{decision}**.",'',
           f"Teacher neighborhoods: {teacher.get('neighborhood_count')} / {teacher.get('anchor_count')} anchors; data classification: {teacher.get('classification')}.",
           f"Sampling: {sampling['decision']} (selected: {sampling.get('selected')}).",
           f"Structured proposal: {structured['decision']}.",
           f"Serial smoke: {serial.get('status')}; rho<0.25 in {sum(r['rho_below_0_25'] for r in serial.get('rows',[]))} of {len(serial.get('rows',[]))} states.",'',
           'No formal Phase C, model retraining, full collection, Phase D/E, or real robot run was performed.',
           'Validation diagnostic states were evaluated without selection; held-out data was excluded.',
           '', '## Artifacts','',
           '- Frozen config: [coverage-config.yaml](coverage-config.yaml)',
           '- Input provenance: [source-provenance.json](source-provenance.json)',
           '- Detailed teacher audit: [teacher-multimodality-details.json](teacher-multimodality-details.json)',
           '- Execution status: `/tmp/eatnav-quality-identifiability-gate/phase-c-coverage/status.json`',
           '- Full machine outputs: `/tmp/eatnav-quality-identifiability-gate/phase-c-coverage/`','',
           '## Figures','']
    lines.extend(f'- [{p.stem}](figures/{p.name})' for p in figures)
    DOC.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n')
    for name in ('source-provenance.json','diagnostic-state-ids.json','teacher-multimodality.json','sampling-coverage-decision.json',
                 'structured-coverage-decision.json','teacher-multimodality-details.json','evaluator-score-spread.json','serial-smoke-results.json','coverage-decision.json'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,DOC/name)
    (DOC/'figures').mkdir(exist_ok=True)
    for p in figures:shutil.copy2(p,DOC/'figures'/p.name)
    files=[p for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name not in ('runner.log','status.json','output-artifact-hashes.json')]
    atomic(ROOT/'output-artifact-hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    atomic(ROOT/'status.json',{'stage':'complete','decision':decision,'formal_phase_c_allowed':False,'time_unix':time.time()})
    shutil.copy2(ROOT/'output-artifact-hashes.json',DOC/'output-artifact-hashes.json')
    shutil.copy2(ROOT/'status.json',DOC/'status.json')


def main():
    ROOT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    try:
        wait_for_core()
        prepare_conditional_sources()
        teacher_details_stage()
        score_stage()
        serial_stage()
        finalize()
    except Exception as e:
        atomic(ROOT/'status.json',{'stage':'failed','error':repr(e),'traceback':traceback.format_exc(),
            'formal_phase_c_allowed':False,'time_unix':time.time()})
        raise

if __name__=='__main__':main()
