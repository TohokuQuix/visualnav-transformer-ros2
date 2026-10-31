"""Create or run a portable frozen-seed GUI reproduction recipe; no ROS commands."""
import argparse,json,shlex,subprocess
from pathlib import Path
from eat_nav_r6_run import read,write,sha,stage_runtime,REPO


def make_recipe(job_directory,output):
 job=Path(job_directory);assignment=read(job/'assignment.json');binding=read(job.parent/'frozen-binding.json')
 actual=job/'sim-output'
 if not actual.exists():actual=Path(read(job/'equivalence-audit.json')['source'])
 manifest=read(actual/'manifest.json')
 recipe=dict(schema='eatnav_r6_gui_reproduction_v1',assignment=assignment,model_sha256=manifest['model_sha256'],actor_sha256=manifest['actor_sha256'],actor_env_sha256=manifest['actor_env_sha256'],config_sha256=manifest['config_sha256'],source_sha256=binding['source_sha256'],source_actual_output=str(actual),source_manifest_sha256=sha(actual/'manifest.json'),scope='rerun complete original batch with fixed seeds and settings; quantitative source outcomes remain in original record')
 write(output,recipe);return recipe


def run_recipe(recipe_file,checkpoint,output,focus_env=0,dry_run=False):
 recipe=read(recipe_file);output=Path(output)
 if recipe.get('schema')!='eatnav_r6_gui_reproduction_v1':raise ValueError('recipe schema mismatch')
 if sha(checkpoint)!=recipe['model_sha256']:raise ValueError('recipe/checkpoint mismatch')
 for relative,digest in recipe['source_sha256'].items():
  if sha(REPO/relative)!=digest:raise ValueError('source changed since frozen experiment: '+relative)
 for relative,key in [('checkpoints/eat_nav/rl_actor/model_10000.pt','actor_sha256'),('checkpoints/eat_nav/rl_actor/env.yaml','actor_env_sha256'),('config/eat_nav.yaml','config_sha256')]:
  if sha(REPO/relative)!=recipe[key]:raise ValueError('low-level dependency differs')
 if output.exists():raise FileExistsError('fresh GUI output required')
 output.mkdir(parents=True);binding=stage_runtime('isaac-lab-onix',output);stage=Path(binding['stage']);job=recipe['assignment'];groups=job['groups'];condition=job['condition'];mode=condition['mode']
 write(stage/'gui-groups.json',dict(groups=groups));write(stage/'gui-profile.json',condition.get('profile',{}))
 for source,name in [(stage/'gui-groups.json','gui-groups.json'),(stage/'gui-profile.json','gui-profile.json'),(Path(checkpoint),'selected.pt')]:subprocess.run(['docker','cp',str(source),'isaac-lab-onix:'+str(stage/name)],check=True)
 envpath=':'.join([str(stage),str(stage/'src'),'/workspace/isaaclab_onix/source/onix_isaac_lab','/workspace/isaaclab/source/isaaclab','/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
 cmd=['docker','exec','-e','PYTHONPATH='+envpath,'isaac-lab-onix','/isaac-sim/python.sh',str(stage/'eat_nav_r6_eval_sim.py'),'--gui','--focus-env',str(focus_env),'--output',str(stage/'gui-result'),'--terrain-type',job['field'],'--level',str(job['level']),'--num-envs',str(len(groups)),'--seed',str(groups[0]['seed']),'--parent-seeds',','.join(str(g['seed']) for g in groups),'--terrain-cols',str(len(groups)),'--groups',str(stage/'gui-groups.json'),'--mode',mode,'--profile',str(stage/'gui-profile.json'),'--checkpoint',str(stage/'selected.pt'),'--old-checkpoint',str(stage/'selected.pt'),'--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),'--eat-config',str(stage/'eat_nav.yaml')]
 write(output/'command.json',cmd);print(shlex.join(cmd),flush=True)
 if dry_run:return dict(state='prepared_only',command=cmd,gui_displayed=False)
 with (output/'simulator.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=3600).returncode
 copied=subprocess.run(['docker','cp','isaac-lab-onix:'+str(stage/'gui-result'),str(output/'sim-output')],capture_output=True,text=True)
 result=read(output/'sim-output/complete.json') if (output/'sim-output/complete.json').exists() else None
 if code or copied.returncode or result is None or (output/'sim-output/failure.json').exists():raise ValueError('GUI reproduction failed; retained command/log; do not claim visible success')
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 make=sub.add_parser('recipe');make.add_argument('--job',type=Path,required=True);make.add_argument('--output',type=Path,required=True)
 run=sub.add_parser('run');run.add_argument('--recipe',type=Path,required=True);run.add_argument('--checkpoint',type=Path,required=True);run.add_argument('--output',type=Path,required=True);run.add_argument('--focus-env',type=int,default=0);run.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 if a.mode=='recipe':make_recipe(a.job,a.output)
 else:print(json.dumps(run_recipe(a.recipe,a.checkpoint,a.output,a.focus_env,a.dry_run),indent=2))
if __name__=='__main__':main()
