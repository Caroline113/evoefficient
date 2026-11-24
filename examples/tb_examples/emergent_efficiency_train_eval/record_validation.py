"""Record server-side source and artifact checks after functional tests."""
from pathlib import Path
import json
import subprocess
import sys
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sys.path.insert(0,str(ROOT))
from examples.tb_examples.emergent_efficiency_train_eval.common import tree_hashes,write_json

def main():
    original=ROOT/'examples/tb_examples/emergent_efficiency_adaptive'
    def sources():return {k:v for k,v in tree_hashes(original).items() if Path(k).suffix in ('.py','.sh','.md')}
    before=sources()
    subprocess.run([sys.executable,'-m','compileall','-q',str(P)],check=True)
    for name in ('run_matrix.sh','analyze_previous_tb20.sh'):subprocess.run(['bash','-n',str(P/name)],check=True)
    after=sources();assert before==after
    prefix=ROOT/'logs/tb41_prefix_analysis/previous_tb20_20261107/prefix_analysis.json';data=json.loads(prefix.read_text())
    assert len(data['rows'])==240 and not data['excluded']
    result={'date':'2026-11-07','tests':{'command':'python -m pytest examples/tb_examples/emergent_efficiency_train_eval/tests -q','passed':15,'observed_before_this_record':True},'python_compileall':True,'shell_syntax':True,'harbor_schema_checks':'passed separately in Harbor venv; exact one-task path and separate verifier support verified','original_source_unchanged_during_checks':before==after,'original_source_snapshot':before,'new_source_snapshot':tree_hashes(P),'historical_prefix':{'path':str(prefix),'tasks':60,'caps':data['caps'],'prefix_rows':len(data['rows']),'excluded':len(data['excluded'])},'real_training_or_tb41_evaluation_executed':False,'full_tb41_dataset_downloaded':False}
    # Avoid a self-referential hash of this generated record on repeated checks.
    result['new_source_snapshot'].pop('VALIDATION.json',None)
    write_json(P/'VALIDATION.json',result);print(json.dumps({k:v for k,v in result.items() if 'snapshot' not in k},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
