import json

from eat_nav_motion_14_pipeline import run


def test_pipeline_stops_without_training_on_collection_failure(tmp_path, monkeypatch):
    collection = tmp_path/'collection'
    collection.mkdir()
    (collection/'status.json').write_text(json.dumps({'state': 'stopped', 'completed': 19}))
    monkeypatch.setattr('eat_nav_motion_14_pipeline.build',
                        lambda *_: (_ for _ in ()).throw(AssertionError('must not build')))
    training = tmp_path/'training'
    status = tmp_path/'pipeline-status.json'
    run(collection, training, status)
    assert not training.exists()
    assert json.loads(status.read_text())['state'] == 'blocked_by_collection_failure'


def test_pipeline_orders_all_training_gates_after_complete(tmp_path, monkeypatch):
    collection = tmp_path/'collection'
    collection.mkdir()
    (collection/'status.json').write_text(json.dumps({'state': 'completed', 'completed': 1152}))
    training = tmp_path/'training'
    order = []

    def build(root, _):
        order.append('build')
        root.mkdir()

    def overfit(_):
        order.append('overfit')

    def train(_):
        order.append('train')

    def evaluate(root):
        order.append('evaluate')
        (root/'results.json').write_text(json.dumps({'demo_adoption': {'decision': 'fail'}}))

    for name, action in [('build', build), ('overfit', overfit),
                         ('train', train), ('evaluate', evaluate)]:
        monkeypatch.setattr('eat_nav_motion_14_pipeline.'+name, action)
    status = tmp_path/'pipeline-status.json'
    run(collection, training, status)
    assert order == ['build', 'overfit', 'train', 'evaluate']
    assert json.loads(status.read_text())['state'] == 'completed'
