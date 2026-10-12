"""The v4 inner host retains frozen dynamics and records an exact GPU bracket."""
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/eat_nav_p1_g3_crossover_v4_inner_host.py'
V3 = SOURCE.with_name('eat_nav_p1_g3_crossover_v3_host.py')


def test_v4_records_both_gpu_wall_boundaries_and_uses_terminal_contract():
    source = SOURCE.read_text()
    assert "'sample_started_wall_ns': gpu['wall_ns']" in source
    assert "'sample_finished_wall_ns': gpu['process_table_after_wall_ns']" in source
    assert 'classify_gpu_bracket(' in source
    assert 'terminal_completion(' in source


def test_v4_preserves_frozen_simulator_and_control_route():
    before, after = V3.read_text(), SOURCE.read_text()
    for text in ('def build_wrapper_command(',
                 "'--headless','--input'", 'def approve_prepath(',
                 'validate_frozen_assignment(assigned, publication',
                 'validate_four_follower_final('):
        assert text in before and text in after


def test_v4_router_identity_can_use_open_fd_if_original_log_path_was_deleted(tmp_path):
    from eat_nav_p1_g3_crossover_v4_inner_host import router_log_bytes
    fd_dir = tmp_path/'proc/10/fd'
    fd_dir.mkdir(parents=True)
    (fd_dir/'1').write_bytes(b'original router stdout')
    assert router_log_bytes(tmp_path/'deleted.log', 10, 1,
                            proc_root=tmp_path/'proc') == b'original router stdout'
