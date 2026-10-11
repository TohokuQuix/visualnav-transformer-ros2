"""Static contracts for the material audit and balanced Path crossover."""

from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest

from visualnav_transformer.eat_nav.p1_path_crossover_v2 import (
    build_crossover, canonical_material, fingerprint_material,
    reassess_direct_repeat, material_values_from_source, read_usd_material_values,
)


def test_material_fingerprint_ignores_callable_address_and_is_strict():
    fields = dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                  friction_combine_mode="multiply", restitution_combine_mode="multiply",
                  compliant_contact_stiffness=0.0, compliant_contact_damping=0.0)
    a = SimpleNamespace(**fields, func=lambda: None)
    b = SimpleNamespace(**fields, func=lambda: None)
    assert str(a) != str(b)
    assert fingerprint_material(a) == fingerprint_material(b)
    assert canonical_material(a) == fields
    b.dynamic_friction = 0.6
    assert fingerprint_material(a) != fingerprint_material(b)
    del b.restitution
    with pytest.raises(ValueError, match="missing"):
        canonical_material(b)


def test_crossover_has_each_path_once_per_seed_env_and_balanced_periods():
    seeds = {"first-differential": (1, 2, 3), "second-graded": (11, 12, 13)}
    schedule = build_crossover(seeds)
    assert len(schedule["processes"]) == 18
    assert sum(len(p["trials"]) for p in schedule["processes"]) == 72
    by_condition = defaultdict(list)
    for process in schedule["processes"]:
        assert len(process["trials"]) == 4
        for row in process["trials"]:
            by_condition[(process["state_group"], process["seed"], row["env_id"])].append(row["path_role"])
    assert len(by_condition) == 24
    assert all(set(roles) == {"direct", "left_20", "right_20"} for roles in by_condition.values())
    assert Counter(row["path_role"] for p in schedule["processes"] for row in p["trials"]) == {
        "direct": 24, "left_20": 24, "right_20": 24}
    direct_duplicate_pairs = sum(
        all(p["trials"][env]["path_role"] == "direct" for env in pair)
        for p in schedule["processes"] for pair in ((0, 1), (2, 3)))
    assert direct_duplicate_pairs == 12


def test_reassess_preserves_legacy_sha_and_matches_24_seed_env_pairs():
    rows = []
    batches = []
    for group, base in (("first-differential", 1), ("second-graded", 11)):
        for seed in range(base, base + 3):
            for repeat in range(2):
                batches.append({"anchor_id": group, "batch_seed": seed,
                                "mesh_sha256": group, "material_sha256": f"legacy-{group}-{seed}-{repeat}"})
                for env in range(4):
                    rows.append({"anchor_id": group, "batch_seed": seed, "repeat_index": repeat,
                                 "env_id": env, "terrain_condition_key": f"{group}:legacy-{repeat}",
                                 "two_second_progress_m": float(seed + env)})
    result = reassess_direct_repeat(rows, batches, "canonical-material")
    assert result["seed_env_mesh_correspondence_pairs"] == 24
    assert result["actual_material_verified_pairs"] == 0
    assert result["independent_seed_conditions"] == 6
    assert result["legacy_material_sha_count"] == 12
    assert result["terrain_mesh_match_all"] is True
    assert result["declared_material_match_all"] is None
    assert result["actual_physx_material_verified"] is False
    assert rows[0]["terrain_condition_key"].endswith("legacy-0")


def test_reassess_fails_closed_on_mesh_drift():
    rows = [{"anchor_id": "g", "batch_seed": 1, "repeat_index": r,
             "env_id": e, "terrain_condition_key": "old"}
            for r in range(2) for e in range(4)]
    batches = [{"anchor_id": "g", "batch_seed": 1, "mesh_sha256": f"mesh-{r}",
                "material_sha256": f"legacy-{r}"} for r in range(2)]
    with pytest.raises(ValueError, match="mesh drift"):
        reassess_direct_repeat(rows, batches, "canonical")


def test_source_material_extraction_uses_defaults_not_function_repr():
    cfg = '''class MySceneCfg:
    terrain = TerrainImporterCfg(physics_material=RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=1.0,
        friction_combine_mode="multiply", restitution_combine_mode="multiply"))
'''
    defaults = '''class RigidBodyMaterialCfg:
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    friction_combine_mode: str = "average"
    restitution_combine_mode: str = "average"
    compliant_contact_stiffness: float = 0.0
    compliant_contact_damping: float = 0.0
'''
    values = material_values_from_source(cfg, defaults)
    assert values["static_friction"] == 1.0
    assert values["restitution"] == 0.0
    assert values["friction_combine_mode"] == "multiply"


def test_usd_material_read_only_extractor_requires_authored_values():
    class Attribute:
        def __init__(self, value, authored=True):
            self.value, self.authored = value, authored
        def IsValid(self):
            return True
        def HasAuthoredValue(self):
            return self.authored
        def Get(self):
            return self.value
    class Prim:
        def __init__(self, attrs):
            self.attrs = attrs
        def IsValid(self):
            return True
        def GetAttribute(self, name):
            return self.attrs.get(name)
    attrs = {
        "physics:staticFriction": Attribute(1.0),
        "physics:dynamicFriction": Attribute(1.0),
        "physics:restitution": Attribute(0.0),
        "physxMaterial:frictionCombineMode": Attribute("multiply"),
        "physxMaterial:restitutionCombineMode": Attribute("multiply"),
        "physxMaterial:compliantContactStiffness": Attribute(0.0),
        "physxMaterial:compliantContactDamping": Attribute(0.0),
    }
    prim = Prim(attrs)
    assert read_usd_material_values(prim)["dynamic_friction"] == 1.0
    assert len(attrs) == 7  # the diagnostic never authors a USD attribute
    attrs["physics:dynamicFriction"].authored = False
    with pytest.raises(ValueError, match="missing"):
        read_usd_material_values(prim)
