"""Versioned contract and prospective logging-pair consistency checks."""

import hashlib
import json
from pathlib import Path

import yaml

from visualnav_transformer.eat_nav.portable_input_v2 import field_order_hash, field_specs


ROOT = Path(__file__).resolve().parents[2] / "docs/quality-identifiability-gate/portable-input-v2-parity"


def test_three_contracts_match_fixture_field_orders_and_do_not_claim_physical_24j():
    names = {"Core": "portable-core-contract-v2.json",
             "Core+2S": "portable-core-2s-contract-v2.json",
             "Core+24J": "simulator-core-24j-ablation-contract.json"}
    hashes = json.loads((ROOT / "field-order-hashes.json").read_text())
    for variant, filename in names.items():
        contract = json.loads((ROOT / filename).read_text())
        assert contract["field_order_hash"] == hashes[variant]["hash"] == field_order_hash(variant)
        assert [(f["name"], tuple(f["shape"])) for f in contract["fields"]] == list(field_specs(variant))
        assert contract["status"] == "proposal_not_frozen"
    assert json.loads((ROOT / names["Core+24J"]).read_text())["physical_24j"] == "unavailable_by_design"


def test_logging_pair_path_and_checkpoint_identity_are_frozen_without_execution():
    config = yaml.safe_load((ROOT / "logging-parity-execution-proposal.yaml").read_text())
    pair = config["pair"]
    assert hashlib.sha256(Path(pair["map_path_file"]).read_bytes()).hexdigest() == pair["map_path_sha256"]
    assert hashlib.sha256(Path(pair["checkpoint"]).read_bytes()).hexdigest() == pair["checkpoint_sha256"]
    assert hashlib.sha256(Path(pair["source_summary_path"]).read_bytes()).hexdigest() == pair["source_summary_sha256"]
    reference = json.loads(Path(pair["source_summary_path"]).read_text())["source_sha256"]
    assert all(reference[k] == v for k, v in pair["source_sha256"].items())
    assert pair["run_order"] == ["logging_off", "logging_on"]
    assert config["execution_requires_separate_approval"] is True
    assert config["status"] == "proposal_only_not_ready_for_execution"


def test_selected_bag_bodies_and_metadata_are_sha_recorded():
    provenance = json.loads((ROOT / "source-provenance.json").read_text())
    assert provenance["sealed_trace_count"] == 63
    assert len(provenance["selected_bags"]) == 3
    assert provenance["bag_inventory_count"] == 70
    assert all(len(x["bag_sha256"]) == len(x["metadata_sha256"]) == 64 for x in provenance["selected_bags"])
