"""Synthetic PhysX shape coverage and canonical material evidence."""

from copy import deepcopy

from visualnav_transformer.eat_nav.p1_g3_material_diagnostic import map_effective_materials


def fixture():
    links=[]; collisions=[]; tensor=[]
    for env in range(4):
        base=f"/World/envs/env_{env}/robot"
        links.append([{"path":base+"/body", "physx_shape_count":2},
                      {"path":base+"/flipper", "physx_shape_count":1}])
        collisions.extend({"env_id":env, "path":base+"/body/collisions/"+name,
                           "shape_type":"Cylinder", "binding_status":"unbound"}
                          for name in ("a", "b"))
        collisions.append({"env_id":env, "path":base+"/flipper/collisions/a",
                           "shape_type":"Cube", "binding_status":"bound"})
        tensor.append([[1.0,1.0,0.0]]*3)
    return links,collisions,tensor


def test_unbound_colliders_have_complete_effective_mapping_when_link_uniform():
    links,collisions,tensor=fixture()
    result=map_effective_materials(links,collisions,tensor,[(1.,1.)]*2+[(0.,0.)])
    assert result["coverage_pass"]
    assert result["startup_ranges_pass"]
    assert result["collision_count"]==12
    assert result["unbound_collision_count"]==8
    assert len(set(result["canonical_fingerprint_by_env"].values()))==1
    assert all(row["effective_material"]==[1.,1.,0.] for row in result["mapped_collisions"])


def test_nonuniform_multishape_link_is_ambiguous_and_fails_closed():
    links,collisions,tensor=fixture()
    tensor[2][1]=[0.8,0.6,0.0]
    result=map_effective_materials(links,collisions,tensor,[(1.,1.)]*2+[(0.,0.)])
    assert not result["coverage_pass"]
    assert result["ambiguous_link_count"]==1
    assert not result["startup_ranges_pass"]


def test_collision_count_or_link_mapping_gap_fails_closed():
    links,collisions,tensor=fixture()
    missing=deepcopy(collisions)
    missing.pop(0)
    result=map_effective_materials(links,missing,tensor,[(1.,1.)]*2+[(0.,0.)])
    assert not result["coverage_pass"]
    assert result["unmatched_tensor_shape_count"]==1
    extra=deepcopy(collisions)
    extra.append({"env_id":0,"path":"/World/envs/env_0/unknown/collisions/a",
                  "shape_type":"Cube","binding_status":"unbound"})
    result=map_effective_materials(links,extra,tensor,[(1.,1.)]*2+[(0.,0.)])
    assert not result["coverage_pass"]
    assert result["unmapped_collision_count"]==1
