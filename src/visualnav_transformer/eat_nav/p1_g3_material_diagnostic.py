"""Read-only P1 material tensor/CollisionAPI coverage audit; no Isaac dependency."""

from __future__ import annotations

import hashlib
import json
import math
import re


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def map_effective_materials(links_by_env: list[list[dict]], collisions: list[dict],
                            tensor, startup_ranges: list[tuple[float,float]]) -> dict:
    """Map tensor slices by ArticulationView link order and per-link shape counts.

    Isaac Lab's startup material event indexes the ArticulationView tensor in
    exactly this link order. Its public API gives per-link shape *counts*, not
    per-shape USD paths. Multi-shape links are attributable only when all
    their effective material triples agree; otherwise fail closed.
    """
    data=tensor.tolist() if hasattr(tensor,"tolist") else tensor
    if len(links_by_env)!=4 or not isinstance(data,list) or len(data)!=4 or \
            len(startup_ranges)!=3:
        raise ValueError("four-env material tensor/links/ranges required")
    shape_count=len(data[0])
    if shape_count<=0 or any(not isinstance(env,list) or len(env)!=shape_count for env in data):
        raise ValueError("material tensor shape invalid")
    for env in data:
        for shape in env:
            if not isinstance(shape,list) or len(shape)!=3 or any(
                    isinstance(value,bool) or not isinstance(value,(float,int)) or
                    not math.isfinite(value) for value in shape):
                raise ValueError("nonfinite/invalid PhysX material tensor")
    mapped=[];gaps=[];ambiguous=[];used=set();fingerprints={}
    for env_id,links in enumerate(links_by_env):
        offset=0;env_rows=[]
        if len({row["path"] for row in links})!=len(links):
            raise ValueError("duplicate PhysX link path")
        for link in links:
            path=link["path"];count=link["physx_shape_count"]
            if not isinstance(count,int) or count<0:
                raise ValueError("invalid per-link PhysX shape count")
            # A nested rigid body must own its descendants. Prefer longest link.
            related=[row for row in collisions if row["env_id"]==env_id and
                     row["path"].startswith(path+"/") and
                     not any(len(other["path"])>len(path) and
                             other["path"].startswith(path+"/") and
                             row["path"].startswith(other["path"]+"/")
                             for other in links)]
            selected=sorted(related,key=lambda row:row["path"])
            values=data[env_id][offset:offset+count]
            if len(selected)!=count or len(values)!=count:
                gaps.append({"env_id":env_id,"link_path":path,"usd_collision_count":len(selected),
                             "physx_shape_count":count,"tensor_start":offset})
            uniform=bool(values) and all(row==values[0] for row in values)
            if count>1 and not uniform:
                ambiguous.append({"env_id":env_id,"link_path":path,"tensor_start":offset,
                                  "physx_shape_count":count})
            if len(selected)==count and (count==1 or uniform):
                for row in selected:
                    key=row["path"]
                    if key in used:raise ValueError("collision assigned twice")
                    used.add(key)
                    item={**row,"physx_link_path":path,
                          "physx_tensor_index_range":[offset,offset+count],
                          "correspondence":"single_shape_exact" if count==1 else "link_set_uniform",
                          "effective_material":list(values[0])}
                    mapped.append(item);env_rows.append(item)
            offset+=count
        if offset!=shape_count:
            gaps.append({"env_id":env_id,"reason":"link_shape_sum_vs_tensor",
                         "link_shape_sum":offset,"tensor_shape_count":shape_count})
        if env_rows and len(env_rows)==shape_count:
            canonical=[{"shape_path":re.sub(r"/env_[0-3]/","/env_*/",row["path"]),
                        "shape_type":row["shape_type"],"effective_material":row["effective_material"]}
                       for row in sorted(env_rows,key=lambda x:x["path"])]
            fingerprints[str(env_id)]=_canonical_sha(canonical)
    unmapped=[row["path"] for row in collisions if row["path"] not in used]
    within=all(low-1e-5<=float(shape[index])<=high+1e-5
               for env in data for shape in env for index,(low,high) in enumerate(startup_ranges))
    return {"tensor_shape":[4,shape_count,3],"tensor_values":data,
            "collision_count":len(collisions),
            "unbound_collision_count":sum(row["binding_status"]=="unbound" for row in collisions),
            "physx_shape_count_total":4*shape_count,
            "mapped_collision_count":len(mapped),
            "unmatched_tensor_shape_count":sum(max(0,row.get("physx_shape_count",0)-
                                                   row.get("usd_collision_count",0)) for row in gaps),
            "unmapped_collision_count":len(unmapped),"unmapped_collision_paths":unmapped,
            "link_count_mismatches":gaps,"ambiguous_links":ambiguous,
            "ambiguous_link_count":len(ambiguous),"mapped_collisions":mapped,
            "canonical_fingerprint_by_env":fingerprints,
            "startup_ranges_pass":within,
            "coverage_pass":not gaps and not ambiguous and not unmapped and
                            len(mapped)==4*shape_count and len(fingerprints)==4}
