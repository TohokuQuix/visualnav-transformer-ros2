"""Choose the fixed eight-pose second-seed pilot from the unchanged R3 order."""
from __future__ import annotations

from .anchor_placement_rules import CLASS_ORDER


def choose_pilot(r3_ordered: list[dict]) -> tuple[list[dict],list[str]]:
    """Return A then B: first two distinct source families per class."""
    first=[];second=[];missing=[]
    for cls in CLASS_ORDER:
        distinct=[];seen=set()
        for item in r3_ordered:
            if item['classification']!=cls or item['source_family'] in seen:
                continue
            distinct.append(item);seen.add(item['source_family'])
            if len(distinct)==2:break
        if len(distinct)<2:
            missing.append(cls)
        else:
            first.append(distinct[0]);second.append(distinct[1])
    return first+second,missing
