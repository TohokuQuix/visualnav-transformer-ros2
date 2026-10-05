"""Small diagrams for the offline M11-M13 review; no new outcomes are computed."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eat_nav_evaluator_multihead_addendum import pareto_front

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-addendum"
OLD = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-freeze"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def flow(name, title, boxes, *, color="#dbe8ea"):
    fig, ax = plt.subplots(figsize=(max(9, len(boxes)*2.2), 2.6))
    ax.set_xlim(0, len(boxes)); ax.set_ylim(0, 1); ax.axis("off")
    for index, label in enumerate(boxes):
        ax.text(index+.5, .5, label, ha="center", va="center", fontsize=10,
                bbox={"boxstyle":"round,pad=.55", "facecolor":color, "edgecolor":"#52717b"})
        if index + 1 < len(boxes):
            ax.annotate("", xy=(index+.95,.5), xytext=(index+.8,.5),
                        arrowprops={"arrowstyle":"->", "color":"#344f59", "lw":1.8})
    ax.set_title(title, fontsize=12, pad=14)
    save(fig, name)


def main():
    summaries=json.loads((OLD/'pairwise-labels.json').read_text())['candidate_summaries']
    by=defaultdict(list)
    for row in summaries:by[row['state_group_id']].append(row)
    fig,ax=plt.subplots(figsize=(8,5))
    colors=plt.get_cmap('tab10')
    for index,(group,rows) in enumerate(sorted(by.items())):
        rows=sorted(rows,key=lambda r:r['candidate_id'])
        x=[r['reward'] for r in rows];y=[r['goal'] for r in rows]
        ax.scatter(x,y,color=colors(index),s=32,label=group,alpha=.8)
        fronts=pareto_front(list(zip(x,y)))
        ax.scatter([x[i] for i in fronts],[y[i] for i in fronts],facecolors='none',edgecolors='black',s=85,linewidths=1)
        direct=next(r for r in rows if r['candidate_id']=='direct_reference')
        ax.scatter([direct['reward']],[direct['goal']],marker='*',s=95,color=colors(index),edgecolors='black',linewidths=.5)
    ax.set_xlabel('Observed 0–2 s discounted reward (higher better)')
    ax.set_ylabel('Goal-distance reduction (m; higher better)')
    ax.set_title('66/147 reward–goal preference conflicts; black rings = group Pareto set')
    ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(1, .5))
    fig.tight_layout();save(fig,'fig-01-conflict-pareto.svg')

    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for ax,title,lines,color in (
        (axes[0],"Phase A: observed/task teachers",
         ["Rγ(0–2 s)","Direct-relative reward","Goal-distance reduction","Reward and goal pairwise","Huber primary; ranking auxiliary"],"#d7e7df"),
        (axes[1],"Phase B: independent ablations",
         ["Endpoint error / encounter / path metrics","Terminal V: hold and canonical","Critic residual: hold and canonical","Reject if primary task ranking worsens","Safety head disabled"],"#e7dfe7")):
        ax.axis('off');ax.set_title(title)
        for i,line in enumerate(lines):
            ax.text(.5,.82-i*.16,line,ha='center',va='center',transform=ax.transAxes,
                    bbox={"boxstyle":"round,pad=.35","facecolor":color,"edgecolor":"#6d7b82"},fontsize=9)
    fig.tight_layout();save(fig,'fig-02-phase-heads.svg')

    flow('fig-03-evaluation-flow.svg','Prospective group-level evaluation; seven current groups are diagnostic only',
         ['Group/seed\nsplit','Train-only\nscaler','Baselines +\nPhase A','Group macro\nmetrics + CI','Phase B\nablation'],color='#dce7ed')
    flow('fig-04-logging-timeline.svg','Future one-step logging contract; exact logical stamps',
         ['Anchor\npre-Path','Pre-command\nobservation','Command adopt +\nActor/Critic input','env.step → reward\nbody/contact','Terminal pre-reset\nvalidity'],color='#eee5d7')
    flow('fig-05-decision-flow.svg','Current decision authorizes design and input preparation only',
         ['Input integration','Field order/unit/\ntime parity','Logging proposal +\nparity gate','Additional groups\ncollection design','Small training\nplan (approval later)'],color='#dfeadd')
    print('5 SVG figures written')


if __name__=='__main__':main()
