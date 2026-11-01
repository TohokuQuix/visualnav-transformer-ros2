# 全評価後の報告生成手順

これは既存の閉ループpipelineを変更しない最終postprocessingです。評価中のruntime source bindingは維持します。

1. scientific-evaluation-complete.json、全validation/testのall_jobs_completed=true、凍結top3とモデル・sourceの一致を確認する。途中表を最終報告として使用しない。
2. 以下を永続worktreeから実行する。出力ディレクトリは新規であること。失敗・既存出力は調査し、既存成果物を無条件に削除しない。

```bash
export PYTHONPATH=src:scripts:tests/eat_nav:.
/home/ubuntu/venvs/isaaclab_py310/bin/python docs/eat-nav/r6-2s-pose-20261030/reporting/raw_peak_metrics.py --queue-root /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/validation-evaluation --output /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/validation-raw-peak-metrics
/home/ubuntu/venvs/isaaclab_py310/bin/python docs/eat-nav/r6-2s-pose-20261030/reporting/raw_peak_metrics.py --queue-root /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/test-evaluation --output /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/test-raw-peak-metrics
/home/ubuntu/venvs/isaaclab_py310/bin/python docs/eat-nav/r6-2s-pose-20261030/reporting/render_report.py --root /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030 --output /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/final-report
```

3. `--interim`を上記の最終診断に付けない。test診断はfrozen-top3とruntime bindingを検証する。最終rendererもqueue完了・split・モデル・完全window分母を検証する。
4. 0.1秒格子の軌跡誤差と、20Hz記録の最大傾斜を両方報告する。自然終端の部分windowは真の2秒MAEに混ぜず、観測過小予測下限を併記する。STOP中転倒も走行転倒集計から落とさない。
5. 完全/部分window、旧モデルheadなし、STOPや同一policy共有の除外を確認。20Hzもsubstep全観測ではなく、候補windowは相関することを記載する。
6. 小規模report.json・report.md/txt・sourcesのSHAをGit成果物へ複製する。windows.jsonl.gz、生走行、大容量データは元PCへ保管し、所在・取得手順・SHAを記載する。
7. 図表・代表GUI再現・独立全体レビュー・回帰/strictload/modelpackage検証を完了してから、既存branchへforceせず新branchをpushし、remote SHAを照合する。

2026-11-01時点では全queue未完了。この手順を保存したことは実行・最終報告完成・push成功の証明ではありません。
