# EAT-Nav validation record

## Task 1: shared configuration

- `PYTHONPATH=src /usr/bin/python3 -m pytest tests/eat_nav/test_config.py -q`: 16 passed.
- The host's default Isaac Python has no pytest; system Python has no torch, so the existing full VNT suite cannot collect `tests/test_robot_inference.py` in this environment yet.

## Task 2: DEM and real GridMap probe

- Pure adapter: `PYTHONPATH=src /usr/bin/python3 -m pytest tests/eat_nav/test_dem.py tests/eat_nav/test_config.py -q`: 25 passed.
- Source data: recorded Onix bag `/home/kojima/Desktop/robocup2026_incheon/2026_07_04-09_04_43`. Its MCAP contains real Livox CustomMsg point clouds and dynamic/static TF, but no recorded `/elevation_map`.
- Probe: replayed 10 seconds of recorded sensor/TF data through the built `livox_to_pointcloud2` and `elevation_mapping` ROS 2 nodes in isolated `ROS_DOMAIN_ID=150`. The large raw point messages were thinned to every 20th point to avoid DDS message loss. The robot's mapping config was not modified.
- Actual emitted `grid_map_msgs/msg/GridMap`: frame `map`, `elevation` layer, 20×20 cells, 0.1 m resolution, 2×2 m extent, identity map orientation. Example map center (12.1, 8.6), circular start indices (19, 14). The elevation Float32MultiArray has 20×20 values, layout dimensions `column_index` (size 20, stride 400) and `row_index` (size 20, stride 20), and NaN unknown cells.
- GridMap library `getPositionFromIndex` measured index (0,0) at map center + (0.95,0.95) m when start indices are zero; its indices increase toward negative world x/y. The ROS adapter reverses those axes and unwinds the circular buffer into a world-axis-increasing SourceGrid. A hand-checked asymmetric circular-buffer fixture passes four ROS tests; the adapter also loaded a GridMap CDR produced by the probe, yielding a 20×20 SourceGrid with origin (11.15,7.65) in `map`.
- Probe output stamps: nine intervals `[0.0998, 0.1001, 0.7001, 0.0998, 0.1001, 0.1001, 0.0998, 0.1001, 0.1001]` s. This shows roughly 10 Hz publication with a replay/processing gap; the thinned probe does not establish the live robot's sustained update-rate distribution. Keep EAT's 0.25 s history sampling configurable and reject missing timestamps.
- The first emitted map had 39/400 valid elevation cells. This is an initialization frame from thinned replay and falls below EAT's 25% validity threshold; it is rejected. It is not representative evidence for the normal robot's valid-cell fraction.
- ROS focused test: `source /home/kojima/onix/install/setup.bash && PYTHONPATH=/tmp/eatnav-vnt/src:$PYTHONPATH /usr/bin/python3 -m pytest onix_automation/test/test_eat_nav_grid_map.py -q`: 4 passed.

The live robot's update-rate and missing-cell distribution remain deployment validation items. Do not infer them from this thinned replay.
