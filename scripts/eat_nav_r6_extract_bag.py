#!/usr/bin/env python3
"""Read-only ROS2 MCAP extraction. Run with ROS-compatible /usr/bin/python3.

No rclpy node is created. Original serialized records and both clocks are retained.
load_observations() is NumPy-only and can be used from the training interpreter.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import numpy as np

DEFAULT_ADAPTER = Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/r6_real_inputs.py'


class CDRReader:
    """Bounded little-endian CDR1 decoder for the embedded GridMap schema."""
    def __init__(self, data):
        self.data = bytes(data)
        if self.data[:4] != b'\x00\x01\x00\x00':
            raise ValueError('expected little-endian CDR1 encapsulation')
        self.offset = 4

    def scalar(self, fmt, alignment):
        self.offset += (4 - self.offset) % alignment
        size = struct.calcsize('<' + fmt)
        if self.offset + size > len(self.data):
            raise ValueError('truncated CDR scalar')
        result = struct.unpack_from('<' + fmt, self.data, self.offset)[0]
        self.offset += size
        return result

    def string(self):
        size = self.scalar('I', 4)
        if size < 1 or self.offset + size > len(self.data) or self.data[self.offset+size-1] != 0:
            raise ValueError('invalid CDR string')
        result = self.data[self.offset:self.offset+size-1].decode('utf8')
        self.offset += size
        return result

    def sequence(self, fn):
        count = self.scalar('I', 4)
        if count > len(self.data):
            raise ValueError('invalid CDR sequence length')
        return [fn() for _ in range(count)]

    def vector(self, names):
        return NS(**{name: self.scalar('d', 8) for name in names})

    def array(self):
        dims = self.sequence(lambda: NS(label=self.string(), size=self.scalar('I', 4), stride=self.scalar('I', 4)))
        offset = self.scalar('I', 4)
        return NS(layout=NS(dim=dims, data_offset=offset), data=self.sequence(lambda: self.scalar('f', 4)))

    def finish(self):
        if self.offset != len(self.data):
            raise ValueError('CDR trailing bytes or wrong schema')

    def grid(self):
        header = NS(stamp=NS(sec=self.scalar('i', 4), nanosec=self.scalar('I', 4)), frame_id=self.string())
        info = NS(resolution=self.scalar('d', 8), length_x=self.scalar('d', 8), length_y=self.scalar('d', 8),
                  pose=NS(position=self.vector('xyz'), orientation=self.vector('xyzw')))
        result = NS(header=header, info=info, layers=self.sequence(self.string), basic_layers=self.sequence(self.string),
                    data=self.sequence(self.array), outer_start_index=self.scalar('H', 2), inner_start_index=self.scalar('H', 2))
        self.finish()
        return result


def causal_index(receive_ns, cutoff_ns):
    index = bisect_right(receive_ns, cutoff_ns) - 1
    return index if index >= 0 else None


def history_indices(receive_ns, cutoff_ns, interval_ns):
    """Four distinct causal frames; monotonic receive clock chooses nearest targets.

    Selection ends at the most recent received frame. Earlier choices are constrained
    to earlier indices, so no frame is duplicated and no future receipt is used.
    """
    latest = causal_index(receive_ns, cutoff_ns)
    if latest is None or latest < 3:
        return None
    ids = [latest]
    for k in range(1, 4):
        target = receive_ns[latest] - k * interval_ns
        upper = ids[-1] - 1
        lower = 3 - k
        ids.append(min(range(lower, upper + 1), key=lambda i: abs(receive_ns[i] - target)))
    return list(reversed(ids))


def initial_goal(position, quaternion):
    x, y, z, w = quaternion
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return [float(position[0]+2*math.cos(yaw)), float(position[1]+2*math.sin(yaw))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_module(path):
    spec = importlib.util.spec_from_file_location('r6_handoff_inputs', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def extract(bag, output, adapter_path=DEFAULT_ADAPTER, interval_sec=1/3, period_sec=2., max_age_sec=1., allow_inpainted_estimates=False):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    import yaml
    bag, output = Path(bag), Path(output)
    if output.exists():
        raise FileExistsError('refusing to replace an existing extraction: ' + str(output))
    adapter = adapter_module(adapter_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    topic_types = {t.name:t.type for t in reader.get_all_topics_and_types()}
    definitions = [{'type':d.topic_type, 'encoding':d.encoding, 'definition':d.encoded_message_definition}
                   for d in reader.get_all_message_definitions()]
    classes = {t:get_message(t) for t in set(topic_types.values()) if t != 'grid_map_msgs/msg/GridMap'}
    records, blobs, grids, streams, tfs = [], [], [], {}, []
    cursor = 0
    while reader.has_next():
        topic, raw, receive = reader.read_next()
        record = dict(topic=topic, receive_ns=int(receive), raw_offset=cursor, raw_length=len(raw))
        blobs.append(raw); cursor += len(raw)
        if topic_types[topic] == 'grid_map_msgs/msg/GridMap':
            msg = CDRReader(raw).grid()
            if hasattr(adapter,'VALIDITY_CONTRACT'):
                grid = adapter.decode_grid_map(msg,'elevation_inpainted',observed_layer=None if allow_inpainted_estimates else 'elevation')
            elif allow_inpainted_estimates:grid=adapter.decode_grid_map(msg,'elevation_inpainted')
            else:raise ValueError('strict observed-elevation validity requires the portableR6 sensor transform')
            if grid.frame_id!='map':raise ValueError('DEM and map->body TF frames differ')
            grids.append(dict(receive_ns=int(receive), grid=grid, message=msg))
            record['header_ns'] = grid.stamp
        else:
            msg = deserialize_message(raw, classes[topic_types[topic]])
            record['message'] = jsonable(message_to_ordereddict(msg))
            if hasattr(msg, 'header'): record['header_ns'] = adapter.stamp_ns(msg.header.stamp)
            streams.setdefault(topic, []).append((int(receive), msg))
            if topic in ('/tf', '/tf_static'):
                for transform in msg.transforms:
                    tfs.append((int(receive), topic, transform))
        records.append(record)
    if not records or not grids: raise ValueError('empty bag or absent DEM')
    counts = Counter(r['topic'] for r in records)
    metadata = yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    expected = {x['topic_metadata']['name']:x['message_count'] for x in metadata['topics_with_message_count']}
    if len(records) != metadata['message_count'] or any(counts[t] != count for t,count in expected.items()):
        raise ValueError('bag metadata count mismatch')
    body_tfs = [(receive, transform) for receive, topic, transform in tfs
                if transform.header.frame_id == 'map' and transform.child_frame_id == 'chassis/body']
    mount_tfs = [(receive, transform) for receive, topic, transform in tfs
                 if transform.header.frame_id == 'chassis/body' and transform.child_frame_id == 'bno055']
    if not body_tfs: raise ValueError('map-to-body TF missing')
    first_tf = body_tfs[0][1]
    first_pos = adapter.xyz(first_tf.transform.translation)
    first_q = adapter.quaternion(first_tf.transform.rotation)
    goal = initial_goal(first_pos, first_q)
    tf_receives = [x[0] for x in body_tfs]
    mount_receives = [x[0] for x in mount_tfs]
    grid_receives = [g['receive_ns'] for g in grids]
    stream_receives = {name:[r for r,m in rows] for name,rows in streams.items()}
    # Rasterize each received map using only a previously received body TF.
    rasterized, raster_tf_receive = [], []
    for row in grids:
        index = causal_index(tf_receives, row['receive_ns'])
        if index is None:
            rasterized.append(None); raster_tf_receive.append(-1); continue
        receive, tf = body_tfs[index]
        q = adapter.quaternion(tf.transform.rotation); r = adapter.rotation(q)
        p = adapter.xyz(tf.transform.translation)
        try: rasterized.append(adapter.sample_dem(row['grid'], (p[0],p[1],math.atan2(r[1,0],r[0,0]))))
        except ValueError: rasterized.append(None)
        raster_tf_receive.append(receive)
    arrays = {name:[] for name in ('dem','actor_state','body_state','decision_receive_ns','dem_receive_ns','dem_header_ns',
              'state_receive_ns','state_header_ns','tf_receive_ns','pose_map','imu_orientation_valid','valid')}
    diagnostics = []
    start, end = records[0]['receive_ns'], records[-1]['receive_ns']
    for cutoff in range(start, end+1, int(period_sec*1e9)):
        ids = history_indices(grid_receives, cutoff, int(interval_sec*1e9))
        reason = None
        if ids is None: reason = 'insufficient_distinct_history'
        elif any(rasterized[i] is None for i in ids): reason = 'DEM_TF_or_coverage_unavailable'
        elif cutoff-grid_receives[ids[-1]] > max_age_sec*1e9: reason = 'stale_DEM_receipt'
        state = []
        for topic in ('/bno055/imu','/joint_states','/diff_drive_controller/odom'):
            index = causal_index(stream_receives.get(topic, []), cutoff)
            if index is None or cutoff-stream_receives[topic][index] > max_age_sec*1e9:
                reason = 'missing_or_stale_state_receipt'; break
            state.append(streams[topic][index])
        ti, mi = causal_index(tf_receives, cutoff), causal_index(mount_receives, cutoff)
        if ti is None or mi is None: reason = 'missing_causal_body_or_IMU_mount_TF'
        elif cutoff-tf_receives[ti] > max_age_sec*1e9: reason = 'stale_body_TF_receipt'
        if reason:
            diagnostics.append(dict(decision_receive_ns=cutoff, valid=False, reason=reason)); continue
        (_,imu),(_,joint),(_,odom) = state
        tr = body_tfs[ti][1].transform
        r = adapter.rotation(adapter.quaternion(tr.rotation))
        mount = adapter.rotation(adapter.quaternion(mount_tfs[mi][1].transform.rotation))
        gravity = r.T @ np.array([0.,0.,-1.])
        angular = mount @ np.array(adapter.xyz(imu.angular_velocity))
        # Odom already has chassis/body as child; reject other frames rather than assume.
        if odom.child_frame_id != 'chassis/body' or imu.header.frame_id != 'bno055':
            raise ValueError('unsupported body/IMU frame contract')
        fold_rows = streams.get('/flipper_fold_mode', [])
        fold_index = causal_index(stream_receives.get('/flipper_fold_mode', []), cutoff)
        fold = fold_rows[fold_index][1].data if fold_index is not None else 0
        actor = adapter.actor_state(joint, gravity, angular, fold)
        body = adapter.body_state(gravity, adapter.xyz(odom.twist.twist.linear), angular)
        pose = adapter.xyz(tr.translation)+[math.atan2(r[1,0],r[0,0])]
        values = dict(dem=np.stack([rasterized[i] for i in ids]),actor_state=actor,body_state=body,
            decision_receive_ns=cutoff,dem_receive_ns=[grid_receives[i] for i in ids],
            dem_header_ns=[grids[i]['grid'].stamp for i in ids],state_receive_ns=[x[0] for x in state],
            state_header_ns=[adapter.stamp_ns(x[1].header.stamp) for x in state],tf_receive_ns=body_tfs[ti][0],
            pose_map=pose,imu_orientation_valid=bool(imu.orientation_covariance[0] != -1), valid=True)
        if any(v > cutoff for v in values['dem_receive_ns']+values['state_receive_ns']+[values['tf_receive_ns']]):
            raise AssertionError('future receipt entered input')
        for name,value in values.items(): arrays[name].append(value)
        diagnostics.append(dict(decision_receive_ns=cutoff,valid=True,dem_indices=ids,
                                dem_raster_tf_receive_ns=[raster_tf_receive[i] for i in ids]))
    arrays = {k:np.asarray(v) for k,v in arrays.items()}
    offsets = {}
    for topic in counts:
        rows = [r for r in records if r['topic']==topic and 'header_ns' in r]
        if rows:
            x = np.array([r['receive_ns']-r['header_ns'] for r in rows])/1e9
            offsets[topic] = dict(min_sec=float(x.min()),median_sec=float(np.median(x)),max_sec=float(x.max()))
    source = {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [bag/'metadata.yaml',bag/'bag_0.mcap',bag.parent/'planner_params.yaml',bag.parent/'record_qos.yaml'] if p.exists()}
    output.mkdir(parents=True)
    np.savez_compressed(output/'observations.npz', **arrays)
    np.savez_compressed(output/'raw_dem.npz', height=np.stack([x['grid'].height for x in grids]),
        valid=np.stack([x['grid'].valid for x in grids]),center=np.stack([x['grid'].center for x in grids]),
        resolution=np.array([x['grid'].resolution for x in grids]),yaw=np.array([x['grid'].yaw for x in grids]),
        receive_ns=np.array(grid_receives,dtype=np.int64),header_ns=np.array([x['grid'].stamp for x in grids],dtype=np.int64))
    (output/'serialized_records.bin').write_bytes(b''.join(blobs))
    with (output/'records.jsonl').open('w') as file:
        for record in records: file.write(json.dumps(record,allow_nan=True)+'\n')
    (output/'message_definitions.json').write_text(json.dumps(definitions,indent=2)+'\n')
    summary = dict(schema='eatnav_r6_bag_intermediate_v1',source=source,extractor_sha256=sha(__file__),
        adapter_sha256=sha(adapter_path),counts=dict(counts),message_count=len(records),receive_start_ns=start,receive_end_ns=end,
        valid_observations=len(arrays['valid']),decisions=diagnostics,log_minus_header_offsets=offsets,
        common_goal=dict(map_xy=goal,initial_map_position=first_pos,initial_xyzw=first_q,distance_m=2.,basis='first received map->chassis/body TF, yaw-forward'),
        contract=dict(dem_shape=[4,2,20,20],actor_shape=[22],body_shape=[9],dem_resolution_m=.1,dem_extent_m=2.,validity_source='selected-layer finite including inpainting; reference sensitivity only' if allow_inpainted_estimates else 'finite observed elevation intersect selected layer; inpainted-only cells unknown',
            nominal_history_interval_sec=interval_sec,training_history_interval_sec=.25,history_match_verified=False,
            decision_period_sec=period_sec,max_receipt_age_sec=max_age_sec,units='m, rad, s; actor angular*.25 speed*.05 effort*.01',
            orientation_source='causal map->body TF; IMU angular transformed using received static mount',
            imu_orientation_covariance_invalid='preserved; not used for gravity',flipper_calibration='user reference outward0/up90deg confirmed and simulation axes independently verified; real encoder offsets not independently measured; raw radians retained without mean imputation',
            fold_source='causal fold message, otherwise handoff default zero; no fold messages in this bag',
            causality='all input records log/receive_ns <= decision; raw header clocks retained without shifting',
            limitations=['TF localization orientation is not independently calibrated truth','receive clock offset does not resolve physical acquisition clock','no executed candidate labels or counterfactual outcomes','real history spacing differs from frozen training']))
    summary['artifacts'] = {p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()}
    (output/'index.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def load_observations(directory):
    """Load causal inputs without ROS dependencies; paths/directions are added by evaluator."""
    path = Path(directory)
    index = json.loads((path/'index.json').read_text())
    if sha(path/'observations.npz') != index['artifacts']['observations.npz']['sha256']:
        raise ValueError('observation SHA mismatch')
    with np.load(path/'observations.npz', allow_pickle=False) as data:
        arrays = {name:data[name].copy() for name in data.files}
    return arrays, index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=Path('/home/ubuntu/eat_nav_planner_20261029_210926/bag'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter-inputs', type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument('--allow-inpainted-estimates',action='store_true',help='reference sensitivity only; default excludes unobserved filled cells')
    args = parser.parse_args()
    result = extract(args.bag,args.output,args.adapter_inputs,allow_inpainted_estimates=args.allow_inpainted_estimates)
    print(json.dumps({k:result[k] for k in ('message_count','valid_observations','counts','common_goal')},indent=2))
