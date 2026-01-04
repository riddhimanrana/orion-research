import json, os, statistics, collections
from pathlib import Path

base = Path('results/lambda_iter_fastvlm_001')
print('Base:', base.resolve())

tracks_path = base/'tracks.jsonl'
filtered_path = base/'tracks_filtered.jsonl'
audit_path = base/'vlm_filter_audit.jsonl'
scene_path = base/'vlm_scene.jsonl'


def read_jsonl(p):
    rows=[]
    if not p.exists(): return []
    with p.open('r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

tracks = read_jsonl(tracks_path)
filtered = read_jsonl(filtered_path)
audit = read_jsonl(audit_path)
scene = read_jsonl(scene_path)

print('\n== Tracks ==')
print('observations:', len(tracks))
track_ids = [r.get('track_id') for r in tracks]
print('unique track_id:', len(set(track_ids)))
by_track=collections.defaultdict(list)
for r in tracks:
    by_track[r.get('track_id')].append(r)
if by_track:
    lengths=[len(v) for v in by_track.values()]
    print('obs/track: min/med/max =', min(lengths), statistics.median(lengths), max(lengths))

classes=collections.Counter(r.get('label') or r.get('class_name') or r.get('class') for r in tracks)
print('top classes:', classes.most_common(10))

frames=sorted({r.get('frame_id') for r in tracks if r.get('frame_id') is not None})
if frames:
    print('unique frames with obs:', len(frames), 'frame_id range:', (frames[0], frames[-1]))

# track continuity stats
print('\n== Track continuity ==')
for tid, rows in sorted(by_track.items(), key=lambda kv: -len(kv[1]))[:8]:
    fids=sorted(r.get('frame_id') for r in rows if r.get('frame_id') is not None)
    gaps=[b-a for a,b in zip(fids,fids[1:])]
    big_gaps=[g for g in gaps if g>10]
    print('track',tid,'len',len(rows),'frames', (fids[0],fids[-1]) if fids else None,'max_gap', max(gaps) if gaps else None,'big_gaps',len(big_gaps))

print('\n== FastVLM filter ==')
print('filtered observations:', len(filtered))
filtered_ids=set(r.get('track_id') for r in filtered)
print('kept track_ids:', len(filtered_ids), '->', sorted(filtered_ids)[:20])

# audit: summarize decisions if present
if audit:
    decisions=collections.Counter((r.get('decision') or r.get('keep') or r.get('action')) for r in audit)
    print('audit decisions:', decisions)
    reasons=collections.Counter((r.get('reason') or r.get('why') or r.get('note')) for r in audit)
    print('top reasons:', [x for x in reasons.most_common(8) if x[0]])

print('\n== Scene captions ==')
print('scene rows:', len(scene))
if scene:
    # print first few captions
    for r in scene[:3]:
        keys=list(r.keys())
        print('keys:', keys)
        # print caption snippet
        if 'caption' in r:
            print('caption:', r['caption'][:100] + '...')
        break

# memory + clusters
mem_path=base/'memory.json'
clusters_path=base/'reid_clusters.json'
if mem_path.exists():
    mem=json.loads(mem_path.read_text())
    print('\n== Memory ==')
    objs=mem.get('objects') or mem.get('memory_objects') or []
    print('memory objects:', len(objs))
    if objs:
        sizes=[len(o.get('track_ids',[])) for o in objs if isinstance(o,dict)]
        if sizes:
            print('tracks/object min/med/max =', min(sizes), statistics.median(sizes), max(sizes))
if clusters_path.exists():
    cl=json.loads(clusters_path.read_text())
    print('\n== ReID clusters ==')
    # could be dict or list
    if isinstance(cl, dict):
        # try common forms
        if 'clusters' in cl and isinstance(cl['clusters'], list):
            clusters=cl['clusters']
        else:
            clusters=list(cl.values())
    else:
        clusters=cl
    print('clusters:', len(clusters) if clusters is not None else None)

# scene graph edges
sg_path=base/'scene_graph.jsonl'
if sg_path.exists():
    sg=read_jsonl(sg_path)
    edge_counts=[]
    node_counts=[]
    for r in sg:
        edges=r.get('edges') or []
        nodes=r.get('nodes') or []
        edge_counts.append(len(edges))
        node_counts.append(len(nodes))
    print('\n== Scene graph ==')
    print('frames:', len(sg),'avg nodes/frame:', sum(node_counts)/len(node_counts) if node_counts else 0,'avg edges/frame:', sum(edge_counts)/len(edge_counts) if edge_counts else 0)
