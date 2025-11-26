#!/usr/bin/env python3
"""
Generate comprehensive video analysis annotations from Orion perception results.

Creates spatial zones, temporal analysis, re-identification patterns, and narrative summaries.
"""

import json
from collections import defaultdict
from pathlib import Path


def generate_video_annotations(results_dir: str) -> None:
    """Generate comprehensive video annotations from perception output."""
    entities_file = Path(results_dir) / "entities.json"
    if not entities_file.exists():
        print(f"❌ Error: {entities_file} not found")
        return

    with open(entities_file, "r") as f:
        entities_data = json.load(f)

    print("🎬 COMPREHENSIVE VIDEO ANALYSIS ANNOTATIONS")
    print("=" * 60)

    # Entity summary
    print(f"\n📊 DETECTED ENTITIES: {entities_data['total_entities']} unique objects")
    print("-" * 50)

    class_counts = defaultdict(int)
    spatial_zones = defaultdict(list)
    temporal_spans = []

    for entity in entities_data.get("entities", []):
        cls = entity["class"]
        class_counts[cls] += 1

        span = entity["last_frame"] - entity["first_frame"]
        temporal_spans.append((cls, span, entity["first_frame"], entity["last_frame"]))

        if cls in {"bed", "chair", "table", "sofa"}:
            spatial_zones["furniture_zone"].append(entity)
        elif cls in {"book", "laptop", "keyboard", "mouse"}:
            spatial_zones["workspace_zone"].append(entity)
        elif cls == "person":
            spatial_zones["human_zone"].append(entity)
        elif cls in {"tv", "vase", "potted plant"}:
            spatial_zones["living_zone"].append(entity)
        else:
            spatial_zones["other_zone"].append(entity)

    for cls, count in sorted(class_counts.items()):
        print(f"  • {cls}: {count}")

    print("\n🏠 SPATIAL ZONES IDENTIFIED:")
    print("-" * 30)
    for zone, items in spatial_zones.items():
        if items:
            classes = sorted({e["class"] for e in items})
            zone_name = zone.replace("_", " ").title()
            print(f"  📍 {zone_name}: {len(items)} objects ({', '.join(classes)})")

    print("\n⏰ TEMPORAL ANALYSIS:")
    print("-" * 20)
    temporal_spans.sort(key=lambda x: x[1], reverse=True)
    print("Longest-present objects:")
    for cls, span, start, end in temporal_spans[:5]:
        duration_sec = span / 30.0
        print(f"  • {cls}: {span} frames ({duration_sec:.1f}s) from frame {start} to {end}")

    print("\n🔄 RE-IDENTIFICATION PATTERNS:")
    print("-" * 30)
    multi_instance = {cls: count for cls, count in class_counts.items() if count > 1}
    if multi_instance:
        for cls, count in multi_instance.items():
            print(f"  • {cls}: {count} instances tracked")
    else:
        print("  • No multi-instance tracking detected")

    print("\n📈 VIDEO NARRATIVE SUMMARY:")
    print("-" * 25)
    print("This video shows a scene with:")
    if "person" in class_counts:
        print(f"  • {class_counts['person']} person(s) interacting with the environment")
    if spatial_zones.get("workspace_zone"):
        print(f"  • A workspace area with {len(spatial_zones['workspace_zone'])} tech items")
    if spatial_zones.get("living_zone"):
        print(f"  • A living area with {len(spatial_zones['living_zone'])} decorative items")
    if spatial_zones.get("furniture_zone"):
        print(f"  • {len(spatial_zones['furniture_zone'])} furniture pieces defining the space")

    print("\n🎯 KEY INSIGHTS:")
    print("-" * 15)
    total_frames = max((e["last_frame"] for e in entities_data.get("entities", [])), default=0)
    avg_lifetime = sum(span for _, span, _, _ in temporal_spans) / max(len(temporal_spans), 1)
    print(f"  • Video duration: ~{total_frames/30:.1f} seconds at 30fps")
    print(f"  • Spatial complexity: {len(spatial_zones)} distinct zones identified")
    print(f"  • Object persistence: Average lifetime = {avg_lifetime:.0f} frames ({avg_lifetime/30:.1f}s)")

    annotations = {
        "video_summary": {
            "total_entities": entities_data.get("total_entities", 0),
            "total_frames": total_frames,
            "duration_seconds": total_frames / 30.0,
            "spatial_zone_count": len(spatial_zones),
            "entity_class_counts": dict(class_counts),
        },
        "spatial_zones": {
            zone: {
                "count": len(items),
                "classes": sorted({e["class"] for e in items}),
            }
            for zone, items in spatial_zones.items()
            if items
        },
        "temporal_analysis": {
            "longest_present": [
                {
                    "class": cls,
                    "frames": span,
                    "seconds": span / 30.0,
                    "start_frame": start,
                    "end_frame": end,
                }
                for cls, span, start, end in temporal_spans[:5]
            ]
        },
        "reidentification": {
            "multi_instance_classes": multi_instance,
        },
    }

    output_file = Path(results_dir) / "video_annotations.json"
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\n💾 Annotations saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate video annotations from Orion results")
    parser.add_argument("--results", default="results/full_video_analysis", help="Results directory")
    args = parser.parse_args()

    generate_video_annotations(args.results)


if __name__ == "__main__":
    main()
