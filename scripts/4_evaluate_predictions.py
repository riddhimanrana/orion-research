#!/usr/bin/env python3
"""
Evaluate Orion predictions against TAO-Amodal ground truth.
Computes: Recall@K, mRecall, BBox IoU@0.5
"""

import json
import os
import numpy as np

GROUND_TRUTH_FILE = 'data/tao_75_test/ground_truth.json'
PREDICTIONS_FILE = 'data/tao_75_test/results/predictions.json'
RESULTS_DIR = 'data/tao_75_test/results'

def compute_bbox_iou(box1, box2):
    """Compute IoU between two bboxes [x, y, w, h]"""
    try:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_end = [x1 + w1, y1 + h1]
        box2_end = [x2 + w2, y2 + h2]
        
        # Intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(box1_end[0], box2_end[0])
        y_bottom = min(box1_end[1], box2_end[1])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def evaluate(gt_data, predictions):
    """Compute evaluation metrics"""
    
    # Build GT index
    gt_by_img = {}
    for ann in gt_data.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id:
            if img_id not in gt_by_img:
                gt_by_img[img_id] = []
            gt_by_img[img_id].append(ann)
    
    # Build predictions index
    pred_by_img = {}
    for pred in predictions:
        img_id = pred.get('image_id')
        if img_id:
            if img_id not in pred_by_img:
                pred_by_img[img_id] = []
            pred_by_img[img_id].append(pred)
    
    recalls_k = {10: [], 20: [], 50: []}
    ious = []
    
    for img_id, gts in gt_by_img.items():
        if not gts:
            continue
        
        preds = sorted(pred_by_img.get(img_id, []), key=lambda x: x.get('score', 0), reverse=True)
        
        # Match predictions to GTs
        gt_matched = [False] * len(gts)
        num_matched = 0
        
        for pred in preds:
            pred_bbox = pred.get('bbox')
            if not pred_bbox:
                continue
                
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue
                iou = compute_bbox_iou(pred_bbox, gt.get('bbox', [0, 0, 0, 0]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5 and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                num_matched += 1
                ious.append(best_iou)
        
        # Compute recall@K
        num_gt = len(gts)
        for k in [10, 20, 50]:
            recall = min(num_matched, min(k, num_gt)) / num_gt if num_gt > 0 else 0
            recalls_k[k].append(recall)
    
    # Aggregate
    metrics = {
        'Recall@10': np.mean(recalls_k[10]) * 100 if recalls_k[10] else 0,
        'Recall@20': np.mean(recalls_k[20]) * 100 if recalls_k[20] else 0,
        'Recall@50': np.mean(recalls_k[50]) * 100 if recalls_k[50] else 0,
        'mRecall': (np.mean(recalls_k[10]) + np.mean(recalls_k[20]) + np.mean(recalls_k[50])) / 3 * 100 if recalls_k[10] else 0,
        'BBox_IoU@0.5': np.mean(ious) if ious else 0,
    }
    
    return metrics

def main():
    print("="*70)
    print("STEP 4: Evaluate Orion Predictions")
    print("="*70)
    
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"\n❌ Predictions file not found: {PREDICTIONS_FILE}")
        print("\nPlease:")
        print("1. Run Orion on TAO videos")
        print("2. Save predictions to: " + PREDICTIONS_FILE)
        print("\nPredictions format:")
        print("""   [
     {"image_id": int, "bbox": [x,y,w,h], "score": float, 
      "category_id": int},
     ...
   ]""")
        return
    
    print("\n1. Loading data...")
    with open(GROUND_TRUTH_FILE, 'r') as f:
        gt_data = json.load(f)
    
    with open(PREDICTIONS_FILE, 'r') as f:
        predictions = json.load(f)
    
    print(f"   Ground truth images: {len(gt_data.get('images', []))}")
    print(f"   Predictions: {len(predictions)}")
    
    print("\n2. Computing metrics...")
    metrics = evaluate(gt_data, predictions)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"\n  Recall@10:    {metrics['Recall@10']:.2f}%")
    print(f"  Recall@20:    {metrics['Recall@20']:.2f}%")
    print(f"  Recall@50:    {metrics['Recall@50']:.2f}%")
    print(f"  mRecall:      {metrics['mRecall']:.2f}%")
    print(f"  BBox IoU@0.5: {metrics['BBox_IoU@0.5']:.4f}")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = f'{RESULTS_DIR}/metrics.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved metrics to: {results_file}")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
