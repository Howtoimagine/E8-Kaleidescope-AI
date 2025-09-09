
#!/usr/bin/env python3
import json, os, math, statistics, collections, itertools, re
from collections import defaultdict, Counter

def read_ndjson(path):
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                xs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return xs

def five_gram_tokens(s):
    toks = re.findall(r"[A-Za-z0-9]+", s.lower())
    return set(zip(*[toks[i:] for i in range(5)])) if len(toks)>=5 else set()

def jaccard(a,b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

def bucketize(vals, bins):
    if not vals: return [0]* (len(bins)-1)
    hist = [0]*(len(bins)-1)
    for v in vals:
        for i in range(len(bins)-1):
            if bins[i] <= v < bins[i+1]:
                hist[i]+=1; break
    return hist

def main():
    base = "/mnt/data"
    ins_path = os.path.join(base, "insights.ndjson")
    prox_path = os.path.join(base, "proximity_alerts.ndjson")
    insights = read_ndjson(ins_path) if os.path.exists(ins_path) else []
    alerts = read_ndjson(prox_path) if os.path.exists(prox_path) else []

    # Ratings / calibration summary
    ratings = [x.get("rating", 0.0) for x in insights if isinstance(x.get("rating", None), (int,float))]
    cal = [x.get("calibrated_rating", None) for x in insights if x.get("calibrated_rating", None) is not None]
    print(f"Insights: {len(insights)}; Alerts: {len(alerts)}")
    if ratings:
        print(f"Rating mean={statistics.mean(ratings):.3f} median={statistics.median(ratings):.3f}")
    if cal:
        print(f"Calibrated mean={statistics.mean(cal):.3f} median={statistics.median(cal):.3f}")

    # Diversity (Jaccard on 5-grams) over sliding windows of 200
    labels = [x.get("label","") for x in insights]
    windows = 0
    if labels:
        W=200
        last = None
        j_scores = []
        for i in range(0, len(labels), W):
            chunk = labels[i:i+W]
            grams = set().union(*[five_gram_tokens(s) for s in chunk])
            if last is not None:
                j_scores.append(jaccard(last, grams))
            last = grams
            windows += 1
        if j_scores:
            print(f"Diversity (lower is better novelty): Jaccard mean={statistics.mean(j_scores):.3f}")
        else:
            print("Diversity: not enough windows")

    # Alert acceptance rate by distance bands (assume 'accepted' flag if present; else treat all as pending)
    distances = [a.get("distance", 0.0) for a in alerts if isinstance(a.get("distance", None),(int,float))]
    bands = [0.0, 0.25, 0.5, 0.75, 0.9, 1.01]
    hist = bucketize(distances, bands)
    print("Alert distance histogram (counts per band):", dict(zip([f"[{bands[i]:.2f},{bands[i+1]:.2f})" for i in range(len(bands)-1)], hist)))

if __name__ == "__main__":
    main()
