
#!/usr/bin/env python3
import os, time, json, urllib.request

HOST = os.getenv("E8_HOST", "http://127.0.0.1:8080")
TOKEN = os.getenv("E8_API_TOKEN", "")

def call(path, data=None):
    url = f"{HOST}{path}"
    headers = {"Content-Type":"application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, headers=headers, data=(json.dumps(data).encode("utf-8") if data else None))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def set_quantizer(name):
    return call("/quantizer", {"mode": name})

def get_metrics():
    return call("/telemetry")

def main():
    quantizers = ["e8","cubic","random","none"]
    results = {}
    for q in quantizers:
        try:
            set_quantizer(q)
            # allow system to run for a short time window
            time.sleep(5)
            t = get_metrics()
            results[q] = {
                "insights": t.get("insights_total"),
                "shell_population": t.get("shell_population"),
                "kdtree_failures": t.get("kdtree_failures"),
                "new_nodes": t.get("recent_new_nodes", 0),
            }
            print(q, results[q])
        except Exception as e:
            print("error on", q, e)
    with open("/mnt/data/quantizer_results.json","w",encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved /mnt/data/quantizer_results.json")

if __name__ == "__main__":
    main()
