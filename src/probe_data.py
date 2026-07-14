import pickle

DATA_DIR = "/home/ubuntu/upload/scotus_results/scotus"

for name in ["profiles.pkl", "opinions_meta.pkl", "las_clusters.pkl",
             "las_features.pkl", "V.pkl", "G.pkl"]:
    try:
        with open(f"{DATA_DIR}/{name}", "rb") as f:
            obj = pickle.load(f)
        t = type(obj)
        if hasattr(obj, "keys"):
            keys = list(obj.keys())[:5]
            print(f"{name}: {t.__name__}, {len(obj)} entries, sample keys: {keys}")
            # inspect first value
            if keys:
                v = obj[keys[0]]
                print(f"  first value type: {type(v)}, value: {str(v)[:200]}")
        elif hasattr(obj, "shape"):
            print(f"{name}: {t.__name__}, shape={obj.shape}, dtype={obj.dtype}")
            print(f"  sample: {obj[:3]}")
        elif hasattr(obj, "__len__"):
            print(f"{name}: {t.__name__}, len={len(obj)}")
            if len(obj) > 0:
                print(f"  first item: {str(obj[0])[:200]}")
        else:
            print(f"{name}: {t.__name__}, value: {str(obj)[:200]}")
    except Exception as e:
        print(f"{name}: ERROR — {e}")
    print()
