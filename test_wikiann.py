from datasets import load_dataset

print("Testing alternatives for WikiANN (mr)...")

try:
    print("Attempting main wikiann...")
    load_dataset("wikiann", "mr", trust_remote_code=True)
    print("Main wikiann worked (unexpected).")
except Exception as e:
    print(f"Main wikiann failed: {e}")

try:
    print("\nAttempting rahular/wikiann...")
    ds = load_dataset("rahular/wikiann", "mr", trust_remote_code=True)
    print("rahular/wikiann loaded successfully.")
    print("Keys:", ds.keys())
    print("Features:", ds['train'].features)
except Exception as e:
    print(f"rahular/wikiann failed: {e}")

# try:
#     print("\nAttempting tner/wikiann...")
#     ds = load_dataset("tner/wikiann", "mr", trust_remote_code=True)
#     print("tner/wikiann loaded successfully.")
# except Exception as e:
#     print(f"tner/wikiann failed: {e}")
