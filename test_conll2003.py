from datasets import load_dataset

print("Testing CoNLL-2003 alternatives...")

# Option 1: Remove trust_remote_code (might work if auto-converted)
try:
    print("\n1. Attempting load_dataset('conll2003') without trust_remote_code...")
    ds = load_dataset("conll2003")
    print("Success! conll2003 (no trust_remote_code) worked.")
    print("Keys:", ds.keys())
except Exception as e:
    print(f"Failed: {e}")

# Option 2: Try tner/conll2003
try:
    print("\n2. Attempting tner/conll2003...")
    ds = load_dataset("tner/conll2003")
    print("Success! tner/conll2003 worked.")
    print("Keys:", ds.keys())
    print("Features:", ds['train'].features)
except Exception as e:
    print(f"Failed: {e}")

# Option 3: Try eriktks/conll2003
try:
    print("\n3. Attempting eriktks/conll2003...")
    ds = load_dataset("eriktks/conll2003")
    print("Success! eriktks/conll2003 worked.")
    print("Keys:", ds.keys())
    print("Features:", ds['train'].features)
except Exception as e:
    print(f"Failed: {e}")
