# Quick Dataset Test
# Run this cell to verify datasets load correctly before running full experiments

print("Testing dataset loading...")

# Test CoNLL-2003
print("\n" + "="*60)
print("Testing CoNLL-2003...")
print("="*60)
try:
    from datasets import load_dataset
    ds_conll = load_dataset("tner/conll2003", trust_remote_code=True, split='train')
    print(f"[OK] CoNLL-2003: {len(ds_conll)} examples")
    print(f"   Features: {list(ds_conll.features.keys())}")
    print(f"   Sample tokens: {ds_conll[0]['tokens'][:5]}...")
    print(f"   Sample tags: {ds_conll[0]['tags'][:5]}...")
except Exception as e:
    print(f"[FAIL] CoNLL-2003 failed: {e}")

# Test WikiANN Hindi
print("\n" + "="*60)
print("Testing WikiANN Hindi...")
print("="*60)
try:
    ds_hindi = load_dataset("unimelb-nlp/wikiann", "hi", split='train')
    print(f"[OK] WikiANN Hindi: {len(ds_hindi)} examples")
    print(f"   Features: {list(ds_hindi.features.keys())}")
    print(f"   Sample tokens: {ds_hindi[0]['tokens'][:5]}...")
except Exception as e:
    print(f"[FAIL] WikiANN Hindi failed: {e}")

# Test WikiANN Marathi
print("\n" + "="*60)
print("Testing WikiANN Marathi...")
print("="*60)
try:
    ds_marathi = load_dataset("unimelb-nlp/wikiann", "mr", split='train')
    print(f"[OK] WikiANN Marathi: {len(ds_marathi)} examples")
except Exception as e:
    print(f"[FAIL] WikiANN Marathi failed: {e}")

print("\n" + "="*60)
print("[OK] All dataset tests completed!")
print("="*60)
print("\nIf all tests passed, you're ready to run experiments!")

