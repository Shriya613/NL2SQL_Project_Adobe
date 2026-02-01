# Translation Filtering Methods

**What's saved:**
- Filtered translations are saved in the main CSV with `translation=None`, `filtered=True`, and `filter_reason`
- They're mixed with successful translations

**Current filtering:**
- Length ratio (0.3-3.0)
- Language detection
- Suspicious patterns

**Semantic Similarity** (Meaning Preservation)

**What it does:**
- Uses embeddings to check if meaning is preserved
- Compares semantic similarity between original and translation

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def check_semantic_similarity(original, translation):
    emb_orig = model.encode(original)
    emb_trans = model.encode(translation)
    similarity = np.dot(emb_orig, emb_trans) / (
        np.linalg.norm(emb_orig) * np.linalg.norm(emb_trans)
    )
    return similarity > 0.6  # Threshold
```

**Pros:**
- Checks meaning, not just words
- Works across languages
- Catches semantic errors

**Cons:**
- Requires model loading
- Slower than basic checks

---

**Back-Translation Check**

**What it does:**
- Translates translation back to English
- Compares with original
- If very different, likely bad translation

**Implementation:**
```python
def check_back_translation(original, translation, back_translator):
    back_translated = back_translator(translation)
    # Compare original with back-translated
    similarity = calculate_similarity(original, back_translated)
    return similarity > 0.7  # Threshold
```

**Pros:**
- No reference needed
- Catches major errors

**Cons:**
- Requires second translation model
- Slow (2 translations per item)

