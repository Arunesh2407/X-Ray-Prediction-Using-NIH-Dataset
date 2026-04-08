# GraphRAG Data

Expected JSON structure:

```json
{
  "Pneumonia": [
    {
      "source": "Pneumonia",
      "relation": "may_cooccur_with",
      "target": "Effusion",
      "explanation": "Parapneumonic effusion can accompany infection."
    }
  ]
}
```

You can also store a flat list of relation objects if you prefer.
