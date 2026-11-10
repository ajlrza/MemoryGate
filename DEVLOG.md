1. Alternative architecture design

```python
    def __init__(self, short_term_capacity):
        self.working_memory: dict[str, any] = {}
        self.memory_space = np.zeros((1000, short_term_capacity))
```

Alternatively, MemoryGate could accomodate user-defined short term capacity, which is an abstraction for how many
types of memory it can hold, but x - context importance and y - context relevance columns must remain. For example,

| **X (Context Importance)** | **Y (Context Relevance)** | **Score** | **Mood** | **Keyword** |
|---|---|---|---|---|
| 0 | 0 | 85% | Energetic | Vibrant |
| 1 | 3 | 40% | Melancholic | Somber |
| 5 | 2 | 95% | Serene | Peaceful |
