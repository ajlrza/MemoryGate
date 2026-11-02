# MemoryGate Documentation

The following is the main module from the MemoryGate Package:

- `gate.py`

---

# AttentionSpace

### `AttentionSpace` **gate.py**
> Manages the underlying working memory dictionary and numpy vector space for conversational memory tracking.

#### Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `working_memory` | `dict[str, str]` | Maps calculated attention embedding strings to attention content. |
| `memory_space` | `np.ndarray` | A $1000 \times 128$ numpy array representing the vector space. |

#### Core Methods

* **`compute_attention(attention_embedding: float)`**: Placeholder method intended for calculating vector space metrics using numpy arrays.
* **`add_memory(attention: str, attention_embedding: float)`**: Stores the attention context into `working_memory` indexed by its float embedding string.

---

# Gate

### `Gate` **gate.py**
> Main controller class managing attention focus, response schemas, and memory state updates for LLM agent interactions.

#### Class Initialization Parameters - NONE

---

### `attach()`
> Attaches and returns the current working memory dictionary from the underlying `AttentionSpace` instance.

#### Function Parameters - NONE

#### Return Value

| Type | Description |
| :--- | :--- |
| `dict[str, str]` | The active `working_memory` dictionary containing attention context mapped by embedding strings. |

---

### `process()`
> Evaluates the agent's schematic output to determine whether to divert or push attention based on the attention flag and coordinate.

#### Function Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `agent_response` | `dict[str, any]` | Yes | Schema dictionary containing `output` (str), `attention` (bool), and `coordinate` (float). |

#### Return Value

| Type | Description |
| :--- | :--- |
| `None` | Evaluates response and triggers `push_attention()` or `divert_attention()`. |

---

### `raw_schema()`
> Returns the raw schema configuration required by the LLM system prompt to format schematic responses.

#### Function Parameters - NONE

#### Return Value

| Type | Description |
| :--- | :--- |
| `dict[str, dict[str, str]]` | Dictionary defining expected schema parameters (`output`, `attention`, `coordinate`). |

---

### `push_attention()` **(async)**
> Appends new attention text to the attention history list and registers it into the `AttentionSpace` memory.

#### Function Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `attention` | `str` | Yes | The attention context string to store in memory. |
| `distance` | `float` | Yes | The calculated embedding coordinate/distance. |

#### Return Value

| Type | Description |
| :--- | :--- |
| `None` | Updates internal state asynchronously. |

---

### `divert_attention()`
> Recalculates attention placement distance using `compute_attention()` and shifts context away from previous focus.

#### Function Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `attention` | `str` | Yes | The new context text to shift focus toward. |
| `distance` | `float` | Yes | Base coordinate distance value. |

#### Return Value

| Type | Description |
| :--- | :--- |
| `None` | Updates state via computed distance values. |

---

### `attention_check()`
> Applies an exponential decay calculation ($e^{-\text{decay\_rate} \times \Delta t}$) to a vector state based on elapsed time to pull extreme states back toward neutral (0).

#### Function Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `saved_state_vector` | `np.ndarray` | Yes | The numpy array representing saved state vectors. |
| `last_timestamp` | `time.struct_time` | Yes | Timestamp of the last evaluation. |
| `decay_rate` | `float` | Yes | Coefficient determining the rate of exponential decay. |

#### Return Value

| Type | Description |
| :--- | :--- |
| `typing.Any \| np.ndarray` | Returns the decayed numpy array state, or `False` if decay preconditions fail. |

---

### `access_attention()`
> Placeholder method for accessing or querying attention states.

#### Function Parameters - NONE

#### Return Value - NONE
