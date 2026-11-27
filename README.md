# MemoryGate
A lightweight memory layer for roleplay chat agents. A gate filters what’s worth keeping from each turn, and a bounded working memory decays old items and builds a short context for the roleplay LLM, so prompts don’t grow with the conversation and establish realism with real-life conversation attention & working memory

<img width="1024" height="687" alt="image" src="https://github.com/user-attachments/assets/549408e4-c3cf-4d58-9cfa-42337d1860a4" />

# MemoryGate: Dev Workflow Guide

MemoryGate uses a two-way data structure system: a dictionary and a numpy array. The dictionary serves as the interface for
non-computational operations, whereas the numpy array serves as the ground truth and computational operations. 

You must use the ```.attach()``` method from ```Gate``` class to feed the current memory attention anchor to the LLM model of your chatbot.
The Gate itself houses the "short-term memory" and feeds this onto the short-term context of the chatbot.

## Prerequisites

Before using MemoryGate, ensure your environment meets the following requirements at [requirements.txt](https://github.com/ajlrza/MemoryGate/pyproject.toml)
If prerequisites are not yet installed or are outdated, follow the steps below

## Installation

You can install the `MemoryGate` package directly from the Git repository using pip.

```bash
pip install git+https://github.com/ajlrza/MemoryGate.git
```

```bash
pip install -r requirements.txt
```

# MemoryGate Example Usage


> Assuming an openAI chat template

```python

from MemoryGate import memorygate

mem_gate = memorygate.Gate()

def roleplay(rp_msg):
    try:
        
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "system",
                    "content": "ROLEPLAY SYSTEM PROMPT HERE"
                },
                {
                    "role": "system",
                    "content": mem_gate.attach()
                }
                {
                    "role": "user",
                    "content": rp_msg,
                },
            ],
            schema = mem_gate.schema
            temperature=0.7,
            max_tokens=150
        )
        
        message = response.choices[0].message.content
        attention = mem_gate.process(message)

        return message

    except Exception as e:
        print(f"An error occurred: {e}")
```

# Example Outputs

```python

# Initial memory space
None

# First interaction with roleplay agent
('1,2', SpaceWorkingMemAttn(Attn='User likes coffee', Score=90.0))

# Second interaction with roleplay agent
{'1,2': SpaceWorkingMemAttn(Attn='User likes coffee', Score=90.0), '3,4': SpaceWorkingMemAttn(Attn='Roleplay in a cafe', Score=75.0)}

# N-times interaction with roleplay agent
{'1,2': SpaceWorkingMemAttn(Attn='User likes coffee', Score=90.0), '3,4': SpaceWorkingMemAttn(Attn='Roleplay in a cafe', Score=75.0), 'N': SpaceWorkingMemAttn(Attn='N-th Attention', Score="N.N")}

```

> [!NOTE]
> **MemoryGate Usage:** MemoryGate offers the Gate class and can be used wherever, as long as ```.attach()``` is called and added within the chatbot's and are not coming from the user itself. Meaning, ```.attach()``` must be glued onto the LLM layer of your software, rather than onto the user's since this could cause an intention mismatch for MemoryGate use-case.

# Core Premise of MemoryGate

MemoryGate uses a concept called "attention anchor", this is for the short-term memory realism.
It is always the first row for structural purposes and as the center in space.
This method returns it as well as the decay rate associated per access.

To mimic human attention span, it can only hold at most 4, so we 
take that concept and apply it to the working memory dict item list.
The working memory dict is in parallel with the memory space for the
easier data access, while memory space serves as the geometric truth.

"Attention" is the text inputs from the roleplay, which comes from you.
The roleplay agent stores your text inputs in the memory space, and therefore
labelling it as attention.

# Example 2D Numpy Memory Space

| Row (`x_row`) | `context importance` | `context relevance` | `score` | `attention` |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `1` | `2` | `90.0` | `'User likes coffee'` |
| `1` | `3` | `4` | `75.0` | `'Roleplay in a cafe'` |
