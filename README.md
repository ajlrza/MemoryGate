# MemoryGate
A lightweight memory layer for roleplay chat agents. A gate filters what’s worth keeping from each turn, and a bounded working memory decays old items and builds a short context for the roleplay LLM, so prompts don’t grow with the conversation and establish realism with real-life conversation attention & working memory

<img width="2760" height="1360" alt="chat_memory_gate_pipeline_white" src="https://github.com/user-attachments/assets/07216840-6ffa-4c93-a076-453370abfb3a" />


# How To Use

Use the library's main gate class to direct your agent outputs 

# Example

```python

from MemoryGate import gate

mem_gate = gate.Gate()

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