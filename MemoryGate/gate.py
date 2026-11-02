import os, sys, time, typing
import numpy as np

class AttentionSpace:

    working_memory: dict[str, str]
    memory_space: np.ndarray

    __slots__ = ['working_memory', 'memory_space']

    def __init__(self):
        self.working_memory = {}
        self.memory_space = np.zeros((1000, 128))
        
    def compute_attention(self, attention_embedding: float):
        pass
        # soon, basically we just use the np ndarray as vector space
        
    def add_memory(self, attention, attention_embedding: float):
        self.working_memory[f'{attention_embedding}'] = attention

class Gate:

     attention: AttentionSpace
     attention_list: list[float]
     attention_count: int
     current_attention = dict[str, any]
     schema = {
        "type": "object",
        "properties": {
            "output": {
                "type": "string",
                "description": "LLM Output"
            },
            "attention": {
                "type": "bool",
                "description": "Attention focus on the output based on LLM thinking"
            },
            "coordinate": {
                "type": "float",
                "description": "Embedding coordinate placement based on LLM thinking"
            }
        }
     }

     __slots__ = ['attention', 'attention_list', 'current_attention', 'attention_count']
     
     def __init__(self):
        
        self.attention = AttentionSpace()
        self.attention_list = []
        self.current_attention = {}
        self.attention_count = len(self.attention_list)

     def attach(self) -> dict[str, str]:

        return self.attention.working_memory

     def process(self, agent_response: dict[str, any]) -> None:
        '''
            The LLM has a system prompt that must determine on its own whether
            to focus on its current response, by outputting a schematic response
            {"output": text, "attention": bool, "coordinate": float}
        '''

        if (agent_response["attention"] == False):
            embedding = -1
            distance = agent_response["coordinate"]
            self.divert_attention(agent_response["output"], embedding * distance)
        
        embedding = 1
        distance = agent_response["coordinate"]
        self.push_attention(agent_response["output"], embedding * distance)

     def raw_schema(self) -> dict[str, dict[str, str]]: 

        return {"output": {
                    "type": "string",
                    "description": "LLM Output"
                },
                "attention": {
                    "type": "bool",
                    "description": "Attention focus on the output based on LLM thinking"
                },
                "coordinate": {
                    "type": "float",
                    "description": "Embedding coordinate placement based on LLM thinking & current attention space"
                }}

     async def push_attention(self, attention: str, distance: float) -> None:

        '''
            Pushing to the attention space is what makes the LLM agent remember what to remember on,
            and rely on for context during roleplays.
        '''

        self.attention_list.append(attention)
        self.attention.add_memory(attention, distance)
                    
     def divert_attention(self, attention: str, distance: float) -> None:

        '''
            Diverting attention means to get rid of the current context in the roleplay session, and focus on newer attentions,
            unless the LLM does not decide so.
        '''

        new_attention_distance = self.attention.compute_attention(distance)
        self.push_attention(attention, new_attention_distance)

     def attention_check(self, saved_state_vector: np.ndarray, last_timestamp: time.struct_time, decay_rate: float) -> typing.Any | np.ndarray:
            """
            Applies exponential decay to the emotional vector based on elapsed time.
            Brings extreme emotions back toward 0 (Neutral).
            """
            current_time = time.time()
            time_delta_seconds = current_time - last_timestamp
            decay_factor = 0
    
            if (time_delta_seconds < current_time and decay_rate >= 1.0): 
                return False
    
                #approaches 0
            if (time_delta_seconds > current_time and decay_rate <= 0): 
                decay_factor = np.exp(-decay_rate * time_delta_seconds)
    
            decayed_state = saved_state_vector * decay_factor
            
            return decayed_state

     def access_attention():
        pass