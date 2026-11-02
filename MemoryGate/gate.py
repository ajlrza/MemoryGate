import os, sys, time
import numpy as np

class AttentionSpace:

    working_memory: dict[str, str]
    memory_space: any

    __slots__ = ['working_memory', 'memory_space']

    def __init__(self):
        self.working_memory = {}
        self.memory_space = np.zeros((1000, 128))
        
    def working_memory(self):
        return len(self.working_memory)
    
    def add_temp_memory(self, attention: str, attention_embedding: float):
        self.working_memory[f'{attention_embedding}'] = attention
        
    def add_memory(self, attention):
        self.working_memory['attention'] = attention

class Gate:

     attention: AttentionSpace
     attention_list: list[float]
     attention_count: int
     current_attention = dict[str, any]

     __slots__ = ['attention', 'attention_list', 'current_attention', 'attention_count']
     
     def __init__(self):
        
        self.attention = AttentionSpace()
        self.attention_list = []
        self.current_attention = {}
        self.attention_count = len(self.attention_list)

     def process(self, agent_response: dict[str, any]) -> None:
        '''
            The LLM has a system prompt that must determine on its own whether
            to focus on its current response, by outputting a schematic response
            {"output": text, "focus": bool}
        '''

        if (agent_response["focus"] == False):
            embedding = -1
            distance = len(agent_response["output"])
            self.divert_attention(agent_response["output"], embedding * distance)
        
        embedding = 1
        distance = len(agent_response["output"])
        self.push_attention(agent_response["output"], embedding * distance)

     async def push_attention(self, attention: str, distance: float):

        '''
            Pushing to the attention space is what makes the LLM agent remember what to remember on,
            and rely on for context during roleplays.
        '''

        for attention in attention:
            self.attention_list.append(attention)
        # For networks to know
        return self.attention.working_memory
                    
     def divert_attention(self, attention: str, distance: float):

        '''
            Diverting attention means to get rid of the current context in the roleplay session, and focus on newer attentions,
            unless the LLM does not decide so.
        '''
               
        while (self.attention.working_memory() < 100):
            self.attention.add_memory(self.attention_list[self.task_count - 1])

            if (len(self.attention.working_memory.keys()) != 100): 
                self.attention.add_memory(self.attention_list[task_count])
                self.attention_list.pop(task_count)
                task_count += 1
            else:
                self.attention.add_temp_memory(self.attention_list[task_count])
                self.attention_list.pop(taskCount)
                taskCount += 1

     def attention_check(self, saved_state_vector: np.array, last_timestamp: time, decay_rate: float):
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
                    self.decay_factor = np.exp(-decay_rate * time_delta_seconds)
        
                decayed_state = saved_state_vector * decay_factor
                
                return decayed_state

     def access_attention():
        pass