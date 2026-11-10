import time, typing, math
import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class coords():
    x: int
    y: int

class AttentionSpace:

    working_memory: dict[str, any]
    memory_space: np.ndarray

    __slots__ = ['working_memory', 'memory_space']

    def __init__(self):
        self.working_memory: dict[str, any] = {}
        self.memory_space = np.zeros((1000, 3))

    def compute_attention(self, attention_coords: coords) -> str:

        if (len(self.memory_space.shape[0] == 0)):
            return "0,0"

        sorted_working_mem = sorted(int(k.replace(",", "")) for k in self.working_memory.keys())
        curr_short_mem = sorted_working_mem[0]
        curr_short_anchor = self.memory_space[0]

        # We compute based on their X - context importance, and Y - context relevance, 
        # so we assume context importance has a relationship with context relevance
        dot = np.dot([attention_coords.x, attention_coords.y], [curr_short_mem[0], curr_short_mem[1]])
        magnitude_importance = math.sqrt((attention_coords.x ** 2 + attention_coords.y[0] ** 2))
        magnitude_relevance = math.sqrt((curr_short_mem[0] ** 2 + curr_short_mem[1] ** 2))
        similarity  = 0
        similarity = dot / (magnitude_importance * magnitude_relevance)

        distance = ((attention_coords.x - curr_short_mem[0])) + ((attention_coords.y - curr_short_mem[1]))
        attention_equation = similarity * distance

        if (attention_equation == 0):
            return attention_coords
        elif (attention_equation > 1):
            return f"{attention_coords.x - curr_short_mem[0]},{attention_coords.y - curr_short_mem[1]}"
        elif (attention_equation < 1):
            return f"{attention_coords.x - curr_short_mem[0] - distance},{attention_coords.y - curr_short_mem[1] - distance}"

    def add_memory(self, attention: str, attn_coords: coords) -> None | bool:
        xy: str = self.compute_attention(attn_coords)

        if not (isinstance(xy, str)):
           return False

        # { ROW attn_coords[0], COL attn_coords[2]:  "I washed away the remaining plates."}

        self.memory_space[attn_coords.x, attn_coords.y] = attention
        self.working_memory[f"{str(attn_coords.x)},{str(attn_coords.y)}"] = attention

    def decay_current(self, decayed_attn_score: float) -> None | bool:

        if (self.working_memory[0, 2] < 40.0):
            return False

        self.working_memory[0, 2] = decayed_attn_score
        return

    def new_short_term_memory(self, attention) -> None:

        # check if any attention embeddings are at most 5 coordinates away, we could keep them.
        # 5 - not too relevant, still a little bit of context and relevance

        match = (self.memory_space[0, 0] > self.memory_space) & (self.memory_space[0, 1] > self.memory_space)
        matching_mems = self.memory_space[match]

        matching_mems_embeds = np.where(match)[0]
        new_memory_space = np.zeros((1000, 3))

        new_memory_space[matching_mems_embeds] = self.memory_space[matching_mems_embeds]

        self.memory_space = np.zeros((1000, 3))
        self.memory_space = new_memory_space

        self.memory_space[0,0] = 0
        self.memory_space[0,1] = 0
        self.memory_space[0,2] = 0.00
        self.working_memory.clear()

        #indices = np.array(matching_mems_embeds)
        #N = len(indices)

        #self.working_memory[f"{indices[:N-3]}, {}"] = indices[:N-3]  
        #self.working_memory[f"{indices[:N-3]}, {}"] = indices[1:N-2]  
        #self.working_memory[f"{indices[:N-3]}, {}"] = indices[2:N-1]  

    def get_curr_attn(self) -> list[np.ndarray, float]:

        if len(self.working_memory.items() > 4):
            return [self.memory_space[0], 15.0]

        return [self.memory_space[0], 5.0]

class Gate:

     attention: AttentionSpace
     attention_list: list[float]

     attention_count: int
     current_attention: dict[str, any]

     schema = {
        "type": "object",
        "properties": {
            "output": {
                "type": "string",
                "description": "LLM Output"
            },
            "attention": {
                "type": "boolean",
                "description": "True or False only. Do not capitalize as True/False if outputting raw JSON. Focus on this output or not = context"
            }, 
            "x": {
                "type": "number",
                "description": "context importance coordinate. Must be a plain decimal number between 0.00 and 1.00 (e.g., 0.55). Do not use scientific notation, NaN, or text. X coordinate -> memoryspace"
            },
            "y": {
                "type": "number",
                "description":  "context relevance coordinate. Must be a plain decimal number between 0.00 and 1.00 (e.g., 0.55). Do not use scientific notation, NaN, or text. Y coordinate -> memoryspace"
            },
        }
     }

     __slots__ = ['attention', 'attention_list', 'current_attention', 'xy', 'time']
     
     def __init__(self):
        
        self.attention = AttentionSpace()
        self.attention_list = []
        self.current_attention = {}
        self.xy = coords(x=0, y=0)

     def attach(self, deep_memory: bool = False) -> dict[str, str]:

        self.time = time.time()

        if (len(self.attention.working_memory == 0)):
            return {"0,0": ""}

        if (deep_memory):
            return self.attention.working_memory

        curr_attn = list(self.attention.working_memory.keys())[-1]
        return self.attention.working_memory[curr_attn]

     def process(self, agent_response: dict[str, any]) -> None:
        '''
            The LLM has a system prompt that must determine on its own whether
            to focus on its current response, by outputting a schematic response
        '''

        self.xy.x = agent_response["x"]
        self.xy.y = agent_response["y"]

        if (agent_response["attention"] == False):
            return
        
        self.push_attention(agent_response["output"], self.xy)

     def raw_schema(self) -> dict[str, dict[str, str]]: 

        return {"output": {
                    "type": "string",
                    "description": "LLM Output"
                },
                "attention": {
                    "type": "bool",
                    "description": "Attention focus on the output based on LLM thinking"
                },
                "x": {
                    "type": "float",
                    "description": "Context importance - X coordinate placement based on LLM thinking & returned working memory"
                },
                "y": {
                    "type": "float",
                    "description": "Context relevance - Y coordinate placement based on LLM thinking & returned working memory"
                },
            }

     def push_attention(self, attention: str, xy: coords) -> None:

        '''
            Pushing to the attention space is what makes the LLM agent remember what to remember on,
            and rely on for context during roleplays.
        '''

        curr_anchor = self.attention.get_curr_attn()

        curr_attn_decay: float = self.attention_check(curr_anchor[0], self.time, curr_anchor[1])
        self.attention.decay_current(curr_attn_decay)
        self.attention.add_memory(attention, xy)
    

     def attention_check(self, curr_attn_decay: float, last_timestamp: float, decay_rate: float) -> typing.Any | float:
            """
            Applies exponential decay to the working memory based on elapsed time.
            """
            current_time = time.time()
            time_delta_seconds = current_time - last_timestamp
            decay_factor = 0
    
            if (time_delta_seconds < current_time and decay_rate >= 1.0): 
                return False
            elif (time_delta_seconds > current_time and decay_rate <= 0): 
                decay_factor = np.exp(1 - decay_rate * time_delta_seconds)
    
            decayed_curr_attn = curr_attn_decay * decay_factor
            
            return decayed_curr_attn