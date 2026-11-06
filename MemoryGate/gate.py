import os, sys, time, typing
import numpy as np

class AttentionSpace:

    working_memory: dict[str, str]
    memory_space: np.ndarray

    __slots__ = ['working_memory', 'memory_space']

    def __init__(self):
        self.working_memory: list[dict[str, any]] = []
        self.memory_space = np.zeros((1000, 128))
        
    def compute_attention(self, attention_coords: tuple[float]) -> str:

        x_max = sorted(float(k) for k in self.working_memory[0].keys())
        y_max = sorted(float(k) for k in self.working_memory[1].keys())

        dot = None # soon
        similarity = None # soon

        distance = ((attention_coords[0] - x_max[-1]) ** 2) + ((attention_coords[1] - y_max[-1]) ** 2)
        attended_memories = []

        if (distance > 1):

            for i in range(int(distance)):
                x_val = x_max[-1] - i
                y_val = y_max[-1] - i

                x = self.working_memory[0][x_val]
                y = self.working_memory[1][y_val]

                # For the current context, this coordinate stores a memory
                # The nearer it is to the current context, the higher its relevance
                self.memory_space[x_val, y_val] = 1

                attended_memories.append(f"({x,y})")
            
            return attended_memories

        x_val = x_max[-1] - distance
        y_val = y_max[-1] - distance

        # For the current context, this coordinate stores a memory
        # The nearer it is to the current context, the higher its relevance
        self.memory_space[x_val, y_val] = 1

        return f"({self.working_memory[0][x_val], self.working_memory[1][y_val]})"

    def add_memory(self, attention, attn_coords: tuple[float]) -> None | bool:
        # need an algo soon, or maybe just ask LLM to give the x and y coordinate
        xy: str | list[str] = self.compute_attention(attn_coords)

        if (not (isinstance(xy, str)) or not isinstance(xy, list[str])):
            return False

        # { "(2,7)": ("(2,5)" "I washed away the remaining plates.")}
        # { "(2,7)": (["(2,5)", "(4,7)", "(3,5)"], "I washed away the remaining plates.")}

        self.working_memory[0][attn_coords[0]] = attention
        self.working_memory[1][attn_coords[1]] = attention
        self.working_memory[2][f"({attn_coords[0], attn_coords[1]})"] = {xy, attention}

    def dilute_memory(self, attention, old_attn_embeds: tuple[float], new_attn_embeds: tuple[float]) -> None | bool:

        if (old_attn_embeds[0] not in self.working_memory[0].keys() and old_attn_embeds[1] not in self.working_memory[1].keys()):
            return False

        del self.working_memory[0][old_attn_embeds[0]]
        del self.working_memory[1][old_attn_embeds[1]]
        del self.working_memory[2][f"({old_attn_embeds[0], old_attn_embeds[1]})"]

        self.working_memory[0][new_attn_embeds[0]] = attention
        self.working_memory[1][new_attn_embeds[1]] = attention

        xy: str | list[str] = self.compute_attention(new_attn_embeds)

        self.working_memory[2][f"({new_attn_embeds[0], new_attn_embeds[1]})"] = (xy, attention)

    def new_short_term_memory(self, attention, attn_embeds: tuple[float]) -> None | bool:

        # check if any attention embeddings are at most 5 coordinates away, we could keep them.
        # 5 - not too relevant, still a little bit of context and relevance
        pass

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
                "type": "bool",
                "description": "Attention focus on the output based on LLM thinking & returned working memory"
            },
            "x": {
                "type": "float",
                "description": "Relevance X coordinate placement based on LLM thinking & returned working memory"
            },
            # Y value is the context impact based on the relevance of the attended memory.
            "y": {
                "type": "float",
                "description": "Context Impact Y coordinate placement based on LLM thinking & returned working memory"
            },
        }
     }

     __slots__ = ['attention', 'attention_list', 'current_attention', 'attention_count', 'schema']
     
     def __init__(self):
        
        self.attention = AttentionSpace()
        self.attention_list = []
        self.current_attention = {}
        self.attention_count = len(self.attention_list)

     def attach(self, deep_memory: bool = False) -> dict[str, str]:

        self.time = time.time()

        if (deep_memory):
            return self.attention.working_memory

        return self.attention.working_memory[-1]

     def process(self, agent_response: dict[str, any]) -> None:
        '''
            The LLM has a system prompt that must determine on its own whether
            to focus on its current response, by outputting a schematic response
        '''

        if (agent_response["attention"] == False):
            self.divert_attention(agent_response["output"], (agent_response["x"], agent_response["y"]))
        
        self.push_attention(agent_response["output"], (agent_response["x"], agent_response["y"]))

     def raw_schema(self) -> dict[str, dict[str, str]]: 

        return {"output": {
                    "type": "string",
                    "description": "LLM Output"
                },
                "attention": {
                    "type": "bool",
                    "description": "Attention focus on the output based on LLM thinking"
                },
                "x_coordinate": {
                    "type": "float",
                    "description": "Embedding X coordinate placement based on LLM thinking & returned working memory"
                },
                "y_coordinate": {
                    "type": "float",
                    "description": "Embedding Y coordinate placement based on LLM thinking & returned working memory"
                },
            }

     def push_attention(self, attention: str, xy: tuple[float]) -> None:

        '''
            Pushing to the attention space is what makes the LLM agent remember what to remember on,
            and rely on for context during roleplays.
        '''

        attention_remembrance = self.attention_check(self.time)
        self.attention.add_memory(attention, xy)
                    
     def divert_attention(self, attention: str, xy: tuple[float]) -> None:

        '''
            Diverting attention means to get rid of the current context in the roleplay session, and focus on newer attentions,
            unless the LLM does not decide so.
        '''

        mem = self.working_memory[curr_xy]

        displacement = (mem[0] - xy[0], mem[1] - xy[1])
        new_attention_distance = self.attention.compute_attention(displacement)
        self.push_attention(attention, new_attention_distance)

     def attention_check(self, saved_state_vector: np.ndarray, last_timestamp: time.struct_time, decay_rate: float) -> typing.Any | np.ndarray:
            """
            Applies exponential decay to the working memory based on elapsed time.
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