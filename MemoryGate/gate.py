import os, sys, time, typing
import numpy as np

class AttentionSpace:

    working_memory: dict[str, any]
    memory_space: np.ndarray

    __slots__ = ['working_memory', 'memory_space']

    def __init__(self):
        self.working_memory: dict[str, any] = {}
        self.memory_space = np.zeros((1000, 128))
        

    # instead of tuple, maybe ill just use "xy" -> "12" then use the static position of 0 and 1 f6r xy... so i can avoid
    # compute heavy shit, also i decided to stop using a list of dict, cuz its gonna make it harder to structure and organize thedata right
    def compute_attention(self, attention_coords: tuple[float]) -> str:
        sorted_mem = sorted(k.replace(",", " ") for k in self.working_memory.keys())

        dot = None # soon
        similarity = None # soon

        # sorted_mem[-1] is the max

        upper_boundary = ((attention_coords[0] - sorted_mem[-1][0]) ** 2) + ((attention_coords[1] - sorted_mem[-1][2]) ** 2)
        

        # For the current context, this coordinate stores a memory
        # The nearer it is to the current context, the higher its relevance
        #self.memory_space[x_val, y_val] = 1

        #return f"({self.working_memory[0][x_val], self.working_memory[1][y_val]})"

    def add_memory(self, attention: str, attn_coords: str) -> None | bool:
        xy: str | list[str] = self.compute_attention(attn_coords)

        if (not (isinstance(xy, str)) or not isinstance(xy, list[str])):
            return False

        # { ROW attn_coords[0], COL attn_coords[2]:  "I washed away the remaining plates."}

        self.working_memory[attn_coords[0], attn_coords[2]] = attention

    def dilute_memory(self, attention, new_attn_embeds: tuple[float]) -> None | bool:

        sorted_mem = sorted(int(k.replace(",", "")) for k in self.working_memory.keys())
        curr_attn_anchor = str(sorted[-1])
        del self.working_memory[curr_attn_anchor[:1]+ "," + curr_attn_anchor[1:]]

        xy: str | list[str] = self.compute_attention(new_attn_embeds)

        self.working_memory[xy] = xy, attention

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

     __slots__ = ['attention', 'attention_list', 'current_attention', 'attention_count', 'time']
     
     def __init__(self):
        
        self.attention = AttentionSpace()
        self.attention_list = []
        self.current_attention = {}

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
            self.push_attention(agent_response["output"], f"{agent_response["x"],agent_response["y"]}", True)
        
        self.push_attention(agent_response["output"], f"{agent_response["x"],agent_response["y"]}")

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

     def push_attention(self, attention: str, xy: str, divert: bool = False) -> None:

        '''
            Pushing to the attention space is what makes the LLM agent remember what to remember on,
            and rely on for context during roleplays.
        '''

        if (divert):
            self.attention.dilute_memory(attention, xy)

        attention_remembrance = self.attention_check(self.time)
        self.attention.add_memory(attention, xy)
    

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