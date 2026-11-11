import time, typing, math
import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class Coords(typing.NamedTuple):
    x: int
    y: int

@dataclass(slots=True)
class DecayedAttn(typing.NamedTuple):
    CurrAttnFocus: np.ndarray
    DecayRate: typing.Literal[5,15]

@dataclass(slots=True)
class WorkingMemAttn(typing.NamedTuple):
    Attn: str
    Score: float

@dataclass(slots=True)
class StoredWorkingMem(typing.NamedTuple):
    Pos: Coords
    Attn: WorkingMemAttn

def AttentionMemAdder(mem_space: np.ndarray[1000, 4], attn_coords: Coords | list[int], attn: str) -> np.ndarray:

    x = attn_coords.x if isinstance(attn_coords, Coords) else attn_coords[0] if isinstance(attn_coords, list) else None
    y = attn_coords.y if isinstance(attn_coords, Coords) else attn_coords[1] if isinstance(attn_coords, list) else None

    mem_space[x, y] = x
    mem_space[x, y + 1] = y
    mem_space[x, y + 2] = 100
    mem_space[x, y + 2] = attn

    return mem_space

class AttentionSpace:
    working_memory: dict[str, WorkingMemAttn]
    memory_space: np.ndarray

    __slots__ = ['working_memory', 'memory_space', 'decay']

    def __init__(self):
        self.working_memory: dict[str, any] = {}
        dt = np.dtype([('context impt.', 'i4'), ('context relv.', 'i4'), ('score', 'f8'), ('attention', 'U10')])
        self.memory_space = np.zeros(1000, dtype=dt)
        self.decay = DecayedAttn(CurrAttn=None, DecayRate=5)

    def compute_attention(self, attention_coords: Coords) -> str:

        if (len(self.memory_space.shape[0]) == 0):
            return "0,0"

        curr_attn_anchor = self.memory_space[0]

        # We compute based on their X - context importance, and Y - context relevance, 
        # so we assume context importance has a relationship with context relevance
        dot = np.dot([attention_coords.x, attention_coords.y], [curr_attn_anchor[0], curr_attn_anchor[1]])
        magnitude_importance = math.sqrt((attention_coords.x ** 2) + (attention_coords.y ** 2))
        magnitude_relevance = math.sqrt((curr_attn_anchor[0] ** 2) + (curr_attn_anchor[1] ** 2))
        similarity = dot / (magnitude_importance * magnitude_relevance)

        distance = ((attention_coords.x - curr_attn_anchor[0])) + ((attention_coords.y - curr_attn_anchor[1]))
        attention_equation = similarity * distance

        if (attention_equation == 0):
            return attention_coords
        elif (attention_equation > 1):
            return f"{attention_coords.x - curr_attn_anchor[0]},{attention_coords.y - curr_attn_anchor[1]}"
        elif (attention_equation < 1):
            return f"{attention_coords.x - curr_attn_anchor[0] - distance},{attention_coords.y - curr_attn_anchor[1] - distance}"

    def add_memory(self, attention: str, attn_coords: Coords) -> None | bool:
        xy: str = self.compute_attention(attn_coords)

        if not (isinstance(xy, str)):
           return False

        # { ROW attn_coords[0], COL attn_coords[2]:  "I washed away the remaining plates."
        self.memory_space = AttentionMemAdder(self.memory_space, attn_coords, attention)

        self.working_memory[f"{str(attn_coords.x)},{str(attn_coords.y)}"] = attention

    def decay_current(self, decayed_attn_score: float) -> None:

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
        self.memory_space = AttentionMemAdder(mem_space=self.memory_space, attn_coords=[0,0], attn="")
        self.working_memory.clear()

    def get_curr_attn(self) -> DecayedAttn | None:

        """Returns the current attention anchor and its decay rate.

        The decay rate is higher (15.0) if working memory exceeds 4 items,
        otherwise it defaults to 5.0.
        """

        if (len(self.memory_space) == 0):
            return None

        self.decay.CurrAttnFocus = self.memory_space[0]

        if len(self.working_memory.items()) > 4:
            self.decay.DecayRate = 15.0
            return self.decay
            
        return self.decay

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
        self.xy = Coords(x=0, y=0)

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

     def push_attention(self, attention: str, xy: Coords) -> None:

        '''
            Pushing to the attention space is what makes the LLM agent remember what to remember on,
            and rely on for context during roleplays.
        '''

        curr_anchor = self.attention.get_curr_attn()
        decay = False

        if (curr_anchor.CurrAttnFocus[2] > 40):
            decay = True

        curr_attn_decay: float | None = self.attention_check(curr_anchor.CurrAttnFocus, self.time, curr_anchor.DecayRate) if decay else None

        if (isinstance(curr_attn_decay, None)):
            self.attention.add_memory(attention, xy)
            return

        self.attention.decay_current(curr_attn_decay)
        self.attention.add_memory(attention, xy)
        return

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