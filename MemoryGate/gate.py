import time, typing, math
import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class Coords():
    x: int
    y: int

@dataclass(slots=True)
class DecayedAttn():
    AttnFocus: np.ndarray
    DecayRate: typing.Literal[5,15]

@dataclass(slots=True)
class DictWorkingMemAttn():
    ContxtImportance: int
    ContxtRelevance: int
    Attn: str
    Score: float

@dataclass(slots=True)
class SpaceWorkingMemAttn():
    Attn: str
    Score: float

@dataclass(slots=True)
class SpaceStoredWorkingMem():
    Pos: Coords
    Attn: SpaceWorkingMemAttn

CONTENT_KEY_NAMES: set[str] = {
    "content",  # OpenAI, Anthropic, Mistral, Groq, Together AI, Perplexity
    "text",  # Google Gemini, Cohere, Anthropic (legacy v1), AWS Bedrock
    "completion",  # Anthropic (legacy completion API), AI21
    "generation",  # Cohere (legacy generation API)
    "output",  # Replicate, Hugging Face TGI
}

def attn_mem_adder(mem_dict: dict[str, any], mem_space: np.ndarray, attn_coords: Coords | list[float], attn: str) -> np.ndarray:

    if (len(mem_dict.items()) > 1000):
        return mem_space

    x = attn_coords.x if isinstance(attn_coords, Coords) else attn_coords[0] if isinstance(attn_coords, list) else None
    y = attn_coords.y if isinstance(attn_coords, Coords) else attn_coords[1] if isinstance(attn_coords, list) else None

    if (x == None or y == None):
        return mem_space

    x_row = int(x)
    y_col = int(y)

    if (x > 1000 or y > 1000 & len(mem_dict.items()) < 1000):
        last_mem_idx = int(list(mem_dict.keys())[-1][0])
        x_row = last_mem_idx + 1

    mem_space['context importance'][x_row] = x_row
    mem_space['context relevance'][x_row] = y_col
    mem_space['score'][x_row] = 100
    mem_space['attention'][x_row] = attn

    return mem_space

dt = np.dtype([('context importance', 'i4'), ('context relevance', 'i4'), ('score', 'f8'), ('attention', 'U10')])

class AttentionSpace:
    working_memory: dict[str, SpaceWorkingMemAttn]
    memory_space: np.ndarray
    decay: DecayedAttn

    __slots__ = ['working_memory', 'memory_space', 'decay']

    def __init__(self):
        self.working_memory: dict[str, SpaceWorkingMemAttn] = {}
        self.memory_space = np.zeros((1000, 4), dtype=dt)
        self.decay = DecayedAttn(AttnFocus=None, DecayRate=5)

    def compute_attention(self, attention_coords: Coords) -> Coords:

        if (len(self.working_memory.items()) == 0):
            return "0,0"

        # We compute based on their X - context importance, and Y - context relevance, 
        # so we assume context importance has a relationship with context relevance
        dot = np.dot([attention_coords.x, attention_coords.y], [self.memory_space[0,0]['context importance'], self.memory_space[0,0]['context relevance']])
        magnitude_importance = math.sqrt((attention_coords.x ** 2) + (attention_coords.y ** 2))
        magnitude_relevance = math.sqrt((self.memory_space[0,0]['context importance'] ** 2) + (self.memory_space[0,0]['context relevance'] ** 2))
        similarity = dot / (magnitude_importance * magnitude_relevance)

        distance = ((attention_coords.x - self.memory_space[0][0]['context importance'])) + ((attention_coords.y - self.memory_space[0,0]['context relevance']))
        attention_equation = similarity * distance

        Coords(x=0,y=0)

        if (attention_equation == 0):
            return attention_coords
        elif (attention_equation > 1):
            return Coords(
                x=attention_coords.x - self.memory_space[0,0]['context importance'],
                y=attention_coords.y - self.memory_space[0,0]['context relevance']
                )
        elif (attention_equation < 1):
            return Coords(
                x=attention_coords.x - self.memory_space[0,0]['context importance'] - distance,
                y=attention_coords.y - self.memory_space[0,0]['context relevance'] - distance
            )

    def add_memory(self, attention: str, attn_coords: Coords) -> None | bool:
        xy: Coords = self.compute_attention(attn_coords)

        if not (isinstance(xy, str)):
           return False

        self.memory_space = attn_mem_adder(
            mem_dict=self.working_memory, 
            mem_space=self.memory_space, 
            attn_coords=xy, 
            attn=attention)

        self.working_memory[f"{str(xy.x)},{str(xy.y)}"].Attn = attention

    def decay_current(self, decayed_attn_score: float) -> None:

        self.working_memory[0, 2] = decayed_attn_score
        return

    def new_short_term_memory(self, attention) -> None:

        # check if any attention embeddings are at most 5 coordinates away, we could keep them.
        # 5 - not too relevant, still a little bit of context and relevance

        match = (self.memory_space[0,0]['context importance'] > self.memory_space['context importance']) & (self.memory_space[0,0]['context relevance'] > self.memory_space['context relevance'])

        matching_mems_embeds = np.where(match)[0]
        new_memory_space = np.zeros((1000, 4), dtype=dt)

        new_memory_space[matching_mems_embeds] = self.memory_space[matching_mems_embeds]

        self.memory_space = new_memory_space
        self.memory_space = attn_mem_adder(mem_dict=self.working_memory,mem_space=self.memory_space, attn_coords=[0,0], attn="")
        self.working_memory.clear()

    def get_curr_attn(self) -> DecayedAttn | None:

        """Returns the current attention anchor and its decay rate.

        The decay rate is higher (15.0) if working memory exceeds 4 items,
        otherwise it defaults to 5.0.
        """

        if (len(self.working_memory.items()) == 0):
           return None
        
        self.decay.AttnFocus= self.memory_space[0]

        if len(self.working_memory.items()) > 4:
            self.decay.DecayRate = 15.0
            return self.decay
            
        return self.decay

    def get_mem_space_dict(self) -> dict[str, SpaceWorkingMemAttn]:
        
        return self.working_memory

    def get_mem_space(self) -> np.ndarray:

        return self.memory_space

class Gate:

     attention: AttentionSpace
     attention_count: int
     current_attention: dict[str, DictWorkingMemAttn]
     schema: dict[str, any]
     time: float
     xy: Coords

     __slots__ = ['attention', 'attention_count', 'current_attention', 'schema', 'time', 'xy']
     
     def __init__(self):
        
        self.attention = AttentionSpace()
        self.current_attention = {}
        self.attention_count = 0
        self.time = 0.0
        # might decouple from gate?
        self.xy = Coords(x=0, y=0)
        self.schema = {
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
                    "description": "context importance coordinate. Must be a plain decimal number between 0.00 and 1.00 (e.g., 0.55) and within 0-50. Do not use scientific notation, NaN, or text. X coordinate -> memoryspace"
                },
                "y": {
                    "type": "number",
                    "description":  "context relevance coordinate. Must be a plain decimal number between 0.00 and 1.00 (e.g., 0.55) and within 0-50. Do not use scientific notation, NaN, or text. Y coordinate -> memoryspace"
                },
            }
        }

     def attach(self, deep_memory: bool = False) -> None | tuple[str, SpaceWorkingMemAttn] | dict[str, SpaceWorkingMemAttn]:

        self.time: float = time.time()
        curr_attn_anchor: dict[str, SpaceWorkingMemAttn] = self.attention.get_mem_space_dict()

        if (len(curr_attn_anchor.items()) == 0):

            # If none, gate object should skip adding it to the API schema.
            return None

        if (deep_memory):
            return curr_attn_anchor

        # In other methods, computing, I would put the one with most attention score on to the first one via sorting.
        # But before all that, I'm planning to, per attention computation, the score column for each row would experience
        # dilution, as they are kept stale or not read much / needed. That's why we had the < 40 thing.

        # Return the first row in the 2D array via dictionary,
        # first row is the attention anchor for the current session
        return list(curr_attn_anchor.items())[0]

     def process(self, agent_response: dict[str, any]) -> None | bool[False]:
        '''
            The LLM has a system prompt that must determine on its own whether
            to focus on its current response, by outputting a schematic response
        '''

        content_key: set[str] = set(agent_response.keys()).intersection(CONTENT_KEY_NAMES)

        if (len(content_key) == 0):
            return False

        self.xy.x = agent_response[content_key[0]]["x"]
        self.xy.y = agent_response[content_key[0]]["y"]

        if (agent_response["attention"] == False):
            return
        
        self.push_attention(agent_response[content_key[0]], self.xy)

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

        curr_anchor: DecayedAttn | None = self.attention.get_curr_attn()

        if (curr_anchor == None):
            self.attention.add_memory(attention, xy)
            return

        decay = False

        if (curr_anchor.AttnFocus['score'][0] > 40):
            decay = True

        curr_attn_decay: float | None = self.attention_check(curr_anchor.AttnFocus['score'][0], self.time, curr_anchor.DecayRate) if decay else None

        if (not isinstance(curr_attn_decay, float)):
            self.attention.add_memory(attention, xy)
            return

        self.attention.decay_current(curr_attn_decay)
        self.attention.add_memory(attention, xy)

        self.current_attention[f"{str(xy.x)},{str(xy.y)}"].ContxtImportance = xy.x
        self.current_attention[f"{str(xy.x)},{str(xy.y)}"].ContxtRelevance = xy.y
        self.current_attention[f"{str(xy.x)},{str(xy.y)}"].Attn = attention
        self.attention_count += 1

        return

     def attention_check(self, curr_attn_decay: float, last_timestamp: float, decay_rate: typing.Literal[5, 15]) -> typing.Any | float:
            """
            Applies exponential decay to the working memory based on elapsed time.
            """
            current_time = time.time()
            time_delta_seconds = current_time - last_timestamp
            decay_factor = 0
    
            if (time_delta_seconds < current_time): 
                return False
            elif (time_delta_seconds > current_time): 
                decay_factor = np.exp(1 - decay_rate * time_delta_seconds)
    
            decayed_curr_attn = curr_attn_decay * decay_factor
            
            return decayed_curr_attn