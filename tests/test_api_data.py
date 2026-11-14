import pytest as py
import time, typing, math
import numpy as np
from MemoryGate.gate import Coords, DecayedAttn, WorkingMemAttn, StoredWorkingMem, AttentionSpace, Gate, AttnMemAdder, dt

def coords_dc(x: int, y: int) -> Coords:
    dc = Coords(x=x, y=y)
    return dc

def decayedattn_dc(focusattn: np.ndarray, rate: typing.Literal[5,15]) -> DecayedAttn:
    dc = DecayedAttn(AttnFocus=focusattn, DecayRate=rate)
    return dc

def workingattn_dc(attn: str, score: float) -> WorkingMemAttn:
    dc = WorkingMemAttn(Attn=attn, Score=score)
    return dc

def stored_dc(pos: Coords, attn: str) -> StoredWorkingMem:
    dc = StoredWorkingMem(Pos=pos, Attn=y)
    return dc


def test_coords_format_add_list(empty_mem: np.ndarray) -> bool:
    result = AttnMemAdder(empty_mem, [1, 2], "Test")
    assert result['context importance'] == 1
    assert result['context relevance'] == 2

def test_coords_format_add_dataclass(coords: Coords, empty_mem: np.ndarray) -> bool:
    result = AttnMemAdder(empty_mem, coords, "Test")
    assert result['context importance'] == 3
    assert result['context relevance'] == 4

def test_attn_space_properties(attnspace_obj: AttentionSpace):

    assert isinstance(attnspace_obj.working_memory, dict) == True
    assert isinstance(attnspace_obj.memory_space, np.ndarray) == True
    assert isinstance(attnspace_obj.decay) == DecayedAttn
    assert isinstance(attnspace_obj.decay.DecayRate) == int

def test_gate_properties(gate: Gate):

    assert isinstance(gate.attention, AttentionSpace) ==  True
    assert isinstance(gate.current_attention, dict) ==  True
    assert isinstance(gate.xy, Coords) ==  True
    assert isinstance(gate.time, float) ==  True
    assert isinstance(gate.schema, dict) ==  True

def test_gate_schema(gate: Gate):

    schema_obj = ["type", "properties"]
    schema_props = ["output", "attention", "x", "y"]

    assert list(gate.schema.keys()) == schema_obj
    assert list(gate.schema["properties"].keys()) == schema_props
    assert gate.schema["properties"]["output"] == "string"
    assert gate.schema["properties"]["attention"] == "boolean"
    assert gate.schema["properties"]["x"] == "number"
    assert gate.schema["properties"]["y"] == "number"

def test_attn_flow(coords: Coords, attnspace_obj: AttentionSpace):

    assert isinstance(attnspace_obj.compute_attention(coords_dc(1,2)), str) ==  True
    attnspace_obj.add_memory("Test", Coords)

def main():

    empty_mem = np.zeros((1000, 4), dt = dt)

    test_coords_format_add_list(empty_mem)
    test_coords_format_add_dataclass(empty_mem)

    space = AttentionSpace()

    test_attn_space_properties(space)

    gate = Gate()

    test_gate_properties(gate)
    test_gate_schema(gate)

if __name__ == "__main__":
    main()