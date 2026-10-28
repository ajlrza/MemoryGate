import time
import numpy as np

def filter_modalities(modalities: object):
     
     accepted_modalities: set = set("TextSensor", "AudioSensor", "VideoSensor")
     
     video_sensor: set = set("Image", "Video")
     text_sensor: set = set("UserText" "AgentText")
     audio_sensor: set = set("UserAudio", "AgentAudio")

     filtered_modalities = []

     for modality in modalities:
          if modality in accepted_modalities:
               filtered_modalities.append(True)
     
     if filtered_modalities >= 1:
          return True
     else:
          return False
     
class CENNetwork:

    __slots__ = ['agent_output', 'working_memory']

    agent_output: any
    working_memory: dict

    def __init__(self, agent_output: object):
 
        if len(agent_output['modalities']) >= 2:
            
            if self.__filter_modalities(agent_output['modalities']):
                self.agent_output = agent_output
            else:
                raise ValueError("Modality not accepted. Only accepts Text, Audio, Video, Image, or VideoSensor.")
        else:
            raise ValueError("CENetwork requires at least 2 modalities.")

    def __filter_modalities(self, modalities: list[str]) -> bool:

        accepted_modalities: set = set("TextSensor", "AudioSensor", "VideoSensor")
     
        video_sensor: set = set("Image", "Video")
        text_sensor: set = set("UserText" "AgentText")
        audio_sensor: set = set("UserAudio", "AgentAudio")

        filtered_modalities = []

        valid_count = sum(1 for mode in modalities if mode in accepted_modalities)
        
        return valid_count >= 1

    def __handle_text_output(self, user_text: str) -> dict:
        spike = False
        tre_neurons = {}
        
        if user_text:
            spike = True
            for count, text_char in enumerate(user_text, start=1):
                neuron_name = f"Neuron {count}"
                if neuron_name not in tre_neurons:
                    tre_neurons[neuron_name] = 0 
                tre_neurons[neuron_name] += 1 * len(user_text)

        return tre_neurons 

    def cen_pathway():
        pass

    # Posterior Parietal Cortex 
    def PPC(self, visual_spikes: dict, text_spikes: dict) -> dict:
        """
        Posterior Parietal Cortex layer.
        Fuses multimodal spike trains before the Central Executive reasons over them.
        """
        fused_state = {}

        return fused_state
    
    # Dorsolateral Prefrontal Cortex
    async def push_attention(self, attention_list: list, task_count: int):
        for attention in attention_list:
            self.working_memory.append(attention)
            # Metadata to track how many attention is being held in working memory
            self.working_memory.count(1)
        # For networks to know
        return self.working_memory

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

