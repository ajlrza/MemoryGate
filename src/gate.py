import os, sys, time
import numpy as np

def filter_modalities(self, modalities: list[str]) -> bool:

        accepted_modalities: set = set("TextSensor", "AudioSensor", "VideoSensor")
     
        video_sensor: set = set("Image", "Video")
        text_sensor: set = set("UserText" "AgentText")
        audio_sensor: set = set("UserAudio", "AgentAudio")

        filtered_modalities = []

        valid_count = sum(1 for mode in modalities if mode in accepted_modalities)
        
        return valid_count >= 1

def handle_text_output(self, user_text: str) -> dict:
        spike = False
        neurons = {}
        
        if user_text:

            spike = True
            for count, text_char in enumerate(user_text, start=1):

                neuron_name = f"Neuron {count}"

                if neuron_name not in neurons:
                    neurons[neuron_name] = 0 

                neurons[neuron_name] += 1 * len(user_text)

        return neurons 

class attention_space:
    current_context: list
    working_memory = {}

    def context(self, environmental_data: list):
        self.current_context = environmental_data
        
    def working_memory(self):
        return len(self.working_memory)
    
    def add_temp_memory(self, attention):
        self.working_memory['temp'] = attention
        
    def add_memory(self, attention):
        self.working_memory['attention'] = attention

class Gate:

     attention: attention_space
     attention_list = []
     current_attention = {}

     __slots__ = ['rag', 'attention_list', 'current_attention']
     
     def __init__(self, agent_output: object):
 
        if len(agent_output['modalities']) >= 2:
            
            if self.__filter_modalities(agent_output['modalities']):
                self.agent_output = agent_output
            else:
                raise ValueError("Modality not accepted. Only accepts Text, Audio, Video, Image, or VideoSensor.")
        else:
            raise ValueError("CENetwork requires at least 2 modalities.")

     def signal_network(self, current_data) -> None:

          for state, data in enumerate(current_data):
               if (state == "Focus"):
                    self.attention_list.append(data)

     async def push_attention(self, attention_list: list, task_count: int):
        for attention in attention_list:
            self.attention_list.append(attention)
        # For networks to know
        return self.attention.working_memory
                    
     def divert_attention(self, check_attention: bool):

          attention_count = 0
          task_count = 0
          
          for attention in self.attention_list:
               
               attention_count += 1
               self.current_attention[attention_count] = attention
               
          while (True):

               if (self.attention.working_memory() < 100):
                  self.attention.add_memory(self.attention_list[task_count - 1])

               elif (check_attention == False):
                   continue
               else:
               # Note: Attention is the data from user, should SLM be involved here hm
               
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