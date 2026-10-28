import os
import sys

class RAG:
    current_context: list
    working_memory = {}

    def context_rag(self, environmental_data: list):
        self.current_context = environmental_data
        
    def working_memory(self):
        return len(self.working_memory)
    
    def add_temp_memory(self, attention):
        self.working_memory['temp'] = attention
        
    def add_memory(self, attention):
        self.working_memory['attention'] = attention

class SALNetwork:

     rag: RAG
     environmental_data = []
     attention_list = []
     current_attention = {}

     __slots__ = ['rag', 'environmental_data', 'attention_list', 'current_attention']
     
     def __init__(self, raw_data: any):
          self.environmental_data.append(raw_data)
          # Get current status of user activity

     def sal_pathway():
        pass

     def signal_network(self) -> None:
          RAG = self.RAG.context_rag(self.environmental_data)
          for state, data in enumerate(self.RAG.current_context):
               if (state == "Focus"):
                    self.attention_list.append(data)
                    
     def divert_attention(self):
          attention_count = 0
          task_count = 0
          
          for attention in self.attention_list:
               attention_count += 1
               self.current_attention[attention_count] = attention
               
          while (True):

               if (self.RAG.working_memory() < 100):
                  self.RAG.add_memory(self.attention_list[task_count - 1])
               elif (self.cen.check_attention == False):
                   continue
               else:
               # Note: Attention is the data from user, should SLM be involved here hm
                  if (self.RAG.working_memory() != 100): 
                     self.RAG.add_memory(self.attention_list[task_count])
                     self.attention_list.pop(task_count)
                     task_count += 1
                  else:
                     self.RAG.add_temp_memory(self.attention_list[task_count])
                     self.attention_list.pop(taskCount)
                     taskCount += 1
               
     async def store_attention(self, attentionList: list, taskCount):
          
          attentionList = attentionList
          add_attention = await self.cen.push_attention(self.attentionList, taskCount)
          return add_attention
     
               
