from transformers import AutoTokenizer
from transformers import GenerationConfig
from auto_gptq import AutoGPTQForCausalLM
import torch
from CFG import CFG_models
import time
from rich.logging import RichHandler
import logging
import torch
from cfg import LLMCFG
torch.manual_seed(1)

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    handlers=[RichHandler()])

class Llama:
    def __init__(self):
        self.llama_model = self.load_LLama_model()
        self.config = GenerationConfig.from_pretrained(LLMCFG.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(LLMCFG.model_name , trust_remote_code=True)
        
    def load_LLama_model(self):
        model = AutoGPTQForCausalLM.from_quantized(LLMCFG.model_name, device_map= LLMCFG.device, trust_remote_code=LLMCFG.trust_remote_code, use_safetensors=LLMCFG.use_safetensors).eval()
        return model def generate_prompt(self, question: str, context: str) -> str: 
        return f"""
                
                ### Pytanie:
                {question}
                ### Kontekst:
                {context}:
                """
                
    def get_answer_alpaca(self, question : str, stream = False) -> str:
        """
        This method takes the question along with the context and generates the answer
        """
        torch.manual_seed(1)
        print(question)
        print("_____________________"*3)
        response, history = self.llama_model.chat(self.tokenizer,
                                                  question, 
                                                  history=None, 
                                                  generation_config=self.config,  
                                                  top_k=1,
                                                  top_p=0.4,
                                                  temperature=0.6,
                                                  system = "Jesteś AI assystentem.")

        return response
