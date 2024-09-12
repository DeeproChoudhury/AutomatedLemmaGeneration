from declarative import load_prompt, bedrock_client, modelId, accept, contentType, load_text
from formaliser import process_ai_message

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import random
import os
import json
import time
import re
import pkg_resources

def extract_code_from_sketch(sketch):
    code_pattern = re.compile(r"```(?:[i|I]sabelle)(.*?)```", re.DOTALL)
    code = "\n".join(code_pattern.findall(sketch)).strip()
    return code

if __name__ == "__main__":
    # package_path = pkg_resources.resource_filename("aime_1983_p9", "")
    sketch = load_text(f"aime_1983_p9/aime_1983_p9_0.txt")
    # print(sketch)
    
    code_pattern = re.compile(r"```(?:[i|I]sabelle)(.*?)```", re.DOTALL)
    # text = message.content[message.content.index("# Formalized Code"):]
    code = "\n".join(code_pattern.findall(sketch)).strip()
    
    print(code)    

    # with open("aime_1983_p9/aime_1983_p9_0.txt", "r") as f:
    #     sketch = f.read()
        
    # print(sketch)