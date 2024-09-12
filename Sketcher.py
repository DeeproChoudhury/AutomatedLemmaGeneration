# from declarative import load_prompt, bedrock_client, modelId, accept, contentType
from dsp_functions import Checker

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Dict
from openai import OpenAI
from langchain_utils import LLMMixture

import random
import os
import json
import time
import re
import logging

class Sketcher:

    def __init__(self, example_path: str = "data/paper_prompt_examples", directory: str = "data/sketches", logger=None, type="mistral"):
    
        self.example_path = example_path
        self.directory = directory
        self.system_prompt = """
As a mathematician familiar with Isabelle, your task is to provide a formal proof in response to a given problem statement.
Your proof should be structured and clearly written, meeting the following criteria:
- It can be verified by Isabelle.
- Each step of the proof should be explained in detail using comments enclosed in "(*" and "*)".
- The explanation for each step should be clear and concise, avoiding any unnecessary or apologetic language.
- You should use `sledgehammer` wherever possible. `sledgehammer` will be use to call Isabelle's automated sledgehammer prover.
"""

        self.human_prompt = """
Here some examples:

{examples}

####################

"""

        self.prefilled = """
Informal:
(* ### Problem
{informal_statement}

### Solution
{informal_proof} *)

Formal:
{formal_statement}"""

        self.client = OpenAI(
            organization='org-31771IWlACvxaZ3zYNEZiHsS',
        )

        self.logger = logger
        self.type=type

    def load_examples(self) -> Dict[str, str]:
        examples = {}
        for file in os.listdir(self.example_path):
            if file.endswith(".json"):
                with open(os.path.join(self.example_path, file), "r") as f:
                    data = json.load(f)
                    prompt = data.get("prompt", "")
                examples[file[:-5]] = prompt
        return examples
    
    def create_message_pair(self, informal_statement: str, informal_proof: str, formal_statement: str) -> Dict[str, str]:
        sketcher_examples = self.load_examples()
        icl_examples = random.sample(list(sketcher_examples.values()), 3)
        icl_examples = "\n\n####################\n\n".join(icl_examples)
        
        human_prompt_template = HumanMessagePromptTemplate.from_template(self.human_prompt)
        human_message = human_prompt_template.format(examples=icl_examples)
        
        assistant_prefilled_template = HumanMessagePromptTemplate.from_template(self.prefilled)
        assistant_prefilled = assistant_prefilled_template.format(
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            formal_statement=formal_statement
        )

        system_message = SystemMessage(self.system_prompt)
        user_message = HumanMessage(human_message.content)
        assistant_message = AIMessage(assistant_prefilled.content)
        
        return [system_message, user_message, assistant_message]
    
    def extract_lemma_and_proof(self, text : str) -> str:
        combined_pattern = re.compile(r'(lemma\s+\w+:\s*.*?proof\s+-\s*.*?qed)', re.DOTALL)
        
        combined_match = combined_pattern.search(text)
        combined_content = combined_match.group(1) if combined_match else None
        
        return combined_content
    
    def create_formal_sketch(self, informal_statement: str, informal_proof: str, formal_statement: str, model="mistral-large"):

        
        llm = LLMMixture(
            model_name=model,
            temperature=0.0,
            request_timeout=120,
            logger=self.logger,
            type=self.type
        )

        messages = self.create_message_pair(
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            formal_statement=formal_statement
        )

        response = llm.query(
            langchain_msgs=messages,
            temperature=0.0,
            max_tokens=2048
        )

        return response


    
    def create_sketch(self, start_index: int, end_index: int, informal_statement: str, informal_proof: str, formal_statement: str, problem_name: str):
        for i in range(start_index, end_index):
            try:
                print(i)
                messages = self.create_message_pair(
                    informal_statement=informal_statement,
                    informal_proof=informal_proof,
                    formal_statement=formal_statement
                )

                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 4000,
                        "system": self.system_prompt,
                        "messages": messages
                    }  
                )

                response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
                
                response_body = json.loads(response.get('body').read())
                print(response_body)

                print(response_body.get('content')[0]['text'])
                
                if not os.path.exists(self.directory):
                    os.makedirs(self.directory)
            
                filename = f"{self.directory}/{problem_name}_{i}.txt"
                theorem = self.extract_lemma_and_proof(formal_statement + (response_body.get('content')[0]['text']))
                os.environ['PISA_PATH'] = '/local/scratch/dc755/Portal-to-ISAbelle/src/main/python'

                checker = Checker(
                    working_dir='/local/scratch/dc755/Isabelle2022/src/HOL/Examples',
                    isa_path='/local/scratch/dc755/Isabelle2022',
                    theory_file='/local/scratch/dc755/Isabelle2022/src/HOL/Examples/Interactive.thy',
                    port=8000
                )
                result = checker.check(theorem)
                print("\n==== Success: %s" % result['success'])
                print("--- Complete proof:\n%s" % result['theorem_and_proof'])

                with open(filename, 'w') as f:
                    f.write(formal_statement + response_body.get('content')[0]['text'] + "\n\n" + result['theorem_and_proof'] + "\n\n" + str(result['success']))


            except Exception as e:
                print(e)
                i -= 1
                time.sleep(5)

    def render_human_message(self):
        sketcher_examples = {}
        for file in os.listdir("data/paper_prompt_examples"):
            if file.endswith(".json"):
                with open(os.path.join("data/paper_prompt_examples", file), "r") as f:
                    data = json.load(f)
                    prompt = data.get("prompt", "")
                sketcher_examples[file[:-5]] = prompt
        
        icl_examples = random.sample(list(sketcher_examples.values()), 3)
        icl_examples = "\n\n####################\n\n".join(icl_examples)
        
        human_prompt_template = HumanMessagePromptTemplate.from_template(self.human_prompt)
        
        human_message = human_prompt_template.format(
            examples=icl_examples
        )

        return human_message

    def render_prefilled(self, informal_statement, informal_proof, formal_statement):
        assistant_prefilled_template = HumanMessagePromptTemplate.from_template(self.prefilled)
            
        assistant_prefilled = assistant_prefilled_template.format(
                informal_statement=informal_statement,
                informal_proof=informal_proof,
                formal_statement=formal_statement
        )

    def ensure_proof_keyword(text):
        proof_keyword = "proof -"
        
        lines = text.split('\n')
        
        proof_start_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(("define", "have", "let", "fix", "assume", "show")):
                proof_start_index = i
                break
        
        if proof_start_index == -1:
            raise ValueError("Proof section not found in the text.")
        
        if proof_start_index > 0 and lines[proof_start_index - 1].strip() == proof_keyword:
            return text  # The "proof -" keyword is already in the correct position
        
        lines.insert(proof_start_index, proof_keyword)
        
        text_with_proof_keyword = '\n'.join(lines)
        
        return text_with_proof_keyword


if __name__ == "__main__":
    informal_statement = "Find the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$. Show that it is 12."
    informal_proof = "Let $y = x \sin x$. It suffices to show that $12 \leq \frac{9y^2 + 4}{y}. It is trivial to see that $y > 0$. Then one can multiply both sides by $y$ and it suffices to show $12y \leq 9y^2 + 4$. This can be done by the sum of squares method."
    formal_statement = """theorem
fixes x::real
assumes "0<x" "x<pi"
shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)\""""

    sketcher = Sketcher("data/paper_prompt_examples", "sin_gt_zero")
    sketch = sketcher.create_formal_sketch(informal_statement, informal_proof, formal_statement)
    print(sketch)