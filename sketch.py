
from declarative import load_prompt, bedrock_client, modelId, accept, contentType

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_utils import LLMMixture

import random
import os
import json
import time
import re
    
system_prompt = """
As a mathematician familiar with Isabelle, your task is to provide a formal proof in response to a given problem statement.
Your proof should be structured and clearly written, meeting the following criteria:
- It can be verified by Isabelle.
- Each step of the proof should be explained in detail using comments enclosed in "(*" and "*)".
- The explanation for each step should be clear and concise, avoiding any unnecessary or apologetic language.
- You should use `sledgehammer` wherever possible. `sledgehammer` will be use to call Isabelle's automated sledgehammer prover.
"""

human_prompt = """
Write the formal proof for this statement using sledgehammer wherever possible for the intermediate conjectures. Here some examples:

{examples}

####################

"""

prefilled = """
Informal:
(* ### Problem
{informal_statement}

### Solution
{informal_proof} *)

Formal:

{formal_statement}"""

def render_formalizer_system_message(self):
    system_template = load_prompt("formalizer")
    return SystemMessage(content=system_template)

def render_formalizer_human_message(
    self,
    skills,
    context,
    informal_proof=None,
    n_example=3,
) -> HumanMessage:
    human_prompt_template = load_prompt("formalizer_human")
    human_prompt_template = HumanMessagePromptTemplate.from_template(human_prompt_template)

    if context["problem_name"] in formalizer_examples:
        formalizer_examples.pop(context["problem_name"])

    examples = random.sample(list(formalizer_examples.values()), n_example)
    examples = "\n\n####################\n\n".join(examples)
    context["informal_statement"] = context["informal_statement"].replace("\n", ' ').strip()
    context["informal_proof"] = context["informal_proof"].replace("\n", " ").strip()

    skills = self.retrieved_example_skills(skills)
    
    human_message = human_prompt_template.format(
        skill_examples = skills,
        examples=examples,
        informal_statement=context["informal_statement"],
        informal_proof=context["informal_proof"] if informal_proof is None else informal_proof,
        formal_statement=context["formal_statement"],
    )

    return human_message
    

def process_ai_message(message):
    # assert isinstance(message, AIMessage)

    retry = 3
    error = None
    while retry > 0:
        try:
            code_pattern = re.compile(r"```(?:[i|I]sabelle)(.*?)```", re.DOTALL)
            text = message.content[message.content.index("# Formalized Code"):]
            code = "\n".join(code_pattern.findall(text)).strip()
            return code
        except Exception as e:
            retry -= 1
            error = e
            time.sleep(1)
    return False
    
if __name__ == "__main__":
    sketcher_examples = {}
    for file in os.listdir("data/paper_prompt_examples"):
        if file.endswith(".json"):
            with open(os.path.join("data/paper_prompt_examples", file), "r") as f:
                data = json.load(f)
                prompt = data.get("prompt", "")
            sketcher_examples[file[:-5]] = prompt
        
    for i in range(51, 52):
        try:
            icl_examples = random.sample(list(sketcher_examples.values()), 3)
            icl_examples = "\n\n####################\n\n".join(icl_examples)
            
            human_prompt_template = HumanMessagePromptTemplate.from_template(human_prompt)
            
            human_message = human_prompt_template.format(
                examples=icl_examples
            )
            
            assistant_prefilled_template = HumanMessagePromptTemplate.from_template(prefilled)
            
            assistant_prefilled = assistant_prefilled_template.format(
                informal_statement="Find the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$. Show that it is 12.",
                informal_proof="""Step 1: Define a new variable $y = x \sin x$. The goal is now to show $12 \leq \frac{9y^2 + 4}{y}$.
Step 2: Show $y > 0$ using the given assumptions $0 < x < \pi$.
Step 3: Multiply both sides by $y$ to get $12y \leq 9y^2 + 4$.
Step 4: Use the sum of squares method to prove $12y \leq 9y^2 + 4$.
Step 5: Conclude that the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$ is 12.""",
                formal_statement="""theorem
          fixes x::real
          assumes "0<x" "x<pi"
          shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)\""""
            )

            system_message = SystemMessage(content=system_prompt)
            user_message = HumanMessage(content=human_message.content)
            
            assistant_message = AIMessage(content=assistant_prefilled.content)
            
            messages = [system_message, user_message, assistant_message]
            
            print(i)

            llm = LLMMixture(
                model_name="gpt-3.5-turbo",
                temperature=1,
                request_timeout=10,
            )

            response = llm.query(
                langchain_msgs=messages,
                temperature=0
            )
            
            directory = "aime_1983_p9_sketch"
            filename = f"{directory}/aime_1983_p9_{i}.txt"
            
            if not os.path.exists(directory):
                os.makedirs(directory)
        
            with open(filename, 'w') as f:
                f.write(response)
        except Exception as e:
            print(e)
            i -= 1
            time.sleep(5)