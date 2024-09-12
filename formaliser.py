
from declarative import load_prompt, bedrock_client, modelId, accept, contentType

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import random
import os
import json
import time
import re

# formalizer_examples = {}
# for file in os.listdir("data/formalizer_examples"):
#     with open(os.path.join("data/formalizer_examples", file), "r") as f:
#         text = f.read()
#     formalizer_examples[file[:-4]] = text
    
system_prompt = """
As a mathematician familiar with Isabelle, your task is to provide a formal proof in response to a given problem statement.
Your proof should be structured and clearly written, meeting the following criteria:
- It can be verified by Isabelle.
- Each step of the proof should be explained in detail using comments enclosed in "(*" and "*)".
- The explanation for each step should be clear and concise, avoiding any unnecessary or apologetic language.
- You should use `sledgehammer` wherever possible. `sledgehammer` will be use to call Isabelle's automated sledgehammer prover.
- You are **strongly encouraged** to create useful and reusable lemmas to solve the problem.
- The lemmas should be as general as possible (generalizable), and be able to cover a large step in proofs (non-trivial).
Please ensure that your proof is well-organized and easy to follow, with each step building upon the previous one.
"""

human_prompt = """
Here some examples:

{examples}

####################

"""

prefilled = """
## Problems
{informal_statement}

## Informal proof
{informal_proof}

## Formal statement
```isabelle
{formal_statement}
```

## Proof"""

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
    formalizer_examples = {}
    for file in os.listdir("data/formalizer_examples"):
        with open(os.path.join("data/formalizer_examples", file), "r") as f:
            text = f.read()
        formalizer_examples[file[:-4]] = text
        
    for i in range(25, 30):
        try:
            icl_examples = random.sample(list(formalizer_examples.values()), 3)
            icl_examples = "\n\n####################\n\n".join(icl_examples)
            
            human_prompt_template = HumanMessagePromptTemplate.from_template(human_prompt)
            
            human_message = human_prompt_template.format(
                examples=icl_examples
        #         informal_statement="Find the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$. Show that it is 12.",
        #         informal_proof="Let $y = x \sin x$. It suffices to show that $12 \leq \frac{9y^2 + 4}{y}. It is trivial to see that $y > 0$. Then one can multiply both sides by $y$ and it suffices to show $12y \leq 9y^2 + 4$. This can be done by the sum of squares method.",
        #         formal_statement="""theorem
        #   fixes x::real
        #   assumes "0<x" "x<pi"
        #   shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)"
        #   """
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
        
            body = json.dumps({
                'prompt': f"{human_message.content}",
                'max_tokens_to_sample': 4000
            })
            
            user_message = {"role": "user", "content": human_message.content}
            
            assistant_message = {"role": "assistant", "content": assistant_prefilled.content}
            # print(assistant_prefilled.content)
            
            messages = [user_message, assistant_message]
            
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "system": system_prompt,
                    "messages": messages
                }  
            )  
            
            print(i)
            
            response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, 
            contentType=contentType)
            
            # print(response)
        
            response_body = json.loads(response.get('body').read())
            # print(response_body)
            
            # text
            # print(response_body.get('content')[0]['text'])
            
            directory = "aime_1983_p9_sketch"
            filename = f"{directory}/aime_1983_p9_{i}.txt"
            
            if not os.path.exists(directory):
                os.makedirs(directory)
        
            with open(filename, 'w') as f:
                f.write(response_body.get('content')[0]['text'])
        except Exception as e:
            i -= 1
            time.sleep(5)