import random
import os

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from file_utils import load_text, load_prompt
from langchain_utils import LLMMixture


class Orienter:

    def __init__(self, model):
        self.model = model

        self.system_prompt = """
As a mathematician and expert in the isabelle theorem prover, your task is to analyze the given theorem (including problem's informal statement, 
human written informal proof, and formal statement). Provide a better structured step by step proof that closer to isabelle. 
and request relevant lemmas, theorems that might help in proving this problem.
"""

        self.human_prompt = """
Here are some examples.

{examples}

####################
"""

        self.prefilled = """
## Problems
{informal_statement}

## Informal proof
{informal_proof}

## Formal statement
```isabelle
{formal_statement}
```"""  

    def get_examples(self):
        lemma_examples = {}
        for file in os.listdir("data/lemma_examples"):
            with open(os.path.join("data/lemma_examples", file), "r") as f:
                text = f.read()
            lemma_examples[file[:-4]] = text
        
        return lemma_examples
    
    def render_messages(self, informal_statement, informal_proof, formal_statement):
        examples = self.get_examples()
        icl_examples = random.sample(list(examples.values()), 3)
        icl_examples = "\n\n####################\n\n".join(icl_examples)

        human_message = HumanMessagePromptTemplate.from_template(self.human_prompt).format(examples=icl_examples)

        system_message = SystemMessagePromptTemplate.from_template(self.system_prompt)

        assistant_prefilled_template = HumanMessagePromptTemplate.from_template(self.prefilled)
        assistant_prefilled = assistant_prefilled_template.format(
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            formal_statement=formal_statement
        )

        system_message = SystemMessage(content=self.system_prompt)

        human_message = HumanMessage(content=human_message.content)

        assistant_message = AIMessage(content=assistant_prefilled.content)

        messages = [system_message, human_message, assistant_message]

        return messages
    
    def orient(self, informal_statement, informal_proof, formal_statement, temperature=0.7):
        messages = self.render_messages(informal_statement, informal_proof, formal_statement)
        llm = LLMMixture(
            model_name=self.model,
            temperature=temperature,
            request_timeout=60,
        )

        response = llm.query(
            langchain_msgs=messages,
            temperature=temperature,
            max_tokens=1024
        )

        return response
    
if __name__ == "__main__":
    orienter = Orienter(
        model="gpt-3.5-turbo"
    )

    informal_statement = "Find the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$. Show that it is 12."
    informal_proof = """Step 1: Define a new variable $y = x \sin x$. The goal is now to show $12 \leq \frac{9y^2 + 4}{y}$.
Step 2: Show $y > 0$ using the given assumptions $0 < x < \pi$.
Step 3: Multiply both sides by $y$ to get $12y \leq 9y^2 + 4$.
Step 4: Use the sum of squares method to prove $12y \leq 9y^2 + 4$.
Step 5: Conclude that the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$ is 12."""
    formal_statement = """theorem
      fixes x::real
      assumes "0<x" "x<pi"
      shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)\""""

    response = orienter.orient(informal_statement, informal_proof, formal_statement)
    print(response)
