
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import os
from openai import OpenAI
from langchain_utils import LLMMixture

class Drafter:
    def __init__(self, prompt_template_path, model, client=None):
        self.prompt_template_path = prompt_template_path
        self.model = model

        self.system_prompt = """
You are an expert in mathematics. Write an informal mathematical proof (in natural language, with Latex formulae)
for the following problem.
The formal Isabelle statement of the problem will also be provided, as our 
goal is to eventually formalize the proof in Isabelle. The proof should be clear and
should justify each step, including the minor ones, thoroughly. As the proof will be eventually formalized in isabelle,
ensure that you do not use techniques which are too complicated for isabelle to handle, such as calculus.
"""

        self.user_prompt = """
Compose a concise proof for the following problem. This proof should be written in natural language,
and should be detailed and clear. Justify each step, including the minor ones, thoroughly.
"""

        self.prefilled = """
## Problem
{informal_statement}

## Formal Statement
{formal_statement}

## Informal Proof
"""

        self.client = client
    
    def write_proof(self, informal_statement, formal_statement):

        user_message = {"role": "user", "content": self.user_prompt}
              
        assistant_prefilled_template = HumanMessagePromptTemplate.from_template(self.prefilled)
            
        assistant_prefilled = assistant_prefilled_template.format(
                informal_statement=informal_statement,
                formal_statement=formal_statement
            )
        assistant_message = {"role": "assistant", "content": assistant_prefilled.content}
            
        messages = [user_message, assistant_message]
            
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "system": self.system_prompt,
                "messages": messages
            }  
        )
        
        # response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, 
        # contentType=contentType)

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        print(response.choices[0].message.content)
    
        # response_body = json.loads(response.get('body').read())
        # # print(response_body)
        
        # # text
        # return response_body.get('content')[0]['text']
        
        # # directory = "aime_1983_p9_sketch"
        # # filename = f"{directory}/aime_1983_p9_{i}.txt"
        
        # # if not os.path.exists(directory):
        # #     os.makedirs(directory)
    
        # # with open(filename, 'w') as f:
        # #     f.write(response_body.get('content')[0]['text'])

    def write_proof_openai(self, informal_statement, formal_statement):
        llm = LLMMixture(
            model_name=self.model,
            temperature=0.7,
            request_timeout=60,
        )

        system_message = SystemMessage(content=self.system_prompt)
        user_message = HumanMessage(content=self.user_prompt)
        assistant_prefilled_template = HumanMessagePromptTemplate.from_template(self.prefilled)

        assistant_prefilled = assistant_prefilled_template.format(
            informal_statement=informal_statement,
            formal_statement=formal_statement
        )

        assistant_message = AIMessage(content=assistant_prefilled.content)

        messages = [system_message, user_message, assistant_message]

        response = llm.query(
            langchain_msgs=messages,
            temperature=0.7,
            max_tokens=1024
        )

        return response

    def generate_informal_proof(self, problem):
        prompt_template = self.load_template(self.prompt_template_path)

        prompt = prompt_template.render(problem=problem)
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        completion = self.client.chat(
            model=self.model,
            messages=messages
        )
        
        informal_proof = completion.choices[0].message.content
        return informal_proof

if __name__ == "__main__":
    informal_statement="Find the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$. Show that it is 12."
    formal_statement="""theorem
fixes x::real
assumes "0<x" "x<pi"
shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)\""""
    

    drafter = Drafter("path/to/prompt_template", "gpt-3.5-turbo")
    informal_proof = drafter.write_proof_openai(informal_statement=informal_statement, formal_statement=formal_statement)
    print(informal_proof)