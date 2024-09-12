from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Dict
from langchain_utils import LLMMixture

import random
import os
import json
import re


class LemmaSketcher:

    def __init__(
        self,
        example_path: str = "data/lemma_formalizer_examples",
        directory: str = "data/sketches",
        lemma_store: Dict[str, str] = {},
        logger=None,
        type="mistral"
    ):

        self.example_path = example_path
        self.directory = directory
        self.system_prompt = """
As a mathematician familiar with Isabelle, your task is to provide a formal proof in response to a given problem statement.
Your proof should be structured and clearly written, meeting the following criteria:
- It can be verified by Isabelle.
- Each step of the proof should be explained in detail using comments enclosed in "(*" and "*)", representing the corresponding parts of the informal proof.
- The explanation for each step should be clear and concise, avoiding any unnecessary or apologetic language.
- You will be provided with helper lemmas which can be used in the proof. Copy them into the proof text. When closing complex subgoals, you should consider whether these lemmas will help.
You should use `sledgehammer` wherever possible as a proof tactic, instead of anything else. `sledgehammer` will be use to call Isabelle's automated sledgehammer prover.
"""

        self.human_prompt = """
Here are some examples which you should follow to write your proof:

{examples}

####################

"""

        self.prefilled = """
## Problems
{informal_statement}

## Informal proof
{informal_proof}

## Formal Statement
{formal_statement}

## Helper Lemmas
{lemmas}

## Proof"""

        self.logger = logger
        self.lemma_store = lemma_store
        self.type = type


    def load_examples(self) -> Dict[str, str]:
        examples = {}
        for file in os.listdir(self.example_path):
            if file.endswith(".txt"):
                with open(os.path.join(self.example_path, file), "r") as f:
                    content = f.read().strip()
                examples[file[:-4]] = content
        return examples


    def create_message_pair(
        self, informal_statement: str, informal_proof: str, formal_statement: str, helper_lemmas: str
    ) -> Dict[str, str]:
        sketcher_examples = self.load_examples()
        icl_examples = random.sample(list(sketcher_examples.values()), 3)
        icl_examples = "\n\n####################\n\n".join(icl_examples)

        human_prompt_template = HumanMessagePromptTemplate.from_template(
            self.human_prompt
        )
        human_message = human_prompt_template.format(examples=icl_examples)

        assistant_prefilled_template = HumanMessagePromptTemplate.from_template(
            self.prefilled
        )
        assistant_prefilled = assistant_prefilled_template.format(
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            lemmas=helper_lemmas,
            formal_statement=formal_statement,
        )

        system_message = SystemMessage(self.system_prompt)
        user_message = HumanMessage(human_message.content)
        assistant_message = AIMessage(assistant_prefilled.content)

        return [system_message, user_message, assistant_message]

    def extract_lemma_and_proof(self, text: str) -> str:
        combined_pattern = re.compile(
            r"(lemma\s+\w+:\s*.*?proof\s+-\s*.*?qed)", re.DOTALL
        )

        combined_match = combined_pattern.search(text)
        combined_content = combined_match.group(1) if combined_match else None

        return combined_content
    
    def render_lemmas(self, lemmas: Dict[str, str]) -> str:
        lemmas_str = ""
        for name, content in lemmas.items():
            lemmas_str += f"lemma {name}: \"{content}\"\n"
        return lemmas_str
    
    def extract_natural_language_statements(self, lemmas_dict):
        natural_language_statements = {}
        
        for lemma_num, content in lemmas_dict.items():
            natural_language_statements[lemma_num] = content.get('informal', '')

        return natural_language_statements

    def extract_formal_code(self, lemmas_dict):
        formal_code_statements = {}
        
        for lemma_num, content in lemmas_dict.items():
            formal_code_statements[lemma_num] = content.get('formal', '')

        return formal_code_statements

    def format_lemma_text(self, lemmas_dict):
        formatted_lemmas = ""

        for lemma_num, content in lemmas_dict.items():
            informal_statement = content.get("informal", "")
            formal_code = content.get("formal", "")
            
            formatted_lemmas += f"(* lemma {lemma_num}. {informal_statement} *)\n"
            formatted_lemmas += formal_code + "\n\n"

        return formatted_lemmas

    def message_to_dict(self, message):
        return {
            "role": message.type,
            "content": message.content
        }

    def get_next_available_index(self, directory: str) -> int:
        existing_files = [f for f in os.listdir(directory) if f.startswith("prompt_") and f.endswith(".json")]
        indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        if not indices:
            return 1
        return max(indices) + 1
    
    def save_message_pair(self, messages, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        index = self.get_next_available_index(directory)
        messages_as_dicts = [self.message_to_dict(msg) for msg in messages]
        filename = os.path.join(directory, f"prompt_{index}.json")
        with open(filename, "w") as f:
            json.dump(messages_as_dicts, f, indent=4)

    def dict_to_message(self, message_dict):
        if message_dict["role"] == "system":
            return SystemMessage(message_dict["content"])
        elif message_dict["role"] == "human":
            return HumanMessage(message_dict["content"])
        elif message_dict["role"] == "ai":
            return AIMessage(message_dict["content"])
        else:
            raise ValueError("Invalid message role")
    
    def load_message_pair(self, filename: str):
        with open(filename, "r") as f:
            messages_as_dicts = json.load(f)
        return [self.dict_to_message(msg) for msg in messages_as_dicts]

    
    def create_formal_sketch(
        self, informal_statement: str, informal_proof: str, formal_statement: str, model_name: str = "gpt-3.5-turbo"
    ):

        llm = LLMMixture(
            model_name=model_name,
            temperature=0.0,
            request_timeout=120,
            logger=self.logger,
            type=self.type,
        )

        messages = self.create_message_pair(
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            formal_statement=formal_statement,
            helper_lemmas = self.format_lemma_text(self.lemma_store)
        )

        response = llm.query(langchain_msgs=messages, temperature=0.0, max_tokens=4000)

        return response

if __name__ == "__main__":
    lemmas = {
    "1": {
        "informal": "It would be helpful for step 1 if there exists lemma that calculate the base 2 logarithms of the original bases in the equations.",
        "formal": """
lemma log_base_to_power_equals_exponent:
  fixes a :: real
  assumes "a > 0" "a \<noteq> 1" "n > 0"
  shows "log a (a^n) = n"
proof -
  have c0: "log a a = 1"
    by (simp add: assms(1) assms(2))
  have "log a (a^n) = n * (log a a)"
    using log_nat_power[of a a n] by (simp add: assms(1))
  then have c1: "log a (a^n) = n"
    using c0 by simp
  then show ?thesis 
    by (simp add: c1)
qed
"""
    },
    "2": {
        "informal": "In step 5 they uses the log definition to calculate the value of a and b, which require to calculate the antilogarithm_identity which calculate the value of $b = a^c$ given $log_a b = c$.",
        "formal": """
lemma antilogarithm_identity:
  assumes "a > 0" and "a \<noteq> 1" and "b > 0" and "log a b = c"
  shows "b = a ^ c"
  by (metis assms(1) assms(2) assms(3) assms(4) powr_log_cancel powr_realpow)
"""
    }
}

    sketcher = LemmaSketcher(
        lemma_store=lemmas
    )
    informal_statement = "Prove that the sum of two even numbers is even."
    informal_proof = """
Let a and b be two even numbers. Then, a = 2m and b = 2n for some integers m and n. The sum of a and b is a + b = 2m + 2n = 2(m + n). Since m + n is an integer, the sum of two even numbers is even.
"""
    formal_statement = """
lemma sum_of_two_even_numbers_is_even:
  fixes a b :: int
  assumes "even a" "even b"
  shows "even (a + b)"
proof -
    have "even a \<and> even b \<Longrightarrow> even (a + b)"
        by simp
    then show ?thesis
        using assms by blast
qed
"""
    response = sketcher.create_message_pair(
        informal_statement=informal_statement,
        informal_proof=informal_proof,
        formal_statement=formal_statement,
        helper_lemmas = sketcher.format_lemma_text(lemmas)
    )
    print(response[0].type)
    print(response[1].type)
    print(response[2].type)
    sketcher.save_message_pair(response, "test_prompts")