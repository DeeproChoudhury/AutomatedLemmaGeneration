
from file_utils import load_text
from Drafter import Drafter
from Sketcher import Sketcher
import re

def extract_proof(text):
    proof_pattern = re.compile(r"## Structured informal proof(.*?)## Lemmas", re.DOTALL)
    proof_match = proof_pattern.search(text)
    structured_informal_proof = proof_match.group(1).strip() if proof_match else ""
    
    return structured_informal_proof

def extract_thoughts(text):
    thoughts_codes_pattern = re.compile(r"### Lemma (\d+)\s*(.*?)### Code \1\s*```isabelle(.*?)```", re.DOTALL)
    thoughts_codes_matches = thoughts_codes_pattern.findall(text)
    
    thoughts_codes = []
    for match in thoughts_codes_matches:
        thought_number = match[0]
        thought = match[1].strip()
        code = match[2].strip()
        thoughts_codes.append({
            "thought_number": thought_number,
            "thought": thought,
            "code": code
        })
    return thoughts_codes

if __name__ == "__main__":
    text = load_text("aime_1983_p9/aime_1983_p9_2.txt")
    proof = extract_thoughts(text)
    print(proof)

    drafter = Drafter("path/to/prompt_template", model="gpt-3.5-turbo")
    informal_proof = drafter.write_proof(informal_statement=proof[0]["thought"], formal_statement=proof[0]["code"])

    print("sketching")
    sketcher = Sketcher("data/paper_prompt_examples", "sin_gt_zero")
    sketch = sketcher.create_sketch(50, 100, informal_statement=proof[0]["thought"], informal_proof=informal_proof, formal_statement=proof[0]["code"], problem_name="sin_gt_zero")

    print(informal_proof)
    print(sketch)