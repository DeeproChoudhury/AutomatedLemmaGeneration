from Drafter import Drafter
from Orienter import Orienter
from Sketcher import Sketcher
from Prover import Prover
from LemmaSketcher import LemmaSketcher
from extract import extract_thoughts, extract_proof
from collections import deque
import logging
import time
import os
import json
from langchain.schema import AIMessage, HumanMessage, SystemMessage

def initialise_logger(name=None):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    start_time = time.strftime("%Y%m%d_%H%M%S")
    if name is not None:
        os.makedirs(f"logs/{name}/", exist_ok=True)
        handler = logging.FileHandler(f"logs/{name}/{start_time}.log")
    else:
        os.makedirs(f"logs/{start_time}_logs", exist_ok=True)
        handler = logging.FileHandler(f"logs/{start_time}_logs/{start_time}.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def thoughts_queue(informal_statement, informal_proof, formal_statement):
    orienter = Orienter(model="gpt-3.5-turbo")

    prompt_logger = initialise_logger("prompt")
    sketcher_logger = initialise_logger("sketcher")

    drafter = Drafter(
        prompt_template_path="path/to/prompt_template", model="gpt-3.5-turbo"
    )

    sketcher = Sketcher(
        example_path="data/paper_prompt_examples",
        directory="sin_gt_zero",
        logger=prompt_logger,
    )

    thoughts_to_process = [(informal_statement, informal_proof, formal_statement, True)]
    initial_problem = (informal_statement, informal_proof, formal_statement)
    all_proofs = []

    while thoughts_to_process:
        (
            current_informal_statement,
            current_informal_proof,
            current_formal_statement,
            is_initial_problem,
        ) = thoughts_to_process.pop(0)
        response = orienter.orient(
            current_informal_statement,
            current_informal_proof,
            current_formal_statement,
            temperature=0,
        )

        sketcher_logger.info(
            "current_informal_statement:\n" + current_informal_statement
        )

        sketcher_logger.info(response)

        print("##### RESPONSE #####")
        print(response)

        thoughts = extract_thoughts(response)
        print("##### LEMMAS #####")
        for i, thought in enumerate(thoughts):
            print(thought)
            sketcher_logger.info(thought)
            thought_informal_statement = thought.get("thought")
            thought_formal_statement = thought.get("code")
            thought_informal_proof = drafter.write_proof_openai(
                informal_statement=thought_informal_statement,
                formal_statement=thought_formal_statement,
            )
            thoughts_to_process.append(
                (
                    thought_informal_statement,
                    thought_informal_proof,
                    thought_formal_statement,
                    False,
                )
            )
            print(thought_informal_statement)
            print(thought_formal_statement)

        proof = extract_proof(response)
        print("##### PROOF #####")
        print(proof)

        if not is_initial_problem:
            informal_proof = drafter.write_proof_openai(
                informal_statement=current_informal_statement,
                formal_statement=current_formal_statement,
            )

            sketcher_logger.info(f"\n Informal proof for thought {i}:\n")
            sketcher_logger.info(informal_proof)
            print(informal_proof)
            print("\n\n")

            sketch = sketcher.create_formal_sketch(
                informal_statement=current_informal_statement,
                informal_proof=informal_proof,
                formal_statement=current_formal_statement,
                model="mistral-large"
            )
            print("##### SKETCH #####")

            sketcher_logger.info("\n##### SKETCH #####\n")
            sketcher_logger.info("\n" + current_formal_statement + "\n" + sketch)
            print(current_formal_statement)
            print(sketch)
            print("\n\n")

    initial_informal_statement, initial_informal_proof, initial_formal_statement = (
        initial_problem
    )
    sketch = sketcher.create_formal_sketch(
        informal_statement=initial_informal_statement,
        informal_proof=initial_informal_proof,
        formal_statement=initial_formal_statement,
    )
    print("##### SKETCH #####")
    sketcher_logger.info("\n##### SKETCH #####\n")
    sketcher_logger.info("\n" + initial_formal_statement + "\n" + sketch)
    print(initial_formal_statement)
    print(sketch)
    print("\n\n")

    combined_proof = combine_proofs(all_proofs, initial_problem)
    
    print("##### COMBINED PROOF #####")
    print(combined_proof)
    sketcher_logger.info("\n##### COMBINED PROOF #####\n")
    sketcher_logger.info(combined_proof)


def thoughts_queue_2(informal_statement, informal_proof, formal_statement, model="gpt-3.5-turbo", type="openai"):
    orienter = Orienter(model=model, type=type)
    prompt_logger = initialise_logger("prompt")
    sketcher_logger = initialise_logger("sketcher")
    drafter = Drafter(prompt_template_path="path/to/prompt_template", model=model)
    sketcher = Sketcher(example_path="data/paper_prompt_examples", directory="sin_gt_zero", logger=prompt_logger, type=type)
    prover = Prover()
    lemma_sketcher = LemmaSketcher(logger=sketcher_logger, type=type)

    lemma_queue = deque([(informal_statement, informal_proof, formal_statement, None, True)])
    proof_hierarchy = {}
    verified_lemmas = {}

    while lemma_queue:
        current_informal_statement, current_informal_proof, current_formal_statement, parent_id, is_original = lemma_queue.popleft()
        
        response = orienter.orient(current_informal_statement, current_informal_proof, current_formal_statement, temperature=0)
        sketcher_logger.info(f"Orienter response for {current_informal_statement[:50]}...: {response}")

        thoughts = extract_thoughts(response)
        proof = extract_proof(response)

        current_id = len(proof_hierarchy)
        proof_hierarchy[current_id] = {
            'informal_statement': current_informal_statement,
            'formal_statement': current_formal_statement,
            'proof': proof,
            'sublemmas': [],
            'parent_id': parent_id,
            'is_original': is_original,
            'verified': False
        }

        if parent_id is not None:
            proof_hierarchy[parent_id]['sublemmas'].append(current_id)

        for thought in thoughts:
            thought_informal_statement = thought.get('thought')
            thought_formal_statement = thought.get('code')
            thought_informal_proof = drafter.write_proof_openai(
                informal_statement=thought_informal_statement,
                formal_statement=thought_formal_statement,
                type=type
            )
            lemma_queue.append((thought_informal_statement, thought_informal_proof, thought_formal_statement, current_id, False))

    def verify_lemma(lemma_id):
        lemma = proof_hierarchy[lemma_id]
        if lemma['verified']:
            return True
        
        # Verify all sublemmas first
        for sublemma_id in lemma['sublemmas']:
            if not verify_lemma(sublemma_id):
                lemma['sublemmas'].remove(sublemma_id)
        
        # All remaining sublemmas are verified, now verify this lemma
        if not lemma['is_original']:
            # Use LemmaSketcher with the current verified_lemmas
            lemma_sketcher.lemma_store = verified_lemmas
            sketch = lemma_sketcher.create_formal_sketch(
                informal_statement=lemma['informal_statement'],
                informal_proof=lemma['proof'],
                formal_statement=lemma['formal_statement'],
                model_name=model
            )
            sketcher_logger.info(f"Sketch for {lemma['informal_statement']}...: {sketch}")

            verification_result = prover.check_proof(sketch)
            if verification_result['success']:
                verified_lemmas[lemma_id] = {
                    'informal': lemma['informal_statement'],
                    'formal': sketch
                }
                lemma['verified'] = True
                sketcher_logger.info(f"Successfully verified lemma: {lemma['informal_statement']}...")
                return True
            else:
                sketcher_logger.info(f"Failed to verify lemma: {lemma['informal_statement'][:50]}...")
                sketcher_logger.info(f"Prover output: {verification_result['theorem_and_proof']}")
                return False
        return True

    # Start verification from the root (original statement)
    verify_lemma(0)

    # Create the final proof sketch for the original statement using verified lemmas
    lemma_sketcher.lemma_store = verified_lemmas
    final_sketch = lemma_sketcher.create_formal_sketch(
        informal_statement=informal_statement,
        informal_proof=informal_proof,
        formal_statement=formal_statement,
        model_name=model
    )

    print("##### FINAL PROOF SKETCH #####")
    print(final_sketch)
    sketcher_logger.info("\n##### FINAL PROOF SKETCH #####\n")
    sketcher_logger.info(final_sketch)

    # Verify the final sketch
    verification_result = prover.check_proof(final_sketch)
    if verification_result['success']:
        print("##### FINAL PROOF SKETCH VERIFIED SUCCESSFULLY #####")
        sketcher_logger.info("\n##### FINAL PROOF SKETCH VERIFIED SUCCESSFULLY #####\n")
    else:
        print("##### FINAL PROOF SKETCH VERIFICATION FAILED #####")
        sketcher_logger.info("\n##### FINAL PROOF SKETCH VERIFICATION FAILED #####\n")
        print("Prover output:")
        print(verification_result['theorem_and_proof'])
        sketcher_logger.info("Prover output:")
        sketcher_logger.info(verification_result['theorem_and_proof'])

    return final_sketch, verified_lemmas, verification_result['success']

def thoughts_queue_3(informal_statement, informal_proof, formal_statement, model="mistral-large", type="openai"):
    orienter = Orienter(model=model, type=type)
    prompt_logger = initialise_logger("prompt")
    sketcher_logger = initialise_logger("sketcher")
    drafter = Drafter(prompt_template_path="path/to/prompt_template", model=model)
    sketcher = Sketcher(example_path="data/paper_prompt_examples", directory="sin_gt_zero", logger=prompt_logger, type=type)
    prover = Prover()
    lemma_sketcher = LemmaSketcher(logger=sketcher_logger, type=type)

    # Step 1: Generate lemmas
    response = orienter.orient(informal_statement, informal_proof, formal_statement, temperature=0)
    sketcher_logger.info(f"Orienter response for original statement: {response}")

    thoughts = extract_thoughts(response)
    lemmas = []

    for thought in thoughts:
        thought_informal_statement = thought.get('thought')
        thought_formal_statement = thought.get('code')
        thought_informal_proof = drafter.write_proof_openai(
            informal_statement=thought_informal_statement,
            formal_statement=thought_formal_statement,
            type=type
        )
        lemmas.append({
            'informal_statement': thought_informal_statement,
            'informal_proof': thought_informal_proof,
            'formal_statement': thought_formal_statement
        })

    # Step 2: Verify lemmas using regular sketcher
    verified_lemmas = {}
    for i, lemma in enumerate(lemmas):
        sketch = sketcher.create_formal_sketch(
            informal_statement=lemma['informal_statement'],
            informal_proof=lemma['informal_proof'],
            formal_statement=lemma['formal_statement']
        )
        sketcher_logger.info(f"Sketch for lemma {i}: {sketch}...")

        verification_result = prover.check_proof(sketch)
        if verification_result['success']:
            verified_lemmas[i] = {
                'informal': lemma['informal_statement'],
                'formal': sketch
            }
            sketcher_logger.info(f"Successfully verified lemma {i}: {lemma['informal_statement']}")
        else:
            sketcher_logger.info(f"Failed to verify lemma {i}: {lemma['informal_statement']}")
            sketcher_logger.info(f"Prover output: {verification_result['theorem_and_proof']}")

    # Step 3: Use verified lemmas to prove the original statement
    lemma_sketcher.lemma_store = verified_lemmas
    final_sketch = lemma_sketcher.create_formal_sketch(
        informal_statement=informal_statement,
        informal_proof=informal_proof,
        formal_statement=formal_statement,
        model_name=model
    )

    print("##### FINAL PROOF SKETCH #####")
    print(final_sketch)
    sketcher_logger.info("\n##### FINAL PROOF SKETCH #####\n")
    sketcher_logger.info(final_sketch)

    # Verify the final sketch
    verification_result = prover.check_proof(final_sketch)
    if verification_result['success']:
        print("##### FINAL PROOF SKETCH VERIFIED SUCCESSFULLY #####")
        sketcher_logger.info("\n##### FINAL PROOF SKETCH VERIFIED SUCCESSFULLY #####\n")
    else:
        print("##### FINAL PROOF SKETCH VERIFICATION FAILED #####")
        sketcher_logger.info("\n##### FINAL PROOF SKETCH VERIFICATION FAILED #####\n")
        print("Prover output:")
        print(verification_result['theorem_and_proof'])
        sketcher_logger.info("Prover output:")
        sketcher_logger.info(verification_result['theorem_and_proof'])

    return final_sketch, verified_lemmas, verification_result['success']
    # Verify the final sketch
    verification_result = prover.check_proof(final_sketch)
    if verification_result['success']:
        print("##### FINAL PROOF SKETCH VERIFIED SUCCESSFULLY #####")
        sketcher_logger.info("\n##### FINAL PROOF SKETCH VERIFIED SUCCESSFULLY #####\n")
    else:
        print("##### FINAL PROOF SKETCH VERIFICATION FAILED #####")
        sketcher_logger.info("\n##### FINAL PROOF SKETCH VERIFICATION FAILED #####\n")
        print("Prover output:")
        print(verification_result['theorem_and_proof'])
        sketcher_logger.info("Prover output:")
        sketcher_logger.info(verification_result['theorem_and_proof'])

    return final_sketch, verified_lemmas, verification_result['success']

def combine_proofs(proof_hierarchy):
    def recursive_combine(proof_id):
        proof_data = proof_hierarchy[proof_id]
        combined = ""

        for sublemma_id in proof_data['sublemmas']:
            combined += recursive_combine(sublemma_id)
            combined += "\n\n"

        combined += f"lemma {proof_data['informal_statement']}:\n"
        combined += f"{proof_data['formal_statement']}\n"
        combined += f"proof -\n{proof_data['proof']}\nqed\n"

        return combined

    main_proof = recursive_combine(0)
    
    return f"\n{main_proof}\n"

def dict_to_message(message_dict):
    if message_dict["role"] == "system":
        return SystemMessage(message_dict["content"])
    elif message_dict["role"] == "human":
        return HumanMessage(message_dict["content"])
    elif message_dict["role"] == "ai":
        return AIMessage(message_dict["content"])
    else:
        raise ValueError("Invalid message role")

def extract_and_query(directory: str):
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r") as f:
                messages_as_dicts = json.load(f)
            messages = [dict_to_message(msg) for msg in messages_as_dicts]
            response = response = llm.query(langchain_msgs=messages, temperature=0.0, max_tokens=4000)

def test_thoughts_queue_2():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    informal_statement = "Prove that for any real numbers a and b, (a + b)^2 = a^2 + 2ab + b^2."
    informal_proof = """
Let's expand the left side of the equation:
(a + b)^2 = (a + b)(a + b)
Using the distributive property, we get:
(a + b)(a + b) = a(a + b) + b(a + b)
= a^2 + ab + ba + b^2
Since multiplication is commutative, ab = ba, so:
= a^2 + ab + ab + b^2
= a^2 + 2ab + b^2
Thus, we have shown that (a + b)^2 = a^2 + 2ab + b^2.
    """
    formal_statement = """
theorem square_of_sum:
    fixes a b :: real
    shows "(a + b)^2 = a^2 + 2*a*b + b^2"
    """

    logger.info("Starting test of thoughts_queue_2 function")
    logger.info(f"Informal statement: {informal_statement}")
    logger.info(f"Informal proof: {informal_proof}")
    logger.info(f"Formal statement: {formal_statement}")

    final_sketch, verified_lemmas, is_verified = thoughts_queue_2(informal_statement, informal_proof, formal_statement, model="mistral-large", type="mistral")

    logger.info("Test completed")
    logger.info("Final sketch:")
    logger.info(final_sketch)
    logger.info("Verified lemmas:")
    for lemma_id, lemma_data in verified_lemmas.items():
        logger.info(f"Lemma {lemma_id}:")
        logger.info(f"  Informal: {lemma_data['informal'][:100]}...")
        logger.info(f"  Formal: {lemma_data['formal'][:100]}...")
    logger.info(f"Final sketch verified: {is_verified}")

def test_thoughts_queue_3():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    informal_statement = "Prove that for any real numbers a and b, (a + b)^2 = a^2 + 2ab + b^2."
    informal_proof = """
Let's expand the left side of the equation:
(a + b)^2 = (a + b)(a + b)
Using the distributive property, we get:
(a + b)(a + b) = a(a + b) + b(a + b)
= a^2 + ab + ba + b^2
Since multiplication is commutative, ab = ba, so:
= a^2 + ab + ab + b^2
= a^2 + 2ab + b^2
Thus, we have shown that (a + b)^2 = a^2 + 2ab + b^2.
    """
    formal_statement = """
theorem square_of_sum:
    fixes a b :: real
    shows "(a + b)^2 = a^2 + 2*a*b + b^2"
    """

    logger.info("Starting test of thoughts_queue_3 function")
    logger.info(f"Informal statement: {informal_statement}")
    logger.info(f"Informal proof: {informal_proof}")
    logger.info(f"Formal statement: {formal_statement}")

    final_sketch, verified_lemmas, is_verified = thoughts_queue_3(informal_statement, informal_proof, formal_statement, model="mistral-large", type="mistral")

    logger.info("Test completed")
    logger.info("Final sketch:")
    logger.info(final_sketch)
    logger.info("Verified lemmas:")
    for lemma_id, lemma_data in verified_lemmas.items():
        logger.info(f"Lemma {lemma_id}:")
        logger.info(f"  Informal: {lemma_data['informal']}...")
        logger.info(f"  Formal: {lemma_data['formal']}...")
    logger.info(f"Final sketch verified: {is_verified}")

if __name__ == "__main__":

    test_thoughts_queue_3()

#     informal_statement = "Find the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$. Show that it is 12."
#     informal_proof = "Let $y = x \sin x$. It suffices to show that $12 \leq \frac{9y^2 + 4}{y}. It is trivial to see that $y > 0$. Then one can multiply both sides by $y$ and it suffices to show $12y \leq 9y^2 + 4$. This can be done by the sum of squares method."
#     formal_statement = """theorem
#   fixes x::real
#   assumes "0<x" "x<pi"
#   shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)\""""

#     orienter = Orienter(model="gpt-3.5-turbo")

#     thoughts_queue_2(informal_statement, informal_proof, formal_statement, model="mistral-large")

    # response = orienter.orient(informal_statement, informal_proof, formal_statement, temperature=0)

    # print("##### RESPONSE #####")
    # print(response)

    # thoughts = extract_thoughts(response)
    # print("##### THOUGHTS #####")
    # for thought in thoughts:
    #     print(thought)
    #     print(thought.get('thought'))
    #     print(thought.get('code'))

    # proof = extract_proof(response)
    # print("##### PROOF #####")
    # print(proof)

    # Drafter = Drafter(
    #     prompt_template_path="path/to/prompt_template",
    #     model="gpt-3.5-turbo"
    # )

    # for thought in thoughts:
    #     informal_statement = thought.get('thought')
    #     formal_statement = thought.get('code')

    #     informal_proof = Drafter.write_proof_openai(informal_statement=informal_statement, formal_statement=formal_statement)
    #     print(informal_proof)
    #     print("\n\n")
