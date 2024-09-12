from langchain_utils import LLMMixture
from LemmaSketcher import LemmaSketcher
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from Orienter import Orienter
from typing import List, Dict
import os
import json
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "mistral-large", "The model to use for the language model")
flags.DEFINE_string("input_directory", "data/full_data/test", "The directory containing the input JSON files")
flags.DEFINE_string("messages_directory", "data/full_data/test_messages", "The directory to save the messages as JSON files")
flags.DEFINE_string("output_directory", "data/full_data/test_output", "The directory to save the responses as JSON files")
flags.DEFINE_float("temperature", 0.0, "The temperature to use for the language model")
flags.DEFINE_integer("request_timeout", 120, "The request timeout to use for the language model")
flags.DEFINE_string("type", "mistral", "The type of language model to use")
flags.DEFINE_integer("num_runs", 100, "The number of runs to use for the language model")

def extract_components_from_json(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    
    problem_name = data.get("problem_name", "")
    category = data.get("category", "")
    metadata = data.get("metadata", {})
    informal_statement = data.get("informal_statement", "")
    informal_proof = data.get("informal_proof", "")
    formal_statement = data.get("formal_statement", "")
    formal_code = data.get("formal_code", "")
    
    return {
        "problem_name": problem_name,
        "category": category,
        "metadata": metadata,
        "informal_statement": informal_statement,
        "informal_proof": informal_proof,
        "formal_statement": formal_statement,
        "formal_code": formal_code
    }

def message_to_dict(message):
    return {
        "role": message.type,
        "content": message.content
    }

def save_messages_to_json(messages: List[Dict[str, str]], directory: str, run_number: int):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"run_{run_number}.json")
    messages_as_dicts = [message_to_dict(msg) for msg in messages]
    with open(filepath, "w") as file:
        json.dump(messages_as_dicts, file, indent=4)

def load_process_and_save_all(input_directory, messages_directory, orienter, runs=100):
    for file in sorted(os.listdir(input_directory)):
        if file.endswith(".json"):
            filepath = os.path.join(input_directory, file)
            components = extract_components_from_json(filepath)
            
            problem_name = os.path.splitext(file)[0]
            problem_directory = os.path.join(messages_directory, problem_name)
            
            for i in range(1, runs + 1):
                messages = orienter.render_messages(
                    informal_statement=components["informal_statement"],
                    informal_proof=components["informal_proof"],
                    formal_statement=components["formal_statement"]
                )
                save_messages_to_json(messages, problem_directory, i)
            print(f"Saved {runs} runs of oriented messages for {file} in {problem_directory}")

def dict_to_message(message_dict):
    print("Message: ", message_dict)
    if message_dict["role"] == "system":
        return SystemMessage(message_dict["content"])
    elif message_dict["role"] == "human":
        return HumanMessage(message_dict["content"])
    elif message_dict["role"] == "ai":
        return AIMessage(message_dict["content"])
    else:
        raise ValueError("Invalid message role")
        
def load_message_pair(filename: str):
    print(filename)
    with open(filename, "r") as f:
        messages_as_dicts = json.load(f)
    # messages_as_dicts = json.loads(messages_as_dicts)
    print("Messages as dicts: ", messages_as_dicts)
    # return dict_to_message(messages_as_dicts)
    return [dict_to_message(msg) for msg in messages_as_dicts]  

def save_responses_to_json(responses: List[str], directory: str, run_number: int):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"run_{run_number}.txt")
    with open(filepath, "w") as file:
        file.write(responses)
        # json.dump(responses, file, indent=4)

def query_extracted_messages(messages_directory: str, responses_directory: str, model_name: str, temperature: float = 0.0, request_timeout: int = 120, logger=None, type="openai"):
    print("Querying extracted messages")
    llm = LLMMixture(
        model_name=model_name,
        temperature=temperature,
        request_timeout=request_timeout,
        logger=logger,
        type=type
    )

    for file in os.listdir(messages_directory):
        if file.endswith(".json"):
            filepath = os.path.join(messages_directory, file)
            print("Filepath: ", filepath)
            messages = load_message_pair(filepath)
            print("Messages: ", messages)
            response = llm.query(langchain_msgs=messages, temperature=temperature, max_tokens=4000)
            save_responses_to_json(response, responses_directory, int(file.split('_')[1].split('.')[0]))

def main(_):
    input_directory = FLAGS.input_directory
    messages_directory = FLAGS.messages_directory
    output_directory = FLAGS.output_directory
    runs = FLAGS.num_runs
    orienter = Orienter(
        model=FLAGS.model
    )

    load_process_and_save_all(input_directory, messages_directory, orienter, runs=runs)
    for directory in os.listdir(messages_directory):
        print("Directory: ", directory)
        query_extracted_messages(f"{messages_directory}/{directory}", output_directory, FLAGS.model, FLAGS.temperature, FLAGS.request_timeout, logger=None, type=FLAGS.type)         
            


if __name__ == "__main__":
    app.run(main)
    # input_directory = "data/full_data/test"
    input_directory = "test_input"
    output_directory = "test_output"
    orienter = Orienter(
        model="mistral-large"
    )

    query_extracted_messages(input_directory, output_directory, "mistral-large", temperature=0.0, request_timeout=120, logger=None, type="mistral")

    # load_process_and_save_all(input_directory, output_directory, orienter, runs=100)