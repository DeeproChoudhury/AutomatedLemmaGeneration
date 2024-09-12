import boto3
import pprint
from botocore.client import Config
import json
import os
from langchain_community.chat_models.bedrock import BedrockChat
from langchain.retrievers.bedrock import AmazonKnowledgeBasesRetriever
import pkg_resources

import collections
import os
import pickle
import sys
import errno
import shutil
import glob

import codecs
import hashlib
import tarfile
import fnmatch
import tempfile
from datetime import datetime
from socket import gethostname
import logging

import random

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import time

kb_id = "YL3GYNLP2E"
##
pp = pprint.PrettyPrinter(indent=2)
session = boto3.session.Session()
region = session.region_name
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_client = boto3.client('bedrock-runtime', region_name = 'us-west-2')
bedrock_agent_client = boto3.client("bedrock-agent-runtime",
                              config=bedrock_config, region_name = 'us-west-2')
                              
                              
def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults: 
        contexts.append(retrievedResult['content']['text'])
    return contexts

# contexts = get_contexts(retrievalResults)
# pp.pprint(contexts)

modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
accept = 'application/json'
contentType = 'application/json'
llm = BedrockChat(model_id=modelId, 
                  client=bedrock_client)
                  
#query = "Provide examples of asylum seekers fearing persecution if deported."
query = "Summarize how aviation law regards the transport of suspicious packages"
retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={"vectorSearchConfiguration": 
                          {"numberOfResults": 4,
                           'overrideSearchType': "SEMANTIC",
                           }
                          },
    )

lemma_examples = {}
for file in os.listdir("data/lemma_examples"):
    with open(os.path.join("data/lemma_examples", file), "r") as f:
        text = f.read()
    lemma_examples[file[:-4]] = text

system_prompt = """
As a mathematician and expert in the isabelle theorem prover, your task is to analyze the given theorem (including problem's informal statement, 
human written informal proof, and formal statement). Provide a better structured step by step proof that closer to isabelle. 
and request relevant lemmas, theorems that might help in proving this problem.
"""

human_prompt = """
Here are some examples.

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
```"""

def host_name():
    "Get host name, alias with ``socket.gethostname()``"
    return gethostname()


def host_id():
    """
    Returns: first part of hostname up to '.'
    """
    return host_name().split(".")[0]


def utf_open(fname, mode):
    """
    Wrapper for codecs.open
    """
    return codecs.open(fname, mode=mode, encoding="utf-8")


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)


def pack_varargs(args):
    """
    Pack *args or a single list arg as list

    def f(*args):
        arg_list = pack_varargs(args)
        # arg_list is now packed as a list
    """
    assert isinstance(args, tuple), "please input the tuple `args` as in *args"
    if len(args) == 1 and is_sequence(args[0]):
        return args[0]
    else:
        return args


def f_not_empty(*fpaths):
    """
    Returns:
        True if and only if the file exists and file size > 0
          if fpath is a dir, if and only if dir exists and has at least 1 file
    """
    fpath = f_join(*fpaths)
    if not os.path.exists(fpath):
        return False

    if os.path.isdir(fpath):
        return len(os.listdir(fpath)) > 0
    else:
        return os.path.getsize(fpath) > 0


def f_expand(fpath):
    return os.path.expandvars(os.path.expanduser(fpath))


def f_exists(*fpaths):
    return os.path.exists(f_join(*fpaths))


def f_join(*fpaths):
    """
    join file paths and expand special symbols like `~` for home dir
    """
    fpaths = pack_varargs(fpaths)
    fpath = f_expand(os.path.join(*fpaths))
    if isinstance(fpath, str):
        fpath = fpath.strip()
    return fpath

def load_text(*fpaths, by_lines=False):
    with open(f_join(*fpaths), "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def load_prompt(prompt):
    package_path = pkg_resources.resource_filename("lego_prover", "")
    return load_text(f"{package_path}/prompts/{prompt}.txt")

if __name__=="__main__":
    
    for i in range(76, 100):
        try:
            lemma_examples = {}
            for file in os.listdir("data/lemma_examples"):
                with open(os.path.join("data/lemma_examples", file), "r") as f:
                    text = f.read()
                lemma_examples[file[:-4]] = text
            
            icl_examples = random.sample(list(lemma_examples.values()), 3)
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
                informal_proof="Let $y = x \sin x$. It suffices to show that $12 \leq \frac{9y^2 + 4}{y}. It is trivial to see that $y > 0$. Then one can multiply both sides by $y$ and it suffices to show $12y \leq 9y^2 + 4$. This can be done by the sum of squares method.",
                formal_statement="""theorem
          fixes x::real
          assumes "0<x" "x<pi"
          shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)\""""
            )
            
            brt = boto3.client(service_name='bedrock-runtime')
        
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
            
            response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, 
            contentType=contentType)
            
            # print(response)
        
            response_body = json.loads(response.get('body').read())
            # print(response_body)
            
            # text
            # print(response_body.get('content')[0]['text'])
            
            directory = "aime_1983_p9"
            filename = f"{directory}/aime_1983_p9_{i}.txt"
            
            if not os.path.exists(directory):
                os.makedirs(directory)
        
            with open(filename, 'w') as f:
                f.write(response_body.get('content')[0]['text'])
            
            # print(human_message.content)
        except Exception as e:
            i -= 1
            time.sleep(5)