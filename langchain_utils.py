from __future__ import annotations
import random
import time
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings.openai import embed_with_retry
# from langchain.chat_models.openai import _create_retry_decorator
from langchain.schema import LLMResult, AIMessage, HumanMessage, SystemMessage, ChatGeneration


import os

from typing import List, Optional, Any

import numpy as np
import openai
import mistralai
import tiktoken

class LLMMixture:
    def __init__(self, model_name, temperature, request_timeout, logger=None, type="openai") -> None:
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.mistral_api_key = os.environ["MISTRAL_API_KEY"]
        # self.organisation = os.environ["OPENAI_ORGANISATION"]
        if type == "openai":
            self.client = openai.OpenAI(
                api_key=self.openai_api_key
            )
        elif type == "mistral":
            self.client = mistralai.Mistral(
                api_key=self.mistral_api_key
            )
        self.type = type
        self.logger = logger
    
    def query(self, langchain_msgs, llm_type="short", n=1, temperature=None, max_tokens=None):
        print("Querying")
        success = False
        max_retry = 50
        messages = []
        for msg in langchain_msgs:
            if isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})
                if self.logger:
                    self.logger.info(f"System: {msg.content}")
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
                if self.logger:
                    self.logger.info(f"User: {msg.content}")
            if isinstance(msg, AIMessage):
                if self.type == "mistral":
                    messages.append({"role": "assistant", "content": msg.content, "prefix": True})
                else:
                    messages.append({"role": "assistant", "content": msg.content})
                if self.logger:
                    self.logger.info(f"Assistant: {msg.content}")
        while max_retry > 0:
            try:
                if "gpt-4" in self.model_name:
                    if llm_type == "short":
                        llm_model = random.choice(["gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4o"])
                    else:
                        assert False
                elif "mistral-large" in self.model_name:
                    llm_model = "mistral-large-latest"
                elif "nemo" in self.model_name:
                    llm_model = "open-mistral-nemo"
                elif "codestral" in self.model_name:
                    llm_model = "codestral-latest"
                else:
                    if llm_type == "short":
                        llm_model = random.choice(["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106"])
                        
                    else:
                        llm_model = "gpt-3.5-turbo-16k"
                    # llm_model = random.choice(["gpt-4-32k","gpt-4-32k-0314","gpt-4-32k-0613"])

                if temperature is None:
                    temperature = self.temperature
                if self.type == "openai":
                    response = self.client.chat.completions.create(
                        model=llm_model,
                        messages=messages,
                        temperature=temperature,
                        n=n,
                        max_tokens=max_tokens,
                    )
                elif self.type == "mistral":
                    print("Mistral")
                    response = self.client.chat.complete(
                        model=llm_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
            except openai.RateLimitError as e:
                print(".", end="", flush=True)
                time.sleep(0.1)
            except openai.APIConnectionError as e:
                time.sleep(random.randint(1,30))
                print(f"Openai Connection{e}")
                max_retry -= 1
            except openai.APIError as e:
                time.sleep(random.randint(1,30))
                if 'Bad gateway. {"error":{"code":502,"message":"Bad gateway."' in str(e):
                    print("-", end="", flush=True)
                else:
                    print(f"APIError: {e}")
                max_retry -= 1
            except Exception as e:
                time.sleep(random.randint(1,30))
                print(f"Exception:{e}")
                max_retry -= 1
            else:
                success = True
                break
        if success:
            if n == 1:
                res = response.choices[0].message.content
                return res
            else:
                res = []
                for ix in range(n):
                    res.append(response.choices[ix].message.content)
                return res
        else:
            return ""

    def __call__(self, messages, temperature=None, max_tokens=1024, n=1) -> Any:
        word_count = 0
        for msg in messages:
            word_count += len(self.encoder.encode(msg.content))
        if "gpt-4" in self.model_name:
            if word_count < 7000:
                results = self.query(messages, "short", temperature=temperature, n=n)
            else:
                assert False, f"query too long, with {word_count} token in total" 
        else:
            if word_count < 3500:
                results = self.query(messages, "short", temperature=temperature, n=n)
            elif word_count < (16385 - 2100):
                results = self.query(messages, "long",  temperature=temperature, max_tokens=max_tokens, n=n)
            else:
                assert False, f"query too long, with {word_count} token in total" 
        
        if n==1:
            return AIMessage(content=results)
        else:
            ret_messages = []
            for res in results:
                ret_messages.append(AIMessage(content=res))
            return ret_messages
            
    
    def generate(self, batch_message, slow_mode=False, temperature=None, max_tokens=1024):
        if slow_mode is False:
   
            n = len(batch_message)
            word_count = 0
            messages = batch_message[0]
            for msg in messages:
                word_count += len(self.encoder.encode(msg.content))
            # print(f"ckpt 2 {word_count}")
            if "gpt-4" in self.model_name:
                if word_count < 7000:
                    results = self.query(messages, "short", n=n, temperature=temperature, max_tokens=max_tokens)
                else:
                    assert False, f"query too long, with {word_count} token in total" 
            else:
                if word_count < 3500:
                    results = self.query(messages, "short", n=n, temperature=temperature, max_tokens=max_tokens)
                elif word_count < 15000:
                    results = self.query(messages, "long", n=n, temperature=temperature, max_tokens=max_tokens)
                else:
                    assert False, f"query too long, with {word_count} token in total" 
            generations = []
            for res in results:
                generations.append([ChatGeneration(message=AIMessage(content=res))])
            # print(f"Here successful with {len(results)}")
            return LLMResult(generations=generations)
        else:
            results = []
            for messages in batch_message:
                word_count = 0
                messages = batch_message[0]
                for msg in messages:
                    word_count += len(self.encoder.encode(msg.content))
                if word_count < 7000:
                    res = self.query(messages, "short")
                else:
                    res = self.query(messages, "long")
                results.append(res)
            generations = []
            for res in results:
                generations.append([ChatGeneration(text=res)])
            return LLMResult(generations=generations)

class LLMMixtureLangchain:

    def __init__(self, model_name, temperature, request_timeout) -> None:
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        if model_name == "gpt-4":
            self.short_models = [
                ChatOpenAI(
                    model_name="gpt-4",
                    temperature=temperature,
                    request_timeout=request_timeout,
                ),
                ChatOpenAI(
                    model_name="gpt-4-0314",
                    temperature=temperature,
                    request_timeout=request_timeout,
                ),
                ChatOpenAI(
                    model_name="gpt-4-0613",
                    temperature=temperature,
                    request_timeout=request_timeout,
                )
            ]
            self.long_models = [
                ChatOpenAI(
                    model_name="gpt-4-32k",
                    temperature=temperature,
                    request_timeout=request_timeout,
                ),
                ChatOpenAI(
                    model_name="gpt-4-32k-0314",
                    temperature=temperature,
                    request_timeout=request_timeout,
                ),
                ChatOpenAI(
                    model_name="gpt-4-32k-0613",
                    temperature=temperature,
                    request_timeout=request_timeout,
                )
            ]
        else:
            raise NotImplementedError
        
    def __call__(self, messages) -> Any:
        word_count = 0
        for msg in messages:
            word_count += len(self.encoder.encode(msg.content))
        if word_count < 7000:
            llm_model = random.sample(self.short_models + self.long_models, 1)[0]
            return llm_model(messages)
        else:
            llm_model = random.sample(self.long_models, 1)[0]
            return llm_model(messages)
    
    def generate(self, batch_message, slow_mode=True):
        if slow_mode is False:
            max_word_count = 0
            for messages in batch_message:
                word_count = 0
                messages = batch_message[0]
                for msg in messages:
                    word_count += len(self.encoder.encode(msg.content))
                max_word_count = max(max_word_count, word_count)
            if max_word_count < 7000:
                llm_model = random.sample(self.short_models + self.long_models, 1)[0]
                return llm_model.generate(batch_message)
            else:
                llm_model = random.sample(self.long_models, 1)[0]
                return llm_model.generate(batch_message)
        else:
            results = []
            for messages in batch_message:
                word_count = 0
                messages = batch_message[0]
                for msg in messages:
                    word_count += len(self.encoder.encode(msg.content))
                if word_count < 7000:
                    llm_model = random.sample(self.short_models + self.long_models, 1)[0]
                    result = llm_model.generate([messages])
                else:
                    llm_model = random.sample(self.long_models, 1)[0]
                    result = llm_model.generate([messages])
                results.extend(result.generations)
            return LLMResult(generations=results)
    
    

class OpenAIEmbedingsWithAdaKey(OpenAIEmbeddings):

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        api_key = os.environ["OPENAI_API_KEY_ADA"]
        # handle large input text
        if len(text) > self.embedding_ctx_length:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            return embed_with_retry(
                self, input=[text], engine=engine, request_timeout=self.request_timeout, api_key=api_key
            )["data"][0]["embedding"]
    
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        api_key = os.environ["OPENAI_API_KEY_ADA"]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        encoding = tiktoken.model.encoding_for_model(self.model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens += [token[j : j + self.embedding_ctx_length]]
                indices += [i]

        batched_embeddings = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = embed_with_retry(
                self,
                input=tokens[i : i + _chunk_size],
                engine=self.deployment,
                request_timeout=self.request_timeout,
                headers=self.headers,
                api_key=api_key,
            )
            batched_embeddings += [r["embedding"] for r in response["data"]]

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = embed_with_retry(
                    self,
                    input="",
                    engine=self.deployment,
                    request_timeout=self.request_timeout,
                    headers=self.headers,
                    api_key=api_key,
                )["data"][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings



if __name__ == "__main__":
    llm = LLMMixture(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        request_timeout=10
    )

    response = llm.query(
        langchain_msgs=[
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of Uganda?")
        ],
        n=3
    )

    print(response)