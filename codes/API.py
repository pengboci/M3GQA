import random
import json
import requests
import jsonlines

from openai import OpenAI

from datasets import load_dataset
import datasets

import time

class OpenAILLM:
    def __init__(self, model_config):
        self.model_config = model_config
        self.client = OpenAI(api_key='',
                             base_url='')

    def llm(self, prompt, temperature=0, max_tokens=100,
                top_p=1, frequency_penalty=0.0, presence_penalty=0.0):
        messages = [{"role": "user", "content": prompt}]
        response = None
        while response is None:
            output = self.client.chat.completions.create(
                model=self.model_config['model'],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            response = output.choices[0].message.content
            if response is None:
                print('-' * 50)
                print(output)
                print('-' * 50)
        return response

    def run(self, prompt, config):
        while True:
            try:
                response = self.llm(prompt=prompt,
                                    temperature=config['temperature'],
                                    max_tokens=config['max_tokens'])
                return response
            except Exception as e:
                print(e)
                print("retrying")
                time.sleep(1)