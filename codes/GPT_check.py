import os
import json
import jsonlines
from datasets import load_dataset
import datasets

from collections import deque, defaultdict
import random

from API import OpenAILLM

reflector_config = {
    'model': 'gpt-4-turbo',
    'temperature': 0.7,
    'max_tokens': 256,
    'top_p': 1,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
}
reflector = OpenAILLM(reflector_config)

input_files = [
         'single_setting.jsonl']

ok_files = [
         'single_setting.jsonl']

no_files = [
         'single_setting.jsonl']

def process_file(input):
    ok_list = []
    no_list = []
    with open(input, 'r') as file:
        for line in file:
            prompt = "Given a question and an answer, please use the given knowledge graph to determine if the answer is correct.\n"
            prompt += "If the answer is correct, please output 'yes', otherwise, output 'no'. No need to output additional statement.\n"

            data = json.loads(line)

            question = data['question']
            answer = data['answer']
            original_answer = data['original_answer']
            graph = data['edges']

            if answer == 'Yes':
                prompt += 'Question: ' + question + '\n'
                prompt += 'Answer: ' + str(original_answer) + '\n'
                prompt += 'Knowledge Graph: ' + str(graph) + '\n'
                prompt += 'Judgement: '

                output = reflector.run(prompt, reflector_config)

                print(output)

                if 'yes' in output.lower():
                    ok_list.append(data)
                else:
                    no_list.append(data)
            else:
                ok_list.append(data)

    return ok_list, no_list

for i in range(len(input_files)):
    input_file = input_files[i]
    ok_file = ok_files[i]
    no_file = no_files[i]

    ok_list, no_list = process_file(input_file)

    with jsonlines.open(ok_file, 'a') as file:
        for data in ok_list:
            file.write(data)

    with jsonlines.open(no_file, 'a') as file:
        for data in no_list:
            file.write(data)

    print(input_file + ' is ok!')