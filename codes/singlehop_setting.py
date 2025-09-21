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
    'temperature': 1.0,
    'max_tokens': 256,
    'top_p': 1,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
}
reflector = OpenAILLM(reflector_config)

def check_single(graph, answer, topic_entities, relation):
    entity_set = []
    for x in graph:
        entity_set.append(x[0])
        entity_set.append(x[2])

    entity_set = list(set(entity_set))

    answers = []

    for entity in entity_set:
        cnt = 0
        for start, relationship, end in graph:
            if end != entity:
                continue
            if start in topic_entities and relationship == relation:
                cnt += 1

        if cnt == len(topic_entities):
            answers.append(entity)

    answers = list(set(answers))
    if len(answers) == 1 and answers[0] == answer:
        return True

    return False

def smooth(question):
    prompt = 'Please polish the following question to make it more natural and more close to reality. Just need to output the question, no need to output additional explanatory statements.\n'
    prompt += 'Question: ' + question + '\n'

    question = reflector.run(prompt, reflector_config)
    return question

def generate_single(graph):
    entity_set = []
    for x in graph:
        entity_set.append(x[0])
        entity_set.append(x[2])

    entity_set = list(set(entity_set))

    if len(entity_set) < 10:
        return None, None, None, None

    init_entity = "m."
    while init_entity[:2] == 'm.' or init_entity[:2] == 'g.':
        init_entity = random.choice(entity_set)

    adjacency_list = []
    relation_cnt = {}
    for start, relationship, end in graph:
        if end == init_entity:
            if start[:2] == 'm.' or start[:2] == 'g.':
                continue
            adjacency_list.append([start, relationship])
            if relationship not in relation_cnt.keys():
                relation_cnt[relationship] = 1
            else:
                relation_cnt[relationship] += 1

    key_relation = []
    for relation, cnt in relation_cnt.items():
        if cnt >= 3:
            key_relation.append(relation)

    if len(key_relation) == 0:
        return None, None, None, None

    relation = random.choice(key_relation)
    st_node = []
    for edge in adjacency_list:
        if edge[1] == relation:
            st_node.append(edge[0])

    num = random.randint(3, 7)
    entities = random.sample(st_node, min(num, len(st_node)))
    edges = [[init_entity, relation, entity] for entity in entities]

    flag = check_single(graph, init_entity, entities, relation)

    if flag is False:
        return None, None, None, None

    prompt = 'Please help me design a multi-entity reasoning question, including following entities: '
    prompt += str(entities) + '\n'
    prompt += 'And the answer of this question must be ' + init_entity + '\n'
    prompt += 'You can create the question according to the following knowledge graph. The question must be natural and close to reality. The question needs to include the relationships in the knowledge graph. Just need to output the question, no need to output additional explanatory statements.\n'
    prompt += 'Knowledge graph: ' + str(edges)
    prompt += 'Question: '

    question = reflector.run(prompt, reflector_config)

    question = smooth(question)

    return question, init_entity, entities, edges

output_file = 'singlehop_setting.jsonl'
graph_file = ['/graphs/new_graphs.jsonl']

graphs = []
for graph in graph_file:
    with open(graph, 'r') as f:
        for line in f:
            graphs.append(json.loads(line))

print(len(graphs))

with jsonlines.open(output_file, mode='a') as writer:
    for item in graphs:
        id = item['id']
        graph = item['graph']
        question, answer, topic_entities, edges = generate_single(graph)
        if question is None:
            continue
        if 'knowledge graph' in question:
            question = question.replace('knowledge graph', '')
        print(question)
        item = {'question': question,
                'answer': answer,
                'topic_entities': topic_entities,
                'edges': edges,
                'graph_id': id}
        writer.write(item)