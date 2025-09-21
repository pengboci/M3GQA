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

def get_path(start_node, end_node, adjacency_list):
    path = []
    while start_node != end_node:
        relation = adjacency_list[start_node][0][0]
        neighbor = adjacency_list[start_node][0][1]
        path.append([start_node, relation, neighbor])
        start_node = neighbor

    return path


def bfs(graph, init_node, k):
    adjacency_list = defaultdict(list)
    reverse_list = defaultdict(list)

    for start, relationship, end in graph:
        adjacency_list[end].append((relationship, start))

    visited = set()
    queue = deque([(init_node, 0)])
    subtree_edges = []

    while queue:
        current_node, current_depth = queue.popleft()

        if current_depth < k:
            if current_node not in visited:
                visited.add(current_node)

                neighbors = adjacency_list[current_node]

                if len(neighbors) < 1:
                    return None, None

                num = random.randint(1, len(neighbors))

                sampled_neighbors = random.sample(neighbors, num)

                for relationship, neighbor in sampled_neighbors:
                    subtree_edges.append([neighbor, relationship, current_node])
                    queue.append((neighbor, current_depth + 1))
                    reverse_list[neighbor].append((relationship, current_node))

    all_entity = list(visited)
    all_entity = [entity for entity in all_entity if entity[:2] != 'm.' and entity[:2] != 'g.' and '.' not in entity]
    num = random.randint(3, 8)
    entities = random.sample(all_entity, min(num, len(all_entity)))
    entities = list(set(entities))
    if init_node in entities:
        entities.remove(init_node)
    times = 0
    while len(entities) < 3:
        entities.append(random.choice(all_entity))
        entities = list(set(entities))
        if init_node in entities:
            entities.remove(init_node)
        times += 1
        if times >= 10:
            break
    if len(entities) < 3:
        return None, None

    path = []
    for entity in entities:
        path.extend(get_path(entity, init_node, reverse_list))

    return path, entities


def smooth(question):
    prompt = 'Please polish the following question to make it more natural and more close to reality. Just need to output the question, no need to output additional explanatory statements.\n'
    prompt += 'Question: ' + question + '\n'

    question = reflector.run(prompt, reflector_config)
    return question


def check(question, init_entity, entities):
    if init_entity.lower() in question.lower():
        return False

    cnt = 0
    for entity in entities:
        if entity.lower() in question.lower():
            cnt += 1
    if cnt < 3:
        return False

    return True

def generate_multihop(graph, k):
    entity_set = []
    for x in graph:
        entity_set.append(x[0])
        entity_set.append(x[2])

    entity_set = list(set(entity_set))

    if len(entity_set) < 10:
        return None, None, None, None

    init_entities = []
    for init_entity in entity_set:
        if init_entity[:2] != 'm.' and init_entity[:2] != 'g.' and '.' not in init_entity:
            init_entities.append(init_entity)

    if len(init_entities) == 0:
        return None, None, None, None

    init_entity = random.choice(init_entities)

    edges, entities = bfs(graph, init_entity, k)

    if entities is None or len(entities) < 3:
        return None, None, None, None

    prompt = 'Please help me design a multi-hop reasoning question.'
    prompt += 'The question must include at least 3 of the following entities: '
    prompt += str(entities) + '\n'
    prompt += 'And the answer of this question must be ' + init_entity + '\n'
    prompt += 'You can create the question according to the following knowledge graph. The question must be natural and close to reality.'
    prompt += 'The question needs to include the relationships in the knowledge graph. Just need to output the question, no need to output additional explanatory statements.\n'
    prompt += 'Knowledge graph: ' + str(edges)
    prompt += 'Question: '

    question = reflector.run(prompt, reflector_config)

    times = 0
    while check(question, init_entity, entities) is False:
        if times >= 3:
            break
        question = reflector.run(prompt, reflector_config)
        times += 1

    if check(question, init_entity, entities) is False:
        return None, None, None, None

    if 'knowledge graph' in question.lower():
        question = question.replace('knowledge graph', '')

    question = smooth(question)

    return question, init_entity, entities, edges

def modify_graph(graph, entity):
    new_graph = []
    for head, relation, tail in graph:
        if entity == head or entity == head:
            continue
        else:
            new_graph.append([head, relation, tail])

    return new_graph

output_file = 'answerability_setting.jsonl'
graph_file = ['/graphs/new_graphs.jsonl']

graphs = []
for graph in graph_file:
    with open(graph, 'r') as f:
        for line in f:
            graphs.append(json.loads(line))

print(len(graphs))

with jsonlines.open(output_file, mode='a') as writer:
    cnt = 0
    cnt1 = 0
    for item in graphs:
        id = item['id']
        graph = item['graph']
        for k in range(2, 6):
            num = random.choice([0, 1])

            question, answer, topic_entities, edges = generate_multihop(graph, k)
            if question is None:
                continue
            print(question)

            cnt += 1
            item = {'question': question,
                    'answer': "Yes",
                    'original_answer': answer,
                    'topic_entities': topic_entities,
                    'edges': edges,
                    'graph': graph}
            writer.write(item)