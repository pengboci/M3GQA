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
        try:
            relation = adjacency_list[start_node][0][0]
            neighbor = adjacency_list[start_node][0][1]
            path.append([start_node, relation, neighbor])
            start_node = neighbor
        except IndexError:
            return []

    return path


def bfs(graph, init_node, k):
    adjacency_list = defaultdict(list)
    reverse_list = defaultdict(list)

    for start, relationship, end in graph:
        adjacency_list[end].append((relationship, start))

    first_hop_neighbor = adjacency_list[init_node[0]]
    for i in range(1, len(init_node)):
        first_hop_neighbor = list(set(first_hop_neighbor) & set(adjacency_list[init_node[i]]))
        first_hop_neighbor = list(set(first_hop_neighbor))
        neighbors = []
        tmp = []
        for relation, node in first_hop_neighbor:
            if node in init_node:
                continue
            if node in neighbors:
                continue
            neighbors.append(node)
            tmp.append((relation, node))
        first_hop_neighbor = tmp

    if len(first_hop_neighbor) == 0:
        return None, None

    visited = set()
    queue = deque()
    for node in init_node:
        visited.add(node)
    subtree_edges = []

    for relationship, neighbor in first_hop_neighbor:
        queue.append((neighbor, 1))
        visited.add(neighbor)
        for node in init_node:
            subtree_edges.append([neighbor, relationship, node])
            reverse_list[neighbor].append((relationship, node))

    while queue:
        current_node, current_depth = queue.popleft()

        if current_depth < k:
            neighbors = adjacency_list[current_node]

            for relation, neighbor in neighbors:
                if neighbor in visited:
                    neighbors.remove((relation, neighbor))
                    continue

            if len(neighbors) == 0:
                return None, None

            sampled_neighbors = random.sample(neighbors, min(len(neighbors), 10))
            sampled_neighbors = list(set(sampled_neighbors))

            for relationship, neighbor in sampled_neighbors:
                if neighbor in visited:
                    continue
                subtree_edges.append([neighbor, relationship, current_node])
                queue.append((neighbor, current_depth + 1))
                reverse_list[neighbor].append((relationship, current_node))
                visited.add(neighbor)

    all_entity = list(visited)
    all_entity = [entity for entity in all_entity if entity[:2] != 'm.' and entity[:2] != 'g.']
    num = random.randint(3, 8)
    entities = random.sample(all_entity, min(num, len(all_entity)))
    entities = list(set(entities))
    for node in init_node:
        if node in entities:
            entities.remove(node)
    times = 0
    while len(entities) < 3:
        entities.append(random.choice(all_entity))
        entities = list(set(entities))
        for node in init_node:
            if node in entities:
                entities.remove(node)
        times += 1
        if times >= 10:
            break
    if len(entities) < 3:
        return None, None

    for node in init_node:
        if node in entities:
            return None, None

    path = []
    for entity in entities:
        cnt = 0
        for relation, node in first_hop_neighbor:
            tmp_path = get_path(entity, node, reverse_list)
            if entity == node or len(tmp_path) > 0:
                for p in init_node:
                    tmp_path.append([node, relation, p])
                path.extend(tmp_path)
                cnt += 1
        if cnt != 1:
            return None, None

    return path, entities


def smooth(question):
    prompt = 'Please polish the following question to make it more natural and more close to reality. Just need to output the question, no need to output additional explanatory statements.\n'
    prompt += 'Question: ' + question + '\n'

    question = reflector.run(prompt, reflector_config)
    return question


def check(question, init_entity, entities):
    for entity in init_entity:
        if entity.lower() in question.lower():
            return False

    cnt = 0
    for entity in entities:
        if entity.lower() in question.lower():
            cnt += 1
    if cnt < 3:
        return False

    return True

def generate_set(graph, init_entity, k):
    entity_set = []
    for x in graph:
        entity_set.append(x[0])
        entity_set.append(x[2])

    entity_set = list(set(entity_set))

    if len(entity_set) < 10:
        return None, None, None

    edges, entities = bfs(graph, init_entity, k)

    if entities is None or len(entities) < 3:
        return None, None, None

    prompt = 'Please help me design a multi-entity multi-hop reasoning question.'
    prompt += 'The question must include at least 3 of the following entities: '
    prompt += str(entities) + '\n'
    prompt += 'And the answer of this question must be ' + str(init_entity) + '\n'
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
        return None, None, None

    if 'knowledge graph' in question.lower():
        question = question.replace('knowledge graph', '')

    question = smooth(question)

    return question, entities, edges

def sample_multi_entities(graph):
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
        if start == init_entity:
            if end[:2] == 'm.' or end[:2] == 'g.':
                continue
            adjacency_list.append([end, relationship])
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

    return st_node

output_file = 'set_setting.jsonl'
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
        answer = sample_multi_entities(graph)

        for k in range(6):
            question, topic_entities, edges = generate_set(graph, answer, k)
            if question is None:
                continue
            print(question)
            item = {'question': question,
                    'answer': answer,
                    'topic_entities': topic_entities,
                    'edges': edges,
                    'graph_id': id}
            writer.write(item)
