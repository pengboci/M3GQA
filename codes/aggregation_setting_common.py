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

def rewrite(question, answer_entities, neighbor_edges):
    prompt = 'I will give you a question and its answer. I am now providing you with the answer set of all neighboring nodes on the knowledge graph. Please help me modify it into a question involving further reasoning, that is, further reasoning on the answer set. Additionally, you also need to provide the answer of modified question to me.\n'
    prompt += 'The output format is a binary (question, answer), where the first item represents the modified question and the second item represents the answer to the question. If the answer is a set, please write the answer part in the form of a list, such as (question, [a1, a2, a3, a4]). '
    prompt += 'If rewriting is not possible, output "No"\n'
    prompt += 'For example, the original question is, which famous rivers pass through China? The answer is Yangtze River, Yellow River, the Yarlung Zangbo River, etc. We can transform it into a question for further reasoning about the answer: ("Among all the famous rivers that pass through China, which one is the longest?", "Yangtze River")\n'
    prompt += 'For example, the original question is, which scientists in the field of deep learning have won the Turing Award? The answer if Hinton, Bengio, Lecun. We can modify it into a question for further reasoning about the answer: ("Who among the Turing Award winning scientists in the field of deep learning graduated from the University of Cambridge?", "Hinton")\n'
    prompt += 'The answer to the modified question can also be a collection. For example: ("which deep learning scientists who have won the Turing Award are Canadian?", "[Hinton, Bengio]")\n'
    prompt += 'Please strictly follow the form: (question, answer)! No need to output additional explanatory statements.\n'
    prompt += 'Original question: ' + question + '\n'
    prompt += 'The answer of the original question: ' + str(answer_entities) + '\n'
    prompt += 'Edges and neighbors related to the answer: ' + str(neighbor_edges) + '\n'

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

def generate_aggregation(graph, init_entity, k):
    entity_set = []
    for x in graph:
        entity_set.append(x[0])
        entity_set.append(x[2])

    entity_set = list(set(entity_set))

    if len(entity_set) < 10:
        return None, None, None, None

    edges, entities = bfs(graph, init_entity, k)

    if entities is None or len(entities) < 3:
        return None, None, None, None

    common_relation = []
    for entity in init_entity:
        tmp_relation = []
        for head, relation, tail in graph:
            if entity == head and tail[:2] != 'm.' and tail[:2] != 'g.':
                tmp_relation.append(relation)
        tmp_relation = list(set(tmp_relation))
        if len(common_relation) == 0:
            common_relation = tmp_relation
        else:
            common_relation = list(set(common_relation) & set(tmp_relation))

    if len(common_relation) == 0:
        return None, None, None, None

    ok_relation = []
    for relation in common_relation:
        flag = True
        for entity in init_entity:
            cnt = 0
            for head, relationship, tail in graph:
                if relation == relationship and entity == head and tail[:2] != 'm.' and tail[:2] != 'g.':
                    cnt += 1
            if cnt > 1:
                flag = False
                break

        if flag:
            ok_relation.append(relation)
    common_relation = ok_relation

    if len(common_relation) == 0:
        return None, None, None, None

    choice_relation = []
    for relationship in common_relation:
        tmp = []
        for head, relation, tail in graph:
            if head in init_entity and relation == relationship and tail[:2] != 'm.' and tail[:2] != 'g.':
                tmp.append(tail)
        tmp = list(set(tmp))
        if len(tmp) != len(init_entity):
            choice_relation.append(relationship)

    if len(choice_relation) == 0:
        return None, None, None, None

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
        return None, None, None, None

    if 'knowledge graph' in question.lower():
        question = question.replace('knowledge graph', '')

    neighbor_edges = []

    choiced_relation = random.choice(choice_relation)
    for head, relation, tail in graph:
        if head in init_entity and tail[:2] != 'm.' and tail[:2] != 'g.' and relation == choiced_relation:
            neighbor_edges.append([head, relation, tail])

    question = rewrite(question, init_entity, neighbor_edges)
    if 'no' in question.lower():
        return None, None, None, None

    print(question)

    try:
        question = eval(question)
    except SyntaxError:
        print(question)
        return None, None, None, None
    except TypeError:
        print(question)
        return None, None, None, None
    except NameError:
        print(question)
        return None, None, None, None

    question, answer = question

    try:
        if '[' in answer and ']' in answer:
            answer = eval(answer)
    except SyntaxError:
        return None, None, None, None
    except NameError:
        return None, None, None, None
    except TypeError:
        return None, None, None, None

    if isinstance(answer, list):
        for ans in answer:
            if ans not in init_entity:
                return None, None, None, None

        if len(answer) == len(init_entity):
            return None, None, None, None

        if len(answer) == 1:
            answer = answer[0]

    question = smooth(question)
    edges.extend(neighbor_edges)

    return question, entities, edges, answer

def sample_multi_entities(graph):
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

    return entities

output_file = 'aggregation_setting_common.jsonl'
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
        for k in range(2, 6):
            answer = None
            cnt2 = 0
            while answer is None:
                answer = sample_multi_entities(graph)
                cnt2 += 1
                if cnt2 >= 5:
                    break
            question, topic_entities, edges, final_answer = generate_aggregation(graph, answer, k)
            if question is None:
                continue

            print(question, final_answer)
            item = {'question': question,
                    'answer_entities': answer,
                    'answer': final_answer,
                    'topic_entities': topic_entities,
                    'edges': edges,
                    'graph_id': id}
            writer.write(item)