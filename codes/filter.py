import json
import jsonlines

from collections import defaultdict

def check_entity_cnt(question, entities):
    cnt = 0
    for entity in entities:
        if entity in question:
            cnt += 1

    if cnt >= 3:
        return True
    return False

def check_multi_question(question):
    cnt = question.count('?')
    if cnt != 1:
        return False
    return True

def check_answer_appear(question, answer):
    if answer in question:
        return False

    return True

def check_tree_contain_answer(answer, edges):
    for head, relation, tail in edges:
        if answer == head or answer == tail:
            return True

    return False

def get_path(start_node, end_node, adjacency_list, first_hop):
    path = []
    while start_node != end_node:
        if start_node in first_hop:
            return []
        relation = adjacency_list[start_node][0][0]
        neighbor = adjacency_list[start_node][0][1]
        path.append((start_node, relation, neighbor))
        start_node = neighbor

    return path

input_file = 'set_setting.jsonl'
output_file = 'set_setting2.jsonl'

output_data = []
answer_list = []

with open(input_file, 'r') as file:
    for line in file:
        data = json.loads(line)

        question = data['question']
        answer = data['answer']
        topic_entities = data['topic_entities']
        edges = data['edges']
        graph_id = data['graph_id']

        edges = [[item[0], item[1], item[2]] for item in edges if isinstance(item, list)]

        if not check_entity_cnt(question, topic_entities):
            continue

        if not check_multi_question(question):
            continue

        flag = True
        for ans in answer:
            if not check_answer_appear(question, ans):
                flag = False

        if not flag:
            continue

        for ans in answer:
            if not check_tree_contain_answer(ans, edges):
                flag = False
        if not flag:
            continue

        if answer in answer_list:
            continue

        answer_list.append(answer)

        new_topic_entities = []
        for entity in topic_entities:
            if entity in question:
                new_topic_entities.append(entity)

        adjacency_list = defaultdict(list)
        reverse_list = defaultdict(list)
        for start, relationship, end in edges:
            adjacency_list[end].append((relationship, start))
            reverse_list[start].append((relationship, end))

        first_hop_neighbor = adjacency_list[answer[0]]
        for i in range(1, len(answer)):
            first_hop_neighbor = list(set(first_hop_neighbor) & set(adjacency_list[answer[i]]))
            first_hop_neighbor = list(set(first_hop_neighbor))
            neighbors = []
            tmp = []
            for relation, node in first_hop_neighbor:
                if node in answer:
                    continue
                if node in neighbors:
                    continue
                neighbors.append(node)
                tmp.append((relation, node))
            first_hop_neighbor = tmp

        new_edges = []
        neighbors = [neighbor for relation, neighbor in first_hop_neighbor]
        for entity in new_topic_entities:
            if entity in neighbors:
                continue

            for relation, node in first_hop_neighbor:
                tmp_path = get_path(entity, node, reverse_list, neighbors)
                if len(tmp_path) == 0:
                    continue

                new_edges.extend(tmp_path)

        for relation, node in first_hop_neighbor:
            for p in answer:
                new_edges.append((node, relation, p))

        new_edges = list(set(new_edges))
        new_edges = [[head, relation, tail] for head, relation, tail in new_edges]

        output_data.append({'question': question,
                            'answer': answer,
                            'topic_entities': new_topic_entities,
                            'edges': new_edges,
                            'graph_id': graph_id})

print(len(output_data))

with jsonlines.open(output_file, mode='a') as writer:
    for data in output_data:
        writer.write(data)