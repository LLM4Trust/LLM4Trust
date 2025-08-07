import random
import numpy as np
import networkx as nx
import utlis
from utlis import build_few_shot_examples

class DyGraphPrompt:
    def __init__(self, args):
        self.args = args
        self.add_role = args.role
        self.add_knowledge = args.knowledge
        self.add_example = args.example
        self.add_cot = args.cot

        self.length_1 = args.length_1
        self.length_2 = args.length_2
        self.hop = args.hop

        self.role = "You are Frederick, an AI expert in trust evaluation.\n" if self.add_role else ""
        self.cot = "You can think it step by step.\n" if self.add_cot else ""

    def fixed_prompt(self, task, knowledge, example, question):
        if self.add_role == 0:
            prompt_q = task + knowledge + example + question + self.cot
            prompt_qs = ""
        else:
            prompt_q = question + self.cot
            prompt_qs = self.role + task + knowledge + example  # Act as system prompt

        return prompt_q, prompt_qs

    def generate_qa(self, edges, src, dst, label, timestamp, tag):
        edge_data = edges
        task = (
            "Graph Instruction: You are given a directed trust graph. Each edge is a 4-tuple (u, v, w, t), meaning node u trusts node v with trust level w at time t.\n"
            "Task Instruction: Your task is to predict the trust level from node u to node v at a given future or missing time t.\n"
            "Answer Instruction: Provide only the predicted trust level (0 or 1) at your last response after 'Answer:', e.g., 'Answer: 1'.\n"
        )

        tr_te_tuple = [tuple(row) for row in edge_data]
        question = (
            "Question: Given a directed trust graph with edges {}, what is the trust level from node {} to node {} at time {}?\n"
            .format(tr_te_tuple, int(src), int(dst), timestamp))

        if tag == 4:
            knowledge = (
                "Expertise: Trust is dynamic, asymmetric, propagative, and composable."
                "Around 90% of cases involve trust, and only 10% involve distrust, creating a strong imbalance that influences reasoning.\n")
        else:
            knowledge = (
                "Expertise: Trust is dynamic, asymmetric, propagative, and composable.\n")

        # few-shot examples
        if self.add_example == 2:
            train_file = "graph/train_edges.csv"
            example = build_few_shot_examples(self.args, train_file, src, dst, tag)
        else:
            example = ""

        # prompt construction
        prompt_q, prompt_qs = self.fixed_prompt(task, knowledge, example, question)
        prompt_a = str(label)

        return {"prompt": prompt_q, "answer": prompt_a, "prompt_system": prompt_qs}
