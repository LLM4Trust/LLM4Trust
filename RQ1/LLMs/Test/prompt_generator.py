class DyGraphPrompt:
    def __init__(self, args):
        self.args = args
        self.add_role = args.role
        self.add_knowledge = args.knowledge
        self.add_example = args.example
        self.add_cot = args.cot

        self.role = "You are Frederick, an AI expert in trust evaluation.\n" if self.add_role else ""
        self.cot = "You can think it step by step.\n" if self.add_cot else ""

    def fixed_prompt(self, task, knowledge, example, question):
        if self.add_role == 0:
            prompt_q = task + knowledge + example + question + self.cot
            prompt_qs = ""
        else:
            prompt_q = question + self.cot
            prompt_qs = self.role + task + knowledge + example  # act as a system

        return prompt_q, prompt_qs


    def generate_dynamic_qa(self, edges, special_nodes, time):
        optimal_path = special_nodes
        edge_data = edges
        # task&question descriptions
        task = ("Graph Instruction: In a directed trust graph, (u, v, t) means that node u has a trust relationship with node v at time t.\n"
                "Task Instruction: Your task is to return when node A first connected to node B.\n"
                "Answer Instruction: Give the answer as an integer number at your response after 'Answer:', e.g., 'Answer: 2'.\n")

        tr_te_tuple = [tuple(row) for row in edge_data]
        question = ("Question: Given a directed trust graph with the edges {}. When is node {} first connected to node {}?\n"
            .format(tr_te_tuple, optimal_path[0], optimal_path[1]))

        # specific prompts for the dynamic property
        knowledge = "Expertise: Trust is dynamic over time; specifically, (u, v, t1) is established earlier than (u, v, t2) if t1 < t2.\n" if self.add_knowledge else ""
        if self.add_example == 1:  # one-shot
            example = "Here is an example: Question: Given a directed trust graph with the edges [(1, 2, 4), (2, 3, 1)]. When is node 1 first connected to node 3? Answer: 4\n"
        elif self.add_example == 2:  # few-shot
            example = (
                "Here is the first example: Question: Given a directed trust graph with the edges [(1, 2, 4), (2, 3, 1)]. When is node 1 first connected to node 3? Answer: 4\n"
                "Here is the second example: Question: Given a directed trust graph with the edges [(0, 1, 2), (1, 4, 2), (4, 2, 3)]. When is node 0 first connected to node 2? Answer: 3\n"
                "Here is the third example: Question: Given a directed trust graph with the edges [(0, 1, 1), (1, 3, 2), (0, 3, 3)]. When is node 0 first connected to node 3? Answer: 2\n")
        else:
            example = ""

        # prompts construction
        prompt_q, prompt_qs = self.fixed_prompt(task, knowledge, example, question)
        prompt_a = str(time)

        return {"prompt": prompt_q, "answer": prompt_a, "prompt_system": prompt_qs}

    def generate_asymmetric_qa(self, edges, special_nodes, weight):
        optimal_path = special_nodes
        edge_data = edges
        # task&question descriptions
        tr_te_tuple = [tuple(row) for row in edge_data]
        answer = weight

        task = ("Graph Instruction: In a directed trust graph, (u, v, w) means that node u trusts node v with the level of w.\n"
                "Task Instruction: Your task is to return the trust level from one node towards another.\n"
                "Answer Instruction: Give the answer as an integer number at your response after 'Answer:', e.g., 'Answer: 2'.\n")

        question = ("Question: Given a directed trust graph with the edges {}. What is the level of trust from node {} to node {}?\n"
                    .format(tr_te_tuple, optimal_path[0], optimal_path[1]))

        # specific prompts for the asymmetric property
        knowledge = "Expertise: Trust is inherently asymmetric; specifically, the trust of node u in node v, represented as (u, v, w), does not imply an equivalent trust of node v in node u, which would be represented as (v, u, w).\n" if self.add_knowledge else ""
        if self.add_example == 1:  # one-shot
            example = "Here is an example: Question: Given a directed trust graph with the edges [(0, 1, 1), (0, 2, 3)]. What is the level of trust from node 1 to node 0? Answer: 0\n"
        elif self.add_example == 2:  # few-shot
            example = (
                "Here is the first example: Question: Given a directed trust graph with the edges [(0, 1, 1), (0, 2, 3)]. What is the level of trust from node 1 to node 0? Answer: 0\n"
                "Here is the second example: Question: Given a directed trust graph with the edges [(1, 3, 3), (2, 1, 4)]. What is the level of trust from node 3 to node 1? Answer: 0\n"
                "Here is the third example: Question: Given a directed trust graph with the edges [(0, 1, 3), (1, 3, 1)]. What is the level of trust from node 3 to node 1? Answer: 0\n")
        else:
            example = ""
        # prompts construction
        prompt_q, prompt_qs = self.fixed_prompt(task, knowledge, example, question)
        prompt_a = str(answer)
        return {"prompt": prompt_q, "answer": prompt_a, "prompt_system": prompt_qs}

    def generate_propagative_qa(self, edges, special_nodes, min_weight):

        optimal_path = special_nodes
        edge_data = edges
        # task&question descriptions
        task = (
            "Graph Instruction: In a directed trust graph, (u, v, w) means that node u trusts node v with the level of w.\n"
            "Task Instruction: Your task is to return the trust level from one node towards another.\n"
            "Answer Instruction: Give the answer as an integer number at your response after 'Answer:', e.g., 'Answer: 2'.\n")

        tr_te_tuple = [tuple(row) for row in edge_data]
        question = (
            "Question: Given a directed trust graph with the edges {}. What is the level of trust from node {} to node {}?\n"
            .format(tr_te_tuple, optimal_path[0], optimal_path[1]))

        # specific prompts for the propagative property
        knowledge = "Expertise: Trust is propagative; specifically, the trust of node u in node v along a given path is determined by the minimum trust level among all edges constituting that path.\n" if self.add_knowledge else ""
        if self.add_example == 1:  # one-shot
            example = "Here is an example: Question: Given a directed trust graph with the edges [(2, 3, 3), (3, 4, 2)]. What is the level of trust from node 2 to node 4? Answer: 2\n"
        elif self.add_example == 2:  # few-shot
            example = (
                "Here is the first example: Question: Given a directed trust graph with the edges [(2, 3, 3), (3, 4, 2)]. What is the level of trust from node 2 to node 4? Answer: 2\n"
                "Here is the second example: Question: Given a directed trust graph with the edges [(1, 2, 1), (2, 3, 3), (3, 4, 2)]. What is the level of trust from node 1 to node 4? Answer: 1\n"
                "Here is the third example: Question: Given a directed trust graph with the edges [(1, 4, 2), (4, 3, 1)]. What is the level of trust from node 1 to node 3?  Answer: 1\n")
        else:
            example = ""

        # prompts construction
        prompt_q, prompt_qs = self.fixed_prompt(task, knowledge, example, question)
        prompt_a = str(min_weight)

        return {"prompt": prompt_q, "answer": prompt_a, "prompt_system": prompt_qs}

    def generate_composable_qa(self, edges, special_nodes, num_min, num_max):

        optimal_path = special_nodes
        edge_data = edges
        # task&question descriptions
        task = (
            "Graph Instruction: In a directed trust graph, (u, v, w) means that node u trusts node v with the level of w.\n"
            "Task Instruction: Your task is to return the range of trust from one node towards another.\n"
            "Answer Instruction: Give the answer as two integer numbers in ascending at your response after 'Answer:', e.g., 'Answer: 1, 3'.\n")

        tr_te_tuple = [tuple(row) for row in edge_data]
        question = (
            "Question: Given a directed trust graph with the edges {}. What is the range of trust from node {} to node {}?\n"
            .format(tr_te_tuple, optimal_path[0], optimal_path[1]))

        knowledge = "Expertise: Trust is composable; specifically, the trust of node u in node v along a single path is determined by the minimum trust level among all edges constituting that path. When there are multiple paths from node u to node v, the trust level from u to v falls within the range defined by the individual path trust levels.\n" if self.add_knowledge else ""
        if self.add_example == 1:  # one-shot
            example = "Here is an example:Question: Given a directed trust graph with the edges [(1, 2, 3), (1, 3, 4), (2, 4, 2), (3, 4, 3)]. What is the range of trust from node 1 to node 4? Answer: 2, 3\n"
        elif self.add_example == 2:  # few-shot
            example = (
                "Here is the first example: Question: Given a directed trust graph with the edges [(1, 2, 3), (1, 3, 4), (2, 4, 2), (3, 4, 3)]. What is the range of trust from node 1 to node 4? Answer: 2, 3\n"
                "Here is the second example: Question: Given a directed trust graph with the edges [(1, 2, 3), (1, 3, 4), (2, 4, 2), (3, 4, 3), (1, 0, 1), (0, 4, 2)]. What is the range of trust from node 1 to node 4? Answer: 1, 3\n"
                "Here is the third example: Question: Given a directed trust graph with the edges [(2, 3, 4), (3, 4, 2), (4, 1, 3), (2, 0, 1), (0, 1, 5)]. What is the range of trust from node 2 to node 1? Answer: 1, 2\n")
        else:
            example = ""

        # prompts construction
        prompt_q, prompt_qs = self.fixed_prompt(task, knowledge, example, question)
        prompt_a = f"{num_min}, {num_max}"

        return {"prompt": prompt_q, "answer": prompt_a, "prompt_system": prompt_qs}

    def generate_context_qa(self, edges, special_nodes, c, w):

        optimal_path = special_nodes
        edge_data = edges
        # task&question descriptions
        task = (
            "Graph Instruction: In a directed trust graph, (u, v, c, w) means that node u trusts node v with a trust level of w in the context of c.\n"
            "Task Instruction:  Your task is to return the trust level from one node towards another.\n"
            "Answer Instruction: Give the answer as an integer number at your response after 'Answer:', e.g., 'Answer: 2'.\n")

        tr_te_tuple = [tuple(row) for row in edge_data]
        question = (
            "Question: Given a directed trust graph with the edges {}. What is the level of trust from node {} to node {} in the context of 3?\n"
            .format(tr_te_tuple, optimal_path[0], optimal_path[1]))

        knowledge = "Expertise: Trust is context-aware; specifically, (u, v, c1, w1) means that node u trusts node v with a trust level of w1 in the context of c1, but it does not imply that u has the same trust in v in the context of c2, which would be represented as (u, v, c2, w2).\n" if self.add_knowledge else ""
        if self.add_example == 1:  # one-shot
            example = "Here is an example: Question: Given a directed trust graph with the edges [(1, 2, 1, 3), (1, 2, 2, 4)]. What is the level of trust from node 1 to node 2 in the in the context of 3? Answer: 0\n"
        elif self.add_example == 2:  # few-shot
            example = (
                "Here is the first example: Question: Given a directed trust graph with the edges [(1, 2, 1, 3), (1, 2, 2, 4)]. What is the level of trust from node 1 to node 2 in the in the context of 3? Answer: 0\n"
                "Here is the second example: Question: Given a directed trust graph with the edges [(1, 3, 1, 4), (1, 3, 2, 5), (3, 1, 1, 2)]. What is the level of trust from node 1 to node 3 in the context of 3? Answer: 0\n"
                "Here is the third example: Question: Given a directed trust graph with the edges [(3, 5, 2, 3), (1, 3, 2, 2), (1, 3, 1, 4)]. What is the level of trust from node 3 to node 5 in the context of 1? Answer: 0\n")
        else:
            example = ""

        # prompts construction
        prompt_q, prompt_qs = self.fixed_prompt(task, knowledge, example, question)
        prompt_a = f"{0}"

        return {"prompt": prompt_q, "answer": prompt_a, "prompt_system": prompt_qs}

