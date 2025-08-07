import numpy as np
from openai import OpenAI
import random
from utlis import cal_metrics, save_prompt, save_response
from tqdm import tqdm
import time
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--model", type=str, default='claude-3-7-sonnet-20250219')
parser.add_argument("--knowledge", type=int, default=1)
parser.add_argument("--example", type=int, default=1)  # 1 for few-shot, while 0 for 0-shot
parser.add_argument("--cot", type=int, default=0)
parser.add_argument("--role", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--path_length", type=int, default=60)
parser.add_argument("--hop", type=int, default=3)
parser.add_argument("--sampling", type=int, default=200)  # 20 for samll-scale evaluation, while 200 for large-scale evaluation
parser.add_argument("--data", type=str, default="advogato")
args = parser.parse_args()

if args.sampling == 20:
    filename = "./{}/small_pathlen{}_hop{}_seed{}.txt".format(args.data, args.path_length, args.hop, args.seed)
elif args.sampling == 200:
    filename = "./{}/large_pathlen{}_hop{}_seed{}.txt".format(args.data, args.path_length, args.hop, args.seed)

client = OpenAI(api_key="YOUR_ANTHROPIC_API_KEY",
                base_url="https://api.anthropic.com/v1/")

# Construct prompt
task = ("Graph Instruction: In a directed trust graph, (u, v, w) means that node u trusts node v with the level of w. "
        "The trust level w should be one of the following values: 1, 2, 3, or 4, with higher values indicating greater trust.\n"
        "Task Instruction: Your task is to predict the trust level from one node towards another.\n"
        "Answer Instruction: Give the answer as an integer number selected from {1, 2, 3, 4} at your last response after 'Answer:', e.g., 'Answer: 2'.\n")
role = "You are Frederick, an AI expert in trust evaluation.\n"
cot = ""  # chain-of-thought prompt (optional)
knowledge = "Expertise: Trust is asymmetric, propagative, and composable.\n"
example = ""

random.seed(args.seed)
with open('./{}/test_paths{}_hop{}.txt'.format(args.data, args.path_length, args.hop), 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
sampled_lines = random.sample(lines, args.sampling)  # randomly sample 20 lines
print(sampled_lines[0], '\n', sampled_lines[-1])

pred_label, test_label = [], []
time_list = []
for line in tqdm(sampled_lines, desc="LLM inference"):
    src_dst_str, label_str, path_list_str = line.split('\t')
    src_dst = eval(src_dst_str)  # Convert (src, dst) string to tuple
    label = int(label_str)  # Convert label to integer
    path_list = eval(path_list_str)  # Convert string to list of (u, v, l)

    if args.example:  # Whether to use few-shot examples
        example = ""
        with open('./{}/train_paths{}_hop{}.txt'.format(args.data, args.path_length, args.hop), 'r') as f:
            train_lines = [line.strip() for line in f if line.strip()]
            sampled_train_lines = random.sample(train_lines, 3)  # Randomly sample 3 lines
            for idx, train_line in enumerate(sampled_train_lines):
                train_src_dst_str, train_label_str, train_path_list_str = train_line.split('\t')
                train_src_dst = eval(train_src_dst_str)
                train_label = int(train_label_str)
                train_path_list = eval(train_path_list_str)
                example += "Here is the {} example: Question: Given a directed trust graph with the edges {}. What is the level of trust from node {} to node {}? Answer: {}\n".format(
                    ["first", "second", "third"][idx], train_path_list, train_src_dst[0], train_src_dst[1], train_label)

    question = "Question: Given a directed trust graph with the edges {}. What is the level of trust from node {} to node {}?".format(path_list, src_dst[0], src_dst[1])
    prompt_qa = {'prompt': question + cot, 'prompt_system': role + task + knowledge + example}
    save_prompt(prompt_qa, filename)  # Save the prompt

    start_time = time.time()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": prompt_qa['prompt_system']},
            {"role": "user", "content": prompt_qa['prompt']},
        ],
        temperature=0,
    )
    time_list.append(time.time() - start_time)

    test_label.append(label)
    pred_label.append(int(response.choices[0].message.content.split("Answer: ")[1]))
    test_pred = str(label) + "," + str(pred_label[-1])
    save_response(response.choices[0].message.content, test_pred, filename)  # Save the response

print("true: ", test_label)
print("pred: ", pred_label)
print(f"inference time: {np.sum(time_list):.4f}")
f1_micro, mae = cal_metrics(pred_label, test_label)
print(f"f1_micro: {f1_micro}, mae: {mae}")
