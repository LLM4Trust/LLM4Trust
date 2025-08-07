from argparse import ArgumentParser
from prompt_generator import DyGraphPrompt
from api import send_prompt
from utlis import save_file
import utlis
import time

parser = ArgumentParser()
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--model", type=str, default='claude-3-7-sonnet-20250219')
# gpt-3.5-turbo, gpt-4o, claude-3-7-sonnet-20250219, deepseek-chat, qwen-max, meta-llama/llama-4-maverick, meta-llama/llama-4-scout
parser.add_argument("--task", type=str, default='dynamic', help="asymmetric or propagative or composable or dynamic or context")
parser.add_argument("--knowledge", type=int, default=1)
parser.add_argument("--example", type=int, default=2, help="0 for zero-shot; 1 for one-shot; 2 for few-shot")
parser.add_argument("--cot", type=int, default=0)
parser.add_argument("--role", type=int, default=1)

args = parser.parse_args()
start_time = time.time()
flag = 0
for run in range(args.run):
    print("-----------------run {}-----------------".format(run))

    for i in range(args.epoch):

        filename = f"graph_{i}.json"

        if args.task == "dynamic":
            edges, special_nodes, t = utlis.load_graph_from_json(args,filename)
            prompt = DyGraphPrompt(args)
            prompt_qa = prompt.generate_dynamic_qa(edges, special_nodes, t)
            save_file(args, prompt_qa)
            output_answer = send_prompt(args, prompt_qa)

        elif args.task == "asymmetric":
            edges, special_nodes, weight = utlis.load_graph_from_json(args,filename)
            prompt = DyGraphPrompt(args)
            prompt_qa = prompt.generate_asymmetric_qa(edges, special_nodes, weight)
            save_file(args, prompt_qa)
            output_answer = send_prompt(args, prompt_qa)
            if not output_answer.isdigit():  # Asymmetry: Outputting "unknown" or "not determined" is also acceptable
                output_answer = '0'

        elif args.task == "propagative":
            edges, special_nodes, min_weight = utlis.load_graph_from_json(args,filename)
            prompt = DyGraphPrompt(args)
            prompt_qa = prompt.generate_propagative_qa(edges, special_nodes, min_weight)
            save_file(args, prompt_qa)
            output_answer = send_prompt(args, prompt_qa)

        elif args.task == "composable":
            edges, special_nodes, num_min, num_max = utlis.load_graph_from_json(args,filename)
            prompt = DyGraphPrompt(args)
            prompt_qa = prompt.generate_composable_qa(edges, special_nodes, num_min, num_max)
            save_file(args, prompt_qa)
            output_answer = send_prompt(args, prompt_qa)

        elif args.task == "context":
            edges, special_nodes, c, w = utlis.load_graph_from_json(args,filename)
            prompt = DyGraphPrompt(args)
            prompt_qa = prompt.generate_context_qa(edges, special_nodes, c, w)
            save_file(args, prompt_qa)
            output_answer = send_prompt(args, prompt_qa)
            if not output_answer.isdigit():
                output_answer = '0'

        if output_answer == prompt_qa["answer"]:
            flag += 1
        print("epoch:", i + 1, "output:", output_answer, "truth:", prompt_qa["answer"], output_answer == prompt_qa["answer"])
    print("{}/{}".format(flag, args.epoch), flag / args.epoch)


print("{}/{}".format(flag, args.run * args.epoch), flag/(args.run * args.epoch))

end_time = time.time()
execution_time = end_time - start_time
print(f"Running time: {execution_time} seconds")