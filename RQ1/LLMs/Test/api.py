from openai import OpenAI
from utlis import save_file


def send_prompt(args, prompt_qa):
    # LLM API key configuration
    if args.model == "claude-3-7-sonnet-20250219":
        client = OpenAI(
            api_key="YOUR_ANTHROPIC_API_KEY",
            base_url="https://api.anthropic.com/v1/",
        )
    elif args.model in ["gpt-3.5-turbo", "gpt-4o"]:
        client = OpenAI(
            api_key="YOUR_OPENAI_API_KEY"
        )
    elif args.model == "deepseek-chat":
        client = OpenAI(
            api_key="YOUR_DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com/v1",
        )
    elif args.model == "qwen-max":
        client = OpenAI(
            api_key="YOUR_ALICLOUD_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif args.model in ["meta-llama/llama-4-maverick", "meta-llama/llama-4-scout"]:
        client = OpenAI(
            api_key="YOUR_OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
        )

    for attempt in range(5):
        try:
            if not args.role:  # No system role
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt_qa['prompt']}],
                    temperature=0,
                )
            else:  # With system role
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": prompt_qa['prompt_system']},
                        {"role": "user", "content": prompt_qa['prompt']},
                    ],
                    temperature=0,
                )

            response_content = response.choices[0].message.content
            save_file(args, response_content)  # Save the output

            try:
                output_answer = response_content.split("Answer: ")[1]  # Extract useful output
            except IndexError:
                try:
                    output_answer = response_content.split("**Answer:** ")[1]  # Handle inconsistent format
                except IndexError:
                    output_answer = "-1"  # Default if no valid answer found

            return output_answer

        except Exception as e:
            print("Connection error...")
