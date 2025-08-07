from openai import OpenAI
from utlis import save_file
import anthropic
import json
import time
import logging

logging.basicConfig(level=logging.INFO)

def send_prompt(args, prompt_qa):
    output_file = args.output_file
    client = OpenAI(
        api_key="<YOUR_API_KEY>",  # Your Anthropic API key
        base_url="https://api.anthropic.com/v1/"
    )

    for attempt in range(5):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": prompt_qa['prompt_system']},
                    {"role": "user", "content": prompt_qa['prompt']},
                ],
                temperature=0,
            )
            time_once = time.time() - start_time

            response_content = response.choices[0].message.content
            save_file(output_file, response_content)

            try:
                output_answer = response_content.split("Answer: ")[1]
            except IndexError:
                try:
                    output_answer = response_content.split("**Answer:** ")[1]
                except IndexError:
                    output_answer = "-1"

            return output_answer, time_once

        except Exception as e:
            print("fail....")
