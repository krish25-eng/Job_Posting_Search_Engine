import sys
import os

from openai import OpenAI

import pickle
import asyncio
from tqdm.autonotebook import tqdm
import numpy as np

build_project_path = os.environ['BUILD_PROJECT_PATH']

data_path = os.path.join(build_project_path, 'synthetic_data', 'data')

with open(os.path.join(data_path, 'initial_prompt.txt'), 'r') as f:
    base_prompt = f.read()

example_query_titles = []
with open(os.path.join(data_path, 'example_query_titles.txt'), 'r') as f:
    for line in f:
        example_query_titles.append(line.strip())

with open(os.path.join(data_path, 'example_assistant_response.txt'), 'r') as f:
    example_assistant_response = f.read()

with open(os.path.join(data_path, 'follow_up_prompt.txt'), 'r') as f:
    follow_up_prompt = f.read()

rng = np.random.default_rng()

def get_client():
    return OpenAI(
        api_key=os.environ["SAMBANOVA_API_KEY"],
        base_url="https://api.sambanova.ai/v1",
    )

def format_query_title_list(query_job_titles):
    output_string = ''
    for i, title in enumerate(query_job_titles):
        output_string += f'{i+1}. `{title}`\n'
    return output_string
    
def generate_prompt(query_job_titles : list[str] | np.ndarray, num_examples_per_title=5):
    return [
        {"role": "system", "content": "You are an expert in recruitment, staffing and HR."},
        {"role": "user", "content": f"{base_prompt.format(format_query_title_list(example_query_titles))}"},
        {"role": "assistant", "content": example_assistant_response},
        {"role": "user", "content": f"{follow_up_prompt.format(num_examples_per_title, format_query_title_list(query_job_titles))}"},
    ]

async def async_make_api_call(client, model_name, messages, perturbation_std=0.0):
    # Adding perturbation to the temperature to avoid cached responses
    response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stop=["<query>"],
            temperature=0.7 + rng.normal(0, perturbation_std)
        )
    return response


# async def async_main_stubborn(query_df, client, model_name, output_path=None, multi_level=False, delay=2, giveup_after=10):

#     responses_dict = {}
#     for row in tqdm(query_df.itertuples(), total=len(query_df)):
#         attempts = 0
#         while attempts < giveup_after:
#             if attempts > 0:
#                 print(f'Attempt {attempts} for job ID {row.ID}')
#             query_job_title = row.TITLE_RAW
#             query_job_description = row.summarised_jd
#             if multi_level:
#                 query_job_education_list= get_education_range(row.MIN_EDULEVELS_NAME, row.MAX_EDULEVELS_NAME)
#                 query_job_education = convert_to_gpt_string(query_job_education_list)
#             else:
#                 query_job_education = row.MIN_EDULEVELS_NAME
#             api_task = asyncio.create_task(async_make_api_call(client, model_name, generate_relevance_messages(query_job_title, query_job_description, query_job_education, multi_level=multi_level),  perturbation_std=0.1))
#             await asyncio.sleep(delay)
#             response = await api_task
#             if multi_level:
#                 parsed_response = parse_gpt_field_lists_multi(relevence_categories, response.choices[0].message.content, len(query_job_education_list))
#             else:
#                 parsed_response = parse_gpt_field_lists(relevence_categories, response.choices[0].message.content)
#             if parsed_response:
#                 if multi_level:
#                     for i, education_level in enumerate(query_job_education_list):
#                         responses_dict[(row.ID, education_level)] = parsed_response[i]
#                 else:
#                     responses_dict[row.ID] = parsed_response
#                 break
#             elif attempts > 2:
#                 print('-------------------------------')
#                 print('Output:')
#                 print(response.choices[0].message.content)
#             attempts += 1

#         if output_path:
#             with open(output_path, 'wb') as f:
#                 pickle.dump(responses_dict, f)

#     return responses_dict