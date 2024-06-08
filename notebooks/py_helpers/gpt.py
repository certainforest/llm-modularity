import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

async def get_prompts(
	prompts: list[list],
	params: dict = {'model': 'gpt-3.5-turbo'},
	batch_size: int = 3,
	max_retries: int = 3, 
	api_key: str = os.environ.get('OPENAI_API_KEY')
	):
    """
    Description:
    	Send async request with retries to LLM endpoints.
    
    Params:
        @prompts: A lists of prompts, where each prompt is a list of messages to send in the reequest.
        @params: Anything other than the messages to pass into the request body, such as model or temperature. 
         See https://platform.openai.com/docs/api-reference/chat/create.
        @batch_size: Max number of prompts to group in a single batch. Prompts in a batch are sent concurrently.
        @max_retries: Max number of retries on failed prompt calls.
        @api_key: The OpenAI API key.
        
    Example:
        prompts_list = [
            [{'role': 'system', 'content': 'Answer all questions with a single number.'}, {'role': 'user', 'content': '1+1?'}],
            [{'role': 'system', 'content': 'Answer all questions with a single number.'}, {'role': 'user', 'content': '1+2?'}],
            [{'role': 'system', 'content': 'Answer all questions with a single number.'}, {'role': 'user', 'content': '1+3?'}],
            [{'role': 'system', 'content': 'Answer all questions with a single number.'}, {'role': 'user', 'content': '1+4?'}]
        ]
        # Send two requests at once
        results = await get_prompts(prompts_list, {'model': 'gpt-3.5-turbo', 'temperature': 1.0}, batch_size = 2)
        print(results)
    """

    async def make_request(session, prompt: list):
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {"Authorization": "Bearer " + api_key}
        async with session.post(url, headers = headers, json = {**{'messages': prompt}, **params}) as response:
            return await response.json()
            
    async def retry_requests(req_prompts, total_retries = 0):
        if total_retries > max_retries:
            raise Exception('Requests failed')
            
        if total_retries > 0:
            print(f'Retry {total_retries} for {len(req_prompts)} failed requests')
            await asyncio.sleep(2 * 2 ** total_retries) # Backoff rate 
    
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(*(make_request(session, prompt) for prompt in req_prompts), return_exceptions=True)
        
        successful_responses = [result for result in results if not isinstance(result, Exception)]
        failed_requests = [request for request, result in zip(req_prompts, results) if isinstance(result, Exception)]
    
        if failed_requests:
            print([result for result in results if isinstance(result, Exception)])
            retry_responses = await retry_requests(failed_requests, total_retries + 1)
            successful_responses.extend(retry_responses)
    
        return successful_responses

    # Split into batches
    chunks = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    # For each chunk, send requests and retry any failed elements
    responses = [await retry_requests(chunk) for chunk in tqdm(chunks)]

    parsed_responses = [item for sublist in responses for item in sublist]  # Flatten the list

    if len(parsed_responses) != len(prompts):
        raise Exception('Error: response length not the same as input length!')

    return parsed_responses