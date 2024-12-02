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
	api_key: str = os.environ.get('OPENAI_API_KEY'),
    verbose = True
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

    # Make request, returns tuple of the index and the await object
    async def make_request(session, prompt: list, index: int):
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {"Authorization": "Bearer " + api_key}
        async with session.post(url, headers=headers, json={**{'messages': prompt}, **params}) as response:
            return index, await response.json()
            
    async def retry_requests(req_prompts_with_indices, total_retries = 0):
        if total_retries > max_retries:
            raise Exception('Requests failed')
            
        if total_retries > 0:
            print(f'Retry {total_retries} for {len(req_prompts_with_indices)} failed requests')
            await asyncio.sleep(2 * 2 ** total_retries) # Backoff rate 
    
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *(make_request(session, prompt, index) for prompt, index in req_prompts_with_indices),
                return_exceptions=True
            )
        
        successful_responses = [result for result in results if not isinstance(result, Exception)]
        failed_requests = [(req_prompts_with_indices[i], result) for i, result in enumerate(results) if isinstance(result, Exception)]
    
        if failed_requests:
            print([result for result in results if isinstance(result, Exception)])
            retry_responses = await retry_requests([request for request, _ in failed_requests], total_retries + 1)
            successful_responses.extend(retry_responses)
    
        return successful_responses

    ### Guarantee the original order
    # Attach original indices to each prompt
    indexed_prompts = [(prompt, i) for i, prompt in enumerate(prompts)]
    # Split into batches
    chunks = [indexed_prompts[i:i + batch_size] for i in range(0, len(indexed_prompts), batch_size)]
    # For each chunk, send requests and retry any failed elements
    responses = [await retry_requests(chunk) for chunk in tqdm(chunks, disable = not verbose)]

    # Flatten the list and sort by the original indices
    parsed_responses = [item for sublist in responses for item in sublist]
    parsed_responses.sort(key = lambda x: x[0])  # Sort by the original index

    if len(parsed_responses) != len(prompts):
        raise Exception('Error: response length not the same as input length!')

    return [response for _, response in parsed_responses]


async def get_prompts_claude(
	prompts: list[list],
	params: dict = {'model': 'claude-3-5-sonnet-20240620'},
	batch_size: int = 3,
	max_retries: int = 3, 
    api_key: str = os.environ.get('CLAUDE_API_KEY'),
    verbose = True
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
            [{'role': 'user', 'content': '1+1?'}],
            [{'role': 'user', 'content': '1+2?'}],
            [{'role': 'user', 'content': '1+3?'}],
            [{'role': 'user', 'content': '1+4?'}]
        ]
        # Send two requests at once
        results = await get_prompts_claude(
            prompts_list, 
            {'model': 'claude-3-5-sonnet-20240620', 'max_tokens': 1024, 'temperature': 0.8, 'system': 'Answer all questions with a single number.'},
            batch_size = 2
        )
        print(results)
    """

    # Make request, returns tuple of the index and the await object
    async def make_request(session, prompt: list, index: int):
        url = 'https://api.anthropic.com/v1/messages'
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        async with session.post(url, headers=headers, json={**{'messages': prompt}, **params}) as response:
            return index, await response.json()
            
    async def retry_requests(req_prompts_with_indices, total_retries = 0):
        if total_retries > max_retries:
            raise Exception('Requests failed')
            
        if total_retries > 0:
            print(f'Retry {total_retries} for {len(req_prompts_with_indices)} failed requests')
            await asyncio.sleep(2 * 2 ** total_retries) # Backoff rate 
    
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *(make_request(session, prompt, index) for prompt, index in req_prompts_with_indices),
                return_exceptions=True
            )
        
        successful_responses = [result for result in results if not isinstance(result, Exception)]
        failed_requests = [(req_prompts_with_indices[i], result) for i, result in enumerate(results) if isinstance(result, Exception)]
    
        if failed_requests:
            print([result for result in results if isinstance(result, Exception)])
            retry_responses = await retry_requests([request for request, _ in failed_requests], total_retries + 1)
            successful_responses.extend(retry_responses)
    
        return successful_responses

    ### Guarantee the original order
    # Attach original indices to each prompt
    indexed_prompts = [(prompt, i) for i, prompt in enumerate(prompts)]
    # Split into batches
    chunks = [indexed_prompts[i:i + batch_size] for i in range(0, len(indexed_prompts), batch_size)]
    # For each chunk, send requests and retry any failed elements
    responses = [await retry_requests(chunk) for chunk in tqdm(chunks, disable = not verbose)]

    # Flatten the list and sort by the original indices
    parsed_responses = [item for sublist in responses for item in sublist]
    parsed_responses.sort(key = lambda x: x[0])  # Sort by the original index

    if len(parsed_responses) != len(prompts):
        raise Exception('Error: response length not the same as input length!')

    return [response for _, response in parsed_responses]


async def get_prompts_deepseek(
	prompts: list[list],
	params: dict = {'model': 'deepseek-chat'},
	batch_size: int = 3,
	max_retries: int = 3, 
    api_key: str = os.environ.get('DEEPSEEK_API_KEY'),
    verbose = True
	):
    """
    Description:
    	Send async request with retries to LLM endpoints.
    
    Params:
        @prompts: A lists of prompts, where each prompt is a list of messages to send in the reequest.
        @params: Anything other than the messages to pass into the request body, such as model or temperature. 
         See https://api-docs.deepseek.com/api/create-chat-completion.
        @batch_size: Max number of prompts to group in a single batch. Prompts in a batch are sent concurrently.
        @max_retries: Max number of retries on failed prompt calls.
        @api_key: The OpenAI API key.
        
    Example:
        prompts_list = [
            [{'role': 'user', 'content': '1+1?'}],
            [{'role': 'user', 'content': '1+2?'}],
            [{'role': 'user', 'content': '1+3?'}],
            [{'role': 'user', 'content': '1+4?'}]
        ]
        # Send two requests at once
        results = await get_prompts_deepseek(
            prompts_list, 
            {'model': 'deepseek-chat', 'max_tokens': 1024, 'temperature': 1.3},
            batch_size = 2
        )
        print(results)
    """

    # Make request, returns tuple of the index and the await object
    async def make_request(session, prompt: list, index: int):
        url = 'https://api.deepseek.com/chat/completions'
        headers = {"Authorization": "Bearer " + api_key, 'Content-Type': 'application/json', 'Accept': 'application/json'}
        async with session.post(url, headers=headers, json={**{'messages': prompt}, **params}) as response:
            return index, await response.json()
            
    async def retry_requests(req_prompts_with_indices, total_retries = 0):
        if total_retries > max_retries:
            raise Exception('Requests failed')
            
        if total_retries > 0:
            print(f'Retry {total_retries} for {len(req_prompts_with_indices)} failed requests')
            await asyncio.sleep(2 * 2 ** total_retries) # Backoff rate 
    
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *(make_request(session, prompt, index) for prompt, index in req_prompts_with_indices),
                return_exceptions=True
            )
        
        successful_responses = [result for result in results if not isinstance(result, Exception)]
        failed_requests = [(req_prompts_with_indices[i], result) for i, result in enumerate(results) if isinstance(result, Exception)]
    
        if failed_requests:
            print([result for result in results if isinstance(result, Exception)])
            retry_responses = await retry_requests([request for request, _ in failed_requests], total_retries + 1)
            successful_responses.extend(retry_responses)
    
        return successful_responses

    ### Guarantee the original order
    # Attach original indices to each prompt
    indexed_prompts = [(prompt, i) for i, prompt in enumerate(prompts)]
    # Split into batches
    chunks = [indexed_prompts[i:i + batch_size] for i in range(0, len(indexed_prompts), batch_size)]
    # For each chunk, send requests and retry any failed elements
    responses = [await retry_requests(chunk) for chunk in tqdm(chunks, disable = not verbose)]

    # Flatten the list and sort by the original indices
    parsed_responses = [item for sublist in responses for item in sublist]
    parsed_responses.sort(key = lambda x: x[0])  # Sort by the original index

    if len(parsed_responses) != len(prompts):
        raise Exception('Error: response length not the same as input length!')

    return [response for _, response in parsed_responses]

