import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from openai_inference.modules.mylogger import init_logging

T = TypeVar("T")


# Function to retry request on error
async def retry_on_error(
    openai_call: Callable[[], Awaitable[T]],
    max_num_trials: int = 5,
    first_wait_time: int = 10,
    log_path="openai_log/latest.log",
) -> T:
    """Retry if an error is returned when using OpenAI API"""
    retry_on_error_logger = init_logging(__name__, log_path=log_path)
    for i in range(max_num_trials):
        try:
            # Run function
            return await openai_call()
        except (APIConnectionError, APIStatusError) as e:
            if i == max_num_trials - 1:
                raise RuntimeError("Maximum number of retries reached")
            retry_on_error_logger.info("Error received: %s", e)
            wait_time_seconds = first_wait_time * (2**i)
            retry_on_error_logger.info("Wait for %s seconds", wait_time_seconds)
            await asyncio.sleep(wait_time_seconds)
    retry_on_error_logger.error("Maximum number of retries reached")

    raise RuntimeError("Maximum number of retries reached")


# Function to send multiple requests in parallel with error handling
async def _async_batch_run_chatgpt(
    messages_list: list[list[ChatCompletionMessageParam]],
    model_name: str,
    temperature: float,
    max_tokens: int | None,
    logprobs: bool,
    top_logprobs: int | None,
    seed: int,
    stop: str | list[str] | None = None,
    log_path="openai_log/latest.log",
) -> list[ChatCompletion]:
    """Send requests in parallel to the OpenAI API"""
    client = AsyncOpenAI()

    def create_completion_function(
        ms: list[ChatCompletionMessageParam],
    ) -> Callable[[], Awaitable[Any]]:
        async def create_completion(
            ms: list[ChatCompletionMessageParam] = ms,
        ) -> Any:
            return await client.chat.completions.create(
                model=model_name,
                messages=ms,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                seed=seed,
                stop=stop,
            )

        return create_completion

    # Store coroutine objects in tasks
    tasks: list[Coroutine[Any, Any, Any]] = [
        retry_on_error(create_completion_function(ms), log_path=log_path)
        for ms in messages_list
    ]

    # Execute asynchronous processing in tasks and collect results
    responses = await asyncio.gather(*tasks)

    await client.close()
    return responses


# Coroutine functions can be executed through the `run` function in `asyncio`
# Wrapper function to execute asynchronous processing functions
def batch_run_chatgpt(
    messages_list: list[list[ChatCompletionMessageParam]],
    model_name: str,
    logprobs: bool,
    top_logprobs: int | None,
    temperature: float = 0,
    max_tokens: int | None = None,
    seed: int = 42,
    stop: str | list[str] | None = None,
    log_path="openai_log/latest.log",
) -> list[ChatCompletion]:
    """Wrapper for executing asynchronous processing functions"""
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(
        _async_batch_run_chatgpt(
            messages_list,
            model_name,
            temperature,
            max_tokens,
            logprobs,
            top_logprobs,
            seed,
            stop,
            log_path=log_path,
        )
    )


def create_messsage(
    system_message: str, prompt: str
) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionUserMessageParam(role="system", content=system_message),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]


def main(
    messages: list[list[ChatCompletionMessageParam]],
    model_name: str,
    logprobs=True,
    top_logprobs=20,
    temperature=0.0,
    max_tokens=1,
    seed=42,
    log_path="openai_log/latest.log",
) -> list[ChatCompletion]:
    logger = init_logging(__name__, log_path=log_path)

    logger.info("Start batch call to OpenAI API")
    answers = batch_run_chatgpt(
        messages,
        model_name=model_name,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        log_path=log_path,
    )
    logger.info("Finished batch call to OpenAI API")

    return answers
