# libraries
from __future__ import annotations

import logging
from typing import Any
import tiktoken
import asyncio

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from gpt_researcher.llm_provider.generic.base import NO_SUPPORT_TEMPERATURE_MODELS, SUPPORT_REASONING_EFFORT_MODELS, ReasoningEfforts

from ..prompts import PromptFamily
from .costs import estimate_llm_cost, precise_llm_cost
from .validators import Subtopics
from .token_tracker import TokenTracker
import os


def get_llm(llm_provider, **kwargs):
    from gpt_researcher.llm_provider import GenericLLMProvider
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def create_chat_completion(
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = 0.4,
        max_tokens: int | None = 4000,
        llm_provider: str | None = None,
        stream: bool = False,
        websocket: Any | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cost_callback: callable = None,
        reasoning_effort: str | None = ReasoningEfforts.Medium.value,
        timeout: float | None = None,
        **kwargs
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion.
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.4.
        max_tokens (int, optional): The max tokens to use. Defaults to 4000.
        llm_provider (str, optional): The LLM Provider to use.
        stream (bool): Whether to stream the response. Defaults to False.
        webocket (WebSocket): The websocket used in the currect request,
        llm_kwargs (dict[str, Any], optional): Additional LLM keyword arguments. Defaults to None.
        cost_callback: Callback function for updating cost.
        reasoning_effort (str, optional): Reasoning effort for OpenAI's reasoning models. Defaults to 'low'.
        timeout (float, optional): Timeout in seconds for the LLM call. Defaults to None (no timeout).
        **kwargs: Additional keyword arguments.
    Returns:
        str: The response from the chat completion.
    """
    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 32001:
        raise ValueError(
            f"Max tokens cannot be more than 16,000, but got {max_tokens}")

    # Get the provider from supported providers
    provider_kwargs = {'model': model}

    if llm_kwargs:
        provider_kwargs.update(llm_kwargs)

    if model in SUPPORT_REASONING_EFFORT_MODELS:
        provider_kwargs['reasoning_effort'] = reasoning_effort

    if model not in NO_SUPPORT_TEMPERATURE_MODELS:
        # NOTE: set temperature to 0.0 for reproducibility
        # provider_kwargs['temperature'] = temperature
        provider_kwargs['temperature'] = 0.0
        provider_kwargs['max_tokens'] = max_tokens
    else:
        provider_kwargs['temperature'] = None
        provider_kwargs['max_tokens'] = None

    if llm_provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            provider_kwargs['openai_api_base'] = base_url

    elif llm_provider == "azure_openai":
        # Load Azure OpenAI configuration from environment or config file
        import json
        try:
            with open('./config/azure.key', 'r') as f:
                config = json.load(f)
                azure_config = config.get('AZURE_OPENAI_CONFIG', {})
        except (FileNotFoundError, json.JSONDecodeError):
            azure_config = {}

        # Find the appropriate configuration for the model
        config_found = False
        for config_group in azure_config.values():
            if model in config_group.get('models', []):
                provider_kwargs['azure_endpoint'] = config_group['endpoint']
                provider_kwargs['api_version'] = config_group['api_version']
                provider_kwargs['api_key'] = config_group['api_key']
                config_found = True
                break

        if not config_found:
            raise ValueError(f"Unsupported Azure OpenAI model: {model}. Please check your config_local.json")

    provider = get_llm(llm_provider, **provider_kwargs)
    response = ""
    
    # create response
    for attempt in range(10):  # maximum of 10 attempts
        try:
            # Apply timeout to the LLM call
            response = await asyncio.wait_for(
                provider.get_chat_response(messages, stream, websocket, **kwargs),
                timeout=timeout
            )
            # If we got a response, break out of retry loop
            if response:
                break
        except asyncio.TimeoutError:
            logging.error(f"LLM call timed out after {timeout}s (attempt {attempt + 1}) for model {model}")
            continue
        except Exception as e:
            logging.error(f"Error in LLM call (attempt {attempt + 1}): {e}")
            raise e

    # Capture the current loop to safely route callbacks from worker threads
    try:
        calling_loop = asyncio.get_running_loop()
    except RuntimeError:
        calling_loop = None

    async def _background_token_accounting(_messages, _response, _model, _cost_callback):
        def _compute():
            try:
                try:
                    _encoding = tiktoken.encoding_for_model(_model)
                except Exception:
                    _encoding = tiktoken.get_encoding("o200k_base")

                _input_text = str(_messages)
                _output_text = str(_response)
                _input_tokens = len(_encoding.encode(_input_text))
                _output_tokens = len(_encoding.encode(_output_text))

                # _llm_costs = estimate_llm_cost(str(_messages), _response)
                _llm_costs = precise_llm_cost(_model, _input_tokens, _output_tokens)

                TokenTracker.track_tokens(_model, _input_tokens, _output_tokens, _llm_costs)

                if _cost_callback:
                    # Run the callback on the original loop thread if possible
                    if calling_loop and calling_loop.is_running():
                        try:
                            calling_loop.call_soon_threadsafe(_cost_callback, _llm_costs)
                        except Exception:
                            _cost_callback(_llm_costs)
                    else:
                        _cost_callback(_llm_costs)
            except Exception as e:
                logging.getLogger(__name__).error(f"Background token accounting failed: {e}")

        # Offload CPU-bound tokenization to a thread to avoid blocking the event loop
        await asyncio.to_thread(_compute)

    # Schedule token accounting without awaiting it to avoid latency
    asyncio.create_task(_background_token_accounting(messages, response, model, cost_callback))

    return response

    logging.error(f"Failed to get response from {llm_provider} API")
    raise RuntimeError(f"Failed to get response from {llm_provider} API")


async def construct_subtopics(
    task: str,
    data: str,
    config,
    subtopics: list = [],
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> list:
    """
    Construct subtopics based on the given task and data.

    Args:
        task (str): The main task or topic.
        data (str): Additional data for context.
        config: Configuration settings.
        subtopics (list, optional): Existing subtopics. Defaults to [].
        prompt_family (PromptFamily): Family of prompts
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of constructed subtopics.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=prompt_family.generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        provider_kwargs = {'model': config.smart_llm_model}

        if config.llm_kwargs:
            provider_kwargs.update(config.llm_kwargs)

        if config.smart_llm_model in SUPPORT_REASONING_EFFORT_MODELS:
            provider_kwargs['reasoning_effort'] = ReasoningEfforts.High.value
        else:
            provider_kwargs['temperature'] = config.temperature
            provider_kwargs['max_tokens'] = config.smart_token_limit

        provider = get_llm(config.smart_llm_provider, **provider_kwargs)

        model = provider.llm

        chain = prompt | model | parser

        output = await chain.ainvoke({
            "task": task,
            "data": data,
            "subtopics": subtopics,
            "max_subtopics": config.max_subtopics
        }, **kwargs)

        return output

    except Exception as e:
        print("Exception in parsing subtopics : ", e)
        logging.getLogger(__name__).error("Exception in parsing subtopics : \n {e}")
        return subtopics
