import tiktoken

# Per OpenAI Pricing Page: https://openai.com/api/pricing/
ENCODING_MODEL = "o200k_base"
INPUT_COST_PER_TOKEN = 0.000005
OUTPUT_COST_PER_TOKEN = 0.000015
IMAGE_INFERENCE_COST = 0.003825
EMBEDDING_COST = 0.02 / 1000000 # Assumes new ada-3-small


# Cost estimation is via OpenAI libraries and models. May vary for other models
def estimate_llm_cost(input_content: str, output_content: str) -> float:
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    input_tokens = encoding.encode(input_content)
    output_tokens = encoding.encode(output_content)
    input_costs = len(input_tokens) * INPUT_COST_PER_TOKEN
    output_costs = len(output_tokens) * OUTPUT_COST_PER_TOKEN
    return input_costs + output_costs


def estimate_embedding_cost(model, docs):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(str(doc))) for doc in docs)
    return total_tokens * EMBEDDING_COST


MODEL_PRICING = {
    # GPT-5 family
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    # GPT-4.1 family
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    # GPT-4o variants
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},  # no cached pricing listed
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    # Realtime / audio
    "gpt-realtime": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
    "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
    "gpt-audio": {"input": 2.50, "output": 10.00},
    "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
    # Reasoning / o-series
    "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-pro": {"input": 150.00, "output": 600.00},  # no cached input price specified
    "o3-pro": {"input": 20.00, "output": 80.00},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-deep-research": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    "o4-mini-deep-research": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
}

def precise_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")

    input_costs = input_tokens / 1000000 * MODEL_PRICING[model]["input"]
    output_costs = output_tokens / 1000000 * MODEL_PRICING[model]["output"]
    return input_costs + output_costs