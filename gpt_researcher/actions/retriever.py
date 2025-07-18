def get_retriever(retriever: str):
    """
    Gets the retriever
    Args:
        retriever (str): retriever name

    Returns:
        retriever: Retriever class

    """
    match retriever:
        case "google":
            from gpt_researcher.retrievers import GoogleSearch

            return GoogleSearch
        case "searx":
            from gpt_researcher.retrievers import SearxSearch

            return SearxSearch
        case "searchapi":
            from gpt_researcher.retrievers import SearchApiSearch

            return SearchApiSearch
        case "serpapi":
            from gpt_researcher.retrievers import SerpApiSearch

            return SerpApiSearch
        case "serper":
            from gpt_researcher.retrievers import SerperSearch

            return SerperSearch
        case "duckduckgo":
            from gpt_researcher.retrievers import Duckduckgo

            return Duckduckgo
        case "bing":
            from gpt_researcher.retrievers import BingSearch

            return BingSearch
        case "arxiv":
            from gpt_researcher.retrievers import ArxivSearch

            return ArxivSearch
        case "tavily":
            from gpt_researcher.retrievers import TavilySearch

            return TavilySearch
        case "exa":
            from gpt_researcher.retrievers import ExaSearch

            return ExaSearch
        case "semantic_scholar":
            from gpt_researcher.retrievers import SemanticScholarSearch

            return SemanticScholarSearch
        case "pubmed_central":
            from gpt_researcher.retrievers import PubMedCentralSearch

            return PubMedCentralSearch
        case "custom":
            from gpt_researcher.retrievers import CustomRetriever

            return CustomRetriever
        case "mcp":
            from gpt_researcher.retrievers import MCPRetriever

            return MCPRetriever
        case "brave":
            from gpt_researcher.retrievers import BraveSearch

            return BraveSearch
        case _:
            raise ValueError(f"Unknown retriever: {retriever}. Please check the retriever name or add it to the retrievers module.")


def get_retrievers(headers: dict[str, str], cfg):
    """
    Determine which retriever(s) to use based on headers, config, or default.

    Args:
        headers (dict): The headers dictionary
        cfg: The configuration object

    Returns:
        list: A list of retriever classes to be used for searching.
    """
    # Check headers first for multiple retrievers
    if headers.get("retrievers"):
        retrievers = headers.get("retrievers").split(",")
    # If not found, check headers for a single retriever
    elif headers.get("retriever"):
        retrievers = [headers.get("retriever")]
    # If not in headers, check config for multiple retrievers
    elif cfg.retrievers:
        # Handle both list and string formats for config retrievers
        if isinstance(cfg.retrievers, str):
            retrievers = cfg.retrievers.split(",")
        else:
            retrievers = cfg.retrievers
        # Strip whitespace from each retriever name
        retrievers = [r.strip() for r in retrievers]
    # If not found, check config for a single retriever
    elif cfg.retriever:
        retrievers = [cfg.retriever]
    # If still not set, use default retriever
    else:
        retrievers = [get_default_retriever().__name__]

    # Convert retriever names to actual retriever classes
    # Use get_default_retriever() as a fallback for any invalid retriever names
    retriever_classes = [get_retriever(r) or get_default_retriever() for r in retrievers]
    
    return retriever_classes


def get_default_retriever():
    from gpt_researcher.retrievers import TavilySearch

    return TavilySearch