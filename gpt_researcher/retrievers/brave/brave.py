# Tavily API Retriever

# libraries
import os
from typing import Literal, Sequence, Optional
import requests
import json


class BraveSearch:
    """
    Brave Search API Retriever
    """

    def __init__(self, query, headers=None, query_domains=None):
        """
        Initializes the BraveSearch object.

        Args:
            query (str): The search query string.
            headers (dict, optional): Additional headers to include in the request. Defaults to None.
            topic (str, optional): The topic for the search. Defaults to "general".
            query_domains (list, optional): List of domains to include in the search. Defaults to None.
        """
        self.query = query
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
           "X-Subscription-Token": self.get_api_key()
        }
        self.query_domains = query_domains or None

    def get_api_key(self):
        """
        Gets the Brave Search API key
        Returns:

        """
        try:
            api_key = os.environ["BRAVE_API_KEY"]
        except KeyError:
            print(
                "Brave Search API key not found, set to blank. If you need a retriver, please set the BRAVE_API_KEY environment variable."
            )
            return ""
        return api_key


    def _search(
        self,
        query: str,
        max_results: int = 10,
    ) -> dict:
        """
        Internal search method to send the request to the API.
        """

        data = {
            "q": query,
            "count": max_results,
            "result_filter": "web",
            "extra_snippets": True,
        }

        response = requests.get(
            self.base_url, headers=self.headers, params=data, timeout=100
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Raises a HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()

    def search(self, max_results=10):
        """
        Searches the query
        Returns:

        """
        try:
            # Search the query
            results = self._search(
                self.query,
                max_results=max_results
            )
            sources = results.get("web", {}).get("results", [])
            if not sources:
                raise Exception("No results found with Brave Search API search.")
            # Return the results
            search_response = [
                {"href": obj["url"], "body": "\n".join(obj.get("extra_snippets", []))} for obj in sources
            ]
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
