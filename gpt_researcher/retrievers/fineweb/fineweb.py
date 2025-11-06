import os
from typing import Any, Dict, List, Optional
import requests
import base64
import json


class FineWebSearch:
    """
    FineWeb Search API Wrapper
    """

    def __init__(self, query: str, query_domains=None):
        self.endpoint = "https://clueweb22.us/fineweb/search"
        self.query = query

    def search(self, max_results: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Performs the search using the custom retriever endpoint.

        :param max_results: Maximum number of results to return (not currently used)
        :return: JSON response in the format:
            [
              {
                "url": "http://example.com/page1",
                "raw_content": "Content of page 1"
              },
              {
                "url": "http://example.com/page2",
                "raw_content": "Content of page 2"
              }
            ]
        """
        try:
            headers = {
                "X-API-Key": os.getenv("FINEWEB_API_KEY", "")
            }
            response = requests.get(self.endpoint, headers=headers, params={'query': self.query, 'k': max_results})
            response.raise_for_status()
            json_data = response.json()

            results = json_data.get("results", [])
            search_response = []
            for returned_document in results:
                # Assuming each document in 'results' is a base64 encoded JSON string
                decoded_result = base64.b64decode(returned_document).decode("utf-8")
                parsed_result = json.loads(decoded_result)
                
                url = parsed_result["url"]
                text = parsed_result["text"]

                search_response.append(
                    {"href": url, "body": text}
                )
            return search_response
        except requests.RequestException as e:
            print(f"Failed to retrieve search results: {e}")
            return None