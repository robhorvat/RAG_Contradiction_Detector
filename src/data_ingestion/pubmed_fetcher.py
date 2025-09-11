import requests
import xml.etree.ElementTree as ET
import json
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times on failure
    wait=wait_exponential(multiplier=1, min=2, max=10)  # Wait 2s, then 4s, etc.
)
def get_paper_details(pubmed_id: str) -> dict:
    """
    Fetches details for a given PubMed ID with exponential backoff retries.

    This function is designed to be resilient to transient network issues by
    automatically retrying failed requests. It also handles PubMed's varied
    XML schemas and gracefully manages entries that may be missing an abstract.

    Args:
        pubmed_id: The unique identifier for a PubMed article.

    Returns:
        A dictionary containing paper details, or an error message if fetching fails.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pubmed_id, "retmode": "xml", "rettype": "abstract"}

    try:
        # The print statement is moved inside the try block to avoid printing on every retry.
        print(f"Fetching data for PubMed ID: {pubmed_id}...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # This will raise an HTTPError for bad responses

        root = ET.fromstring(response.content)

        article = root.find(".//PubmedArticle")
        if article is None:
            article = root.find(".//PubmedBookArticle")

        if article is None:
            return {"error": f"No valid PubmedArticle or PubmedBookArticle found for ID {pubmed_id}."}

        title_element = article.find(".//ArticleTitle")
        title = title_element.text if title_element is not None else "No Title Found"

        abstract_elements = article.findall(".//AbstractText")
        abstract = "\n".join(
            [elem.text.strip() for elem in abstract_elements if elem.text]) if abstract_elements else ""

        author_list = article.findall(".//Author")
        authors = [
            f"{author.find('.//LastName').text}, {author.find('.//ForeName').text}"
            for author in author_list if
            author.find('.//LastName') is not None and author.find('.//ForeName') is not None
        ]

        return {
            "pubmed_id": pubmed_id,
            "title": title.strip(),
            "abstract": abstract.strip(),
            "authors": authors,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
        }

    except requests.exceptions.RequestException as e:
        # We print the error and then re-raise it. Tenacity's @retry decorator
        # will catch this exception and trigger a retry.
        print(f"HTTP request failed for {pubmed_id}: {e}. Retrying...")
        raise
    except ET.ParseError as e:
        # We don't retry on parsing errors, as they are likely permanent.
        return {"error": f"Failed to parse XML response: {e}"}


# This block is for direct script testing and demonstration.
if __name__ == '__main__':
    test_id = "29276945"
    details = get_paper_details(test_id)
    print(json.dumps(details, indent=2))