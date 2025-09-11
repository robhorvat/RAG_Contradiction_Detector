import requests
import xml.etree.ElementTree as ET
import json
from dotenv import load_dotenv

load_dotenv()


def get_paper_details(pubmed_id: str) -> dict:
    """
    Fetches details for a given PubMed ID.

    This function is designed to be resilient to PubMed's varied XML schemas by
    checking for both standard journal articles (<PubmedArticle>) and book
    chapters (<PubmedBookArticle>). It also gracefully handles entries that
    may be missing an abstract.

    Args:
        pubmed_id: The unique identifier for a PubMed article.

    Returns:
        A dictionary containing paper details, or an error message if fetching fails.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pubmed_id, "retmode": "xml", "rettype": "abstract"}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Accommodate different publication types in the PubMed XML schema.
        article = root.find(".//PubmedArticle")
        if article is None:
            article = root.find(".//PubmedBookArticle")

        if article is None:
            return {"error": f"No valid PubmedArticle or PubmedBookArticle found for ID {pubmed_id}."}

        title_element = article.find(".//ArticleTitle")
        title = title_element.text if title_element is not None else "No Title Found"

        # Abstracts can be split into multiple tags, so we join them.
        # If no abstract tag exists, return an empty string.
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
        return {"error": f"An HTTP error occurred: {e}"}
    except ET.ParseError as e:
        return {"error": f"Failed to parse XML response: {e}"}


# This block is for direct script testing and demonstration.
if __name__ == '__main__':
    # Example usage with a known good ID.
    test_id = "29276945"
    details = get_paper_details(test_id)

    # Using json.dumps for pretty-printing the output dictionary.
    print(json.dumps(details, indent=2))
