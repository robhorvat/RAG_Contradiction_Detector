import requests
import xml.etree.ElementTree as ET
import json
from dotenv import load_dotenv

load_dotenv()


def get_paper_details(pubmed_id: str) -> dict:
    """
    Fetches details of a paper from PubMed, handling journal articles, book chapters, and missing abstracts.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pubmed_id, "retmode": "xml", "rettype": "abstract"}

    try:
        print(f"Fetching data for PubMed ID: {pubmed_id}...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        article = root.find(".//PubmedArticle")
        if article is None:
            article = root.find(".//PubmedBookArticle")

        if article is None:
            # This happens for IDs that return an empty PubmedArticleSet
            return {"error": f"No valid PubmedArticle or PubmedBookArticle found for ID {pubmed_id}."}

        title_element = article.find(".//ArticleTitle")
        title = title_element.text if title_element is not None else "No title found"

        abstract_elements = article.findall(".//AbstractText")
        if abstract_elements:
            abstract = "\n".join([elem.text.strip() for elem in abstract_elements if elem.text is not None])
        else:
            abstract = ""  # Return an empty string if no abstract is found

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


# A clean, simple example for future reference.
if __name__ == '__main__':
    test_id = "29276945"  # A known good ID (the coffee paper)
    details = get_paper_details(test_id)
    print(json.dumps(details, indent=2))