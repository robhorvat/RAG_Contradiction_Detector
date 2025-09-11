import sys
import os

# This script is designed to be run from the command line to quickly check
# the abstracts of PubMed IDs before using them in the main application.

# Add the project's src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from data_ingestion.pubmed_fetcher import get_paper_details
except ImportError:
    print(
        "Error: Could not import the pubmed_fetcher. Make sure you run this script from the project's root directory.")
    sys.exit(1)


def verify(pmid: str):
    """Fetches and prints the title and abstract for a given PubMed ID."""
    print("-" * 50)
    print(f"Verifying PubMed ID: {pmid}")
    data = get_paper_details(pmid)
    if "error" in data:
        print(f"  Error: {data['error']}")
    else:
        print(f"  Title: {data.get('title', 'N/A')}")
        print(f"  Abstract: {data.get('abstract', 'N/A')[:250]}...")  # Print first 250 chars
    print("-" * 50)


if __name__ == "__main__":
    # This is an interactive script. It will ask you for the IDs to check.
    print("--- PubMed Paper Verification Tool ---")

    id1 = input("Enter the first PubMed ID to verify: ").strip()
    if id1:
        verify(id1)

    id2 = input("Enter the second PubMed ID to verify: ").strip()
    if id2:
        verify(id2)