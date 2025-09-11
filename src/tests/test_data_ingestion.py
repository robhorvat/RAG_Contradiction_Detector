import pytest

# With pytest.ini configured, we no longer need the sys.path hack.
# We can import directly from our source modules.
from data_ingestion.pubmed_fetcher import get_paper_details

def test_get_paper_details_success():
    """
    Tests the successful fetching and parsing of a known, valid PubMed ID.
    This is a "happy path" test.
    """
    # GIVEN a known valid PubMed ID
    pubmed_id = "29276945"

    # WHEN we call the function
    result = get_paper_details(pubmed_id)

    # THEN the result should be a valid dictionary with expected content
    assert "error" not in result
    assert result["pubmed_id"] == pubmed_id
    assert "Coffee Consumption and Cardiovascular Disease" in result["title"]
    assert len(result["abstract"]) > 50

def test_get_paper_details_invalid_id():
    """
    Tests how the function handles a non-existent ID.
    """
    # GIVEN a PubMed ID that is syntactically valid but does not exist
    pubmed_id = "99999999"

    # WHEN we call the function
    result = get_paper_details(pubmed_id)

    # THEN the function should return a specific error message
    assert "error" in result
    assert "No valid PubmedArticle or PubmedBookArticle found" in result["error"]

def test_get_paper_details_empty_response():
    """
    Tests the function's resilience against an ID that returns an empty XML set,
    which we discovered during our debugging journey.
    """
    # GIVEN a PubMed ID that is known to return an empty response
    pubmed_id = "31505598"

    # WHEN we call the function
    result = get_paper_details(pubmed_id)

    # THEN the function should correctly identify this and return an error
    assert "error" in result