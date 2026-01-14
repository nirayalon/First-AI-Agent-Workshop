from datetime import datetime

from langchain_core.tools import Tool, tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_the_web",
    func=search.run,
    description="Useful for when you need to look up current information on the web. "
                "Input should be a search query.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def save_to_text_file(content: str, filename: str = None) -> str:
    """
    Saves content to a text file and returns the filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_content = f"--- Research Output --- \n --- Timestamp: {timestamp} ---\n\n{content}"
    if not filename:
        filename = f"research_output_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    return f"Data successfuly saved to {filename}"
