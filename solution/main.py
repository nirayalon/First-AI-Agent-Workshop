from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel
from pydantic import Field

from solution.tools import search_tool, wikipedia_tool, save_to_text_file

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str = Field(description="The research topic.")
    summary: str = Field(description="A brief summary of the research findings.")
    sources: list[str] = Field(description="A list of sources URLs used for the research.")
    tool_used: list[str] = Field(description="A list of tools used during the research.")


llm = ChatOpenAI(model="gpt-5.2")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a research assistant that will help generate a research paper answer the user query and use the necessary tools. 
         Wrap the output in the following format and provide no other text:\n{format_instructions}
         provide a link to the source of the information in the sources field.
         """,
         ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Print the format instructions
# print(parser.get_format_instructions())

tools = [
    search_tool,
    wikipedia_tool,
    save_to_text_file,
]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# get query from the user input
query = input("Enter your research query: ")

raw_response = agent_executor.invoke(
    {
        "query": query,
    }
)

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(f"Topic is: {structured_response.topic}")
    print(f"Summary is: {structured_response.summary}")
    print(f"Sources are: {structured_response.sources}")
    print(f"tool_used are: {structured_response.tool_used}")
except Exception as e:
    print(f"Error parsing response: {e} Raw response is {raw_response}")
