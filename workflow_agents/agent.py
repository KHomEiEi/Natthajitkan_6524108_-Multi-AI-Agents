import os
import logging
import google.cloud.logging

from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool  # import
from google.adk.models import Gemini
from google.genai import types


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()
model_name = os.getenv("MODEL")
print(model_name)

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)


# Tools
from google.adk.tools import exit_loop
from google.adk.models import Gemini


def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key.


    Args:
        field (str): a field name to append to
        response (str): a string to append to the field


    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}


def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(content)
    return {"status": "success"}




# Agents
the_Admirer = Agent(
    name="the_Admirer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Search and collect only success, achievements and positive",
    instruction="""
     PROMPT: 
{ PROMPT? }


    INSTRUCTIONS:
    Write a report that compiles only the success, achievements, or positive aspects of the person described in PLOT_OUTLINE.
    CRITICAL RULE:
    - You MUST format your response as a bulleted list.
    - Use bold headings for each bullet point.
    - Limit your report to the 5 most significant positive points to maintain balance.""",
    output_key="Positive_report"
)




the_Critic = Agent(
    name="the_Critic",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Search and collect only Negative aspects and Controversies",
    instruction="""
     PROMPT: 
{ PROMPT? }


INSTRUCTIONS:
    Write a report that compiles only the negative aspects and controversies of the person described in PLOT_OUTLINE.
    CRITICAL RULE:
    - You MUST format your response as a bulleted list.
    - Use bold headings for each bullet point.
    - Limit your report to the 5 most significant negative points to maintain balance.
    """,
    output_key="Negative_report"
)
investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[
        the_Admirer,
        the_Critic
    ]
)


the_Trial = Agent(
    name="the_Trial",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Reviews the outline so that it can be improved.",
    instruction="""
    INSTRUCTIONS:
    Consider these questions about the PLOT_OUTLINE:
    - Is the narrative factual, objective, and neutral?
    - Are both positive achievements and negative controversies adequately addressed?
    - Does it sufficiently incorporate historical details from the RESEARCH?
    - Does it feel grounded in a real time period in history?
    - Is the visual formatting and length of the positive and negative reports perfectly balanced? (e.g., Do they both use bullet points? Do they have a similar number of points?)


    If the PLOT_OUTLINE does a good job with these questions, exit the writing loop with your 'exit_loop' tool.
    If significant improvements can be made, use the 'append_to_state' tool to add your feedback to the field 'CRITICAL_FEEDBACK'.
    Explain your decision and briefly summarize the feedback you have provided.

    PLOT_OUTLINE:
    { PLOT_OUTLINE? }

    RESEARCH:
    { research? }
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[append_to_state, exit_loop]
)


Review = Agent(
    name="Review",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="As a reviewer, write a synopsis and list the pros and cons of a historical figure's story.",
    instruction="""
    INSTRUCTIONS:
    Your goal is to write a factual account of the person's history, both positive and negative, in a judicial format to achieve the greatest possible neutrality described by the PROMPT: { PROMPT? }
   
    - If there is CRITICAL_FEEDBACK, use those thoughts to improve upon the outline.
    - If there is RESEARCH provided, feel free to use details from it, but you are not required to use it all.
    - If there is a PLOT_OUTLINE, improve upon it.
    - Use the 'append_to_state' tool to write your synopsis and list the pros and cons of a historical figure's story to the field 'PLOT_OUTLINE'.
    - Ensure that the final account represents an objective balance. The length and level of detail devoted to positive achievements MUST be exactly equal to the length and detail devoted to negative controversies. Do not let one side overshadow the other visually or contextually.
    - Summarize what you focused on in this pass.


    PROMPT: 
    { PROMPT? }
    
    POSITIVE_REPORT:
    { Positive_report? }

    NEGATIVE_REPORT:
    { Negative_report? }


    RESEARCH:
    { research? }


    PLOT_OUTLINE:
    { PLOT_OUTLINE? }


    CRITICAL_FEEDBACK:
    { CRITICAL_FEEDBACK? }
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[append_to_state],
)

researcher = Agent(
    name="researcher",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Answer research questions using Wikipedia.",
    instruction="""
   PROMPT:
    { PROMPT? }
    
    POSITIVE_REPORT:
    { Positive_report? }


    NEGATIVE_REPORT:
    { Negative_report? }


    PLOT_OUTLINE:
    { PLOT_OUTLINE? }


    CRITICAL_FEEDBACK:
    { CRITICAL_FEEDBACK? }
    INSTRUCTIONS:
    - You are researching to support the judge. Review the POSITIVE_REPORT and NEGATIVE_REPORT.
    - When searching Wikipedia, use the PROMPT name along with keywords to verify or expand on the claims in the reports.
    - If there is a CRITICAL_FEEDBACK, use your Wikipedia tool to do research to solve those suggestions.
    - If there is a PLOT_OUTLINE, use your Wikipedia tool to find historical details to enrich it.
    - Use the 'append_to_state' tool to add your research to the field 'research'.
    - Summarize what you have learned.
    Now, use your Wikipedia tool to do research.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)
The_Judge = LoopAgent(
    name="The_Judge",
    description="Iterates through research and writing to improve the fact.",
    sub_agents=[
        researcher,
        Review,
        the_Trial
    ],
    max_iterations=5,
)


file_writer = Agent(
    name="file_writer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Creates a final historical trial report and saves it as a document.",
    instruction="""
    INSTRUCTIONS:
    - Use your 'write_file' tool to create a new txt file with the following arguments:
        - for a filename, use the name of the historical figure or event described in the PROMPT (ensure it ends with .txt and replace spaces with underscores if necessary).
        - Write to the 'Judge_Fact' directory.
        - For the 'content' to write, format it nicely and include:
            - The Final Factual Account (PLOT_OUTLINE)
            - The Positive Report
            - The Negative Report


PROMPT:
    { PROMPT? }


    PLOT_OUTLINE:
    { PLOT_OUTLINE? }


    Positive_report:
    { Positive_report? }


    Negative_report:
    { Negative_report? }
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[write_file],
)

The_Inquiry = SequentialAgent(
    name="The_Inquiry",
    description="Write a film plot outline and save it as a text file.",
    sub_agents=[
        investigation_team,
        The_Judge,
        file_writer
    ],
)

root_agent = Agent(
    name="greeter",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Guides the user in analyzing history.",
    instruction="""
    - Let the user know you will help them analyze history in a mock trial. Ask them for  
      a historical figure or event to analyze.
    - When they respond, use the 'append_to_state' tool to store the user's response
      in the 'PROMPT' state key and transfer to the 'The_Inquiry' agent
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[append_to_state],
    sub_agents=[The_Inquiry],
)
