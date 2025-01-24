import google.generativeai as genai
from phi.model.google import Gemini
from phi.agent import Agent
from dotenv import load_dotenv
import os

from phi.tools.airflow import AirflowToolkit
from phi.tools.googlesearch import GoogleSearch
from phi.tools.youtube_tools import YouTubeTools

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_google_search_agent(gemini_model='gemini-2.0-flash-exp', show_tool_calls=False, debug_mode=False):
    """
    Create and return a Google Search agent for retrieving top search results.

    This agent is designed to perform efficient and precise searches using Google.
    It processes user queries, retrieves the top 5 most relevant results, and presents them
    in a concise, formatted manner. Each response includes a short summary of the search result
    along with its source in brackets for credibility and reference.

    Parameters
    ----------
    gemini_model : str, optional
        The identifier for the Gemini model to be used by the agent (default is 'gemini-2.0-flash-exp').
    show_tool_calls : bool, optional
        If True, displays the intermediate tool calls made by the agent during its execution (default is False).
    debug_mode : bool, optional
        If True, enables debugging mode to provide detailed logs for troubleshooting (default is False).

    Returns
    -------
    Agent
        An instance of the `Agent` class configured for Google Search queries.

    Description
    -----------
    - The agent retrieves the top 5 Google search results for a given query.
    - Results are presented as a numbered list with a brief summary and the source in brackets.
    - Responses are always in English and formatted using markdown for readability.

    Instructions
    ------------
    1. Perform a Google search based on the user's query.
    2. Retrieve the top 5 results and format them as a numbered list.
    3. Include a short summary of each result, followed by the source in brackets
       (e.g., 'Summary of result (Source)').
    4. Ensure responses are concise, accurate, and from credible sources.
    5. Use markdown formatting to structure the output for clarity.

    Example
    -------
    >>> agent = get_google_search_agent(show_tool_calls=True, debug_mode=True)
    >>> agent.print_response("What are the latest advancements in artificial intelligence?", markdown=True)

    Example Output:
    ---------------
    1. Researchers are making breakthroughs in generative AI, with new models offering increased capabilities. (Source: MIT Technology Review)
    2. OpenAI announces advancements in multimodal AI systems capable of understanding text and images. (Source: OpenAI Blog)
    3. AI is being increasingly adopted in healthcare for diagnostics and patient monitoring. (Source: Nature)
    4. The role of AI in climate change mitigation is gaining prominence, with several innovative applications. (Source: Science Daily)
    5. Google introduces advanced AI tools for workplace productivity in its latest announcement. (Source: Google AI Blog)

    Notes
    -----
    - Ensure an active internet connection for Google Search queries.
    - The agent can be integrated into larger workflows requiring real-time information retrieval.
    """
    return Agent(
        model=Gemini(id=gemini_model),
        tools=[GoogleSearch()],
        description=(
            "You are a search agent designed to retrieve the top 5 results for any given query from Google. "
            "Your responses are concise, clear, and include the source for each result in brackets."
        ),
        instructions=[
            "1. Perform a Google search based on the user's query.",
            "2. Retrieve the top 5 results and format them as a numbered list.",
            "3. Include a short summary of each result, followed by the source in brackets, e.g., 'Summary of result (Source)'.",
            "4. Provide all responses in English and ensure sources are accurate and credible.",
            "5. Use markdown formatting for clear presentation."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
    )


def get_youtube_agent(get_youtube_video_captions=True, gemini_model='gemini-2.0-flash-exp',
                      show_tool_calls=False, debug_mode=False):
    """
    Create and return a YouTube agent for retrieving captions and metadata from YouTube videos.

    This agent specializes in extracting captions and metadata from YouTube videos. It can summarize video content,
    extract specific information from captions, and answer user questions about the video's content.
    The agent ensures concise and accurate responses with an emphasis on clarity and usability.

    Parameters
    ----------
    get_youtube_video_captions : bool, optional
        If True, enables the agent to retrieve captions for YouTube videos (default is True).
    gemini_model : str, optional
        The identifier for the Gemini model to be used by the agent (default is 'gemini-2.0-flash-exp').
    show_tool_calls : bool, optional
        If True, displays the intermediate tool calls made by the agent during its execution (default is False).
    debug_mode : bool, optional
        If True, enables debugging mode to provide detailed logs for troubleshooting (default is False).

    Returns
    -------
    Agent
        An instance of the `Agent` class configured to handle YouTube video queries.

    Description
    -----------
    - The agent retrieves captions and metadata from YouTube videos when provided with a video URL.
    - It uses the retrieved captions to summarize videos or answer user queries about the content.
    - Relevant metadata, such as video title and duration, is included when helpful.
    - If captions are unavailable, the agent notifies the user and offers metadata instead.

    Instructions
    ------------
    1. Retrieve captions and metadata when provided with a YouTube video URL.
    2. Use captions to summarize the video or answer specific user queries.
    3. Include metadata such as the video title and duration when relevant.
    4. Notify the user politely if captions are unavailable, offering metadata as an alternative.
    5. Format responses in markdown for clarity and readability.
    6. Ensure summaries focus on the key points or central theme of the video.

    Example
    -------
    >>> agent = get_youtube_agent(show_tool_calls=True, debug_mode=True)
    >>> agent.print_response("Summarize this video: https://www.youtube.com/watch?v=Iv9dewmcFbs", markdown=True)

    Example Output:
    ---------------
    **Video Title**: "Understanding Artificial Intelligence in 10 Minutes"
    **Duration**: 10:25
    **Summary**:
    The video explains the fundamentals of artificial intelligence, including its history, key concepts,
    and applications. It covers topics like machine learning, neural networks, and real-world use cases.
    Captions are used to provide additional context and enhance the summary.

    Notes
    -----
    - The agent requires an active internet connection to retrieve captions and metadata from YouTube.
    - This tool is ideal for summarizing video content, answering specific questions, and extracting video metadata.
    """
    return Agent(
        model=Gemini(id=gemini_model),
        tools=[YouTubeTools()],
        description=(
            "You are a YouTube agent specializing in retrieving captions and metadata from YouTube videos. "
            "You can summarize videos, extract specific information from captions, and answer user questions about video content."
        ),
        instructions=[
            "1. If provided with a YouTube video URL, retrieve its captions and metadata.",
            "2. Use the captions to summarize the video or answer specific user queries.",
            "3. Include relevant metadata such as the video title and duration when helpful.",
            "4. If captions are unavailable, notify the user politely and offer to provide metadata instead.",
            "5. Ensure all responses are concise, accurate, and formatted in markdown for clarity.",
            "6. When summarizing, focus on the key points or central theme of the video."
        ],
        debug_mode=debug_mode,
        get_youtube_video_captions=get_youtube_video_captions,
        show_tool_calls=show_tool_calls
    )


def get_file_read_write_agent(save_dag=True, read_dag=True, dir_name="files", gemini_model='gemini-2.0-flash-exp',
                              show_tool_calls=False, debug_mode=False):
    """
    Create and return a file read/write agent for handling data in specified files.

    This agent is designed to manage file operations within a given directory.
    It can save user-provided data to files, read the content of existing files,
    and ensure safe and efficient file handling. The agent prevents accidental overwrites
    and validates input data before saving it.

    Parameters
    ----------
    save_dag : bool, optional
        If True, enables the agent to save data to files (default is True).
    read_dag : bool, optional
        If True, allows the agent to read data from files (default is True).
    dir_name : str, optional
        The directory where files will be saved or read from (default is "files").
    gemini_model : str, optional
        The identifier for the Gemini model to be used by the agent (default is 'gemini-2.0-flash-exp').
    show_tool_calls : bool, optional
        If True, displays the intermediate tool calls made by the agent during its execution (default is False).
    debug_mode : bool, optional
        If True, enables debugging mode to provide detailed logs for troubleshooting (default is False).

    Returns
    -------
    Agent
        An instance of the `Agent` class configured for file read/write operations.

    Description
    -----------
    - The agent allows saving data to files and reading data from existing files within a specified directory.
    - File operations are performed safely, with validation to avoid accidental overwrites.
    - If the specified directory or file does not exist, the agent will inform the user.
    - The agent ensures that user-provided data is formatted correctly before saving it.

    Instructions
    ------------
    1. Save user-provided data to a file in the specified directory upon request.
    2. Read and return the content of an existing file from the directory when requested.
    3. Ensure all file operations are conducted in the specified directory.
    4. Validate input data before saving to ensure proper formatting and prevent errors.
    5. Return full file content unless the user specifies otherwise.
    6. Notify the user if the file or directory cannot be found or accessed.
    7. Prevent overwriting files during save operations by confirming or appending to the file.
    8. Use markdown formatting for responses to enhance clarity and readability.

    Example
    -------
    >>> agent = get_file_read_write_agent(show_tool_calls=True, debug_mode=True)
    >>> agent.print_response("Save the following data to 'myfile.txt': Sample data to be stored in the file.", markdown=True)

    Example Output:
    ---------------
    **File Save Successful**
    Data saved to 'myfile.txt' in the 'files' directory.

    >>> agent.print_response("Read the content of 'myfile.txt'.", markdown=True)

    Example Output:
    ---------------
    **File Content**
    "Sample data to be stored in the file."

    Notes
    -----
    - Ensure the specified directory exists or the agent will notify the user if it cannot be found.
    - This agent is suitable for file management workflows where reading from and writing to files is required.
    - Can be used in larger systems for organizing and handling data stored in files.
    """
    return Agent(model=Gemini(id=gemini_model),
                 description=(
                     "You are a file management agent designed to read and write data to files. "
                     "You can save provided data to specified files in a directory, as well as read the content of existing files for review or further use. "
                     "This allows efficient and organized file handling in various workflows."
                 ),
                 tools=[AirflowToolkit(dags_dir=dir_name,
                                       save_dag=save_dag,
                                       read_dag=read_dag)
                        ],
                 instructions=[
                     "1. Save user-provided data to a file in the specified directory when requested.",
                     "2. Read and return the content of existing files from the specified directory upon request.",
                     "3. Ensure all file operations are performed in the directory provided by the user.",
                     "4. For 'save' operations, validate the input to ensure the data is correctly formatted and saved without errors.",
                     "5. For 'read' operations, retrieve the full file content unless the user specifies otherwise.",
                     "6. Notify the user if the specified file or directory does not exist or cannot be accessed.",
                     "7. Prevent accidental overwrites during save operations by confirming or appending to existing files if necessary.",
                     "8. Use markdown formatting for responses to ensure clarity and readability."
                 ],
                 show_tool_calls=show_tool_calls,
                 markdown=debug_mode
                 )
