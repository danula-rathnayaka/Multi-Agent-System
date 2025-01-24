import google.generativeai as genai
from phi.model.google import Gemini
from phi.agent import Agent
from dotenv import load_dotenv
import os

from phi.tools.airflow import AirflowToolkit
from phi.tools.arxiv_toolkit import ArxivToolkit
from phi.tools.calculator import Calculator
from phi.tools.googlesearch import GoogleSearch
from phi.tools.hackernews import HackerNews
from phi.tools.newspaper4k import Newspaper4k
from phi.tools.phi import PhiTools
from phi.tools.python import PythonTools
from phi.tools.wikipedia import WikipediaTools
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


def get_research_search_tool(gemini_model='gemini-2.0-flash-exp',
                             show_tool_calls=False, debug_mode=False):
    """
        Creates and returns a research search agent to find academic publications on a specified topic.

        This agent is designed to perform targeted searches across a comprehensive academic database,
        returning relevant research papers and scholarly articles based on a user’s query. The agent allows
        searching for academic work, reading paper summaries, and downloading publications if required.

        Parameters
        ----------
        gemini_model : str, optional
            The identifier for the Gemini model to be used by the agent (default: 'gemini-2.0-flash-exp').
        show_tool_calls : bool, optional
            If True, displays the intermediate tool calls made by the agent during its execution (default: False).
        debug_mode : bool, optional
            If True, enables debugging mode to provide detailed logs for troubleshooting (default: False).

        Returns
        -------
        Agent
            An instance of the Agent class configured to search for academic papers and publications.

        Description
        -----------
        - This agent allows the user to search for scholarly articles and papers related to a specific research topic.
        - It can search an academic database, retrieve relevant results, and offer information on the articles found.
        - Supports downloading and reading full-text papers, subject to availability.
        - Results are presented with summaries, publication details, and options to read or download the papers.

        Instructions
        -------------
        1. Perform a search based on the user's query for relevant academic publications.
        2. Retrieve the top articles and present the findings in a list format, including the paper title, authors, and summary.
        3. Provide the publication’s metadata and relevant links when available.
        4. If the user requests, provide access to the full text or a downloadable version of the paper, if possible.
        5. Ensure all responses are clear, concise, and presented in markdown for readability.
        6. When searching, prioritize scholarly and credible sources.

        Example Usage
        --------------
        agent = get_research_search_tool(show_tool_calls=True, debug_mode=True)
        agent.print_response("Search for recent research on 'language models'", markdown=True)

        Example Response
        -----------------
        1. **Title:** Advances in Language Models for Natural Language Processing
           **Authors:** John Doe, Jane Smith
           **Summary:** This paper discusses the recent advancements in language models for NLP tasks, focusing on transformer-based architectures.
           **Link:** [Read Paper](link_to_paper)
        2. **Title:** Language Models: A Comprehensive Review
           **Authors:** Mark Lee, Sarah Wang
           **Summary:** A review of current language models, their applications in various domains, and the challenges they face.
           **Link:** [Read Paper](link_to_paper)
        3. **Title:** Scaling Language Models for Large-Scale NLP Applications
           **Authors:** Emily Brown, Michael Davis
           **Summary:** Exploring the scalability of language models and their use in real-world NLP applications.
           **Link:** [Read Paper](link_to_paper)

        Notes
        ------
        - Ensure that the agent has internet access to perform searches.
        - The agent can be used in conjunction with other tools for comprehensive research workflows.
        - The availability of full-text papers may vary based on the publication’s access permissions.
    """
    return Agent(
        model=Gemini(id=gemini_model),
        description=(
            "You are a research search agent designed to retrieve academic publications based on user queries. "
            "You can search for scholarly articles, summarize key findings, and provide access to papers and metadata. "
            "The agent helps users explore the latest research in various fields, making academic search easy and efficient. "
            "You provide detailed information about the papers, including titles, authors, summaries, and links to the full text or downloads when available."
        ),
        tools=[ArxivToolkit()],
        instructions=[
            "1. Perform a search based on the user's query for relevant academic publications.",
            "2. Retrieve the top articles and present the findings in a list format, including the paper title, authors, and summary.",
            "3. Provide the publication’s metadata and relevant links when available.",
            "4. If the user requests, provide access to the full text or a downloadable version of the paper, if possible.",
            "5. Ensure all responses are clear, concise, and presented in markdown for readability.",
            "6. When searching, prioritize scholarly and credible sources."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode
    )


def get_calculator_agent(gemini_model='gemini-2.0-flash-exp',
                         show_tool_calls=False, debug_mode=False, markdown=True):
    """
        Creates and returns a mathematical computation agent capable of performing various operations.

        This agent is designed to handle both basic and advanced mathematical tasks, making it a powerful tool for solving arithmetic and algebraic problems.
        The agent supports addition, subtraction, multiplication, division, exponentiation, factorial calculation, prime number checking, and square root computation.

        Parameters
        ----------
        gemini_model : str, optional
            The identifier for the Gemini model to be used by the agent (default: 'gemini-2.0-flash-exp').
        show_tool_calls : bool, optional
            If True, displays the intermediate tool calls made by the agent during its execution (default: False).
        debug_mode : bool, optional
            If True, enables debugging mode to provide detailed logs for troubleshooting (default: False).
        markdown : bool, optional
            If True, the results will be presented using markdown formatting for improved readability (default: False).

        Returns
        -------
        Agent
            An instance of the Agent class configured to perform mathematical operations.

        Description
        -----------
        - The agent can perform a wide range of mathematical operations, including basic arithmetic (addition, subtraction, multiplication, division),
          as well as advanced functions like exponentiation, factorials, prime number checking, and square roots.
        - Results are presented clearly and concisely in markdown format for easy understanding.

        Instructions
        -------------
        1. Perform basic arithmetic operations such as addition, subtraction, multiplication, and division.
        2. Compute exponentiation (raising numbers to a power).
        3. Calculate the factorial of a number.
        4. Check if a number is prime.
        5. Compute the square root of a given number.
        6. Ensure all results are returned clearly and concisely.
        7. Use markdown formatting for presenting the answers in a clean and readable manner.
        8. If an operation is not supported or invalid, provide an appropriate error message.

        Example Usage
        --------------
        agent = get_calculator_agent(show_tool_calls=True, debug_mode=True)
        agent.print_response("What is 5 + 3?", markdown=True)
        agent.print_response("Calculate the square root of 16.", markdown=True)

        Example Response
        -----------------
        1. 5 + 3 = 8
        2. The square root of 16 is 4.

        Notes
        ------
        - The agent handles a wide range of mathematical tasks and is ideal for use in educational, scientific, or engineering applications.
        - Ensure that input values are valid numbers for the operations to work correctly.
    """
    return Agent(
        model=Gemini(id=gemini_model),
        description=(
            "You are a mathematical computation agent capable of performing a wide range of arithmetic and algebraic operations. "
            "You can perform basic arithmetic operations such as addition, subtraction, multiplication, and division. "
            "Additionally, you support advanced operations including exponentiation, factorial calculation, prime number checking, and square root computation. "
            "This makes you a versatile tool for solving both simple and complex mathematical problems."
        ),
        tools=[
            Calculator(
                add=True,
                subtract=True,
                multiply=True,
                divide=True,
                exponentiate=True,
                factorial=True,
                is_prime=True,
                square_root=True,
            )
        ],
        instructions=[
            "1. Perform basic arithmetic operations such as addition, subtraction, multiplication, and division.",
            "2. Compute exponentiation (raising numbers to a power).",
            "3. Calculate the factorial of a number.",
            "4. Check if a number is prime.",
            "5. Compute the square root of a given number.",
            "6. Ensure all results are returned clearly and concisely.",
            "7. Use markdown formatting for presenting the answers in a clean and readable manner.",
            "8. If an operation is not supported or invalid, provide an appropriate error message."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
        markdown=markdown,
    )


def get_hacker_news_agent(gemini_model='gemini-2.0-flash-exp',
                          show_tool_calls=False, debug_mode=False, markdown=True):
    """
        Creates and returns a Hacker News agent designed to fetch and present top stories and user details from the Hacker News platform.

        This agent allows you to search for the latest popular stories on Hacker News, along with summaries and related user information.
        It provides a convenient way to stay updated with the most relevant content and engage with the stories posted by users.

        Parameters
        ----------
        gemini_model : str, optional
            The identifier for the Gemini model to be used by the agent (default: 'gemini-2.0-flash-exp').
        show_tool_calls : bool, optional
            If True, displays the intermediate tool calls made by the agent during its execution (default: False).
        debug_mode : bool, optional
            If True, enables debugging mode to provide detailed logs for troubleshooting (default: False).
        markdown : bool, optional
            If True, the results will be presented using markdown formatting for improved readability (default: True).

        Returns
        -------
        Agent
            An instance of the Agent class configured to retrieve top stories and user details from Hacker News.

        Description
        -----------
        - The agent retrieves the top stories from Hacker News and provides engaging summaries of the content.
        - If requested, it can also provide detailed user information for the users who submitted the stories.
        - The results are returned in a clear and concise markdown format for easy reading.

        Instructions
        -------------
        1. Retrieve the top stories from Hacker News based on the latest posts.
        2. Present the top stories along with a brief summary and key details about the content.
        3. If requested, fetch user details of those who submitted the stories, including their username and activity on the platform.
        4. Use markdown formatting for presenting the stories and user information in a clear and readable manner.
        5. Ensure the responses are concise, engaging, and focused on the most relevant information.
        6. If no user details are available or requested, provide only the top stories with summaries.

        Example Usage
        --------------
        agent = get_hacker_news_agent(show_tool_calls=True, debug_mode=True)
        agent.print_response("Write an engaging summary of the users with the top 2 stories on hackernews. Please mention the stories as well.", markdown=True)

        Example Response
        -----------------
        1. "Story 1 Title" - Summary: A brief description of the first story's content. (Posted by: user123)
        2. "Story 2 Title" - Summary: A brief description of the second story's content. (Posted by: user456)

        Notes
        ------
        - The agent fetches only the top stories from Hacker News based on the latest activity.
        - Ensure that the agent has internet access to fetch the data in real-time.
    """
    return Agent(
        model=Gemini(id=gemini_model),
        description=(
            "You are a Hacker News agent designed to retrieve top stories and provide insights into users on the Hacker News platform. "
            "You can fetch the latest stories, summarize them, and provide additional details about users who posted the stories. "
            "The agent is equipped to present the top stories along with summaries and relevant user information, offering a comprehensive view of the most popular content."
        ),
        tools=[HackerNews()],
        instructions=[
            "1. Retrieve the top stories from Hacker News based on the latest posts.",
            "2. Present the top stories along with a brief summary and key details about the content.",
            "3. If requested, fetch user details of those who submitted the stories, including their username and activity on the platform.",
            "4. Use markdown formatting for presenting the stories and user information in a clear and readable manner.",
            "5. Ensure the responses are concise, engaging, and focused on the most relevant information.",
            "6. If no user details are available or requested, provide only the top stories with summaries."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
        markdown=markdown,
    )


def get_news_reader_agent(gemini_model='gemini-2.0-flash-exp',
                          show_tool_calls=False, debug_mode=False, markdown=True):
    """
        Creates and returns a News Reader agent designed to retrieve and summarize articles from online sources.

        This agent retrieves articles from the provided URLs and generates summaries using the Newspaper4k library.
        The agent can return both the full article content and concise summaries based on the user's request.
        The summaries focus on key points while ensuring the original meaning is preserved.

        Parameters
        ----------
        gemini_model : str, optional
            The identifier for the Gemini model to be used by the agent (default: 'gemini-2.0-flash-exp').
        show_tool_calls : bool, optional
            If True, displays the intermediate tool calls made by the agent during its execution (default: False).
        debug_mode : bool, optional
            If True, enables debugging mode to provide detailed logs for troubleshooting (default: False).
        markdown : bool, optional
            If True, the results will be presented using markdown formatting for improved readability (default: True).

        Returns
        -------
        Agent
            An instance of the Agent class configured to retrieve and summarize news articles from online sources.

        Description
        -----------
        - The agent retrieves the full text of an article from the provided URL.
        - The agent can generate a concise summary of the article's main points.
        - The summary and/or full content is presented in markdown format for readability.
        - If an error occurs during article retrieval, the agent provides a failure message and suggests potential fixes.

        Instructions
        -------------
        1. Retrieve the full text of an article from the provided URL.
        2. If the article is successfully retrieved, generate a summary of its content in a concise and accurate manner.
        3. Ensure that the summary captures the essential points and message of the article.
        4. Provide the option to return the full article text or just the summary, depending on user preference.
        5. Use markdown formatting for presenting the summary or full content to ensure clarity and readability.
        6. In case of errors (e.g., invalid URL or failure to retrieve the article), notify the user and explain the issue.
        7. Ensure that the summaries are based solely on the article's content, without adding personal opinions.

        Example Usage
        --------------
        agent = get_news_reader_agent(show_tool_calls=True, debug_mode=True)
        agent.print_response("Please summarize https://www.rockymountaineer.com/blog/experience-icefields-parkway-scenic-drive-lifetime")

        Example Response
        -----------------
        "The Icefields Parkway is one of the most scenic drives in the world, offering breathtaking views of glaciers, mountains, and pristine lakes. The experience is truly unforgettable, providing opportunities for sightseeing and wildlife observation."

        Notes
        ------
        - Ensure the provided URL is a valid article link.
        - The agent can process multiple articles but one URL should be processed at a time.
    """

    return Agent(
        description=(
            "You are a news reader agent designed to retrieve and summarize articles from various online news sources. "
            "The agent uses the Newspaper4k library to extract relevant content and generate concise summaries. "
            "You can provide a URL, and the agent will retrieve and summarize the article text, focusing on key points, and present them in a readable format. "
            "Additionally, the agent can return the full content of the article upon request."
        ),
        model=Gemini(id=gemini_model),
        tools=[Newspaper4k(include_summary=True)],
        instructions=[
            "1. Retrieve the full text of an article from the provided URL.",
            "2. If the article is successfully retrieved, generate a summary of its content in a concise and accurate manner.",
            "3. Ensure that the summary captures the essential points and message of the article.",
            "4. Provide the option to return the full article text or just the summary, depending on user preference.",
            "5. Use markdown formatting for presenting the summary or full content to ensure clarity and readability.",
            "6. In case of errors (e.g., invalid URL or failure to retrieve the article), notify the user and explain the issue.",
            "7. Ensure that the summaries are based solely on the article's content, without adding personal opinions."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
        markdown=markdown,
    )


def get_phi_data_tools_agent(gemini_model='gemini-2.0-flash-exp',
                             show_tool_calls=False, debug_mode=False, markdown=True):
    """
        Creates and returns a Phi Data Tools agent for managing phidata workspaces.

        This agent provides functionality to create, validate, and start workspaces in a phidata environment. It allows users to create new applications from templates (e.g., llm-app, api-app, django-app, streamlit-app),
        and validates if Phi is ready to run commands. Additionally, it can start existing workspaces for users.

        Parameters
        ----------
        gemini_model : str, optional
            The identifier for the Gemini model to be used by the agent (default: 'gemini-2.0-flash-exp').
        show_tool_calls : bool, optional
            If True, displays the intermediate tool calls made by the agent during its execution (default: False).
        debug_mode : bool, optional
            If True, enables debugging mode to provide detailed logs for troubleshooting (default: False).
        markdown : bool, optional
            If True, the results will be presented using markdown formatting for improved readability (default: True).

        Returns
        -------
        Agent
            An instance of the Agent class configured for creating, managing, and starting Phi workspaces.

        Description
        -----------
        - The agent allows the creation of new phidata workspaces from templates such as llm-app, api-app, django-app, and streamlit-app.
        - The agent validates whether the Phi environment is ready to execute commands.
        - The agent can start user-specific workspaces by invoking the corresponding start function.
        - Operations related to workspace management are performed in a seamless and efficient manner.

        Instructions
        -------------
        1. Validate that the Phi environment is ready and able to run commands by using the 'validate_phi_is_ready' function.
        2. Create new phidata workspaces for various application templates (e.g., llm-app, api-app, django-app, streamlit-app) using 'create_new_app'.
        3. Start a workspace for a user by calling 'start_user_workspace' with the appropriate workspace name.
        4. Ensure that all workspace operations (creation, validation, starting) are completed successfully before proceeding with further tasks.
        5. Provide informative and concise responses to the user, including the status of workspace creation and operations.
        6. Use markdown formatting to enhance readability and structure of responses.

        Example Usage
        --------------
        agent = get_phi_data_tools_agent(show_tool_calls=True, debug_mode=True)
        agent.print_response("Create a new agent-app called agent-app-turing", markdown=True)

        Example Response
        -----------------
        "Successfully created a new agent-app called agent-app-turing."

        Notes
        ------
        - Ensure that Phi is properly configured before using this agent.
        - Each workspace created or started should be validated before proceeding with further tasks.
    """
    return Agent(
        description=(
            "You are a workspace management agent designed to create, manage, and start phidata workspaces. "
            "Using the Phi toolkit, you can create new applications from templates (like llm-app, api-app, django-app, and streamlit-app), "
            "validate that the Phi environment is ready, and start existing workspaces for users. "
            "This agent streamlines the process of working with phidata workspaces, making it easier to manage your applications."
        ),
        model=Gemini(id=gemini_model),
        tools=[PhiTools()],
        instructions=[
            "1. Validate that the Phi environment is ready and able to run commands by using the 'validate_phi_is_ready' function.",
            "2. Create new phidata workspaces for various application templates (e.g., llm-app, api-app, django-app, streamlit-app) using 'create_new_app'.",
            "3. Start a workspace for a user by calling 'start_user_workspace' with the appropriate workspace name.",
            "4. Ensure that all workspace operations (creation, validation, starting) are completed successfully before proceeding with further tasks.",
            "5. Provide informative and concise responses to the user, including the status of workspace creation and operations.",
            "6. Use markdown formatting to enhance readability and structure of responses."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
        markdown=markdown
    )


def get_python_agent(gemini_model='gemini-2.0-flash-exp',
                     show_tool_calls=False, debug_mode=False, markdown=True):
    """
        Creates and returns a Python agent that can write, save, run, and manage Python code.

        This agent provides functionality to generate Python scripts based on user input, save them to files, run them,
        and return the results. It also supports managing Python packages by installing them via pip,
        and executing scripts or Python code in a safe environment.

        Parameters
        ----------
        gemini_model : str, optional
            The identifier for the Gemini model to be used by the agent (default: 'gemini-2.0-flash-exp').
        show_tool_calls : bool, optional
            If True, displays the intermediate tool calls made by the agent during its execution (default: False).
        debug_mode : bool, optional
            If True, enables debugging mode to provide detailed logs for troubleshooting (default: False).
        markdown : bool, optional
            If True, the results will be presented using markdown formatting for improved readability (default: True).

        Returns
        -------
        Agent
            An instance of the Agent class configured for writing, saving, and running Python code.

        Description
        -----------
        - The agent can create Python code and save it to a file for execution.
        - It can run Python code in the current environment and return the results or error messages.
        - Supports the installation of required Python packages using pip before running the code.
        - It can list, read, and run files in the base directory, with options to handle safe globals and locals.
        - The agent is capable of executing Python scripts after saving them to a file and returning the result.

        Instructions
        -------------
        1. Write Python code based on the user's request.
        2. Save the Python code to a file and execute it if 'save_and_run' is enabled.
        3. If requested, list all files in the base directory or run specific Python files.
        4. Ensure that all code is executed in a secure environment by using 'safe_globals' and 'safe_locals' to limit available variables.
        5. Allow users to install packages using pip before running code if 'pip_install' is enabled.
        6. Ensure that the code output is returned to the user in a readable format, either as the variable's value or a success message.
        7. Notify the user if there are any errors during the script execution, including details about what went wrong.
        8. Use markdown formatting for clear presentation of code, results, and errors.

        Example Usage
        --------------
        agent = get_python_agent(show_tool_calls=True, debug_mode=True)
        agent.print_response("Write a python script for fibonacci series and display the result till the 10th number")

        Example Response
        -----------------
        "Successfully saved and executed the Python script. The Fibonacci series up to the 10th number is: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"

        Notes
        ------
        - Ensure that the Python environment is configured with necessary dependencies for successful execution.
        - Use 'pip_install' to install any required packages before executing the code.
    """
    return Agent(
        description=(
            "You are a Python scripting agent designed to write, run, and manage Python code. "
            "Using the PythonTools library, you can create Python scripts, save them to files, run them, and return the results. "
            "The agent can also handle Python package installations and perform file management operations, allowing for seamless execution of Python scripts in various environments."
        ),
        model=Gemini(id=gemini_model),
        tools=[PythonTools()],
        instructions=[
            "1. Write Python code as requested by the user.",
            "2. Save the Python code to a file and execute it if 'save_and_run' is enabled.",
            "3. If requested, list all files in the base directory or run specific Python files.",
            "4. Ensure that all code is executed in a secure environment by using 'safe_globals' and 'safe_locals' to limit available variables.",
            "5. Allow users to install packages using pip before running code if 'pip_install' is enabled.",
            "6. Ensure that the code output is returned to the user in a readable format, either as the variable's value or a success message.",
            "7. Use markdown formatting for clear presentation of code, results, and errors.",
            "8. Notify the user if there are any errors during the script execution, including details about what went wrong."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
        markdown=markdown
    )


def get_wikipedia_agent(gemini_model='gemini-2.0-flash-exp',
                        show_tool_calls=False, debug_mode=False, markdown=True):
    """
        Creates and returns an agent capable of searching Wikipedia and adding the retrieved information to the knowledge base.

        This agent uses the WikipediaTools library to search for topics on Wikipedia, retrieve relevant article content,
        and add the retrieved data to the agent's knowledge base for further use. The agent can search and update the knowledge base
        with the most recent information available from Wikipedia.

        Parameters
        ----------
        gemini_model : str, optional
            The identifier for the Gemini model to be used by the agent (default: 'gemini-2.0-flash-exp').
        show_tool_calls : bool, optional
            If True, displays the intermediate tool calls made by the agent during its execution (default: False).
        debug_mode : bool, optional
            If True, enables debugging mode to provide detailed logs for troubleshooting (default: False).
        markdown : bool, optional
            If True, the results will be presented using markdown formatting for improved readability (default: True).

        Returns
        -------
        Agent
            An instance of the Agent class configured for searching Wikipedia and updating its knowledge base.

        Description
        -----------
        - The agent can search Wikipedia for a given topic or query.
        - It retrieves the content of the relevant Wikipedia article and adds it to the agent's knowledge base.
        - The agent can then use this content for further processing, summarization, or presenting the information to the user.
        - The knowledge base is continuously updated with the most recent and relevant Wikipedia information.

        Instructions
        -------------
        1. Search Wikipedia for the user's query and retrieve the relevant article content.
        2. Update the knowledge base with the retrieved Wikipedia article data.
        3. If necessary, summarize the Wikipedia article and return the summarized content to the user.
        4. Ensure the results are presented in markdown format for clarity and readability.
        5. Notify the user if the query does not return relevant Wikipedia results or if an error occurs during the search.

        Example Usage
        --------------
        agent = get_wikipedia_agent(show_tool_calls=True, debug_mode=True)
        agent.print_response("Search Wikipedia for 'Artificial Intelligence'")

        Example Response
        -----------------
        "Search results for 'Artificial Intelligence' from Wikipedia: [Article Summary with Key Information]"

        Notes
        ------
        - Ensure that the Wikipedia library is properly installed (pip install -U wikipedia) for the agent to function correctly.
        - The knowledge base is automatically updated with the most recent content retrieved from Wikipedia.
    """
    return Agent(
        description=(
            "You are a Wikipedia search agent designed to retrieve information from Wikipedia and add the contents to the knowledge base. "
            "Using the WikipediaTools library, you can search for topics on Wikipedia and gather relevant information to enhance the agent's knowledge base. "
            "This agent is helpful in retrieving reliable data from Wikipedia articles and using it for further processing or summarization."
        ),
        model=Gemini(id=gemini_model),
        tools=[WikipediaTools()],
        instructions=[
            "1. Search Wikipedia for a specified topic or query provided by the user.",
            "2. Retrieve the relevant article content from Wikipedia based on the search query.",
            "3. Add the retrieved content to the knowledge base to enhance the agent's understanding.",
            "4. Present the relevant article content or summary to the user in a clear and concise manner.",
            "5. Use markdown formatting for better presentation and readability of the results.",
            "6. Notify the user if the search did not return relevant results or if there were any errors during the search.",
            "7. Ensure that the retrieved Wikipedia content is up-to-date and accurate, based on the most recent version of the article."
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
        markdown=markdown
    )
