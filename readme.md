# Multi Agent System (MAS) - Phidata

## Overview

**Phidata** is a versatile framework that enables the use of intelligent agents to automate tasks across various domains. It supports seamless integration and orchestration of different agents to create more complex workflows. These agents are configurable and can be used individually or together to meet specific needs. This system is designed to enhance productivity, streamline processes, and provide real-time insights.

## Agents
The code for all the agents can be found in the following GitHub repository:

[Phidata Agents Code](https://github.com/danula-rathnayaka/Multi-Agent-System/blob/main/Agents.py)

### Main Agent (Coordinator)
- **Description**: Manages the overall coordination of tasks and ensures efficient integration between different agents in the system.

### Google Search Agent
- **Description**: Performs web searches and returns the top results for a given query. Summarizes the content and includes source links.
- **Key Functions**: Search for any query and provide concise summaries.

### YouTube Agent
- **Description**: Extracts captions and metadata from YouTube videos, summarizes content, and answers questions related to the video.
- **Key Functions**: Summarizes video content and handles unavailable captions.

### File Read/Write Agent (DAG Agent)
- **Description**: Saves and reads data from files within a specified directory. Ensures that the files are properly handled, preventing overwrites.
- **Key Functions**: Safe file handling, data validation, and saving.

### Research Agent
- **Description**: Searches academic publications and research papers, aggregates data, and organizes the relevant information for research purposes.
- **Key Functions**: Retrieves research data and summarizes findings.

### Calculator Agent
- **Description**: Performs a wide variety of mathematical functions including basic arithmetic, exponentiation, prime checking, factorials, and more.
- **Key Functions**: Addition, subtraction, multiplication, division, prime checking, square roots, factorials.

### Hacker News Agent
- **Description**: Retrieves and summarizes top stories from Hacker News. Can also fetch user details related to the stories.
- **Key Functions**: Get top stories, summaries, and user information.

### News Reader Agent
- **Description**: Fetches and summarizes articles from URLs using the Newspaper4k library, providing concise overviews of article content.
- **Key Functions**: Retrieve and summarize articles.

### Python Agent
- **Description**: Runs Python code in a specified directory, saving scripts, executing code, and managing files.
- **Key Functions**: Code execution, file handling, package installations.

### Wikipedia Agent
- **Description**: Searches Wikipedia for topics, adds results to the knowledge base, and provides concise information from the search.
- **Key Functions**: Search Wikipedia and update knowledge base.

### Finance (Yahoo Finance) Agent
- **Description**: Retrieves financial data from Yahoo Finance, including stock prices, company information, income statements, and more.
- **Key Functions**: Fetch stock prices, income statements, financial ratios, analyst recommendations, and company news.

### PhiData Agent
- **Description**: Manages Phidata workspaces and helps with the creation of apps like LLM, API, Django, and Streamlit applications.
- **Key Functions**: Create and manage Phidata applications, validate workspace readiness.

## Installation

To use this project, ensure that Python is installed and use the following command to install the necessary dependencies:

```bash
pip install -r requirements.txt
