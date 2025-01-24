# Multi Agent System (MAS) - Phidata

## Overview

**Phidata** is a powerful and versatile framework for building intelligent agents to handle various tasks across different domains. The project includes specialized agents for tasks. This system is designed to make it easy to integrate and utilize multiple agents within one cohesive solution.

Each agent is configurable and can be used individually or as part of larger workflows, making MAS a flexible and scalable solution for automating tasks.


## Agents
**Main Agent (Coordinator)**:
  - Manage all agents collectively, allowing seamless integration and coordination between tasks.

**Google Search Agent**:
  - Retrieve the top search results for a given query.
  - Provide concise summaries and include sources in the response.
  - Supports real-time and reliable web search.

**YouTube Agent**:
  - Extract captions and metadata from YouTube videos.
  - Summarize video content or answer questions related to the video.
  - Handle situations where captions are unavailable.

**File Read/Write Agent (DAG Agent)**:
  - Save and read data from files within a specified directory.
  - Ensure safe file handling to avoid accidental overwrites.
  - Validate input data before saving it to ensure proper formatting.


## Installation
To use this project, you need to have Python installed. Install the necessary dependencies using the following command:

```bash
pip install -r "requirements.txt"
```