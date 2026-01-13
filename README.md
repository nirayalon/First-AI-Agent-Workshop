# Build an AI Agent From Scratch in Python - Workshop Notebook

**Workshop Goal**: Learn to build an AI agent capable of answering questions, using tools,
and structuring its output.

What You'll Build: A practical research assistant with access to web search and
Wikipedia, capable of saving its findings to a file.

# Prerequisites

## Setting Up Your Development Environment
Before we begin coding, let's ensure our development environment is correctly set up.

**Prerequisites:**
*   **Python 3.13** should be installed on your system. You can check your
    Python version by running `python --version` or `python3 --version` in your terminal.
*   A **code editor** is recommended. 
    
## Managing Dependencies & Your Virtual Environment

To keep our project dependencies isolated and manageable, we'll use a **virtual environment**.

**`requirements.txt`:**
First, create a new file named `requirements.txt` in your project folder. This file will
list all the Python packages our project depends on. Paste the following lines into it.


```python
langchain 
wikipedia 
langchain-community 
langchain-openai
langchain-anthropic
python-dotenv
pydantic
duckduckgo-search
```


**Virtual Environment (`venv`):**
1.  Open your terminal or integrated terminal in VS Code/Pycharm (Terminal > New Terminal).
2.  Navigate to your project directory in the terminal.
3.  Create a virtual environment named `venv` by running the following command:
    *   **Mac/Linux:** `python -m venv ai_agent_venv` or `python3.13 -m venv ai_agent_venv`
    *   **Windows:** `python -m venv venv`
    After running this, you should see a new folder named `venv` in your project directory.

4.  Activate the virtual environment:
    *   **Mac/Linux:** `source ai_agent_venv/bin/activate`
    *   **Windows:** `.\venv\Scripts\activate`
    Once activated, you should see `(venv)` at the beginning of your terminal prompt.

5.  Install the project dependencies from `requirements.txt`:

    `pip install -r requirements.txt`
    
    or

    `pip3 install -r requirements.txt`

    This command will install all the necessary packages into your virtual environment.


## Obtaining and Using API Keys

Our AI agent will interact with **Language Models (LLMs)**, such as those from OpenAI or
Anthropic. To use these services, you'll need to obtain **API keys**.

**Why API Keys?**
API keys are required to authenticate your requests to the LLM providers and to track
your usage.

**Obtaining Keys:**
*   **OpenAI:**
    1.  Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).
    2.  You might need to sign up or log in.
    3.  Click on "Create new secret key".
    4.  Give your key a name (optional) and click "Create secret key".
    5.  **Copy the key and store it securely.** You won't be able to see it again.
*   **Anthropic (Claude):**
    1.  Go to [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
    2.  You might need to sign up or log in.
    3.  Click on "Create Key".
    4.  Give your key a name (optional) and click "Create Key".
    5.  **Copy the key and store it securely.**

**`.env` File:**
To avoid hardcoding your API keys directly in your Python code, we'll use a `.env` file
to store them as environment variables.

1.  Create a new file named `.env` in your project directory.
2.  Open the `.env` file and add your API keys in the following format, replacing
    `YOUR_OPENAI_KEY` and `YOUR_ANTHROPIC_KEY` with your actual keys:

```python
OPENAI_API_KEY=YOUR_OPENAI_KEY 
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
```

If you only intend to use one of the LLM providers, you only need to include the
corresponding API key.

**Loading Keys:**
To load these environment variables into our Python environment, we'll use the
`python-dotenv` library. Create a new Python file named `main.py` in your project
directory and add the following import:


```python
from dotenv import load_dotenv
import os
load_dotenv()
```
