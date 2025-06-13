# llm_ensemble_classifier# Multi-Agent LLM Relabeling/Classification Framework
## 1. Overview
This project provides a flexible framework for relabeling scientific content, specifically tailored for astrophysics and multi-messenger astronomy. It leverages multiple Large Language Model (LLM) agents (both from OpenAI and local Ollama instances) to classify text-based articles against a predefined taxonomy (T-DAMM).
The core of the system is a dynamic voting mechanism that aggregates the classifications from all participating agents to determine a final, consensus-based set of labels for each article. The framework is designed for extensibility, detailed logging, and performance analysis, producing comprehensive reports to evaluate the effectiveness of different models and the ensemble itself.
## 2. Key Features
**Multi-Agent Architecture:** Simultaneously query multiple LLM agents (e.g., GPT-4o, Llama3, Qwen3).
**Hybrid Model Support:** Seamlessly integrates with both OpenAI's cloud-based models and locally hosted models via Ollama.
**Dynamic Voting System:**
**Majority Vote:** Assigns labels that receive consensus from more than half of the agents.
**Union Vote:** Assigns all unique labels suggested by any agent.
**Comprehensive Output:** Generates multiple files for analysis:
A clean, relabeled final dataset.
Detailed JSON logs of the entire process for each item.
    a. Per-agent performance reports.
    b. An aggregate classification report with standard metrics (Precision, Recall, F1-score, Jaccard Score).
**Configuration-Driven:** Easily configure models, API keys, and other parameters through command-line arguments and environment variables.
**Asynchronous Processing:** Leverages asyncio for efficient, parallel querying of LLM agents, significantly speeding up the process.
## 3. Project Structure
The project is organized into several focused Python modules:
main.py: The main entry point of the application. It handles command-line argument parsing and orchestrates the overall workflow.
orchestrator.py: The core logic controller. It reads the input data, manages the asynchronous calls to LLM agents, invokes the voting process, and calls the output writers.
config.py: Central hub for all configurations. It defines static configurations like the label taxonomy, manages environment variables, and sets up the command-line argument parser.
llm_agents.py: Handles the initialization of LLM clients (OpenAI, Ollama) and constructs the system prompt used to instruct the models.
voter.py: Implements the voting logic (majority or union) to aggregate predictions from all agents and determine the final labels.
data_models.py: Contains all dataclasses used throughout the application, ensuring data consistency for logs, reports, and final outputs.
metrics_calculator.py: Calculates classification metrics (e.g., F1-score, Jaccard score) for individual predictions and generates an aggregate report for the entire run.
output_writer.py: Manages the creation of all output files, including the final JSON, detailed logs, and performance reports.
## 4. Input Data Format:
Input data consists of a .json file of articles in this format. 
```
[
    {
        "link": "https://science.nasa.gov/ems/12_gammarays/",
        "title": "Gamma Rays - NASA Science",
        "full_text": "Gamma Rays - NASA Science\n\nSkip to main content ...",
        "labels": [
            "Gamma rays"
        ]
    },
    {
        "link": "https://science.nasa.gov/mission/burstcube/",
        "title": "BurstCube - NASA Science",
        "full_text": "BurstCube - NASA Science\n\nSkip to main content ...
        "labels": [
            "Gamma rays"
        ]
    }
]
```
## 5. Setup and Installation
Clone the Repository:
```
git clone <your-repository-url>
cd <repository-directory>
```

Install Dependencies:
The script relies on several Python libraries. A ```pyproject.toml``` file is provided. :
```
pip install openai python-dotenv scikit-learn numpy pandas

```
Set Up Environment Variables:
```
Create a file named .env in the root of the project directory. This file will store your secret API keys and local configuration.
# .env file

# Required for using OpenAI models
OPENAI_API_KEY="your-openai-api-key"

# Optional: If you are running a local Ollama server on a different address
OLLAMA_BASE_URL="http://localhost:11434/v1"
```

(Optional) Set Up Local LLMs with Ollama:
If you plan to use local models, ensure you have Ollama installed and have downloaded the models you intend to use.
```
# Example: Pulling Llama3 and Qwen2 models
ollama pull llama3:8b
ollama pull qwen3:8b
```

## 6. Usage
The script is executed from the command line. You must provide an input file and specify at least one LLM agent to use.
```
Command-Line Arguments
--input / -i: (Required) Path to the input JSON dataset to be relabeled.
--openai: A space-separated list of OpenAI model names to use (e.g., gpt-4o-mini gpt-4-turbo).
--local: A space-separated list of local/Ollama model names to use (e.g., llama3:8b qwen2:7b).
--voting-scheme: The voting method to use. Choices are majority (default) or union.
First, ensure you have defined any custom prompts you want to use in the PROMPT_CONFIG dictionary inside config.py.
```

**Scenario 1:** Use a default system prompt for all models.
Simply provide the model names without any prompt keys. The system will automatically use the "default" prompt for all of them.
```
python main.py -i path/to/your/data.json --openai gpt-4o-mini gpt-4-turbo --local llama3:8b
```

**Scenario 2:** Use a different prompt for each model.
Append a colon and the prompt key (from PROMPT_CONFIG) to each model name.
```
# This uses the 'strict' prompt for gpt-4o-mini and the 'summary_focused' prompt for llama3
python main.py -i path/to/data.json --openai gpt-4o-mini:strict --local llama3:8b:summary_focused
```
**Scenario 3:** Use different prompts for the same model.
Just list the same model multiple times with different prompt keys. The application will create a separate, unique "agent" for each entry.
```
# This runs gpt-4o-mini twice: once with a 'creative' prompt and once with a 'strict' prompt.
# Both will participate in the voting process as independent agents.
python main.py -i path/to/data.json --openai gpt-4o-mini:creative gpt-4o-mini:strict
```

**Scenario 4:** Use same prompt for one model multiple times.
Just list the same model multiple times with same prompt keys. The application will create a separate, unique "agent" for each entry.
```
# This runs gpt-4o-mini twice with the same prompt. 
# Both will participate in the voting process as independent agents.
python main.py -i path/to/data.json --openai gpt-4o-mini:summary_focused gpt-4o-mini:summary_focused
```

## 7. Output Files

After a successful run, the framework generates several files in the output/, tmp/logs/, and tmp/reports/ directories.
Final Relabeled Data is stored in the (/output) directory. 



