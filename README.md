## Summary ##
[ISARIC - Github](https://github.com/ISARICResearch)

This repo is forked from VERTEX, ISARIC's web-based application designed to present graphs and tables based on relevant research questions that need to be quickly answered during an outbreak. VERTEX uses reproducible analytical pipelines. Currently, we have pipelines for identifying the spectrum of clinical features in a disease and determining risk factors for patient outcomes. New questions will be added by the ISARIC team and the wider scientific community, enabling the creation and sharing of new pipelines. See VERTEX Tutorial to learn how to use the tool. Additionally, see VERTEX GitHub To download the code that can be used for ARC-structured data visualization.

This fork adds some exploratory AI features to Vertex's existing capabilities. It allows for easy demos and code changes without impacting the researchers regularly using Vertex. After an initial exploratory phase, the AI features may be deployed in the original repository.

## Usage ##
* VERTEX tutorial can be found at https://isaricresearch.github.io/Training/vertex_starting
* Follow the instructions in the tutorial, subsituting the clone command with `git clone https://github.com/ed-phillips/VERTEX-AI.git`
* Groq
  * The AI component currently uses the free tier of the Groq API for LLM requests.
  * You can get your own API key by creating an account and generating a key at console.groq.com/keys
  * Once this is done, create a `.env` file at the top level of the repository, with the token specified as follows: `GROQ_API_KEY="your_token_here"`
* Then run the `descriptive_dashboard.py` file, and navigate to the "Clinical Presentation: Day 0..." panel to test out the new AI insights feature.

## Requirements ##
python3.10.15
