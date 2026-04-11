# AutoStream: Intelligent Video Automation Agent

AutoStream is a production-grade AI agent designed to handle customer inquiries about video automation services and automate the lead generation process. Built using LangGraph, it features a robust state machine that handles task-switching between a knowledge base (RAG) and a lead-capture tool.

## Key Features

* **Intelligent Intent Classification:** Automatically routes user queries into four categories: Greeting, Inquiry, Lead Generation, or Out-of-Scope.
* **Knowledge Retrieval (RAG):** Uses Pinecone and Gemini 2.0 Flash to provide accurate answers about pricing, plans, and features.
* **Stateful Form Filling:** A sequential lead-capture flow that collects user information (Name, Email, Platform) without losing context.
* **Domain Guardrails:** Respectfully refuses to answer non-business related questions (e.g., "What is Twitter?" or cooking recipes) to maintain professional focus.
* **High-Speed Logic:** Implements short-circuit Python logic for greetings to ensure sub-second response times.
* **Cloud Native:** Fully containerized with Docker and deployed on Microsoft Azure App Service.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Orchestration** | LangGraph (Stateful Multi-turn Logic) |
| **LLM** | Google Gemini 2.0 Flash (via OpenRouter) |
| **Vector Database** | Pinecone (Serverless) |
| **UI / Frontend** | Streamlit |
| **Cloud / Infrastructure** | Microsoft Azure App Service |
| **Containerization** | Docker |
| **Embeddings** | OpenAI text-embedding-3-small |

## Project Structure

```text
autostream-agent/
├── app.py              # Streamlit Frontend UI & Session Management
├── agent_logic.py      # LangGraph state machine, RAG & Tool logic
├── Dockerfile          # Container configuration for Azure deployment
├── requirements.txt    # Python dependencies
└── .env                # API Keys (OpenRouter, Pinecone)
```

## Setup and Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/autostream-agent.git
   cd autostream-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a .env file in the root directory:
   ```text
   OPENROUTER_API_KEY=your_openrouter_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX=your_index_name
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Dockerization

To build the image locally and test the container:

```bash
# Build the image
docker build -t playwrightregistryaman.azurecr.io/autostream-agent:v1 .

# Run the container
docker run -p 8501:8501 --env-file .env playwrightregistryaman.azurecr.io/autostream-agent:v1
```

## Azure Deployment Steps

This project is deployed to Azure App Service using Azure Container Registry (ACR).

1. **Login to Azure and ACR:**
   ```bash
   az login
   az acr login --name playwrightregistryaman
   ```
2. **Push the Image:**
   ```bash
   docker push playwrightregistryaman.azurecr.io/autostream-agent:v1
   ```
3. **App Service Configuration:**
   * Deploy as a Web App for Containers.
   * Set WEBSITES_PORT to 8501 in the Azure Portal Configuration.
   * Add your .env variables to the App Settings in Azure.

## Demo Questions to Try

* **Greeting:** "Hello!"
* **Knowledge:** "What features are included in the Pro plan?"
* **Guardrail:** "What is Twitter?" (Expected: Respectful refusal)
* **Lead Gen:** "I want to sign up." (Follow the multi-turn name/email prompts)

