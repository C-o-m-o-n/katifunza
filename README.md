# Katifunza AI

Katifunza AI is an intelligent system designed to help explain and summarize information from various documents, such as research papers, constitutions, and software documentation. It uses AI agents to source and explain information, making it accessible to a broader audience.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Features

- **PDF Processing**: Load and process PDF documents to extract relevant information.
- **AI Agents**: Use agents to source and explain information.
- **Vector Store**: Store document embeddings for efficient retrieval.
- **Django REST API**: Provide endpoints to interact with the AI system.

## Installation

### Prerequisites

- Python 3.8+
- Django 3.2+
- Django Rest Framework
- Cohere API Key (you can use any other model like gemini)
- and a bunch more AI packages. Just install everything in the requirements.txt

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/katifunza.git
    cd katifunza
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:

    Create a `.env` file in the project root and add your Cohere API key:

    ```plaintext
    #if you used chatGPT: OPENAI_API_KEY instead
    COHERE_API_KEY=your_cohere_api_key
    ```

5. **Run migrations**:

    ```bash
    python manage.py migrate
    ```

6. **Start the Django development server**:

    ```bash
    python manage.py runserver
    ```

## Usage

### Processing and Storing Document Embeddings

1. **Load and process the PDF document**:

    Ensure the PDF document is placed in the appropriate directory.

    ```python
    from data_magic.data_job import PreProcess
    load_dotenv()  # take environment variables from .env.
    preprocessor = PreProcess()

    vector_store = preprocessor.store_embeddings("path/to/constitution.pdf")
    ```

### Run the crew against the processed and stored embeddings
2. **retreave the embeddings and pass hem to the crew**:
    ```python
    if __name__ == "__main__": 
      print("## Welcome to Katifunza AI")
      print("-------------------------------")
      question = input(dedent("""Enter your question: """))

      custom_crew = OurCrew(question, vector_store)
      result = custom_crew.run()
      print("########################\n")
      print(result)
    ```

### API Endpoints

#### Prompt Agents

- **URL**: `/app/prompt/`
- **Method**: POST
- **Payload**:

    ```json
    {
      "question": "What are the rights of the people?"
    }
    ```

- **Response**:

    ```json
    {
      "result": "..."
    }
    ```

## Contributing

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License.
