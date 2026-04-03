import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# OpenAI Configuration
open_api_key = os.getenv("OPENAI_API_KEY")

# Pinecone Configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Model     # This is the model that will be used for the chatbot
model = os.getenv("MODEL")


# General App Configuration
# APP_NAME = "Vector Store Bot"
# DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Ensure critical variables are set
if not open_api_key:
    raise ValueError("Environment variable 'open_api_key' is not set!")
if not pinecone_api_key:
    raise ValueError("Environment variable 'pinecone_api_key' is not set!")
if not pinecone_environment:
    raise ValueError("Environment variable 'pinecone_environment' is not set!")
if not model:
    raise ValueError("Environment variable 'model' is not set!")

