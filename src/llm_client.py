import openai
import os
from openai import OpenAI 
from src.config import open_api_key
from src.config import model

openai.api_key = open_api_key
# model = model


def generate_response(user_input: str, context: str) -> str:
    """
    Generate a response using OpenAI GPT-4o model, combining user input and vector store context.

    Args:
        user_input (str): The user's question or message.
        context (str): Relevant context retrieved from the vector store.

    Returns:
        str: The generated response from OpenAI.
    """
    try:
        # Load the OpenAI API key from environment variable
        if not openai.api_key:
            raise ValueError("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")

        # Define the maximum number of tokens for the prompt
        max_tokens = 4096 - 500  # Reserve 500 tokens for the response

        # Truncate the context if it is too large
        truncated_context = context[:max_tokens]

        # Combine the user input and retrieved context into a structured prompt
        prompt = f"Context: {truncated_context}\n\nQuestion: {user_input}"


        print("about to call openai")  # Debugging
        client = OpenAI(api_key=openai.api_key)
        system_prompt = "You are a helpful assistant. You can combine data from the prompt that is provided with other data from the web or any other resourses you have available to you to generate an approopriate response for the user"
        # system_prompt = "You are a helpful assistant. You can only use data provided to you from the vector store. Do no combine data from the prompt that is provided with other data from the web or any other resourses you have available to you to generate an approopriate response for the user"

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract and return the assistant's reply
        return completion.choices[0].message.content


    except openai.error.OpenAIError as e:
        # Handle OpenAI API errors
        error_message = f"OpenAI API error: {str(e)}"
        print(error_message)
        return error_message

    except Exception as e:
        # Handle any other errors
        error_message = f"Error: {str(e)}"
        print(error_message)
        return error_message