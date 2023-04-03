import openai
import re

# Set up your OpenAI API key
openai.api_key = "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"

def ag(text):
    # Remove any non-ASCII characters from the text
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    
    # Use the OpenAI API to correct the text
    response = openai.create(
        engine="davinci",
        prompt="Please correct the following text:\n" + text + "\n\nSuggested correction:",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    # Get the suggested correction from the OpenAI response
    suggestion = response.choices[0].text.strip()
    
    # Return the corrected text
    return suggestion
