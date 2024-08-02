import boto3
import json

# Define the input text and configuration for text generation
input_text = "Explain MLOps to a non-IT person in simple terms."

# Configuration for text generation
text_generation_config = {
    "inputText": input_text,
    "textGenerationConfig": {
        "maxTokenCount": 4096,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1
    }
}

# Create a Bedrock client using boto3
bedrock_client = boto3.client('bedrock-runtime')

# Model ID for Amazon Titan
model_id = "amazon.titan-text-lite-v1"

# Convert the payload to a JSON string
body = json.dumps(text_generation_config)

try:
    # Invoke the model
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body
    )
    
    # Read and parse the response
    response_body = json.loads(response['body'].read())
    
    # Check if 'generatedText' is in the response
    if 'generatedText' in response_body:
        response_text = response_body['generatedText']
        print("Generated Text:", response_text)
    else:
        print("No text generated. Full response:", response_body)

except Exception as e:
    # Print any exception encountered
    print("An error occurred:", str(e))
