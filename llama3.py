import boto3
import json

# Define the prompt data
prompt_data = """
Write c program to check number is even or odd.
"""

# Create a Bedrock client using boto3
bedrock = boto3.client(service_name="bedrock-runtime")

# Create the payload for the model invocation
payload = {
    "prompt": prompt_data,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

# Convert the payload to a JSON string
body = json.dumps(payload)

# Specify the model ID
model_id = "meta.llama3-70b-instruct-v1:0"

# Invoke the model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse the response body and extract the generated text
response_body = json.loads(response.get("body").read())
response_text = response_body['generation']

# Print the generated text
print(response_text)
