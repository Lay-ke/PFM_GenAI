"""
Lambda function for handling Bedrock Agent queries.
Processes incoming queries and manages agent interactions.
"""
import json
import boto3
import logging
import uuid
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

# Hardcoded configuration
BEDROCK_AGENT_ID = 'GMJGK6RO4S'
BEDROCK_AGENT_ALIAS_ID = 'TSTALIASID'

def get_cors_headers():
    """Get CORS headers for response."""
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Amz-Date, X-Api-Key',
        'Access-Control-Max-Age': '300'
    }

def lambda_handler(event, context):
    """
    Main Lambda handler for processing Bedrock Agent queries.
    """
    cors_headers = get_cors_headers()
    
    try:
        # Handle CORS preflight requests
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': ''
            }
        
        # Log request details for debugging
        logger.info(f"Lambda invocation - Request ID: {context.aws_request_id}")
        logger.info(f"Event: {json.dumps(event)}")
        
        # Parse request body
        try:
            body_str = event.get('body', '{}')
            if isinstance(body_str, str):
                body = json.loads(body_str)
            else:
                body = body_str
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request body: {e}")
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Invalid request format',
                    'message': 'Request body must be valid JSON'
                })
            }
        
        # Extract and validate parameters
        query = body.get('query', '').strip()
        session_id = body.get('sessionId', '').strip()
        
        if not query:
            logger.warning("Empty query received")
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Missing query parameter',
                    'message': 'Please provide a question about equipment maintenance'
                })
            }
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")
        
        logger.info(f"Processing query: {query[:100]}... (Session: {session_id})")
        logger.info(f"Using Agent ID: {BEDROCK_AGENT_ID}, Alias: {BEDROCK_AGENT_ALIAS_ID}")
        
        # Invoke Bedrock Agent
        try:
            response = bedrock_agent_runtime.invoke_agent(
                agentId=BEDROCK_AGENT_ID,
                agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                sessionId=session_id,
                inputText=query
            )
            
            logger.info("Bedrock agent invoked successfully")
            
            # Process the streaming response
            full_response = ""
            chunk_count = 0
            
            for event in response['completion']:
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count}: {event.keys()}")
                
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        chunk_text = chunk['bytes'].decode('utf-8')
                        full_response += chunk_text
                        logger.info(f"Added chunk text: {chunk_text[:100]}...")
            
            logger.info(f"Agent response complete: {len(full_response)} characters, {chunk_count} chunks")
            
            if full_response.strip():
                return {
                    'statusCode': 200,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'response': full_response.strip(),
                        'sessionId': session_id,
                        'timestamp': context.aws_request_id
                    })
                }
            else:
                logger.warning("Empty response from Bedrock Agent")
                return {
                    'statusCode': 200,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'response': "I received your query but couldn't generate a response. Please try rephrasing your question.",
                        'sessionId': session_id,
                        'timestamp': context.aws_request_id
                    })
                }
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            logger.error(f"AWS ClientError: {error_code} - {error_message}")
            logger.error(f"Full error response: {e.response}")
            
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': f'Bedrock Agent Error: {error_code}',
                    'message': error_message,
                    'details': f'Agent ID: {BEDROCK_AGENT_ID}, Alias: {BEDROCK_AGENT_ALIAS_ID}'
                })
            }
            
    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }