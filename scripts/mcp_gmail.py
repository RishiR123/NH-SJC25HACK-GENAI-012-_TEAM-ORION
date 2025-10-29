import os
import json
import base64
import traceback
import pickle
import psycopg2
import psycopg2.pool
import requests # NEW: For Microsoft Graph API calls
import msal # NEW: For Microsoft Authentication Library (MSAL)
from datetime import datetime, timedelta # NEW: For Outlook subscription expiry
from dotenv import load_dotenv
from flask import Flask, request, jsonify, redirect, Response
from flask_cors import CORS
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from googleapiclient.errors import HttpError
import logging
# ==============================================================================
# --- CONFIGURATION & INITIALIZATION ---
# ==============================================================================
load_dotenv()


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Database Configuration ---
DB_NAME = os.getenv("DB_NAME", "user_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1269")
DB_HOST = os.getenv("DB_HOST", "localhost")

# --- Google OAuth Configuration (Gmail) ---
GMAIL_SCOPES = [
    'openid',
    'https://mail.google.com/',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]
GMAIL_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_JSON", "credentials.json")
GMAIL_INTEGRATION_KEY = 'gmail_credentials'
REDIRECT_URI = os.getenv("MCP_REDIRECT_URI", "https://salesos.orionac.in/settings/oauth2callback") # Shared callback URL

# --- Microsoft OAuth Configuration (Outlook) ---
# --- Microsoft OAuth Configuration (Outlook) ---

# LIST 1: The Master List (for reference)
# Contains every permission we will ever ask for.
# --- Microsoft OAuth Configuration (Outlook) ---

# LIST 1: The Master List (for reference)
# --- Microsoft OAuth Configuration (Outlook) ---

# LIST 1: The Master List (for reference)
# Scopes used for the user-facing login URL (NO reserved scopes)
OUTLOOK_AUTH_URL_SCOPES = [
    'User.Read',
    'Mail.ReadWrite',
    'Mail.Send',
    'MailboxSettings.Read'
]

# Scopes used for token exchange (backend)
OUTLOOK_TOKEN_SCOPES = [
    'openid',
    'offline_access',
    'email',
    'User.Read',
    'Mail.ReadWrite',
    'Mail.Send',
    'MailboxSettings.Read'
]

# Scopes used for API calls (token refreshes)
OUTLOOK_API_SCOPES = [
    'email',
    'User.Read',
    'Mail.ReadWrite',
    'Mail.Send',
    'MailboxSettings.Read'
]



# ... rest of your Outlook configuration ...

# ... rest of your Outlook configuration ...
OUTLOOK_CLIENT_ID =  "493ceaa5-e0fe-4840-9c2a-d1c968f44852"
OUTLOOK_CLIENT_SECRET = "z_98Q~aA6FbQ~d0AeRSDtIoyVWXJMuMDhSe5rawp"  
OUTLOOK_INTEGRATION_KEY = 'outlook'
OUTLOOK_AUTHORITY = "https://login.microsoftonline.com/common" # Common endpoint for multi-tenant/personal accounts

RESERVED_SCOPES = ['openid', 'offline_access', 'profile'] 


# --- MSAL Confidential Client Application ---
# NOTE: Uses the Outlook Client ID and Secret

# --- Microsoft OAuth Configuration (Outlook) ---
# ... (OUTLOOK_CLIENT_ID and OUTLOOK_CLIENT_SECRET)

# ADD THIS LINE FOR DEBUGGIN

print(f"DEBUG: MSAL Secret Loaded: {OUTLOOK_CLIENT_SECRET[:5]}...{OUTLOOK_CLIENT_SECRET[-5:]}") 


# --- MSAL Confidential Client Application ---
msal_app = msal.ConfidentialClientApplication(
    OUTLOOK_CLIENT_ID,
    authority=OUTLOOK_AUTHORITY,
    client_credential=OUTLOOK_CLIENT_SECRET
)

# ==============================================================================
# --- DATABASE CONNECTION POOL ---
# ==============================================================================
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=10,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
    )
    print("MCP Server: Database connection pool created.")
except psycopg2.OperationalError as e:
    print(f"MCP Server: FATAL - Could not create database pool. Error: {e}")
    db_pool = None

def execute_db_query(query, params=(), fetch_one=False, is_write=False):
    """Executes a database query using a connection from the pool."""
    if not db_pool:
        raise ConnectionError("Database pool is not available.")
    conn = None
    try:
        conn = db_pool.getconn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if is_write:
                    return cur.rowcount
                if fetch_one:
                    return cur.fetchone()
                return cur.fetchall()
    except psycopg2.Error as e:
        print(f"MCP Server: Database query failed: {e}")
        raise e
    finally:
        if conn:
            db_pool.putconn(conn)

# ==============================================================================
# --- GMAIL HELPER AND ACTION FUNCTIONS ---
# ==============================================================================

def get_gmail_credentials(user_id):
    """Fetches and deserializes Gmail credentials, refreshing token if necessary."""
    query = "SELECT credentials FROM crm_schema.user_integrations WHERE user_id = %s AND integration_key = %s"
    result = execute_db_query(query, (int(user_id), GMAIL_INTEGRATION_KEY), fetch_one=True)
    if not result or not result[0]:
        raise ValueError("No Gmail credentials found for this user.")
    
    base64_creds = result[0]
    serialized_creds = base64.b64decode(base64_creds)
    creds = pickle.loads(serialized_creds)
    
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Save refreshed credentials back to DB
        serialized_creds = pickle.dumps(creds)
        base64_encoded_creds = base64.b64encode(serialized_creds).decode('utf-8')
        _save_credentials(user_id, GMAIL_INTEGRATION_KEY, base64_encoded_creds)
        
    return creds

def send_gmail_email(user_id, recipient, subject, body):
    """Uses stored credentials to send an email via the Gmail API with detailed error handling."""
    try:
        creds = get_gmail_credentials(user_id)
        service = build('gmail', 'v1', credentials=creds)

        message = MIMEText(body)
        message['to'] = recipient
        message['subject'] = subject
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {'raw': encoded_message}
        sent_message = service.users().messages().send(userId='me', body=create_message).execute()

        print(f"MCP Worker: Gmail Email sent. Message ID: {sent_message['id']}")
        return {"status": "ok", "message": "Gmail Email sent successfully.", "message_id": sent_message['id']}

    except HttpError as error:
        # This is the critical change: Specifically catch Google API HTTP errors.
        print(f"MCP Worker: A Google API HttpError occurred for user {user_id}: {error}")
        try:
            # The error response from Google is JSON with valuable details.
            error_details = json.loads(error.content.decode('utf-8'))
            error_message = error_details.get("error", {}).get("message", "An unknown error occurred.")
            status_code = error_details.get("error", {}).get("code", 500)
            # Now we can return the REAL error message from Google.
            return {"status": "error", "message": f"Google API Error: {error_message}", "code": status_code}
        except (json.JSONDecodeError, AttributeError):
            # Fallback if the error content isn't the expected JSON format.
            return {"status": "error", "message": f"Google API HttpError: {str(error)}"}

    except Exception as e:
        # General fallback for other errors (e.g., credential loading failed).
        print(f"MCP Worker: A non-API error occurred sending Gmail for user {user_id}: {e}")
        traceback.print_exc() # Print full traceback to console for debugging
        return {"status": "error", "message": str(e)}

# ==============================================================================
# --- OUTLOOK HELPER AND ACTION FUNCTIONS ---
# ==============================================================================

def _save_credentials(user_id, integration_key, base64_creds):
    """Helper to save the refreshed MSAL token cache or Google creds."""
    query = """
        UPDATE crm_schema.user_integrations
        SET credentials = %s, updated_at = NOW()
        WHERE user_id = %s AND integration_key = %s;
    """
    execute_db_query(query, (base64_creds, int(user_id), integration_key), is_write=True)

# DELETE THIS ENTIRE FUNCTION


def send_outlook_email(user_id, recipient, subject, body, content_type="Text"):
    """Sends an email using the Microsoft Graph API (Mail.Send)."""
    try:
        access_token = get_outlook_token(user_id)
        
        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        
        email_payload = {
            "message": {
                "subject": subject,
                "body": {"contentType": content_type, "content": body},
                "toRecipients": [{"emailAddress": {"address": recipient}}]
            },
            "saveToSentItems": "true"
        }

        graph_url = "https://graph.microsoft.com/v1.0/me/sendMail"
        response = requests.post(graph_url, headers=headers, json=email_payload)
        response.raise_for_status()
        
        return {"status": "ok", "message": "Outlook Email sent successfully."}
        
    except requests.exceptions.HTTPError as http_err:
        error_details = http_err.response.json()
        error_message = error_details.get("error", {}).get("message", "Unknown Graph API error")
        return {"status": "error", "message": f"Graph API Error: {error_message}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_gmail_emails(user_id, query, top=10):
    """Fetches a list of emails from a user's Gmail account (Mail.Read)."""
    try:
        # --- NEW LOGS ---
        logging.warning(f"[search_gmail_emails] User: {user_id}. Query: '{query}'. Top: {top}")
        
        creds = get_gmail_credentials(user_id)
        service = build('gmail', 'v1', credentials=creds)
        
        logging.warning(f"[search_gmail_emails] User: {user_id}. Credentials loaded. Calling Google API...")
        # --- END NEW LOGS ---
        
        # List messages matching the query
        results = service.users().messages().list(userId='me', q=query, maxResults=top).execute()
        messages_summary = results.get('messages', [])
        
        # --- NEW LOG ---
        logging.warning(f"[search_gmail_emails] User: {user_id}. Google API returned {len(messages_summary)} messages.")
        # --- END NEW LOG ---
        
        if not messages_summary:
            logging.warning(f"[search_gmail_emails] User: {user_id}. No messages found for query '{query}'. Returning empty list.")
            return {"status": "ok", "emails": []}
        
        parsed_emails = []
        logging.warning(f"[search_gmail_emails] User: {user_id}. Parsing {len(messages_summary)} messages...")
        
        # We must fetch each message individually to get metadata
        for msg_summary in messages_summary:
            msg_id = msg_summary['id']
            msg = service.users().messages().get(
                userId='me', 
                id=msg_id, 
                format='metadata', 
                metadataHeaders=['Subject', 'From', 'Date']
            ).execute()
            
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
            received = next((h['value'] for h in headers if h['name'].lower() == 'date'), None)

            parsed_emails.append({
                'id': msg['id'],
                'subject': subject,
                'from': sender,
                'received': received,
                'is_read': 'UNREAD' not in msg.get('labelIds', []),
                'body_preview': msg['snippet']
            })
        
        logging.warning(f"[search_gmail_emails] User: {user_id}. Successfully parsed {len(parsed_emails)} emails.")
        return {"status": "ok", "emails": parsed_emails}

    except HttpError as error:
        logging.error(f"MCP Worker: A Google API HttpError occurred for user {user_id}: {error}", exc_info=True)
        try:
            error_details = json.loads(error.content.decode('utf-8'))
            error_message = error_details.get("error", {}).get("message", "An unknown error occurred.")
            status_code = error_details.get("error", {}).get("code", 500)
            return {"status": "error", "message": f"Google API Error: {error_message}", "code": status_code}
        except (json.JSONDecodeError, AttributeError):
            return {"status": "error", "message": f"Google API HttpError: {str(error)}"}
    except Exception as e:
        logging.error(f"MCP Worker: A non-API error occurred searching Gmail for user {user_id}: {e}", exc_info=True)
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def get_gmail_single_email(user_id, message_id):
    """Fetches the full content of a single email (Mail.Read)."""
    try:
        creds = get_gmail_credentials(user_id)
        service = build('gmail', 'v1', credentials=creds)
        
        msg = service.users().messages().get(userId='me', id=message_id, format='full').execute()
        
        payload = msg['payload']
        headers = payload['headers']
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
        
        body_content = ""
        body_type = "text/plain" # Default
        
        def decode_body(part):
            """Helper to decode a single email part."""
            data = part['body'].get('data')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8')
            return ""

        if 'parts' in payload:
            # Multipart message
            # First, try to find plain text
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    body_content = decode_body(part)
                    body_type = 'text/plain'
                    break
            
            # If no plain text, find HTML
            if not body_content:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/html':
                        body_content = decode_body(part)
                        body_type = 'text/html'
                        break
        
        elif 'body' in payload:
            # Single-part message
            body_content = decode_body(payload)
            body_type = payload.get('mimeType', 'text/plain')
        
        return {
            "status": "ok", 
            "id": msg['id'],
            "subject": subject,
            "body_content": body_content,
            "body_type": body_type,
            "sender": sender
        }
        
    except HttpError as error:
        # ... (Same error handling as above)
        print(f"MCP Worker: A Google API HttpError occurred for user {user_id}: {error}")
        try:
            error_details = json.loads(error.content.decode('utf-8'))
            error_message = error_details.get("error", {}).get("message", "An unknown error occurred.")
            status_code = error_details.get("error", {}).get("code", 500)
            return {"status": "error", "message": f"Google API Error: {error_message}", "code": status_code}
        except (json.JSONDecodeError, AttributeError):
            return {"status": "error", "message": f"Google API HttpError: {str(error)}"}
    except Exception as e:
        print(f"MCP Worker: A non-API error occurred reading Gmail for user {user_id}: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def reply_to_gmail_email(user_id, message_id, reply_body):
    """Replies to a specific email message (Mail.Send)."""
    try:
        creds = get_gmail_credentials(user_id)
        service = build('gmail', 'v1', credentials=creds)
        
        # 1. Get original message for headers and threadId
        original_msg = service.users().messages().get(
            userId='me', 
            id=message_id, 
            format='metadata', 
            metadataHeaders=['Subject', 'From', 'Message-ID', 'References']
        ).execute()
        
        thread_id = original_msg['threadId']
        headers = original_msg['payload']['headers']
        
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
        if not subject.lower().startswith('re:'):
            subject = f"Re: {subject}"
            
        original_from = next((h['value'] for h in headers if h['name'].lower() == 'from'), None)
        if not original_from:
            raise ValueError("Could not find 'From' header in original email to reply to.")
            
        original_msg_id = next((h['value'] for h in headers if h['name'].lower() == 'message-id'), None)
        references = next((h['value'] for h in headers if h['name'].lower() == 'references'), None)

        # 2. Create the reply MIME message
        message = MIMEText(reply_body)
        message['to'] = original_from
        message['subject'] = subject
        
        if original_msg_id:
            message['In-Reply-To'] = original_msg_id
        if references:
            message['References'] = f"{references} {original_msg_id}"
        elif original_msg_id:
            message['References'] = original_msg_id

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # 3. Send the reply using the threadId
        create_message = {
            'raw': encoded_message,
            'threadId': thread_id
        }
        
        sent_message = service.users().messages().send(userId='me', body=create_message).execute()
        
        return {"status": "ok", "message": f"Reply to message {message_id} sent successfully.", "message_id": sent_message['id']}

    except HttpError as error:
        # ... (Same error handling as above)
        print(f"MCP Worker: A Google API HttpError occurred for user {user_id}: {error}")
        try:
            error_details = json.loads(error.content.decode('utf-8'))
            error_message = error_details.get("error", {}).get("message", "An unknown error occurred.")
            status_code = error_details.get("error", {}).get("code", 500)
            return {"status": "error", "message": f"Google API Error: {error_message}", "code": status_code}
        except (json.JSONDecodeError, AttributeError):
            return {"status": "error", "message": f"Google API HttpError: {str(error)}"}
    except Exception as e:
        print(f"MCP Worker: A non-API error occurred replying to Gmail for user {user_id}: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# --- NEW GMAIL ACTION ENDPOINTS ---

@app.route("/actions/get_gmail_emails", methods=['GET'])
def handle_get_gmail_emails():
    """API endpoint to get a list of Gmail emails."""
    try:
        user_id = request.args.get('user_id')
        query = request.args.get('query', '') # 'query' from app.py tool
        top = int(request.args.get('top', 10)) # 'top' from app.py tool
        
        if not user_id: 
            return jsonify({"status": "error", "message": "user_id is required"}), 400
            
        # This call is what is crashing
        result = search_gmail_emails(user_id, query, top)
        
        # This return is only reached if 'result' is successful
        return jsonify(result), (500 if result.get("status") == "error" else 200)

    except Exception as e:
        # --- THIS BLOCK CATCHES THE CRASH ---
        # If search_gmail_emails (or anything above) fails, this will run
        print(f"!!! UNHANDLED EXCEPTION in handle_get_gmail_emails: {e}")
        traceback.print_exc() # Print the full error to the microservice log
        
        # Return a clean JSON error instead of crashing
        return jsonify({
            "status": "error", 
            "message": f"Microservice crash: {str(e)}"
        }), 500

@app.route("/actions/get_gmail_single_email", methods=['GET'])
def handle_get_gmail_single_email():
    """API endpoint to get the full content of one Gmail email."""
    user_id = request.args.get('user_id')
    message_id = request.args.get('message_id')
    
    if not all([user_id, message_id]): 
        return jsonify({"status": "error", "message": "user_id and message_id are required"}), 400
        
    result = get_gmail_single_email(user_id, message_id)
    return jsonify(result), (500 if result.get("status") == "error" else 200)

@app.route("/actions/reply_gmail_email", methods=['POST'])
def handle_reply_gmail_email():
    """API endpoint to send a reply to an existing Gmail email."""
    data = request.get_json()
    if not all([data, data.get('user_id'), data.get('message_id'), data.get('reply_body')]):
        return jsonify({"status": "error", "message": "Missing required parameters: user_id, message_id, reply_body"}), 400
        
    result = reply_to_gmail_email(data['user_id'], data['message_id'], data['reply_body'])
    return jsonify(result), (500 if result.get("status") == "error" else 200)


# --- NEW OUTLOOK ACTION ENDPOINTS ---

def get_outlook_emails(user_id, folder_id='inbox', top=20):
    """Fetches a list of emails from a user's specified folder (Mail.ReadWrite)."""
    try:
        access_token = get_outlook_token(user_id)
        
        headers = {'Authorization': f'Bearer {access_token}'}
        
        query_params = f"$top={top}&$select=id,subject,from,receivedDateTime,isRead,bodyPreview&$orderby=receivedDateTime desc"
        graph_url = f"https://graph.microsoft.com/v1.0/me/mailFolders/{folder_id}/messages?{query_params}"
        
        response = requests.get(graph_url, headers=headers)
        response.raise_for_status()
        
        messages = response.json().get('value', [])
        
        parsed_emails = [{
            'id': msg['id'],
            'subject': msg['subject'],
            'from': msg['from']['emailAddress']['address'],
            'received': msg['receivedDateTime'],
            'is_read': msg['isRead'],
            'body_preview': msg['bodyPreview']
        } for msg in messages]
        
        return {"status": "ok", "emails": parsed_emails}
        
    except requests.exceptions.HTTPError as http_err:
        error_details = http_err.response.json()
        error_message = error_details.get("error", {}).get("message", "Unknown Graph API error")
        return {"status": "error", "message": f"Graph API Error: {error_message}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_outlook_single_email(user_id, message_id):
    """Fetches the full content of a single email (Mail.ReadWrite)."""
    try:
        access_token = get_outlook_token(user_id)
        
        headers = {'Authorization': f'Bearer {access_token}'}
        graph_url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}"
        
        response = requests.get(graph_url, headers=headers)
        response.raise_for_status()
        
        message = response.json()
        
        return {
            "status": "ok", 
            "id": message['id'],
            "subject": message['subject'],
            "body_content": message['body']['content'],
            "body_type": message['body']['contentType'],
            "sender": message['from']['emailAddress']['address']
        }
        
    except requests.exceptions.HTTPError as http_err:
        error_details = http_err.response.json()
        error_message = error_details.get("error", {}).get("message", "Unknown Graph API error")
        return {"status": "error", "message": f"Graph API Error: {error_message}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def reply_to_outlook_email(user_id, message_id, reply_body, content_type="Text", save_to_sent=True):
    """Replies to a specific email message (Mail.ReadWrite)."""
    try:
        access_token = get_outlook_token(user_id)
        
        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        
        reply_payload = {
            "message": {
                "body": {"contentType": content_type, "content": reply_body}
            },
            "saveToSentItems": save_to_sent
        }

        graph_url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/reply"
        response = requests.post(graph_url, headers=headers, json=reply_payload)
        response.raise_for_status() 
        
        return {"status": "ok", "message": f"Reply to message {message_id} sent successfully."}
        
    except requests.exceptions.HTTPError as http_err:
        error_details = http_err.response.json()
        error_message = error_details.get("error", {}).get("message", "Unknown Graph API error")
        return {"status": "error", "message": f"Graph API Error: {error_message}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def delete_outlook_email(user_id, message_id):
    """Deletes a specific email message (Mail.ReadWrite)."""
    try:
        access_token = get_outlook_token(user_id)
        
        headers = {'Authorization': f'Bearer {access_token}'}
        graph_url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}"
        
        response = requests.delete(graph_url, headers=headers)
        response.raise_for_status() # Expects 204 No Content
        
        return {"status": "ok", "message": f"Message {message_id} deleted successfully."}
        
    except requests.exceptions.HTTPError as http_err:
        error_details = http_err.response.json()
        error_message = error_details.get("error", {}).get("message", "Unknown Graph API error")
        return {"status": "error", "message": f"Graph API Error: {error_message}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================================================================
# --- OAUTH2 WEB FLOW ENDPOINTS (UNIFIED) ---
# ==============================================================================

@app.route("/connect/gmail", methods=['GET'])
def connect_gmail():
    """Step 1: Generate Google authorization URL, tagging state with provider."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"}), 400

    try:
        flow = Flow.from_client_secrets_file(
            GMAIL_CREDENTIALS_FILE, scopes=GMAIL_SCOPES, redirect_uri=REDIRECT_URI
        )
        # Modified state to include provider prefix
        state_with_provider = f"gmail:{user_id}"
        authorization_url, state = flow.authorization_url(
            access_type='offline', prompt='consent', state=state_with_provider
        )
        return jsonify({"status": "ok", "authorization_url": authorization_url})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": f"'{GMAIL_CREDENTIALS_FILE}' not found."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# --- TEMPORARILY DELETE THE GLOBAL MSAL_FLOW_CACHE = {} LINE ---
# This is no longer needed.

def get_outlook_token(user_id):
    """Fetches stored Outlook token, uses refresh_token to get a new access_token."""
    query = "SELECT credentials FROM crm_schema.user_integrations WHERE user_id = %s AND integration_key = %s"
    result = execute_db_query(query, (int(user_id), OUTLOOK_INTEGRATION_KEY), fetch_one=True)
    if not result or not result[0]:
        raise ValueError("No Outlook credentials found for this user. Please connect Outlook.")
    
    # Decode the stored JSON data
    base64_creds = result[0]
    token_data_json = base64.b64decode(base64_creds).decode('utf-8')
    token_data = json.loads(token_data_json)

    # Use the refresh token to acquire a new access token
    result = msal_app.acquire_token_by_refresh_token(
        refresh_token=token_data['refresh_token'],
        scopes=OUTLOOK_API_SCOPES  # Use the list that includes API scopes + openid
    )

    if "access_token" not in result:
        raise Exception(f"Failed to refresh Outlook token. Re-authentication may be required. Error: {result.get('error_description')}")

    # The result of a refresh token grant often includes a new refresh token, so we save the whole new result.
    new_token_data_json = json.dumps(result)
    new_base64_encoded_creds = base64.b64encode(new_token_data_json.encode('utf-8')).decode('utf-8')
    _save_credentials(user_id, OUTLOOK_INTEGRATION_KEY, new_base64_encoded_creds)

    return result['access_token']

@app.route("/connect/outlook", methods=['GET'])
def connect_outlook():
    """Step 1: Generate Outlook authorization URL, tagging state with provider."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"}), 400

    try:
        state = f"outlook:{user_id}:{os.urandom(16).hex()}"

        # Use OUTLOOK_AUTH_URL_SCOPES to satisfy the msal library's validation
        auth_url = msal_app.get_authorization_request_url(
            scopes=OUTLOOK_AUTH_URL_SCOPES,
            state=state,
            redirect_uri=REDIRECT_URI
        )

        return jsonify({"status": "ok", "authorization_url": auth_url})

    except Exception as e:
        print("--- FATAL ERROR IN /connect/outlook ---")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal server error generating auth URL."}), 500

@app.route("/oauth2callback", methods=['GET'])
def oauth2callback():
    """Step 2: Handles callback for BOTH Gmail and Outlook, parsing provider from state."""
    user_id_state = request.args.get('state', '')
    code = request.args.get('code')

    if not all([user_id_state, code]):
        return "<html><body><h1>Error: Missing required parameters.</h1></body></html>"

    try:
        parts = user_id_state.split(':', 2)
        provider = parts[0]
        user_id = parts[1]
    except (ValueError, IndexError):
        return "<html><body><h1>Error: Invalid state parameter format.</h1></body></html>"

    if provider == 'gmail':
        # --- GMAIL OAUTH LOGIC (Unaffected) ---
        try:
            # Your existing Gmail logic remains here, no changes needed.
            flow = Flow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, scopes=GMAIL_SCOPES, redirect_uri=REDIRECT_URI)
            flow.fetch_token(code=code)
            creds = flow.credentials
            service = build('oauth2', 'v2', credentials=creds)
            user_info = service.userinfo().get().execute()
            service_email = user_info.get('email')
            if not service_email:
                return "<html><body><h1>Error: Could not retrieve Gmail email.</h1></body></html>"
            serialized_creds = pickle.dumps(creds)
            base64_encoded_creds = base64.b64encode(serialized_creds).decode('utf-8')
            query = """INSERT INTO crm_schema.user_integrations (user_id, integration_key, credentials, integration_email) VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, integration_key) DO UPDATE SET credentials = EXCLUDED.credentials, integration_email = EXCLUDED.integration_email, updated_at = NOW();"""
            execute_db_query(query, (int(user_id), GMAIL_INTEGRATION_KEY, base64_encoded_creds, service_email), is_write=True)
            return f"<html><body style='font-family: sans-serif; text-align: center; padding: 40px;'><h1>Success!</h1><p>Your Gmail account ({service_email}) is connected. You can close this window.</p><script>setTimeout(() => window.close(), 2000);</script></body></html>"
        except Exception as e:
            return f"<html><body><h1>Gmail Error: {e}</h1></body></html>"

    elif provider == 'outlook':
        # --- FINAL, SIMPLIFIED OUTLOOK LOGIC ---
        try:
            # STEP 1: TOKEN EXCHANGE
            token_request_data = {
                'grant_type': 'authorization_code', 'client_id': OUTLOOK_CLIENT_ID,
                'client_secret': OUTLOOK_CLIENT_SECRET, 'code': code,
                'redirect_uri': REDIRECT_URI, 'scope': " ".join(OUTLOOK_TOKEN_SCOPES)
            }
            token_endpoint = f"{OUTLOOK_AUTHORITY}/oauth2/v2.0/token"
            token_response = requests.post(token_endpoint, data=token_request_data)
            token_response.raise_for_status()
            token_result = token_response.json() # This is the dictionary we will save

            # STEP 2: FETCH USER PROFILE
            access_token = token_result['access_token']
            graph_url = "https://graph.microsoft.com/v1.0/me"
            headers = {'Authorization': f'Bearer {access_token}'}
            profile_response = requests.get(graph_url, headers=headers)
            profile_response.raise_for_status()
            profile_data = profile_response.json()
            user_email = profile_data.get('mail') or profile_data.get('userPrincipalName')
            if not user_email:
                raise ValueError("Could not determine user's email from Graph API /me endpoint.")

            # STEP 3: SAVE THE TOKEN **DATA**, NOT THE CACHE
            token_data_json = json.dumps(token_result)
            base64_encoded_creds = base64.b64encode(token_data_json.encode('utf-8')).decode('utf-8')
            query = """INSERT INTO crm_schema.user_integrations (user_id, integration_key, credentials, integration_email) VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, integration_key) DO UPDATE SET credentials = EXCLUDED.credentials, integration_email = EXCLUDED.integration_email, updated_at = NOW();"""
            execute_db_query(query, (int(user_id), OUTLOOK_INTEGRATION_KEY, base64_encoded_creds, user_email), is_write=True)
            
            return f"""<html><body style='font-family: sans-serif; text-align: center; padding: 40px;'><h1>✅ Success!</h1><p>Your Outlook/Office 365 account ({user_email}) is connected. You can close this window.</p><script>setTimeout(() => window.close(), 2000);</script></body></html>"""
        
        except Exception as e:
            error_trace = traceback.format_exc()
            return f"""<html><body style="font-family: monospace; white-space: pre-wrap; padding: 20px;"><h1>An unexpected Outlook error occurred</h1><p><strong>Error Details:</strong> {e}</p><hr><h3>Full Traceback:</h3><pre>{error_trace}</pre></body></html>"""
        

@app.route("/disconnect/gmail", methods=['POST'])
def disconnect_gmail():
    """Deletes a user's Gmail integration credentials."""
    data = request.get_json()
    
    # --- FIX: Add this check ---
    if not data or not data.get('user_id'):
        return jsonify({"status": "error", "message": "Invalid request: Missing JSON payload or user_id."}), 400
    
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"}), 400

    try:
        query = "DELETE FROM crm_schema.user_integrations WHERE user_id = %s AND integration_key = %s;"
        rows_deleted = execute_db_query(query, (int(user_id), GMAIL_INTEGRATION_KEY), is_write=True)
        
        if rows_deleted > 0:
            return jsonify({"status": "ok", "message": "Gmail integration successfully disconnected."})
        else:
            return jsonify({"status": "ok", "message": "No active Gmail integration found to disconnect."})

    except Exception as e:
        return jsonify({"status": "error", "message": "An internal error occurred while trying to disconnect."}), 500

@app.route("/disconnect/outlook", methods=['POST'])
def disconnect_outlook():
    """Deletes a user's Outlook integration credentials."""
    data = request.get_json()
    
    # --- THIS IS THE CRITICAL CHECK ---
    if not data or not data.get('user_id'):
        return jsonify({
            "status": "error", 
            "message": "Invalid request: Missing JSON payload or user_id."
        }), 400
    # ------------------------------------
    
    user_id = data.get('user_id')

    try:
        query = "DELETE FROM crm_schema.user_integrations WHERE user_id = %s AND integration_key = %s;"
        rows_deleted = execute_db_query(query, (int(user_id), OUTLOOK_INTEGRATION_KEY), is_write=True)
        
        if rows_deleted > 0:
            return jsonify({"status": "ok", "message": "Outlook integration successfully disconnected."})
        else:
            return jsonify({"status": "ok", "message": "No active Outlook integration found to disconnect."})

    except Exception as e:
        return jsonify({"status": "error", "message": "An internal error occurred while trying to disconnect."}), 500
# ==============================================================================
# --- ACTION ENDPOINTS (UNIFIED) ---
# ==============================================================================

@app.route("/actions/send_email", methods=['POST'])
def handle_send_email():
    """API endpoint to handle requests to send an email (for Theta, uses Gmail)."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400
        
    user_id = data.get('user_id')
    recipient = data.get('recipient')
    subject = data.get('subject')
    body = data.get('body')

    if not all([user_id, recipient, subject, body]):
        return jsonify({"status": "error", "message": "Missing required parameters: user_id, recipient, subject, body"}), 400

    # NOTE: Since the Theta tool in main.py currently points to /actions/send_email
    # which historically meant Gmail, this endpoint will continue to use Gmail logic.
    result = send_gmail_email(user_id, recipient, subject, body)
    
    if result.get("status") == "error":
        return jsonify(result), 500
    
    return jsonify(result), 200

# --- NEW OUTLOOK ACTION ENDPOINTS ---

@app.route("/actions/send_outlook_email", methods=['POST'])
def handle_send_outlook_email():
    """API endpoint to send a new Outlook email."""
    data = request.get_json()
    if not all([data, data.get('user_id'), data.get('recipient'), data.get('subject'), data.get('body')]):
         return jsonify({"status": "error", "message": "Missing required parameters: user_id, recipient, subject, body"}), 400

    result = send_outlook_email(data['user_id'], data['recipient'], data['subject'], data['body'])
    return jsonify(result), (500 if result.get("status") == "error" else 200)

@app.route("/actions/get_outlook_emails", methods=['GET'])
def handle_get_outlook_emails():
    """API endpoint to get a list of Outlook emails."""
    user_id = request.args.get('user_id')
    folder = request.args.get('folder', 'inbox')
    top = int(request.args.get('top', 20))
    
    if not user_id: return jsonify({"status": "error", "message": "user_id is required"}), 400
        
    result = get_outlook_emails(user_id, folder, top)
    return jsonify(result), (500 if result.get("status") == "error" else 200)

@app.route("/actions/get_outlook_single_email", methods=['GET'])
def handle_get_outlook_single_email():
    """API endpoint to get the full content of one Outlook email."""
    user_id = request.args.get('user_id')
    message_id = request.args.get('message_id')
    
    if not all([user_id, message_id]): return jsonify({"status": "error", "message": "user_id and message_id are required"}), 400
        
    result = get_outlook_single_email(user_id, message_id)
    return jsonify(result), (500 if result.get("status") == "error" else 200)

@app.route("/actions/reply_outlook_email", methods=['POST'])
def handle_reply_outlook_email():
    """API endpoint to send a reply to an existing Outlook email."""
    data = request.get_json()
    if not all([data, data.get('user_id'), data.get('message_id'), data.get('reply_body')]):
        return jsonify({"status": "error", "message": "Missing required parameters: user_id, message_id, reply_body"}), 400
        
    result = reply_to_outlook_email(data['user_id'], data['message_id'], data['reply_body'])
    return jsonify(result), (500 if result.get("status") == "error" else 200)

@app.route("/actions/delete_outlook_email", methods=['POST'])
def handle_delete_outlook_email():
    """API endpoint to delete an Outlook email."""
    data = request.get_json()
    if not all([data, data.get('user_id'), data.get('message_id')]):
        return jsonify({"status": "error", "message": "Missing required parameters: user_id, message_id"}), 400
        
    result = delete_outlook_email(data['user_id'], data['message_id'])
    return jsonify(result), (500 if result.get("status") == "error" else 200)

# ==============================================================================
# --- REAL-TIME RECEIVING: MICROSOFT GRAPH WEBHOOK HANDLER ---
# ==============================================================================
# NOTE: This endpoint is included for completeness in receiving real-time mail
@app.route("/outlook_webhook", methods=['POST', 'GET'])
def outlook_webhook_listener():
    """Handles the Graph API change notifications (webhooks)."""
    
    # 1. Handle Subscription Validation
    validation_token = request.args.get('validationToken')
    if validation_token:
        print("MCP Server: Received Outlook Webhook Validation Request.")
        return Response(validation_token, mimetype='text/plain')

    # 2. Handle Actual Notifications
    try:
        data = request.get_json()
        for notification in data.get('value', []):
            user_id = notification.get('clientState') 
            resource = notification.get('resource')
            
            if resource and resource.startswith('me/messages/'):
                message_id = resource.split('/')[-1]
                print(f"New mail detected for user {user_id}. Fetch Message ID: {message_id}")
                # In a real app, this is where you'd queue a background job:
                # fetch_and_process_mail(user_id, message_id)
                
        # Must return 202 Accepted quickly
        return '', 202
        
    except Exception as e:
        print(f"MCP Server: Outlook Webhook processing error: {e}")
        return '', 202

# ==============================================================================
# --- RUN APPLICATION ---
# ==============================================================================
# ==============================================================================
# --- RUN APPLICATION ---
# ==============================================================================
if __name__ == "__main__":
    print("DEBUG: Script is starting...") # ADDED

    if not db_pool:
        print("DEBUG: EXITING - Database connection pool failed to create.") # MODIFIED
    elif not OUTLOOK_CLIENT_ID or not OUTLOOK_CLIENT_SECRET:
        print("DEBUG: EXITING - Missing OUTLOOK_CLIENT_ID or OUTLOOK_CLIENT_SECRET.") # MODIFIED
    else:
        print("DEBUG: All checks passed. Starting Flask server...") # ADDED
        port = int(os.getenv("MCP_PORT", 5005))
        app.run(host="0.0.0.0", port=port)
