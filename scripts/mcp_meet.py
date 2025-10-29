import os
import json
import base64
import traceback
import pickle
import psycopg2
import psycopg2.pool
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify, redirect, Response
from flask_cors import CORS
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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

# --- Google OAuth Configuration (Google Meet / Calendar) ---
# Scopes are updated to request permission for Google Calendar events.
GMEET_SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/calendar.events'
]
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_JSON", "credentials.json")
GMEET_INTEGRATION_KEY = 'gmeet'
REDIRECT_URI = os.getenv("MCP_REDIRECT_URI", "https://salesos.orionac.in/settings/oauth2callback")

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
# --- GOOGLE MEET HELPER AND ACTION FUNCTIONS ---
# ==============================================================================

def _save_credentials(user_id, integration_key, base64_creds):
    """Helper to save or update Google credentials in the database."""
    query = """
        UPDATE crm_schema.user_integrations
        SET credentials = %s, updated_at = NOW()
        WHERE user_id = %s AND integration_key = %s;
    """
    execute_db_query(query, (base64_creds, int(user_id), integration_key), is_write=True)

def get_gmeet_credentials(user_id):
    """Fetches and deserializes Google credentials, refreshing the token if necessary."""
    query = "SELECT credentials FROM crm_schema.user_integrations WHERE user_id = %s AND integration_key = %s"
    result = execute_db_query(query, (int(user_id), GMEET_INTEGRATION_KEY), fetch_one=True)
    if not result or not result[0]:
        raise ValueError("No Google Meet/Calendar credentials found for this user.")

    base64_creds = result[0]
    serialized_creds = base64.b64decode(base64_creds)
    creds = pickle.loads(serialized_creds)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Save refreshed credentials back to DB
        serialized_creds = pickle.dumps(creds)
        base64_encoded_creds = base64.b64encode(serialized_creds).decode('utf-8')
        _save_credentials(user_id, GMEET_INTEGRATION_KEY, base64_encoded_creds)

    return creds

def create_gmeet_meeting(user_id, summary, description, start_time_iso, end_time_iso, attendees_emails):
    """Uses stored credentials to create a Google Calendar event with a Google Meet link."""
    try:
        creds = get_gmeet_credentials(user_id)
        service = build('calendar', 'v3', credentials=creds)

        event = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time_iso,
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time_iso,
                'timeZone': 'UTC',
            },
            'attendees': [{'email': email} for email in attendees_emails],
            'conferenceData': {
                'createRequest': {
                    'requestId': str(uuid.uuid4()),
                    'conferenceSolutionKey': {
                        'type': 'hangoutsMeet'
                    }
                }
            }
        }

        created_event = service.events().insert(
            calendarId='primary', 
            body=event,
            conferenceDataVersion=1
        ).execute()

        print(f"MCP Worker: Google Meet created. Event ID: {created_event['id']}")
        return {
            "status": "ok", 
            "message": "Google Meet event created successfully.", 
            "event_id": created_event.get('id'),
            "meet_link": created_event.get('hangoutLink'),
            "html_link": created_event.get('htmlLink')
        }

    except HttpError as error:
        print(f"MCP Worker: A Google API HttpError occurred for user {user_id}: {error}")
        try:
            error_details = json.loads(error.content.decode('utf-8'))
            error_message = error_details.get("error", {}).get("message", "An unknown error occurred.")
            status_code = error_details.get("error", {}).get("code", 500)
            return {"status": "error", "message": f"Google API Error: {error_message}", "code": status_code}
        except (json.JSONDecodeError, AttributeError):
            return {"status": "error", "message": f"Google API HttpError: {str(error)}"}

    except Exception as e:
        print(f"MCP Worker: A non-API error occurred creating Google Meet for user {user_id}: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ==============================================================================
# --- OAUTH2 WEB FLOW ENDPOINTS ---
# ==============================================================================

@app.route("/connect/gmeet", methods=['GET'])
def connect_gmeet():
    """Step 1: Generate Google authorization URL for GMeet/Calendar."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"}), 400

    try:
        flow = Flow.from_client_secrets_file(
            GOOGLE_CREDENTIALS_FILE, scopes=GMEET_SCOPES, redirect_uri=REDIRECT_URI
        )
        state_with_provider = f"gmeet:{user_id}"
        authorization_url, state = flow.authorization_url(
            access_type='offline', prompt='consent', state=state_with_provider
        )
        return jsonify({"status": "ok", "authorization_url": authorization_url})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": f"'{GOOGLE_CREDENTIALS_FILE}' not found."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/oauth2callback", methods=['GET'])
def oauth2callback():
    """Step 2: Handles callback for Google authentication."""
    user_id_state = request.args.get('state', '')
    code = request.args.get('code')

    if not all([user_id_state, code]):
        return "<html><body><h1>Error: Missing required parameters.</h1></body></html>"

    try:
        provider, user_id = user_id_state.split(':', 1)
    except (ValueError, IndexError):
        return "<html><body><h1>Error: Invalid state parameter format.</h1></body></html>"

    if provider == 'gmeet':
        try:
            flow = Flow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, scopes=GMEET_SCOPES, redirect_uri=REDIRECT_URI)
            flow.fetch_token(code=code)
            creds = flow.credentials
            
            # Get user's email to store with the integration
            service = build('oauth2', 'v2', credentials=creds)
            user_info = service.userinfo().get().execute()
            service_email = user_info.get('email')
            if not service_email:
                return "<html><body><h1>Error: Could not retrieve Google account email.</h1></body></html>"

            serialized_creds = pickle.dumps(creds)
            base64_encoded_creds = base64.b64encode(serialized_creds).decode('utf-8')
            
            query = """
                INSERT INTO crm_schema.user_integrations (user_id, integration_key, credentials, integration_email) 
                VALUES (%s, %s, %s, %s) 
                ON CONFLICT (user_id, integration_key) 
                DO UPDATE SET credentials = EXCLUDED.credentials, integration_email = EXCLUDED.integration_email, updated_at = NOW();
            """
            execute_db_query(query, (int(user_id), GMEET_INTEGRATION_KEY, base64_encoded_creds, service_email), is_write=True)
            
            return f"<html><body style='font-family: sans-serif; text-align: center; padding: 40px;'><h1>Success!</h1><p>Your Google account ({service_email}) is connected for Google Meet. You can close this window.</p><script>setTimeout(() => window.close(), 2000);</script></body></html>"
        except Exception as e:
            return f"<html><body><h1>Google Meet Auth Error: {e}</h1></body></html>"
    else:
        return "<html><body><h1>Error: Unknown provider in state.</h1></body></html>"

@app.route("/disconnect/gmeet", methods=['POST'])
def disconnect_gmeet():
    """Deletes a user's Google Meet integration credentials."""
    data = request.get_json()
    if not data or not data.get('user_id'):
        return jsonify({"status": "error", "message": "Invalid request: Missing JSON payload or user_id."}), 400
    
    user_id = data.get('user_id')

    try:
        query = "DELETE FROM crm_schema.user_integrations WHERE user_id = %s AND integration_key = %s;"
        rows_deleted = execute_db_query(query, (int(user_id), GMEET_INTEGRATION_KEY), is_write=True)
        
        if rows_deleted > 0:
            return jsonify({"status": "ok", "message": "Google Meet integration successfully disconnected."})
        else:
            return jsonify({"status": "ok", "message": "No active Google Meet integration found to disconnect."})
    except Exception as e:
        return jsonify({"status": "error", "message": "An internal error occurred while trying to disconnect."}), 500

# ==============================================================================
# --- ACTION ENDPOINTS ---
# ==============================================================================

@app.route("/actions/create_meeting", methods=['POST'])
def handle_create_meeting():
    """API endpoint to handle requests to create a Google Meet event."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400
        
    user_id = data.get('user_id')
    summary = data.get('summary')
    description = data.get('description', '') # Optional
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    attendees = data.get('attendees', []) # Optional, list of emails

    if not all([user_id, summary, start_time, end_time]):
        return jsonify({"status": "error", "message": "Missing required parameters: user_id, summary, start_time, end_time"}), 400

    result = create_gmeet_meeting(user_id, summary, description, start_time, end_time, attendees)
    
    if result.get("status") == "error":
        return jsonify(result), 500
    
    return jsonify(result), 200

# ==============================================================================
# --- RUN APPLICATION ---
# ==============================================================================
if __name__ == "__main__":
    if not db_pool:
        print("MCP Server: EXITING - Database connection pool failed to create.")
    else:
        print("MCP Server: Starting Flask server for Google Meet integration...")
        port = int(os.getenv("MCP_PORT", 5006))
        app.run(host="0.0.0.0", port=port)

