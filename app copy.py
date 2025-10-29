## --- Meritto CRM API Endpoints ---
# Place all Meritto CRM endpoints below Flask app initialization
# (Move these after 'app = Flask(...)' and all imports)
import json
import psycopg2
import bcrypt
import jwt
import datetime
import random
import string
import pprint
import os
import requests
import re
import csv
import io
import base64
import logging
import json_repair
from flask import Flask, request, jsonify, render_template, g
from functools import wraps
import functools
from flask_cors import CORS
from dotenv import load_dotenv
from psycopg2.extras import execute_values
import pandas as pd
from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field,field_validator, ValidationError
from typing import Literal
from typing import Optional
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import agent_tools
from bs4 import BeautifulSoup
import feedparser
from urllib.parse import quote_plus
from psycopg2.extras import DictCursor

# --- CONFIGURATION & INITIALIZATION ---
# ==============================================================================
load_dotenv()

# --- MODIFICATION: Specify template folder for Flask ---
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
)
CORS(app)

# --- Secrets and Config from Environment ---
SECRET = os.getenv("FLASK_SECRET", "default-super-secret-key")
DB_NAME = os.getenv("DB_NAME", "user_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1269")
DB_HOST = os.getenv("DB_HOST", "localhost")
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", 5005))
GMEET_MCP_PORT = int(os.getenv("GMEET_MCP_PORT", 5006))
NIM_API_URL="http://13.202.235.154:8000/v1/chat/completions"

NIM_BASE_URL = NIM_API_URL.replace("/chat/completions", "")
# --- Global variable to hold user data for the current request ---
_CURRENT_USER = {}
chat_histories = {}

class LeadInsight(BaseModel):
    score: Literal['Hot', 'Warm', 'Cold'] = Field(description="The calculated score: 'Hot', 'Warm', or 'Cold'.")
    insight: str = Field(description="A concise, one-sentence analysis of the lead's potential.")
    next_best_action: str = Field(description="A clear, actionable next step for a sales representative.")

    # Optional: Add a  field_validator to ensure insight is concise if needed
    @field_validator('insight')
    def insight_must_be_concise(cls, v):
        if len(v.split('.')) > 2 or len(v) > 200: # Example: max 1 sentence, 200 chars
             # Note: In production, you might just log this warning or
             # try to truncate instead of raising an error during validation
             # For now, let's keep it strict for demonstration
             # raise ValueError('Insight must be concise (ideally one sentence, under 200 chars)')
             print(f"Warning: AI insight may be too long: '{v}'")
        return v
    
class MeetingBriefing(BaseModel):
    insight: str = Field(description="Key insights about the lead and opportunity (2-3 sentences)")
    next_best_action: str = Field(description="Specific recommended actions for the meeting")
    opportunity_assessment: str = Field(description="Assessment of deal potential and confidence level")
    conversation_starters: List[str] = Field(description="Suggested talking points and questions (list of 2-3 strings)")
    key_challenges: List[str] = Field(description="Potential challenges or objections to address (list of strings)")
    meeting_objectives: List[str] = Field(description="Recommended goals for this meeting (list of strings)")

class LeadImportItem(BaseModel):
    # Define fields matching TARGET SCHEMA, allow None for optional ones
    name: Optional[str] = Field(None, description="The full name of the contact.")
    email: Optional[str] = Field(None, description="The contact's email address.")
    phone: Optional[str] = Field(None, description="The contact's phone number.")
    status: Optional[Literal['New', 'Contacted', 'Won', 'Lost']] = Field('New', description="Status ('New', 'Contacted', 'Won', 'Lost'). Defaults to 'New'.")
    value: Optional[float] = Field(None, description="Potential monetary value (numeric).")
    organization_name: Optional[str] = Field(None, description="Company name.")
    notes: Optional[str] = Field(None, description="Combined notes.")
    # AI Generated fields - ensure they match the schema exactly
    ai_score: Literal['Hot', 'Warm', 'Cold'] = Field('Cold', description="Score ('Hot', 'Warm', 'Cold'). Defaults to 'Cold'.")
    ai_insight: str = Field(description="Concise analysis.")
    ai_next_action: str = Field(description="Specific next best action.")

    

    @field_validator('value', mode='before')
    def clean_value(cls, v):
        if v is None or str(v).strip() == '':
            return None
        try:
            value_str = str(v).strip().lower()
            multiplier = 1
            if value_str.endswith('k'): multiplier = 1000; value_str = value_str[:-1]
            elif value_str.endswith('m'): multiplier = 1000000; value_str = value_str[:-1]
            value_str = re.sub(r'[$,\s]', '', value_str)
            if value_str: return float(value_str) * multiplier
            return None
        except (ValueError, TypeError):
            logging.warning(f"Pydantic value cleaning failed for '{v}'. Returning None.")
            return None

    @field_validator('status', mode='before')
    def clean_status(cls, v):
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ['new', 'lead']: return 'New'
            if v_lower in ['contacted', 'engaged', 'working']: return 'Contacted'
            if v_lower in ['won', 'closed won', 'customer']: return 'Won'
            if v_lower in ['lost', 'closed lost', 'disqualified']: return 'Lost'
        return 'New' # Default if mapping fails


class LeadsImport(BaseModel):
    leads: List[LeadImportItem] = Field(description="A list of processed lead objects.")



# ==============================================================================
# --- LEAD ENRICHMENT HELPER FUNCTIONS ---
# ==============================================================================

def get_domain_from_email(email):
    """Extracts the domain from an email address."""
    if not email or '@' not in email:
        return None
    try:
        domain = email.split('@')[-1]
        # Avoid generic domains
        if domain in ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']:
            return None
        return domain
    except Exception:
        return None

def scrape_website(domain):
    """Scrapes the title and description from a company's website."""
    if not domain:
        return "No domain provided."
    try:
        url = f"https://{domain}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('title').get_text().strip() if soup.find('title') else 'No title found.'
        description_tag = soup.find('meta', attrs={'name': 'description'})
        description = description_tag.get('content', 'No description found.').strip() if description_tag else 'No description found.'
            
        return f"Website Title: {title}\nWebsite Description: {description}"
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to scrape {domain}: {e}")
        return f"Could not access website: {e}"
    except Exception as e:
        logging.warning(f"Error parsing {domain}: {e}")
        return f"Error parsing website: {e}"

def get_news(company_name):
    """Fetches top 3 news headlines for a company from Google News RSS."""
    if not company_name:
        return "No company name provided."
    try:
        # URL-encode the company name for the query
        query = quote_plus(company_name)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(url)
        headlines = []
        # Get up to 3 headlines
        for entry in feed.entries[:3]:
            headlines.append(entry.title)
            
        if not headlines:
            return "No recent news found."
        
        return "\n- ".join(["Recent News Headlines:"] + headlines)
    except Exception as e:
        logging.warning(f"Failed to fetch news for {company_name}: {e}")
        return f"Error fetching news: {e}"


# ==============================================================================
# --- DATABASE & HELPERS ---
# ==============================================================================
def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    return g.db

def onboarding_token_required(f):
    """Decorator to protect routes with the temporary onboarding token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").split(" ")[-1] if "Authorization" in request.headers else None
        if not token:
            return jsonify({"message": "Onboarding token is missing"}), 401
        try:
            data = jwt.decode(token, SECRET, algorithms=["HS256"])
            # Store data in g for the route to access
            g.onboarding_user = data 
        except Exception as e:
            return jsonify({"message": f"Onboarding token is invalid: {e}"}), 401
        return f(*args, **kwargs)
    return decorated

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").split(" ")[-1] if "Authorization" in request.headers else None
        if not token:
            return jsonify({"message": "Token is missing"}), 401
        try:
            data = jwt.decode(token, SECRET, algorithms=["HS256"])
            current_user = data
        except Exception as e:
            return jsonify({"message": f"Token is invalid: {e}"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

###############################################
# Lead Resource Endpoints: Notes, Documents, Tickets, Calls
###############################################

# Notes Endpoint
@app.route("/leads/<int:lead_id>/notes", methods=["GET", "POST"])
@token_required
def manage_lead_notes(current_user, lead_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to lead
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}), 400
                user_org_id = org_id_tuple[0]
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple or user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Lead not found or access denied."}), 404
                if request.method == "GET":
                    cur.execute("""
                        SELECT id, content, created_at, user_id FROM crm_schema.lead_notes
                        WHERE lead_id = %s
                        ORDER BY created_at DESC
                    """, (lead_id,))
                    notes = []
                    for row in cur.fetchall():
                        notes.append({
                            "id": row[0], "content": row[1], "created_at": row[2], "user_id": row[3]
                        })
                    return jsonify(notes)
                elif request.method == "POST":
                    data = request.json
                    content = data.get("content", "")
                    if not content:
                        return jsonify({"message": "Note content is required."}), 400
                    cur.execute("""
                        INSERT INTO crm_schema.lead_notes (lead_id, user_id, content)
                        VALUES (%s, %s, %s)
                        RETURNING id
                    """, (lead_id, user_id, content))
                    note_id = cur.fetchone()[0]
                    return jsonify({"message": "Note added successfully", "id": note_id}), 201
    except psycopg2.Error as e:
        return jsonify({"message": "Database error managing notes.", "error": str(e)}), 500

# Documents Endpoint
@app.route("/leads/<int:lead_id>/documents", methods=["GET", "POST"])
@token_required
def manage_lead_documents(current_user, lead_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to lead
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}), 400
                user_org_id = org_id_tuple[0]
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple or user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Lead not found or access denied."}), 404
                if request.method == "GET":
                    cur.execute("""
                        SELECT id, filename, url, uploaded_at, user_id FROM crm_schema.lead_documents
                        WHERE lead_id = %s
                        ORDER BY uploaded_at DESC
                    """, (lead_id,))
                    documents = []
                    for row in cur.fetchall():
                        documents.append({
                            "id": row[0], "filename": row[1], "url": row[2], "uploaded_at": row[3], "user_id": row[4]
                        })
                    return jsonify(documents)
                elif request.method == "POST":
                    data = request.json
                    filename = data.get("filename", "")
                    url = data.get("url", "")
                    if not filename or not url:
                        return jsonify({"message": "Filename and URL are required."}), 400
                    cur.execute("""
                        INSERT INTO crm_schema.lead_documents (lead_id, user_id, filename, url)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (lead_id, user_id, filename, url))
                    doc_id = cur.fetchone()[0]
                    return jsonify({"message": "Document added successfully", "id": doc_id}), 201
    except psycopg2.Error as e:
        return jsonify({"message": "Database error managing documents.", "error": str(e)}), 500

# Tickets Endpoint
@app.route("/leads/<int:lead_id>/tickets", methods=["GET", "POST"])
@token_required
def manage_lead_tickets(current_user, lead_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to lead
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}), 400
                user_org_id = org_id_tuple[0]
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple or user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Lead not found or access denied."}), 404
                if request.method == "GET":
                    cur.execute("""
                        SELECT id, title, description, status, created_at, user_id FROM crm_schema.lead_tickets
                        WHERE lead_id = %s
                        ORDER BY created_at DESC
                    """, (lead_id,))
                    tickets = []
                    for row in cur.fetchall():
                        tickets.append({
                            "id": row[0], "title": row[1], "description": row[2], "status": row[3], "created_at": row[4], "user_id": row[5]
                        })
                    return jsonify(tickets)
                elif request.method == "POST":
                    data = request.json
                    title = data.get("title", "")
                    description = data.get("description", "")
                    status = data.get("status", "open")
                    if not title:
                        return jsonify({"message": "Ticket title is required."}), 400
                    cur.execute("""
                        INSERT INTO crm_schema.lead_tickets (lead_id, user_id, title, description, status)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (lead_id, user_id, title, description, status))
                    ticket_id = cur.fetchone()[0]
                    return jsonify({"message": "Ticket created successfully", "id": ticket_id}), 201
    except psycopg2.Error as e:
        return jsonify({"message": "Database error managing tickets.", "error": str(e)}), 500

# Calls Endpoint
@app.route("/leads/<int:lead_id>/calls", methods=["GET", "POST"])
@token_required
def manage_lead_calls(current_user, lead_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to lead
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}), 400
                user_org_id = org_id_tuple[0]
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple or user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Lead not found or access denied."}), 404
                if request.method == "GET":
                    cur.execute("""
                        SELECT id, call_type, duration, notes, created_at, user_id FROM crm_schema.lead_calls
                        WHERE lead_id = %s
                        ORDER BY created_at DESC
                    """, (lead_id,))
                    calls = []
                    for row in cur.fetchall():
                        calls.append({
                            "id": row[0], "call_type": row[1], "duration": row[2], "notes": row[3], "created_at": row[4], "user_id": row[5]
                        })
                    return jsonify(calls)
                elif request.method == "POST":
                    data = request.json
                    call_type = data.get("call_type", "outbound")
                    duration = data.get("duration", 0)
                    notes = data.get("notes", "")
                    cur.execute("""
                        INSERT INTO crm_schema.lead_calls (lead_id, user_id, call_type, duration, notes)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (lead_id, user_id, call_type, duration, notes))
                    call_id = cur.fetchone()[0]
                    return jsonify({"message": "Call logged successfully", "id": call_id}), 201
    except psycopg2.Error as e:
        return jsonify({"message": "Database error managing calls.", "error": str(e)}), 500

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def proxy_to_mcp(endpoint, current_user, method="GET", data=None, params=None, port=None):
    user_id = current_user.get("user_id")
    if not user_id:
        return jsonify({"message": "User ID not found in token"}), 401

    target_port = port if port is not None else MCP_PORT
    mcp_url = f"http://{MCP_HOST}:{target_port}/{endpoint}"
    
    if method == "GET":
        params = dict(params or {})
        params["user_id"] = user_id
    elif method in ["POST", "PUT", "DELETE"]:
        data = data or {}
        data["user_id"] = user_id
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.request(method, mcp_url, json=data, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        if response.headers.get('Content-Type', '').startswith('application/json'):
            return jsonify(response.json()), response.status_code
        else:
            return response.text, response.status_code
        
    except requests.exceptions.RequestException as e:
        error_message = f"Error contacting integration service: {e}"
        logging.error(error_message)
        # Attempt to get more detail from the response if it exists
        try:
            error_details = e.response.json()
        except (ValueError, AttributeError):
            error_details = {"message": "Could not connect to or received an invalid response from the integration service."}
        status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 503
        return jsonify(error_details), status_code

@app.route("/actions/get_gmail_emails", methods=['GET'])
@token_required
def proxy_get_gmail_emails(current_user):
    """Proxy for getting Gmail emails list."""
    return proxy_to_mcp("actions/get_gmail_emails", current_user, method="GET", params=request.args)

@app.route("/actions/get_gmail_single_email", methods=['GET'])
@token_required
def proxy_get_gmail_single_email(current_user):
    """Proxy for getting single Gmail content."""
    return proxy_to_mcp("actions/get_gmail_single_email", current_user, method="GET", params=request.args)

@app.route("/actions/reply_gmail_email", methods=['POST'])
@token_required
def proxy_reply_gmail_email(current_user):
    """Proxy for replying to a Gmail."""
    return proxy_to_mcp("actions/reply_gmail_email", current_user, method="POST", data=request.get_json())

def process_google_notification(data):
    """
    (Placeholder) Processes the actual notification data from Pub/Sub.
    This logic will eventually fetch changes from Google APIs.
    """
    try:
        user_email = data.get('emailAddress')
        history_id = data.get('historyId') # For Gmail changes

        if not user_email or not history_id:
             logging.warning(f"Webhook received insufficient data in notification: {data}")
             return

        logging.info(f"Received notification trigger for {user_email}, historyId: {history_id}. Processing logic TBD.")

        # --- TODO: ---
        # 1. Look up user_id from user_email.
        # 2. Fetch user's Google credentials (via MCP).
        # 3. Use Gmail API history.list(userId='me', startHistoryId=<last_known_history_id>, historyId=history_id)
        # 4. Process changes (fetch messages, parse, match, deduplicate, insert into crm_schema.activity).
        # 5. Update the user's last_known_history_id in your database.

    except Exception as e:
        logging.error(f"Error in placeholder process_google_notification for {data.get('emailAddress')}: {e}", exc_info=True)

@app.route("/webhooks/google/notifications", methods=["POST"])
def google_push_notification_handler():
    """Receives push notifications from Google Cloud Pub/Sub."""
    # --- IMPORTANT: ADD REQUEST VERIFICATION LATER ---
    # You MUST verify that the request comes from Google Pub/Sub.
    # Google recommends verifying the JWT token in the Authorization header for push subscriptions.
    # See Google Cloud documentation for details. Skipping for initial setup simplicity.
    # Example (Conceptual - requires google-auth library):
    # try:
    #     auth_header = request.headers.get('Authorization')
    #     if auth_header:
    #         token = auth_header.split('Bearer ')[1]
    #         # Use google.oauth2.id_token.verify_oauth2_token(token, ...)
    #         # Verify audience, issuer, etc.
    # except Exception as auth_error:
    #     logging.error(f"Webhook authentication failed: {auth_error}")
    #     return "Unauthorized", 401

    # Check if the request has JSON data
    if not request.is_json:
        logging.error("Webhook received non-JSON request.")
        return "Bad Request: Expected JSON", 400

    # Parse the Pub/Sub envelope
    envelope = request.get_json()
    logging.debug(f"Received webhook payload envelope: {envelope}") # Log the full envelope for debugging

    if not envelope or 'message' not in envelope:
        logging.error("Received invalid Pub/Sub message format (missing 'message' key).")
        # Return 400 - Google Pub/Sub might retry, but the format is wrong.
        return "Bad Request: Invalid Pub/Sub envelope", 400

    pubsub_message = envelope['message']
    message_data_encoded = pubsub_message.get('data')

    if message_data_encoded:
        try:
            # Decode the actual notification data
            message_data_str = base64.b64decode(message_data_encoded).decode('utf-8')
            message_data = json.loads(message_data_str)
            logging.info(f"Successfully decoded Google Notification data: {message_data}")

            # --- Trigger Background Processing ---
            # !! WARNING: Processing directly in the request handler is NOT recommended for production.
            # !! It can lead to timeouts and Pub/Sub retries. Use a task queue (Celery/RQ) later.
            # For now, we call the function directly for testing.
            process_google_notification(message_data)

        except (base64.binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
            logging.error(f"Error decoding or parsing Pub/Sub message data: {e}")
            # Return 400 as the data itself is malformed.
            return "Bad Request: Invalid message data encoding or format", 400
        except Exception as e:
            # Catch errors during the *synchronous* call to process_google_notification
            logging.error(f"Error during synchronous processing of notification: {e}", exc_info=True)
            # Return 500 - indicates an issue on our end during processing. Pub/Sub *will* retry.
            return "Internal Server Error during processing", 500
    else:
        logging.warning("Received Pub/Sub message with no data field.")
        # Still acknowledge, as it's a valid Pub/Sub message format, just empty.

    # --- Acknowledge the message to Pub/Sub ---
    # Return a success status code (200, 201, 204 are common) quickly.
    # 204 No Content is suitable here.
    logging.info("Acknowledging Pub/Sub message receipt.")
    return "OK", 204

try:
    llm = ChatOpenAI(
        model="./Qwen3-14B-Instruct", #
        openai_api_base="http://13.202.235.154:8000/v1", #
        openai_api_key="not-used", #
        temperature=0.1
    )
    print("--- ReAct Agent LLM Initialized ---")
except Exception as e:
    print(f"--- FAILED TO INITIALIZE LLM --- Error: {e}")
    llm = None

# ==============================================================================
# --- ALL OTHER FLASK ENDPOINTS (UNCHANGED) ---
# ==============================================================================

@app.route("/agent/invoke", methods=["POST"])
@token_required
def agent_invoke(current_user):
    """
    New agent endpoint using a Pre-Orchestrated ReAct Frame.
    """
    if llm is None:
        return jsonify({"error": "Agent LLM not initialized"}), 500
        
    user_id = current_user["user_id"]
    user_input = request.json.get("command", "")

    tools = [
        Tool(
            name="send_email",
            func=functools.partial(agent_tools.send_email_logic, current_user),
            # --- (UPDATED DESCRIPTION) ---
            description="Sends a new email. Requires 'recipient' (email) and 'subject' (string). The 'body' (string) is optional. If the user does not provide a 'body', you MUST provide a 'user_intent' (string) describing what the email should say, and the tool will generate the body."
        ),
        Tool(
            name="get_emails",
            func=functools.partial(agent_tools.get_emails_logic, current_user),
            description="Get a list of emails. Requires *either* a 'query' (string) to search Gmail *or* a 'folder' (string) to get from Outlook."
        ),
        Tool(
            name="reply_to_email",
            func=functools.partial(agent_tools.reply_email_logic, current_user),
            description="Reply to a specific email. Requires: 'message_id' (string), 'body' (string), and optional 'tool_choice' ('gmail' or 'outlook')."
        )
    ]
    
    agent = create_react_agent(llm, tools)
    
    messages = chat_histories.get(user_id, [])
    messages.append(HumanMessage(content=user_input))
    
    try:
        # --- (UPDATED) Increase recursion limit ---
        response = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 30} # Give it more steps to think
        ) 
        
        chat_histories[user_id] = response["messages"]
        return jsonify({"response": response["messages"][-1].content})
        
    except Exception as e:
        logging.error(f"Agent invoke error for user {user_id}: {e}", exc_info=True)
        return jsonify({"status": "error", "response": f"An error occurred: {e}"}), 500

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

# ... (All your other Flask routes for /register, /login, /leads, /deals, etc. go here)
# ... (They do not need to be changed from your original file)
@app.route("/auth")
@app.route("/auth/")
def auth():
    return render_template("auth.html")

@app.route("/calendar")
@app.route("/calendar/")
def calendar_page():
    return render_template("calendar.html")


@app.route("/privacy-policy")
@app.route("/privacy-policy/")
def privacy_policy():
    return render_template("privacy-policy.html")

@app.route("/sales/post-sales")
@app.route("/sales/post-sales/")
def post_sales():
    return render_template("post-sales.html")

@app.route("/sales/pre-sales")
@app.route("/sales/pre-sales/")
def pre_sales():
    return render_template("pre-sales.html")

@app.route("/mini-crm")
@app.route("/mini-crm/")
def mini_crm():
    return render_template("mini-crm.html")

@app.route("/mini-crm/leads")
@app.route("/mini-crm/leads/")
def mini_mini_leads():
    return render_template("leads.html")

@app.route("/mini-crm/deals")
@app.route("/mini-crm/deals/")
def mini_mini_deals():
    return render_template("deals.html")

@app.route("/mini-crm/sales-funnel")
@app.route("/mini-crm/sales-funnel/")
def mini_mini_sales_funnel():
    return render_template("sales-funnel.html")

@app.route("/space")
@app.route("/space/")
def space():
    return render_template("space.html")

@app.route("/settings")
@app.route("/settings/")
def settings_page():
    return render_template("settings.html")

@app.route("/onboarding")
@app.route("/onboarding/")
def onboarding_page():
    return render_template("onboarding.html")

    
@app.route("/register", methods=["POST"])
def register_user():
    data = request.json
    email, username, password = data.get("email"), data.get("username"), data.get("password")
    if not all([email, username, password]):
        return jsonify({"message": "Email, username, and password are required"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM crm_schema.users WHERE email=%s OR username=%s", (email, username))
                if cur.fetchone():
                    return jsonify({"message": "Email or username already taken"}), 400

                cur.execute(
                    "INSERT INTO crm_schema.users (email, username, password, role, first_name) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (email, username, hashed, "user", username)
                )
                user_id = cur.fetchone()[0]
                
                # --- MODIFICATION START ---
                # Create a temporary onboarding token
                temp_token = jwt.encode({
                    "user_id": user_id,
                    "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15) # 15-minute expiry
                }, SECRET, algorithm="HS256")

                return jsonify({
                    "user_id": user_id, 
                    "onboarding_token": temp_token,
                    "message": "Registered successfully. Please complete your profile."
                }), 201
                # --- MODIFICATION END ---
    except psycopg2.Error as e:
        return jsonify({"message": "Database error during registration.", "error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login_user():
    data = request.json
    email, password = data.get("email"), data.get("password")

    if not all([email, password]):
        return jsonify({"message": "Email and password are required"}), 400

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id, password, role, first_name FROM crm_schema.users WHERE email=%s", (email,))
            row = cur.fetchone()

            if row and bcrypt.checkpw(password.encode("utf-8"), row[1].encode("utf-8")):
                token = jwt.encode({
                    "user_id": row[0],
                    "email": email,
                    "role": row[2],
                    "name": row[3], # Add name to token payload
                    "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
                }, SECRET, algorithm="HS256")
                return jsonify({"token": token})

            return jsonify({"message": "Invalid credentials"}), 401
    except psycopg2.Error as e:
        return jsonify({"message": "Database error during login.", "error": str(e)}), 500

@app.route("/user/<int:user_id>", methods=["GET"])
@token_required
def get_user_profile(current_user, user_id):
    """Fetches a specific user's profile."""
    if current_user["user_id"] != user_id:
        return jsonify({"message": "Access forbidden"}), 403
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, username, first_name, role FROM crm_schema.users WHERE id = %s", (user_id,))
            user = cur.fetchone()
            if not user:
                return jsonify({"message": "User not found"}), 404
            
            user_data = {
                "id": user[0],
                "email": user[1],
                "username": user[2],
                "name": user[3],
                "role": user[4]
            }
            return jsonify(user_data)
    except psycopg2.Error as e:
        return jsonify({"message": "Database error fetching user.", "error": str(e)}), 500


# app.py: around line 506
@app.route("/update_user_details/<int:user_id>", methods=["PUT"])
def update_user_details(user_id):
    data = request.json
    first_name = data.get("first_name")
    last_name = data.get("last_name")

    if not all([first_name, last_name]):
        return jsonify({"message": "First and last name are required"}), 400

    full_name = f"{first_name} {last_name}".strip()
    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE crm_schema.users SET first_name = %s WHERE id = %s", (full_name, user_id))
                return jsonify({"user_id": user_id, "message": "Profile updated. Proceed to organization setup."}),
                # --- END NEW CODE ---
    except psycopg2.Error as db_error:
        print(f"Auth Server: Database error in update_user_details: {db_error}")
        return jsonify({"message": "A database error occurred.", "error": str(db_error)}), 500
    except Exception as e:
        print(f"Auth Server: An unexpected error occurred in update_user_details: {e}")
        return jsonify({"message": "An unexpected server error occurred.", "error": str(e)}), 500

@app.route("/connect/gmail", methods=["GET"])
@token_required
def connect_gmail_via_mcp(current_user):
    """Proxy to MCP worker for Gmail OAuth connection."""
    return proxy_to_mcp("connect/gmail", current_user, method="GET", params=request.args)

@app.route("/disconnect/gmail", methods=["POST"])
@token_required
def disconnect_gmail_via_mcp(current_user):
    """Proxy to MCP worker to disconnect Gmail."""
    return proxy_to_mcp("disconnect/gmail", current_user, method="POST")

# --- NEW OUTLOOK PROXY ENDPOINTS (CONNECT/DISCONNECT) ---

@app.route("/connect/outlook", methods=["GET"])
@token_required
def connect_outlook_via_mcp(current_user):
    """Proxy to MCP worker for Outlook OAuth connection."""
    return proxy_to_mcp("connect/outlook", current_user, method="GET", params=request.args)

@app.route("/disconnect/outlook", methods=["POST"])
@token_required
def disconnect_outlook_via_mcp(current_user):
    """Proxy to MCP worker to disconnect Outlook."""
    return proxy_to_mcp("disconnect/outlook", current_user, method="POST")

# --- NEW OUTLOOK PROXY ENDPOINTS (ACTIONS) ---

@app.route("/actions/send_outlook_email", methods=['POST'])
@token_required
def proxy_send_outlook_email(current_user):
    """Proxy for sending an Outlook email."""
    return proxy_to_mcp("actions/send_outlook_email", current_user, method="POST", data=request.get_json())

@app.route("/actions/get_outlook_emails", methods=['GET'])
@token_required
def proxy_get_outlook_emails(current_user):
    """Proxy for getting Outlook emails list (receive/read)."""
    # Pass through query params like 'folder' and 'top'
    return proxy_to_mcp("actions/get_outlook_emails", current_user, method="GET", params=request.args)

@app.route("/actions/get_outlook_single_email", methods=['GET'])
@token_required
def proxy_get_outlook_single_email(current_user):
    """Proxy for getting single Outlook email content (read)."""
    # Pass through query param 'message_id'
    return proxy_to_mcp("actions/get_outlook_single_email", current_user, method="GET", params=request.args)

@app.route("/actions/reply_outlook_email", methods=['POST'])
@token_required
def proxy_reply_outlook_email(current_user):
    """Proxy for replying to an Outlook email."""
    return proxy_to_mcp("actions/reply_outlook_email", current_user, method="POST")

@app.route("/actions/delete_outlook_email", methods=['POST'])
@token_required
def proxy_delete_outlook_email(current_user):
    """Proxy for deleting an Outlook email."""
    return proxy_to_mcp("actions/delete_outlook_email", current_user, method="POST")

@app.route("/settings/details", methods=["GET"])
@token_required
def get_user_settings(current_user):
    user_id = current_user["user_id"]
    settings_data = {
        "profile": {},
        "organization": None,
        "integrations": {},
        "primary_mail_key": None, 
        "primary_meeting_key": None
    }
    try:
        conn = get_db()
        with conn.cursor() as cur:
            # Profile and primary mail key
            cur.execute("""
                SELECT username, email, first_name, primary_integration_key
                FROM crm_schema.users WHERE id = %s
            """, (user_id,))
            profile = cur.fetchone()
            if profile:
                settings_data["profile"] = {"username": profile[0], "email": profile[1], "name": profile[2]}
                settings_data["primary_mail_key"] = profile[3]

            # Organization
            cur.execute("""
                SELECT o.id, o.name, o.industry, o.website, o.phone, o.address, o.city, o.country, uo.role_in_org
                FROM crm_schema.organizations o JOIN crm_schema.user_organizations uo ON o.id = uo.organization_id
                WHERE uo.user_id = %s
            """, (user_id,))
            org_data = cur.fetchone()
            if org_data:
                settings_data["organization"] = {
                    "id": org_data[0], "name": org_data[1], "industry": org_data[2], "website": org_data[3],
                    "phone": org_data[4], "address": org_data[5], "city": org_data[6], "country": org_data[7],
                    "user_role": org_data[8]
                }

            # Integrations
            cur.execute("SELECT integration_key, integration_email FROM crm_schema.user_integrations WHERE user_id = %s", (user_id,))
            integrations = cur.fetchall()
            for integ in integrations:
                settings_data["integrations"][integ[0]] = {"connected": True, "email": integ[1]}
                
            return jsonify(settings_data)
    except psycopg2.Error as e:
        logging.error(f"Database error in get_user_settings: {e}")
        return jsonify({"message": "Database error fetching settings.", "error": str(e)}), 500

@app.route("/settings/primary_mail", methods=["PUT"])
@token_required
def set_primary_mail(current_user):
    user_id = current_user.get("user_id")
    data = request.json
    integration_key = data.get("integration_key")

    try:
        conn = get_db()
        with conn.cursor() as cur:
            # UPDATE the value
            cur.execute(
                "UPDATE crm_schema.users SET primary_integration_key = %s WHERE id = %s",
                (integration_key, user_id)
            )
        conn.commit()
        
        # After saving, call the get_user_settings logic to return the fresh data
        # This re-uses the same logic and returns the full, updated settings object
        return get_user_settings()

    except psycopg2.Error as e:
        logging.error(f"!!! DATABASE ERROR in set_primary_mail: {e} !!!")
        if conn:
            conn.rollback()
        return jsonify({"message": "Database error updating setting.", "error": str(e)}), 500
@app.route("/settings/profile", methods=["PUT"])
@token_required
def update_user_profile(current_user):
    user_id = current_user["user_id"]
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not all([name, email]):
        return jsonify({"message": "Full name and email are required"}), 400

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE crm_schema.users SET first_name = %s, email = %s WHERE id = %s",
                    (name, email, user_id)
                )
                if password and "•" not in password:
                    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                    cur.execute(
                        "UPDATE crm_schema.users SET password = %s WHERE id = %s",
                        (hashed, user_id)
                    )
                return jsonify({"message": "Profile updated successfully"})
    except psycopg2.Error as e:
        return jsonify({"message": "Database error updating settings.", "error": str(e)}), 500

@app.route("/organization/create", methods=["POST"])
@token_required
def create_organization_for_user(current_user):
    """Creates an organization for an existing, logged-in user."""
    user_id = current_user["user_id"]
    data = request.json
    name = data.get("name")
    website = data.get("website")
    industry = data.get("industry")

    if not name:
        return jsonify({"message": "Organization name is required"}), 400

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # Check if user is already in an organization
                cur.execute("SELECT 1 FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                if cur.fetchone():
                    return jsonify({"message": "User is already in an organization."}),

                # Create the organization
                cur.execute(
                    "INSERT INTO crm_schema.organizations (name, industry, website, created_by) VALUES (%s, %s, %s, %s) RETURNING id",
                    (name, industry, website, user_id)
                )
                org_id = cur.fetchone()[0]

                # Link the user as Admin
                cur.execute(
                    "INSERT INTO crm_schema.user_organizations (user_id, organization_id, role_in_org) VALUES (%s, %s, %s)",
                    (user_id, org_id, 'Admin')
                )
        return jsonify({"message": "Organization created successfully!"}), 201
    except psycopg2.Error as e:
        logging.error(f"DB error in /organization/create: {e}")
        return jsonify({"message": "Database error creating organization.", "error": str(e)}), 500


@app.route("/create_organization", methods=["POST"])
@onboarding_token_required
def create_organization_and_login():
    """
    Final step of the 3-step registration: creates the organization, links the user
    as Admin, and issues the final JWT token to log them in.
    """
    data = request.json
    user_id = data.get("user_id") # User ID passed from the frontend
    name = data.get("name")
    website = data.get("website")
    
    # Optional fields default to None if not passed in the simple registration step
    industry, phone, address, city, country = None, None, None, None, None 

    if not all([user_id, name]):
        return jsonify({"message": "User ID and Organization name are required"}), 400

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # 1. Check if user is already in an organization (prevent re-running)
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                if cur.fetchone() is not None:
                    # User is already set up, proceed directly to login/token generation
                    # Fall-through to token creation after cleanup
                    pass 
                else:
                    # 2. CREATE THE ORGANIZATION
                    cur.execute(
                        """
                        INSERT INTO crm_schema.organizations 
                        (name, industry, website, phone, address, city, country, created_by) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                        """,
                        (name, industry, website, phone, address, city, country, user_id)
                    )
                    org_id = cur.fetchone()[0]

                    # 3. LINK THE USER AS ADMIN
                    cur.execute(
                        """
                        INSERT INTO crm_schema.user_organizations (user_id, organization_id, role_in_org) 
                        VALUES (%s, %s, %s)
                        """,
                        (user_id, org_id, 'Admin')
                    )

                # 4. GENERATE FINAL JWT TOKEN (Log the user in)
                cur.execute("SELECT email, role, first_name FROM crm_schema.users WHERE id = %s", (user_id,))
                user_details = cur.fetchone()
                if not user_details:
                       return jsonify({"message": "User details not found for login."}),
                
                email, role, full_name = user_details
                
                token = jwt.encode({
                    "user_id": user_id,
                    "email": email,
                    "role": role,
                    "name": full_name,
                    "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
                }, SECRET, algorithm="HS256")
                
                return jsonify({"message": "Organization created and login successful!", "token": token}), 201
                
    except psycopg2.Error as e:
        return jsonify({"message": "Database error creating organization.", "error": str(e)}), 500
    
# --- NEW: ORGANIZATION INVITE LINK GENERATION ---

@app.route("/organization/generate_invite", methods=["POST"])
@token_required
def generate_invite_link(current_user):
    """Generates a single-use invitation link for the user's organization."""
    user_id = current_user["user_id"]
    
    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # 1. Get the user's organization ID
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not part of an organization."}),400
                org_id = org_id_tuple[0]

                # 2. Generate a secure, unique token
                invite_token = ''.join(random.choices(string.ascii_letters + string.digits, k=40))
                
                # 3. Set an expiration time (e.g., 72 hours from now)
                expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=72)

                # 4. Store the token in the new database table
                cur.execute(
                    """
                    INSERT INTO crm_schema.invitations (organization_id, token, created_by_user_id, expires_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (org_id, invite_token, user_id, expires_at)
                )

                # 5. Return the token to the frontend
                # The frontend will construct the full URL (e.g., https://yoursite.com/auth?invite_token=...)
                return jsonify({"invite_token": invite_token, "message": "Invite link generated successfully."}), 200
    except psycopg2.Error as e:
        logging.error(f"Database error generating invite link: {e}")
        return jsonify({"message": "Database error generating invite link.", "error": str(e)}), 500


# --- NEW: ACCEPT ORGANIZATION INVITE ---

@app.route("/organization/accept_invite", methods=["POST"])
@onboarding_token_required # Use the onboarding token to identify the new user
def accept_invite():
    """
    Allows a new user to join an organization using a valid invite token.
    This is the final step of their onboarding.
    """
    data = request.json
    invite_token = data.get("invite_token")
    user_id = g.onboarding_user['user_id'] # Get user_id from the onboarding token

    if not invite_token:
        return jsonify({"message": "Invite token is required."}),

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # 1. Find the invitation and validate it
                cur.execute(
                    "SELECT organization_id, expires_at, is_used FROM crm_schema.invitations WHERE token = %s",
                    (invite_token,)
                )
                invite = cur.fetchone()

                if not invite:
                    return jsonify({"message": "This invitation is invalid or does not exist."}),
                
                org_id, expires_at, is_used = invite
                if is_used:
                    return jsonify({"message": "This invitation has already been used."}),
                if expires_at < datetime.datetime.now(datetime.timezone.utc):
                    return jsonify({"message": "This invitation has expired."}),

                # 2. Check if the new user is already in an organization
                cur.execute("SELECT 1 FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                if cur.fetchone():
                    return jsonify({"message": "This account is already part of an organization."}),

                # 3. Add the user to the organization
                cur.execute(
                    "INSERT INTO crm_schema.user_organizations (user_id, organization_id, role_in_org) VALUES (%s, %s, %s)",
                    (user_id, org_id, 'Member') # Default role is 'Member'
                )

                # 4. Mark the invitation as used
                cur.execute("UPDATE crm_schema.invitations SET is_used = TRUE WHERE token = %s", (invite_token,))

                # 5. Issue the final login token for the new user
                cur.execute("SELECT email, role, first_name FROM crm_schema.users WHERE id = %s", (user_id,))
                user_details = cur.fetchone()
                if user_details:
                    email, role, full_name = user_details
                    final_token = jwt.encode({
                        "user_id": user_id, "email": email, "role": role, "name": full_name,
                        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
                    }, SECRET, algorithm="HS256")
                    return jsonify({"token": final_token, "message": "Successfully joined organization!"}), 200
                else:
                    return jsonify({"message": "Could not retrieve user details to finalize login."}),

    except psycopg2.Error as e:
        logging.error(f"Database error accepting invite: {e}")
        return jsonify({"message": "Database error.", "error": str(e)}), 500

@app.route("/organization/join", methods=["POST"])
@token_required # Use the standard token for logged-in users
def join_organization(current_user):
    """
    Allows an authenticated, logged-in user to join an organization using a valid invite token.
    """
    data = request.json
    invite_token = data.get("invite_token")
    user_id = current_user['user_id']

    if not invite_token:
        return jsonify({"message": "Invite token is required."}),

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # 1. Find the invitation and validate it
                cur.execute(
                    "SELECT organization_id, expires_at, is_used FROM crm_schema.invitations WHERE token = %s",
                    (invite_token,)
                )
                invite = cur.fetchone()

                if not invite:
                    return jsonify({"message": "This invitation is invalid or does not exist."}),
                
                org_id, expires_at, is_used = invite
                if is_used:
                    return jsonify({"message": "This invitation has already been used."}),
                if expires_at < datetime.datetime.now(datetime.timezone.utc):
                    return jsonify({"message": "This invitation has expired."}),

                # 2. Check if the current user is already in an organization
                cur.execute("SELECT 1 FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                if cur.fetchone():
                    return jsonify({"message": "This account is already part of an organization."}),

                # 3. Add the user to the organization
                cur.execute(
                    "INSERT INTO crm_schema.user_organizations (user_id, organization_id, role_in_org) VALUES (%s, %s, %s)",
                    (user_id, org_id, 'Member') # Default role is 'Member'
                )

                # 4. Mark the invitation as used
                # 4. Mark the invitation as used
                cur.execute("UPDATE crm_schema.invitations SET is_used = TRUE WHERE token = %s", (invite_token,))

                # 5. No new token needed, just a success message
                return jsonify({"message": "Successfully joined organization!"}), 200

    except psycopg2.Error as e:
        logging.error(f"Database error joining organization: {e}")
        return jsonify({"message": "Database error.", "error": str(e)}), 500


@app.route("/organization/users/<int:org_id>", methods=["GET"])
@token_required
def get_organization_users(current_user, org_id):
    """Fetches all users belonging to a specific organization."""
    user_id = current_user["user_id"]
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM crm_schema.user_organizations WHERE user_id = %s AND organization_id = %s", (user_id, org_id))
            if cur.fetchone() is None:
                return jsonify({"message": "Forbidden: You are not a member of this organization."}),

            cur.execute("""
                SELECT u.id, u.username, u.first_name, u.email, uo.role_in_org 
                FROM crm_schema.users u
                JOIN crm_schema.user_organizations uo ON u.id = uo.user_id
                WHERE uo.organization_id = %s
            """, (org_id,))
            users = []
            for row in cur.fetchall():
                users.append({
                    "id": row[0],
                    "username": row[1],
                    "name": row[2],
                    "email": row[3],
                    "role": row[4]
                })
            return jsonify(users)
    except psycopg2.Error as e:
        return jsonify({"message": "Database error fetching organization users.", "error": str(e)}), 500

@app.route("/organization/invite", methods=["POST"])
@token_required
def invite_to_organization(current_user):
    """Invites a user (by email) to the current user's organization."""
    user_id = current_user["user_id"]
    data = request.json
    invitee_email = data.get("email")
    role = data.get("role", "Member") 

    if not invitee_email:
        return jsonify({"message": "Email of the user to invite is required."}), 400

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not part of an organization."}), 400
                org_id = org_id_tuple[0]

                cur.execute("SELECT id FROM crm_schema.users WHERE email = %s", (invitee_email,))
                invitee_tuple = cur.fetchone()
                if not invitee_tuple:
                    return jsonify({"message": "User to invite not found."}), 404
                invitee_id = invitee_tuple[0]
                
                cur.execute("SELECT 1 FROM crm_schema.user_organizations WHERE user_id = %s", (invitee_id,))
                if cur.fetchone() is not None:
                    return jsonify({"message": "User is already in an organization."}), 400

                cur.execute(
                    "INSERT INTO crm_schema.user_organizations (user_id, organization_id, role_in_org) VALUES (%s, %s, %s)",
                    (invitee_id, org_id, role)
                )
                return jsonify({"message": f"Successfully invited {invitee_email} to the organization."})
    except psycopg2.Error as e:
        return jsonify({"message": "Database error during invitation.", "error": str(e)}), 500


@app.route("/api/leads", methods=["GET"])
@token_required
def get_leads(current_user):
    user_id = current_user["user_id"]
    timeframe = request.args.get('timeframe', 'all')
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You must be in an organization to view leads."}),
            org_id = org_id_tuple[0]
            
            query = """
                SELECT id, name, email, phone, status, value, created_at, user_id, organization_name, notes 
                FROM crm_schema.leads 
                WHERE organization_id = %s
            """
            params = [org_id]

            if timeframe == 'weekly':
                query += " AND created_at >= NOW() - INTERVAL '7 days'"
            elif timeframe == 'monthly':
                query += " AND created_at >= NOW() - INTERVAL '1 month'"
            elif timeframe == 'quarterly':
                query += " AND created_at >= NOW() - INTERVAL '3 months'"

            cur.execute(query, tuple(params))
            leads = []
            for row in cur.fetchall():
                leads.append({
                    "id": row[0], "name": row[1], "email": row[2], "phone": row[3],
                    "status": row[4], "value": row[5], "created_at": row[6],
                    "user_id": row[7], "organization": row[8], "notes": row[9]
                })
            return jsonify(leads)
    except psycopg2.errors.UndefinedTable:
         return jsonify({"message": "Leads feature not fully configured (table missing)."}), 500
    except psycopg2.Error as e:
        return jsonify({"message": "Database error fetching leads.", "error": str(e)}), 500

@app.route("/leads", methods=["POST"])
@token_required
def create_lead(current_user):
    user_id = current_user["user_id"]
    data = request.json
    name, email, status = data.get("name"), data.get("email"), data.get("status", "New")
    phone, value = data.get("phone"), data.get("value")
    organization_name = data.get("organization") 
    notes = data.get("notes")

    if not name or not email or not name.strip() or not email.strip():
        return jsonify({"message": "Lead name and email are required."}), 400

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You must be in an organization to create leads."}),
                org_id = org_id_tuple[0]

                # Check if lead with same email already exists in this organization
                cur.execute("SELECT id FROM crm_schema.leads WHERE email = %s AND organization_id = %s", (email, org_id))
                existing_lead = cur.fetchone()
                if existing_lead:
                    return jsonify({"message": "A lead with this email already exists in your organization."}), 400

                cur.execute(
                    """
                    INSERT INTO crm_schema.leads (name, email, phone, status, value, organization_id, user_id, organization_name, notes) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """,
                    (name, email, phone, status, value, org_id, user_id, organization_name, notes)
                )
                lead_id = cur.fetchone()[0]
                return jsonify({"message": "Lead created successfully", "lead_id": lead_id}), 201
    except psycopg2.Error as e:
        import traceback
        print("[Lead Creation DB Error]", str(e))
        traceback.print_exc()
        return jsonify({"message": "Database error creating lead.", "error": str(e)}), 500

@app.route("/actions/send_gmail_email", methods=['POST'])
@token_required
def proxy_send_gmail_email(current_user):
    """Proxy for sending a Gmail email from the manual draft UI."""
    # This proxies to the 'actions/send_email' endpoint in mcp_gmail.py,
    # which is the correct endpoint for sending Gmail.
    return proxy_to_mcp("actions/send_email", current_user, method="POST", data=request.get_json())

@app.route("/leads/<int:lead_id>", methods=["GET", "PUT", "DELETE"])
@app.route("/api/leads/<int:lead_id>", methods=["GET", "PUT", "DELETE"])
@token_required
def manage_lead(current_user, lead_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}),
                user_org_id = org_id_tuple[0]
                
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple:
                    return jsonify({"message": "Lead not found"}), 404
                
                if user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Forbidden: You do not have access to this lead."}),

                if request.method == "GET":
                    cur.execute("""
    SELECT id, name, email, phone, status, value, created_at, user_id, organization_name, notes, 
           title, source, last_contact_date, tags, enrichment_data
    FROM crm_schema.leads WHERE id = %s
""", (lead_id,))
                    row = cur.fetchone()
                    if not row: return jsonify({"message": "Lead not found"}), 404
                    lead = {
    "id": row[0], "name": row[1], "email": row[2], "phone": row[3],
    "status": row[4], "value": row[5], "created_at": row[6],
    "user_id": row[7], "organization": row[8], "notes": row[9],
    "title": row[10], "source": row[11], "last_contact_date": row[12], "tags": row[13],
    "enrichment_data": row[14]  # <-- ADD THIS LINE
}
                    
                    # Get communication logs
                    cur.execute("""
                        SELECT cl.id, cl.communication_type, cl.subject, cl.content, cl.direction, 
                               cl.created_at, u.first_name, u.last_name
                        FROM crm_schema.communication_logs cl
                        LEFT JOIN crm_schema.users u ON cl.user_id = u.id
                        WHERE cl.lead_id = %s
                        ORDER BY cl.created_at DESC
                        LIMIT 10
                    """, (lead_id,))
                    communications = []
                    for comm_row in cur.fetchall():
                        communications.append({
                            "id": comm_row[0], "type": comm_row[1], "subject": comm_row[2],
                            "content": comm_row[3], "direction": comm_row[4], "created_at": comm_row[5],
                            "user_name": f"{comm_row[6] or ''} {comm_row[7] or ''}".strip() or "Unknown"
                        })
                    
                    # Get activities/timeline
                    cur.execute("""
                        SELECT a.id, a.activity_type, a.title, a.description, a.created_at, 
                               u.first_name, u.last_name
                        FROM crm_schema.activities a
                        LEFT JOIN crm_schema.users u ON a.user_id = u.id
                        WHERE a.lead_id = %s
                        ORDER BY a.created_at DESC
                        LIMIT 20
                    """, (lead_id,))
                    activities = []
                    for act_row in cur.fetchall():
                        activities.append({
                            "id": act_row[0], "type": act_row[1], "title": act_row[2],
                            "description": act_row[3], "created_at": act_row[4],
                            "user_name": f"{act_row[5] or ''} {act_row[6] or ''}".strip() or "Unknown"
                        })
                    
                    # Get tasks
                    cur.execute("""
                        SELECT t.id, t.title, t.description, t.status, t.priority, t.due_date, 
                               t.created_at, t.completed_at, u.first_name, u.last_name
                        FROM crm_schema.tasks t
                        LEFT JOIN crm_schema.users u ON t.user_id = u.id
                        WHERE t.lead_id = %s
                        ORDER BY t.created_at DESC
                    """, (lead_id,))
                    tasks = []
                    for task_row in cur.fetchall():
                        tasks.append({
                            "id": task_row[0], "title": task_row[1], "description": task_row[2],
                            "status": task_row[3], "priority": task_row[4], "due_date": task_row[5],
                            "created_at": task_row[6], "completed_at": task_row[7],
                            "user_name": f"{task_row[8] or ''} {task_row[9] or ''}".strip() or "Unknown"
                        })
                    
                    lead["communications"] = communications
                    lead["activities"] = activities
                    lead["tasks"] = tasks
                    return jsonify(lead)
                
                if request.method == "PUT":
                    # --- MODIFICATION START ---
                    # First, get the current status of the lead before making changes.
                    cur.execute("SELECT status FROM crm_schema.leads WHERE id = %s", (lead_id,))
                    old_status_tuple = cur.fetchone()
                    if not old_status_tuple:
                        # This check is redundant given the earlier check, but good for safety
                        return jsonify({"message": "Cannot update a non-existent lead."}), 404
                    old_status = old_status_tuple[0]
                    # --- END OF PRE-FETCH ---

                    data = request.json
                    
                    # Validate status value if provided
                    if "status" in data:
                        valid_statuses = ['New', 'Contacted', 'Won', 'Lost']
                        if data["status"] not in valid_statuses:
                            return jsonify({"message": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"}), 400
                    
                    # Validate email format if provided
                    if "email" in data and data["email"]:
                        import re
                        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        if not re.match(email_pattern, data["email"]):
                            return jsonify({"message": "Invalid email format"}), 400
                    
                    # Validate value if provided
                    if "value" in data and data["value"] is not None:
                        try:
                            if isinstance(data["value"], str):
                                data["value"] = float(data["value"])
                            if data["value"] < 0:
                                return jsonify({"message": "Value cannot be negative"}), 400
                        except (ValueError, TypeError):
                            return jsonify({"message": "Invalid value format"}), 400
                    
                    fields, params = [], []
                    for key in ["name", "email", "phone", "status", "value", "organization", "notes", "title", "source", "tags"]:
                        if key in data:
                            db_key = "organization_name" if key == "organization" else key
                            fields.append(f"{db_key} = %s")
                            params.append(data[key])
                    
                    if fields:
                        params.append(lead_id)
                        query = f"UPDATE crm_schema.leads SET {', '.join(fields)} WHERE id = %s"
                        cur.execute(query, tuple(params))
                        
                        # Log activity for lead update
                        cur.execute("""
                            INSERT INTO crm_schema.activities (lead_id, user_id, activity_type, title, description, created_by, metadata, organization_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (lead_id, user_id, 'field_updated', 'Lead updated', 
                              f'Lead information was updated', user_id,
                              json.dumps({'updated_fields': list(data.keys())}), user_org_id))

                    # --- MODIFIED DEAL CREATION/DELETION LOGIC ---
                    new_status = data.get("status")

                    # Case 1: If status changes TO 'Won', create a deal if one doesn't exist.
                    if new_status == 'Won' and old_status != 'Won':
                        cur.execute("SELECT id FROM crm_schema.deals WHERE lead_id = %s", (lead_id,))
                        if cur.fetchone() is None:
                            cur.execute("SELECT name, value, organization_name FROM crm_schema.leads WHERE id = %s", (lead_id,))
                            lead_details = cur.fetchone()
                            if lead_details:
                                lead_name, lead_value, lead_org_name = lead_details
                                deal_name = f"Deal for {lead_org_name or lead_name}"
                                cur.execute(
                                    """
                                    INSERT INTO crm_schema.deals (name, value, stage, lead_id, org_id, user_id)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    """,
                                    (deal_name, lead_value, 'Qualification', lead_id, user_org_id, user_id)
                                )
                    
                    # Case 2: If status changes FROM 'Won' to something else, delete the associated deal.
                    elif old_status == 'Won' and new_status and new_status != 'Won':
                        cur.execute("DELETE FROM crm_schema.deals WHERE lead_id = %s", (lead_id,))
                    # --- MODIFICATION END ---
                        
                    return jsonify({"message": "Lead updated successfully"})

                if request.method == "DELETE":
                    cur.execute("DELETE FROM crm_schema.deals WHERE lead_id = %s", (lead_id,))
                    cur.execute("DELETE FROM crm_schema.leads WHERE id = %s", (lead_id,))
                    # Optionally, update the success message
                    return jsonify({"message": "Lead and associated deal deleted successfully"})
    except psycopg2.Error as e:
        import traceback
        print("[Lead Update DB Error]", str(e))
        traceback.print_exc()
        
        # Provide more specific error messages based on error type
        if "column" in str(e).lower() and "does not exist" in str(e).lower():
            return jsonify({"message": "Database schema error: Column not found. Please contact administrator."}), 500
        elif "foreign key" in str(e).lower():
            return jsonify({"message": "Database constraint error: Invalid reference to related data."}), 500
        elif "check constraint" in str(e).lower():
            return jsonify({"message": "Data validation error: Invalid data format or value."}), 400
        elif "connection" in str(e).lower():
            return jsonify({"message": "Database connection error. Please try again later."}), 500
        else:
            return jsonify({"message": "Database error managing lead.", "error": str(e)}), 500

@app.route("/leads/<int:lead_id>/communications", methods=["GET", "POST"])
@token_required
def manage_lead_communications(current_user, lead_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to lead
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}), 400
                user_org_id = org_id_tuple[0]
                
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple or user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Lead not found or access denied."}), 404
                
                if request.method == "GET":
                    cur.execute("""
                        SELECT cl.id, cl.communication_type, cl.subject, cl.content, cl.direction, 
                               cl.created_at, u.first_name, u.last_name
                        FROM crm_schema.communication_logs cl
                        LEFT JOIN crm_schema.users u ON cl.user_id = u.id
                        WHERE cl.lead_id = %s
                        ORDER BY cl.created_at DESC
                    """, (lead_id,))
                    communications = []
                    for row in cur.fetchall():
                        communications.append({
                            "id": row[0], "type": row[1], "subject": row[2],
                            "content": row[3], "direction": row[4], "created_at": row[5],
                            "user_name": f"{row[6] or ''} {row[7] or ''}".strip() or "Unknown"
                        })
                    return jsonify(communications)
                
                elif request.method == "POST":
                    data = request.json
                    comm_type = data.get("type", "note")
                    subject = data.get("subject", "")
                    content = data.get("content", "")
                    direction = data.get("direction", "outbound")
                    
                    cur.execute("""
                        INSERT INTO crm_schema.communication_logs 
                        (lead_id, user_id, communication_type, subject, content, direction)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (lead_id, user_id, comm_type, subject, content, direction))
                    
                    comm_id = cur.fetchone()[0]
                    
                    # Log activity
                    cur.execute("""
                        INSERT INTO crm_schema.activities (lead_id, user_id, activity_type, title, description, created_by, organization_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (lead_id, user_id, 'communication_added', 
                          f'{comm_type.title()} added', 
                          f'{comm_type.title()}: {subject or content[:50]}...', user_id, user_org_id))
                    
                    return jsonify({"message": "Communication logged successfully", "id": comm_id}), 201
                    
    except psycopg2.Error as e:
        return jsonify({"message": "Database error managing communications.", "error": str(e)}), 500

# ==============================================================================
# --- LEAD ENRICHMENT ENDPOINT ---
# ==============================================================================

@app.route("/api/leads/<int:lead_id>/enrich", methods=["POST"])
@token_required
def enrich_lead(current_user, lead_id):
    """
    Enriches a lead with public data and AI analysis.
    """
    if llm is None:
        return jsonify({"error": "AI service not available"}), 503

    user_id = current_user["user_id"]
    conn = get_db()
    
    try:
        with conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                # --- 1. Fetch Lead Info ---
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}), 400
                user_org_id = org_id_tuple['organization_id']

                cur.execute(
                    "SELECT name, email, organization_name FROM crm_schema.leads WHERE id = %s AND organization_id = %s",
                    (lead_id, user_org_id)
                )
                lead = cur.fetchone()
                if not lead:
                    return jsonify({"message": "Lead not found or access denied."}), 404

                lead_name = lead['name']
                company_name = lead['organization_name']
                
                # --- 2. Gather Public Data ---
                domain = get_domain_from_email(lead['email'])
                if not domain and company_name:
                    # Fallback: try to guess domain from company name
                    domain_guess = re.sub(r'\W+', '', company_name.lower().split(' ')[0]) + '.com'
                    domain = domain_guess
                
                website_data = scrape_website(domain)
                news_data = get_news(company_name)
                
                context_data = f"""
                ---
                Lead Name: {lead_name}
                Company Name: {company_name}
                Company Domain: {domain}
                ---
                Website Scraping:
                {website_data}
                ---
                Public News:
                {news_data}
                ---
                """
                
                logging.info(f"Enrichment context for lead {lead_id}:\n{context_data}")

                # --- 3. AI Reasoning (Theta) ---
                system_prompt = f"""
                You are "Theta", a sales intelligence analyst. Your task is to analyze a lead and their company based on public data.
                Provide your analysis ONLY in the following JSON format. Do not include any other text or explanation.

                JSON Format:
                {{
                  "summary": "A concise, one-sentence summary of what the company does.",
                  "persona": "Infer the probable role or persona of the lead (e.g., 'Marketing Director', 'CTO', 'Founder').",
                  "intent_score": "An estimated buying intent score from 0 (no intent) to 100 (ready to buy), as an integer.",
                  "pain_points": [
                    "Probable pain point 1",
                    "Probable pain point 2",
                    "Probable pain point 3"
                  ],
                  "outreach_message": "A 2-sentence, personalized outreach message. Use '{{name}}' as a placeholder for the lead's name."
                }}

                Analyze the following data and generate the JSON output:
                {context_data}
                """
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the JSON analysis for this lead."}
                ]
        
                response = llm.invoke(messages)
                ai_response_str = response.content if hasattr(response, 'content') else str(response)

                # --- 4. Parse and Store Result ---
                try:
                    # Use json_repair to fix any malformed JSON from the LLM
                    enriched_data = json_repair.loads(ai_response_str)
                except Exception as e:
                    logging.error(f"Failed to parse AI JSON for lead {lead_id}: {e}\nRaw response: {ai_response_str}")
                    return jsonify({"message": "AI failed to return valid data.", "raw": ai_response_str}), 500

                # --- 5. Persist to DB ---
                cur.execute(
                    "UPDATE crm_schema.leads SET enrichment_data = %s WHERE id = %s",
                    (json.dumps(enriched_data), lead_id)
                )

                return jsonify(enriched_data), 200

    except psycopg2.Error as e:
        logging.error(f"Database error during lead enrichment: {e}")
        return jsonify({"message": "Database error.", "error": str(e)}), 500
    except Exception as e:
        logging.error(f"General error during lead enrichment: {e}", exc_info=True)
        return jsonify({"message": "An unexpected error occurred.", "error": str(e)}), 500

@app.route("/leads/<int:lead_id>/tasks", methods=["GET", "POST"])
@token_required
def manage_lead_tasks(current_user, lead_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to lead
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You are not in an organization."}), 400
                user_org_id = org_id_tuple[0]
                
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple or user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Lead not found or access denied."}), 404
                
                if request.method == "GET":
                    cur.execute("""
                        SELECT t.id, t.title, t.description, t.status, t.priority, t.due_date, 
                               t.created_at, t.completed_at, u.first_name, u.last_name
                        FROM crm_schema.tasks t
                        LEFT JOIN crm_schema.users u ON t.user_id = u.id
                        WHERE t.lead_id = %s
                        ORDER BY t.created_at DESC
                    """, (lead_id,))
                    tasks = []
                    for row in cur.fetchall():
                        tasks.append({
                            "id": row[0], "title": row[1], "description": row[2],
                            "status": row[3], "priority": row[4], "due_date": row[5],
                            "created_at": row[6], "completed_at": row[7],
                            "user_name": f"{row[8] or ''} {row[9] or ''}".strip() or "Unknown"
                        })
                    return jsonify(tasks)
                
                elif request.method == "POST":
                    data = request.json
                    title = data.get("title", "")
                    description = data.get("description", "")
                    priority = data.get("priority", "medium")
                    due_date = data.get("due_date")
                    
                    if not title:
                        return jsonify({"message": "Task title is required."}), 400
                    
                    cur.execute("""
                        INSERT INTO crm_schema.tasks 
                        (lead_id, user_id, title, description, priority, due_date)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (lead_id, user_id, title, description, priority, due_date))
                    
                    task_id = cur.fetchone()[0]
                    
                    # Log activity
                    cur.execute("""
                        INSERT INTO crm_schema.activities (lead_id, user_id, activity_type, title, description, created_by, organization_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (lead_id, user_id, 'task_created', 
                          'Task created', 
                          f'Task: {title}', user_id, user_org_id))
                    
                    return jsonify({"message": "Task created successfully", "id": task_id}), 201
                    
    except psycopg2.Error as e:
        return jsonify({"message": "Database error managing tasks.", "error": str(e)}), 500

@app.route("/tasks/<int:task_id>", methods=["PUT"])
@token_required
def update_task(current_user, task_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to task
                cur.execute("""
                    SELECT t.lead_id, t.user_id FROM crm_schema.tasks t
                    JOIN crm_schema.leads l ON t.lead_id = l.id
                    JOIN crm_schema.user_organizations uo ON l.organization_id = uo.organization_id
                    WHERE t.id = %s AND uo.user_id = %s
                """, (task_id, user_id))
                task_info = cur.fetchone()
                if not task_info:
                    return jsonify({"message": "Task not found or access denied."}), 404
                
                lead_id = task_info[0]
                data = request.json
                
                fields, params = [], []
                for key in ["title", "description", "status", "priority", "due_date", "is_pinned"]:
                    if key in data:
                        fields.append(f"{key} = %s")
                        params.append(data[key])
                
                if fields:
                    params.append(task_id)
                    query = f"UPDATE crm_schema.tasks SET {', '.join(fields)} WHERE id = %s"
                    cur.execute(query, tuple(params))
                    
                    # Log activity
                    cur.execute("""
                        INSERT INTO crm_schema.activities (lead_id, user_id, activity_type, title, description, created_by, organization_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (lead_id, user_id, 'task_updated', 
                          'Task updated', 
                          f'Task was updated', user_id, user_org_id))
                
                return jsonify({"message": "Task updated successfully"})
                    
    except psycopg2.Error as e:
        return jsonify({"message": "Database error updating task.", "error": str(e)}), 500

@app.route("/tasks/<int:task_id>", methods=["DELETE"])
@token_required
def delete_task(current_user, task_id):
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verify user has access to task
                cur.execute("""
                    SELECT t.lead_id FROM crm_schema.tasks t
                    JOIN crm_schema.leads l ON t.lead_id = l.id
                    JOIN crm_schema.user_organizations uo ON l.organization_id = uo.organization_id
                    WHERE t.id = %s AND uo.user_id = %s
                """, (task_id, user_id))
                task_info = cur.fetchone()
                if not task_info:
                    return jsonify({"message": "Task not found or access denied."}), 404
                
                cur.execute("DELETE FROM crm_schema.tasks WHERE id = %s", (task_id,))
                return jsonify({"message": "Task deleted successfully"})
                    
    except psycopg2.Error as e:
        return jsonify({"message": "Database error deleting task.", "error": str(e)}), 500
    

# --- NEW: PRE-SALES & AI INSIGHT ENDPOINTS ---

@app.route("/leads/<int:lead_id>/ai_tasks", methods=["PUT"])
@token_required
def update_lead_ai_tasks(current_user, lead_id):
    # ... (This endpoint is unchanged from the previous step)
    data = request.json
    task_name = data.get("task")
    is_enabled = data.get("enabled")
    if task_name is None or is_enabled is None:
        return jsonify({"message": "Task name and enabled status are required."}),
    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE crm_schema.leads SET ai_tasks = ai_tasks || %s::jsonb WHERE id = %s RETURNING ai_tasks",
                            (json.dumps({task_name: is_enabled}), lead_id))
                updated_tasks = cur.fetchone()[0]
                return jsonify({"message": "AI tasks updated", "tasks": updated_tasks})
    except psycopg2.Error as e:
        return jsonify({"message": "Database error updating AI tasks.", "error": str(e)}), 500




# ==============================================================================
# --- LEADS IMPORT ENDPOINT (FULLY MODIFIED) ---
# ==============================================================================

@app.route("/leads/import", methods=["POST"])
@token_required
def import_leads_with_ai(current_user):
    """
    Parses, analyzes, and imports leads from CSV/Excel using AI,
    with robust JSON handling (repair, Pydantic list validation, self-correction).
    """
    user_id = current_user["user_id"]
    if 'file' not in request.files:
        logging.warning("Import attempt failed: No file part in request.")
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("Import attempt failed: No file selected.")
        return jsonify({"message": "No file selected"}), 400

    # --- 1. File Handling & Reading ---
    filename = file.filename.lower()
    logging.info(f"Starting import: {filename} by user {user_id}")
    df = None
    try:
        if filename.endswith('.csv'):
            try: df = pd.read_csv(file.stream, encoding='utf-8', engine='python', on_bad_lines='warn')
            except UnicodeDecodeError: file.stream.seek(0); df = pd.read_csv(file.stream, encoding='latin-1', engine='python', on_bad_lines='warn')
            except pd.errors.ParserError as pe: raise ValueError(f"Error parsing CSV: {pe}")
        elif filename.endswith(('.xls', '.xlsx')): df = pd.read_excel(file.stream, engine='openpyxl')
        else: return jsonify({"message": "Invalid file type. Please upload CSV or Excel."}), 400
        if df is None or df.empty: return jsonify({"message": "File is empty or unreadable."}), 400
    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}", exc_info=True)
        return jsonify({"message": f"Error reading file: {e}"}), 400

    # --- Optional: Email Cleaning Logic (same as before) ---
    email_column_name = next((col for col in df.columns if 'email' in str(col).lower()), None)
    if email_column_name:
        logging.info(f"Attempting to clean email column: '{email_column_name}'")
        def extract_email(value):
             if isinstance(value, str):
                 value_stripped = value.strip()
                 match = re.search(r'\[\s*(.+?@[^\]\s]+)\s*\]\(mailto:[^\)]+\)', value_stripped)
                 if match: return match.group(1).strip()
                 if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value_stripped): return value_stripped
             return value
        try:
            original_emails = df[email_column_name].copy()
            df[email_column_name] = df[email_column_name].apply(extract_email)
            changed_count = (original_emails != df[email_column_name]).sum()
            logging.info(f"Cleaned {changed_count} emails in column: {email_column_name}")
        except Exception as e:
            logging.error(f"Error cleaning email column '{email_column_name}': {e}") # Log but continue

    # --- Prepare data for AI ---
    file_headers = df.columns.tolist()
    MAX_PROMPT_ROWS = 50 # Limit rows sent to AI
    df_to_prompt = df.head(MAX_PROMPT_ROWS)
    try: sample_data_md = df_to_prompt.to_markdown(index=False)
    except ImportError: sample_data_md = df_to_prompt.to_string(index=False)

    # --- 2. Prepare AI Prompt (Initial Attempt) ---
    #
    #    MODIFICATION: This section now needs to be re-wired to call the 
    #    CORHA system instead of an inline LLM.
    #
    #    The old logic (calling llm.invoke) has been removed as it
    #    depended on the deleted 'llm' variable.
    #
    # --- 3. Call AI and Process Response (with Retry Loop) ---

    logging.warning(f"TODO: /leads/import needs to be re-wired to call CORHA system.")
    
    # --- 4. Database Insertion ---
    conn = None
    inserted_count = 0
    skipped_count = 0
    leads_to_insert = []
    
    # ---
    # --- MOCKUP / PLACEHOLDER ---
    # --- This section should be replaced by a call to CORHA
    # --- For now, we return a TODO message.
    # ---
    
    # --- BEGINNING OF OLD LOGIC (FOR REFERENCE, BUT NOW REMOVED/REPLACED) ---
    #
    # try:
    #     # --- 2. Prepare AI Prompt (Initial Attempt) ---
    #     initial_system_prompt = f"""
    #     ... (prompt definition) ...
    #     """
    #     human_prompt = f"Please process the data provided in the system prompt..."
    #
    #     # --- 3. Call AI and Process Response (with Retry Loop) ---
    #     ... (while loop calling llm.invoke, json_repair, Pydantic) ...
    #
    #     # --- 4. Database Insertion ---
    #     ... (database insertion logic) ...
    #
    #     return jsonify({"message": success_message}), 200
    #
    # except psycopg2.Error as e:
    #     ...
    # except Exception as e:
    #     ...
    #
    # --- END OF OLD LOGIC ---

    # --- NEW PLACEHOLDER RESPONSE ---
    return jsonify({
        "message": "TODO: This import endpoint must be re-wired to use the CORHA reasoning engine."
    }), 501 # 501 Not Implemented


# ==============================================================================
# --- DEALS API ENDPOINTS (NEW SECTION) ---
# ==============================================================================

@app.route("/deals", methods=["GET"])
@token_required
def get_deals(current_user):
    user_id = current_user["user_id"]
    timeframe = request.args.get('timeframe', 'weekly')
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You must be in an organization to view deals."}),
            org_id = org_id_tuple[0]

            query = """
                SELECT d.id, d.name, d.value, d.stage, d.lead_id, l.organization_name
                FROM crm_schema.deals d
                LEFT JOIN crm_schema.leads l ON d.lead_id = l.id
                WHERE d.org_id = %s
            """
            params = [org_id]

            if timeframe == 'weekly':
                query += " AND d.created_at >= NOW() - INTERVAL '7 days'"
            elif timeframe == 'monthly':
                query += " AND d.created_at >= NOW() - INTERVAL '1 month'"
            elif timeframe == 'quarterly':
                query += " AND d.created_at >= NOW() - INTERVAL '3 months'"
            
            cur.execute(query, tuple(params))
            deals = []
            for row in cur.fetchall():
                deals.append({
                    "id": row[0], "name": row[1],
                    "value": float(row[2]) if row[2] is not None else None,
                    "stage": row[3], "lead_id": row[4], "lead_name": row[5] or ''
                })
            return jsonify(deals)
    except psycopg2.Error as e:
        return jsonify({"message": "Database error fetching deals.", "error": str(e)}), 500

@app.route("/deals", methods=["POST"])
@token_required
def create_deal(current_user):
    user_id = current_user["user_id"]
    data = request.json
    name, value, lead_id = data.get("name"), data.get("value"), data.get("lead_id")
    stage = data.get("stage", "Qualification")

    if not name or not lead_id:
        return jsonify({"message": "Deal name and associated lead ID are required."}),

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You must be in an organization to create deals."}),
                org_id = org_id_tuple[0]

                cur.execute(
                    """
                    INSERT INTO crm_schema.deals (name, value, stage, lead_id, org_id, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                    """,
                    (name, value, stage, lead_id, org_id, user_id)
                )
                deal_id = cur.fetchone()[0]
                return jsonify({"message": "Deal created successfully", "deal_id": deal_id}), 201
    except psycopg2.Error as e:
        return jsonify({"message": "Database error creating deal.", "error": str(e)}), 500


@app.route("/deals/<int:deal_id>", methods=["PUT"])
@token_required
def update_deal_stage(current_user, deal_id):
    user_id = current_user["user_id"]
    data = request.json
    new_stage = data.get("stage")

    if not new_stage:
        return jsonify({"message": "New stage is required."}),

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                user_org_id = cur.fetchone()[0]
                
                cur.execute("SELECT org_id FROM crm_schema.deals WHERE id = %s", (deal_id,))
                deal_org_tuple = cur.fetchone()
                if not deal_org_tuple:
                    return jsonify({"message": "Deal not found"}), 404
                
                if user_org_id != deal_org_tuple[0]:
                    return jsonify({"message": "Forbidden: You do not have access to this deal."}),

                cur.execute("UPDATE crm_schema.deals SET stage = %s WHERE id = %s", (new_stage, deal_id))
                return jsonify({"message": f"Deal {deal_id} moved to {new_stage}"})
    except psycopg2.Error as e:
        return jsonify({"message": "Database error updating deal.", "error": str(e)}), 500

# --- NEW: POST-SALES ENDPOINTS ---

@app.route("/postsales/accounts", methods=["GET"])
@token_required
def get_postsales_accounts(current_user):
    """Fetches all 'won' deals to be treated as active accounts."""
    user_id = current_user["user_id"]
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "User not part of an organization."}),
            org_id = org_id_tuple[0]

            # Query deals and join with leads to get contact person's name
            cur.execute("""
                SELECT 
                    d.id, d.name, d.stage, d.ai_health_score, d.renewal_date, d.ai_summary, d.ai_tasks,
                    l.name as contact_name,
                    l.email as contact_email
                FROM crm_schema.deals d
                LEFT JOIN crm_schema.leads l ON d.lead_id = l.id
                WHERE d.org_id = %s AND d.stage = 'Closed-Won'
            """, (org_id,))
            
            accounts = [dict(zip([desc[0] for desc in cur.description], row)) for row in cur.fetchall()]
            return jsonify(accounts)

    except psycopg2.Error as e:
        logging.error(f"Database error in get_postsales_accounts: {e}")
        return jsonify({"message": "Database error fetching accounts.", "error": str(e)}), 500

@app.route("/deals/<int:deal_id>/generate_summary", methods=["POST"])
@token_required
def generate_deal_summary(current_user, deal_id):
    # This is a mock AI call. In a real scenario, you'd call your AI service.
    mock_summary = "High product usage and positive feedback. AI proactively sent Q3 performance report. No outstanding issues."
    mock_health_score = "Excellent"
    # Set a mock renewal date one year from today
    mock_renewal_date = datetime.date.today() + datetime.timedelta(days=365)

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE crm_schema.deals SET ai_summary = %s, ai_health_score = %s, renewal_date = %s WHERE id = %s",
                    (mock_summary, mock_health_score, mock_renewal_date, deal_id)
                )
                # Fetch and return the updated deal/account info
                cur.execute("""
                    SELECT d.id, d.name, d.stage, d.ai_health_score, d.renewal_date, d.ai_summary, d.ai_tasks,
                           l.name as contact_name, l.email as contact_email
                    FROM crm_schema.deals d LEFT JOIN crm_schema.leads l ON d.lead_id = l.id WHERE d.id = %s
                """, (deal_id,))
                account = dict(zip([desc[0] for desc in cur.description], cur.fetchone()))
                return jsonify(account)
    except psycopg2.Error as e:
        return jsonify({"message": "Database error generating summary.", "error": str(e)}), 500

@app.route("/deals/<int:deal_id>/ai_tasks", methods=["PUT"])
@token_required
def update_deal_ai_tasks(current_user, deal_id):
    data = request.json
    task_name = data.get("task")
    is_enabled = data.get("enabled")

    if task_name is None or is_enabled is None:
        return jsonify({"message": "Task name and enabled status are required."}),

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # Atomically update the JSONB field for the specified deal
                cur.execute("UPDATE crm_schema.deals SET ai_tasks = ai_tasks || %s::jsonb WHERE id = %s RETURNING ai_tasks",
                            (json.dumps({task_name: is_enabled}), deal_id))
                updated_tasks = cur.fetchone()[0]
                return jsonify({"message": "AI tasks for deal updated", "tasks": updated_tasks})
    except psycopg2.Error as e:
        return jsonify({"message": "Database error updating deal AI tasks.", "error": str(e)}), 500

# ==============================================================================
# --- DASHBOARD API ENDPOINTS (NEW SECTION) ---
# ==============================================================================

@app.route("/space/data", methods=["GET"])
@token_required
def get_space_data(current_user):
    """Provides REAL, time-series aggregated data for the space dashboard."""
    user_id = current_user["user_id"]
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "User not part of any organization"}), 403
            org_id = org_id_tuple[0]

            # --- 1. AGGREGATE METRIC CALCULATIONS (for the main numbers) ---
            cur.execute("SELECT SUM(value) FROM crm_schema.deals WHERE stage = 'Closed-Won' AND org_id = %s", (org_id,))
            total_revenue = cur.fetchone()[0] or 0
            cur.execute("SELECT COUNT(*) FROM crm_schema.deals WHERE stage = 'Closed-Won' AND org_id = %s", (org_id,))
            deals_won = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM crm_schema.deals WHERE org_id = %s", (org_id,))
            opportunities = cur.fetchone()[0]

            # --- 2. TIME-SERIES QUERIES (for the charts) ---
            # Define the last 6 months for our chart labels
            today = datetime.date.today()
            labels = []
            for i in range(5, -1, -1):
                month = today - datetime.timedelta(days=i*30)
                labels.append(month.strftime("%b")) # e.g., "Oct", "Nov"

            # Query for monthly revenue
            cur.execute("""
                SELECT 
                    to_char(date_trunc('month', created_at), 'Mon') as month,
                    SUM(value) as monthly_revenue
                FROM crm_schema.deals
                WHERE stage = 'Closed-Won' AND org_id = %s AND created_at > NOW() - INTERVAL '6 months'
                GROUP BY date_trunc('month', created_at)
                ORDER BY date_trunc('month', created_at);
            """, (org_id,))
            revenue_by_month_raw = {row[0].strip(): float(row[1]) for row in cur.fetchall()}
            
            # Query for monthly deals won
            cur.execute("""
                SELECT 
                    to_char(date_trunc('month', created_at), 'Mon') as month,
                    COUNT(*) as deals_count
                FROM crm_schema.deals
                WHERE stage = 'Closed-Won' AND org_id = %s AND created_at > NOW() - INTERVAL '6 months'
                GROUP BY date_trunc('month', created_at)
                ORDER BY date_trunc('month', created_at);
            """, (org_id,))
            deals_won_by_month_raw = {row[0].strip(): int(row[1]) for row in cur.fetchall()}
            
            # Helper to map query results to our fixed 6-month labels
            def map_data_to_labels(labels_list, data_dict):
                return [data_dict.get(label, 0) for label in labels_list]

            revenue_chart_data = map_data_to_labels(labels, revenue_by_month_raw)
            deals_won_chart_data = map_data_to_labels(labels, deals_won_by_month_raw)
            # For now, opportunities will just be a count over time
            # A more complex query could be added later
            opportunities_chart_data = [0] * len(labels)
            
            # --- 3. FINAL DATA STRUCTURE ---
            data = {
                "revenue": {
                    "labels": labels, "data": revenue_chart_data, 
                    "value": f"${int(total_revenue):,}", "comparison": "Last 6 months"
                },
                "closedWon": {
                    "labels": labels, "data": deals_won_chart_data, 
                    "value": str(deals_won), "comparison": "Last 6 months"
                },
                "opportunityCreation": {
                    "labels": labels, "data": opportunities_chart_data, 
                    "value": str(opportunities), "comparison": "Live Data"
                },
                "demoBookedRate": {"labels": labels, "data": [0]*len(labels), "value": "0%", "comparison": "Not Tracked"},
                "responseRate": {"labels": labels, "data": [0]*len(labels), "value": "0%", "comparison": "Not Tracked"},
                "openReplyRate": {"labels": labels, "data": [0]*len(labels), "value": "0%", "comparison": "Not Tracked"},
            }
            return jsonify(data)

    except psycopg2.Error as e:
        logging.error(f"Database error in get_space_data: {e}")
        return jsonify({"message": "Database error fetching dashboard data.", "error": str(e)}), 500
    
@app.route("/leads/<int:lead_id>/activity", methods=["GET"])
@token_required
def get_lead_activity(current_user, lead_id):
    """Fetches the activity timeline for a specific lead."""
    user_id = current_user["user_id"]
    conn = get_db()
    try:
        with conn.cursor() as cur:
            # First, verify the user has access to this lead's organization
            cur.execute("""
                SELECT 1 FROM crm_schema.leads l
                JOIN crm_schema.user_organizations uo ON l.organization_id = uo.organization_id
                WHERE l.id = %s AND uo.user_id = %s
            """, (lead_id, user_id))
            if cur.fetchone() is None:
                return jsonify({"message": "Forbidden: You do not have access to this lead."}),

            # Fetch the activity history
            query = """
                SELECT id, activity_type, title, description, created_at, user_id
                FROM crm_schema.activities
                WHERE lead_id = %s
            """
            params = [lead_id]
            activity_filter = request.args.get('type')
            if activity_filter and activity_filter != 'all':
                query += " AND activity_type = ANY(%s)"
                params.append(activity_filter.split(','))

            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            limit = int(request.args.get('limit', 20))
            offset = int(request.args.get('offset', 0))
            params.extend([limit, offset])

            cur.execute(query, tuple(params))
            
            activities = []
            columns = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                activities.append(dict(zip(columns, row)))
            
            return jsonify({"activities": activities, "has_more": len(activities) == limit})

    except psycopg2.Error as e:
        logging.error(f"Database error in get_lead_activity: {e}")
        return jsonify({"message": "Database error fetching lead activity.", "error": str(e)}), 500

@app.route("/settings/oauth2callback", methods=["GET"])
def gmail_oauth_callback():
    """
    Proxy Google OAuth callback from /settings/oauth2callback
    to the correct MCP worker's /oauth2callback endpoint based on the state.
    """
    state = request.args.get('state', '')
    target_port = GMEET_MCP_PORT if state.startswith('gmeet') else MCP_PORT
    
    mcp_url = f"http://{MCP_HOST}:{target_port}/oauth2callback"

    try:
        # Forward the request with all its original arguments to the correct worker
        response = requests.get(mcp_url, params=request.args, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes
        # Return the exact response from the worker
        return (response.text, response.status_code, {'Content-Type': response.headers.get('Content-Type')})
    except requests.exceptions.RequestException as e:
        logging.error(f"OAuth callback proxy failed: {e}")
        return "<h1>Callback Error</h1><p>Failed to communicate with the integration service.</p>", 502


@app.route("/calendar/events", methods=["GET"])
@token_required
def get_calendar_events(current_user):
    """
    Fetches calendar events for the logged-in user's organization from the database.
    """
    user_id = current_user["user_id"]
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify([]), 200
            org_id = org_id_tuple[0]

            cur.execute("""
                SELECT id, title, start_time, end_time, extended_props 
                FROM crm_schema.calendar_events 
                WHERE org_id = %s
                ORDER BY start_time ASC
            """, (org_id,))
            
            events = []
            for row in cur.fetchall():
                extended_props = row[4] or {}
                events.append({
                    "id": str(row[0]),
                    "title": row[1],
                    "start": row[2].isoformat() if row[2] else None,
                    "end": row[3].isoformat() if row[3] else None,
                    "extendedProps": extended_props
                })
            return jsonify(events)
    except psycopg2.Error as e:
        logging.error(f"Database error fetching calendar events: {e}")
        return jsonify({"message": "Database error fetching events."}),

@app.route("/calendar/events", methods=["POST"])
@token_required
def create_calendar_event(current_user):
    """
    Creates a new calendar event for the user in the database and syncs it
    with their primary external calendar if one is connected.
    """
    user_id = current_user["user_id"]
    data = request.json
    if not data or not data.get('title') or not data.get('start'):
        return jsonify({"message": "Title and start time are required."}),

    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # --- 1. Get User and Org Info ---
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "Cannot create event: User not in an organization."}),
                org_id = org_id_tuple[0]

                cur.execute("SELECT primary_meeting_key FROM crm_schema.users WHERE id = %s", (user_id,))
                primary_meeting_key = cur.fetchone()[0]

                # --- 2. Sync with External Calendar (if configured) ---
                external_event_id = None
                meeting_link = None
                
                # Check if user wants to create an online meeting for this specific event
                create_link = data.get('create_meeting_link', False)

                if primary_meeting_key and create_link:
                    attendees = data.get('extendedProps', {}).get('attendees', [])
                    
                    # Prepare payload for the MCP
                    mcp_payload = {
                        "summary": data['title'],
                        "start_time": data['start'],
                        "end_time": data.get('end'),
                        "attendees": attendees
                    }
                    
                    # Determine which MCP to call
                    port = None
                    if primary_meeting_key == 'gmeet':
                        port = GMEET_MCP_PORT
                    elif primary_meeting_key == 'outlook_calendar':
                        # Assuming you have a different port or endpoint for outlook
                        port = MCP_PORT 

                    if port:
                        try:
                            # Use the proxy function to handle the request
                            sync_response, status_code = proxy_to_mcp(
                                "actions/create_meeting", 
                                current_user, 
                                method="POST", 
                                data=mcp_payload, 
                                port=port
                            )
                            if status_code in [200, 201]:
                                response_data = sync_response.get_json()
                                external_event_id = response_data.get("id")
                                meeting_link = response_data.get("htmlLink")
                                logging.info(f"Successfully created external event {external_event_id} via {primary_meeting_key}")
                            else:
                                logging.warning(f"Failed to sync event to {primary_meeting_key}. Status: {status_code}, Response: {sync_response.get_data(as_text=True)}")
                        except Exception as e:
                            logging.error(f"Error during external calendar sync: {e}")
                            # Decide if you want to fail the whole operation or just log the error
                            # For now, we'll just log it and continue creating the internal event

                # --- 3. Create Internal Calendar Event ---
                extended_props = data.get('extendedProps', {})
                if external_event_id:
                    extended_props['external_event_id'] = external_event_id
                if meeting_link:
                    extended_props['meetingLink'] = meeting_link

                cur.execute(
                    """
                    INSERT INTO crm_schema.calendar_events 
                    (title, start_time, end_time, user_id, org_id, extended_props) 
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id, title, start_time, end_time, extended_props
                    """,
                    (
                        data['title'], data['start'], data.get('end'),
                        user_id, org_id, json.dumps(extended_props)
                    )
                )
                new_event_data = cur.fetchone()
                new_event = {
                    "id": str(new_event_data[0]), 
                    "title": new_event_data[1],
                    "start": new_event_data[2].isoformat() if new_event_data[2] else None,
                    "end": new_event_data[3].isoformat() if new_event_data[3] else None,
                    "extendedProps": new_event_data[4] or {}
                }
                return jsonify(new_event), 201
    except psycopg2.Error as e:
        logging.error(f"Database error creating calendar event: {e}")
        return jsonify({"message": "Database error creating event."}),
    except Exception as e:
        logging.error(f"An unexpected error occurred in create_calendar_event: {e}")
        return jsonify({"message": f"An unexpected server error occurred: {str(e)}"}), 500
    


@app.route("/calendar/events/<string:event_id>", methods=["PUT"])
@token_required
def update_calendar_event(current_user, event_id):
    """
    Updates an existing calendar event and syncs changes to the external provider.
    """
    user_id = current_user["user_id"]
    data = request.json
    
    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                # --- 1. Fetch Existing Event & User Permissions ---
                cur.execute("""
                    SELECT ce.extended_props, u.primary_meeting_key 
                    FROM crm_schema.calendar_events ce
                    JOIN crm_schema.user_organizations uo ON ce.org_id = uo.organization_id
                    JOIN crm_schema.users u ON uo.user_id = u.id
                    WHERE ce.id = %s AND uo.user_id = %s
                """, (event_id, user_id))
                
                event_info = cur.fetchone()
                if event_info is None:
                    return jsonify({"message": "Forbidden or event not found."}),
                
                existing_props, primary_meeting_key = event_info
                existing_props = existing_props or {}

                # --- 2. Sync Update with External Calendar ---
                external_event_id = existing_props.get('external_event_id')
                if primary_meeting_key and external_event_id:
                    mcp_payload = {
                        "summary": data.get('title'),
                        "start_time": data.get('start'),
                        "end_time": data.get('end'),
                        "attendees": data.get('extendedProps', {}).get('attendees', [])
                    }
                    
                    port = GMEET_MCP_PORT if primary_meeting_key == 'gmeet' else MCP_PORT
                    
                    try:
                        # The endpoint for updating is often actions/update_meeting/:event_id
                        update_endpoint = f"actions/update_meeting/{external_event_id}"
                        sync_response, status_code = proxy_to_mcp(
                            update_endpoint, 
                            current_user, 
                            method="PUT", 
                            data=mcp_payload, 
                            port=port
                        )
                        if status_code not in [200, 201]:
                            logging.warning(f"Failed to sync event update to {primary_meeting_key}. Status: {status_code}, Response: {sync_response.get_data(as_text=True)}")
                        else:
                            logging.info(f"Successfully synced update for event {external_event_id} to {primary_meeting_key}")
                    except Exception as e:
                        logging.error(f"Error during external calendar update sync: {e}")

                # --- 3. Update Internal Calendar Event ---
                # Merge new props with existing ones, but let the new data take precedence
                updated_props = {**existing_props, **data.get('extendedProps', {})}

                cur.execute(
                    """
                    UPDATE crm_schema.calendar_events SET
                    title = %s, start_time = %s, end_time = %s, extended_props = %s
                    WHERE id = %s
                    RETURNING id, title, start_time, end_time, extended_props
                    """,
                    (
                        data.get('title'), data.get('start'), data.get('end'),
                        json.dumps(updated_props), event_id
                    )
                )
                updated_event_data = cur.fetchone()
                if updated_event_data:
                    updated_event = {
                        "id": str(updated_event_data[0]),
                        "title": updated_event_data[1],
                        "start": updated_event_data[2].isoformat() if updated_event_data[2] else None,
                        "end": updated_event_data[3].isoformat() if updated_event_data[3] else None,
                        "extendedProps": updated_event_data[4] or {}
                    }
                    return jsonify(updated_event)
                # This part is unlikely to be reached due to the RETURNING clause, but it's safe to keep
                return jsonify({"message": "Event updated but could not retrieve updated data."}),
    except psycopg2.Error as e:
        logging.error(f"Database error updating event {event_id}: {e}")
        return jsonify({"message": "Database error updating event."}),
    except Exception as e:
        logging.error(f"An unexpected error occurred in update_calendar_event: {e}")
        return jsonify({"message": f"An unexpected server error occurred: {str(e)}"}), 500

@app.route("/calendar/events/<string:event_id>", methods=["DELETE"])
@token_required
def delete_calendar_event(current_user, event_id):
    """
    Deletes a calendar event from the database.
    """
    user_id = current_user["user_id"]
    try:
        conn = get_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM crm_schema.calendar_events 
                    WHERE id = %s AND org_id = (SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s)
                    """, 
                    (event_id, user_id)
                )
                if cur.rowcount == 0:
                    return jsonify({"message": "Event not found or you do not have permission to delete it."}),
                
                return jsonify({"message": "Event deleted successfully"}), 200
    except psycopg2.Error as e:
        logging.error(f"Database error deleting event {event_id}: {e}")
        return jsonify({"message": "Database error deleting event."}),
    
@app.route("/connect/gmeet", methods=["GET"])
@token_required
def connect_gmeet_via_mcp(current_user):
    """Proxy to GMeet MCP worker for Google Meet OAuth connection."""
    return proxy_to_mcp("connect/gmeet", current_user, method="GET", params=request.args, port=GMEET_MCP_PORT)

@app.route("/disconnect/gmeet", methods=["POST"])
@token_required
def disconnect_gmeet_via_mcp(current_user):
    """Proxy to GMeet MCP worker to disconnect Google Meet."""
    return proxy_to_mcp("disconnect/gmeet", current_user, method="POST", port=GMEET_MCP_PORT)

@app.route("/actions/create_meeting", methods=['POST'])
@token_required
def proxy_create_meeting(current_user):
    """Proxy for creating a Google Meet event."""
    return proxy_to_mcp("actions/create_meeting", current_user, method="POST", port=GMEET_MCP_PORT)

@app.route("/api/calendar", methods=["GET"])
@token_required
def get_lead_calendar_events(current_user):
    """
    Fetches calendar events related to a specific lead.
    """
    user_id = current_user["user_id"]
    lead_id = request.args.get('leadId')
    
    if not lead_id:
        return jsonify({"message": "leadId query parameter is required."}), 400
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            # Verify user has access to lead
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify([]), 200
            user_org_id = org_id_tuple[0]
            
            cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
            lead_org_tuple = cur.fetchone()
            if not lead_org_tuple or user_org_id != lead_org_tuple[0]:
                return jsonify({"message": "Lead not found or access denied."}), 404
            
            # Get events where extended_props contains the lead_id
            cur.execute("""
                SELECT id, title, start_time, end_time, extended_props 
                FROM crm_schema.calendar_events 
                WHERE org_id = %s AND extended_props::text LIKE %s
                ORDER BY start_time ASC
            """, (user_org_id, f'%{lead_id}%'))
            
            events = []
            for row in cur.fetchall():
                extended_props = row[4] or {}
                # Double-check that this event is actually related to the lead
                if str(extended_props.get('lead_id', '')) == str(lead_id):
                    events.append({
                        "id": str(row[0]),
                        "title": row[1],
                        "start": row[2].isoformat() if row[2] else None,
                        "end": row[3].isoformat() if row[3] else None,
                        "extendedProps": extended_props
                    })
            return jsonify(events)
    except psycopg2.Error as e:
        logging.error(f"Database error fetching lead calendar events: {e}")
        return jsonify({"message": "Database error fetching events."}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Catches all unhandled exceptions and returns a JSON error response."""
    logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
    # Customize error message for specific exception types if needed
    if isinstance(e, psycopg2.Error):
        return jsonify(message="A database error occurred.", error=str(e)), 500
    if isinstance(e, jwt.ExpiredSignatureError):
        return jsonify(message="Session expired. Please log in again.", error="token_expired"), 401
        
    return jsonify(message="An internal server error occurred.", error=str(e)), 500

@app.route("/settings/primary_meeting", methods=["PUT"])
@token_required
def set_primary_meeting(current_user):
    user_id = current_user.get("user_id")
    data = request.json
    integration_key = data.get("integration_key")

    if not user_id or not integration_key:
        return jsonify({"message": "User ID and integration key are required."}),

    try:
        conn = get_db()
        with conn.cursor() as cur:
            # Update the new primary_meeting_key column
            cur.execute(
                "UPDATE crm_schema.users SET primary_meeting_key = %s WHERE id = %s",
                (integration_key, user_id)
            )
        conn.commit()
        
        # Return the complete, updated settings object to the frontend
        return get_user_settings()

    except psycopg2.Error as e:
        logging.error(f"DATABASE ERROR in set_primary_meeting: {e}")
        if conn:
            conn.rollback()
        return jsonify({"message": "Database error updating setting.", "error": str(e)}), 500
    
@app.route("/calendar/sync", methods=["POST"])
def handle_calendar_sync():
    """
    Webhook endpoint for bi-directional calendar sync.
    Handles notifications from external providers (e.g., Google, Outlook).
    """
    # The logic here will depend heavily on the provider (Google, Outlook, etc.)
    # and the structure of the webhook payload.
    
    # 1. Authenticate the webhook (e.g., by checking a secret header)
    # This is crucial for security to ensure the request is from a trusted source.
    
    # 2. Parse the incoming notification
    notification_data = request.json
    logging.info(f"Received calendar sync notification: {json.dumps(notification_data)}")

    # 3. Determine the event type (created, updated, deleted)
    event_state = notification_data.get('state') # e.g., 'exists', 'cancelled' 
    external_event_id = notification_data.get('id')
    
    # 4. Find the corresponding user and internal event
    # This might involve looking up the user by email or a channel ID provided
    # in the webhook, and then finding the event by its external_event_id.

    # 5. Apply changes to the internal database
    # - If event is new, create it in `crm_schema.calendar_events`.
    # - If event is updated, modify the existing record.
    # - If event is cancelled, delete the record.
    
    # Placeholder response
    return jsonify({"status": "received"}), 200

# ==============================================================================
# --- RUN APPLICATION ---
# ==============================================================================
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)