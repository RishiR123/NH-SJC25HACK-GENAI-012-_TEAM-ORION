
import json
import psycopg2
import psycopg2.errors
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
import decimal
import json_repair
from flask import Flask, request, jsonify, render_template, g , Response , stream_with_context, redirect
from functools import wraps
import functools
from flask_cors import CORS
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from psycopg2.extras import DictCursor
import pandas as pd
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field,field_validator, ValidationError
from typing import Literal
from typing import Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain import hub # <-- ADDED
import agent_tools

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

class ChartDataset(BaseModel):
    """A single dataset for a Chart.js chart."""
    label: str = Field(description="The human-readable name for this dataset (e.g., 'Revenue', 'Leads').")
    data: List[Any] = Field(description="The list of numerical data points for this dataset.")

class ChartGenerationQuery(BaseModel):
    """The SQL query and metadata needed to generate a chart."""
    sql_query: str = Field(description="A valid, read-only PostgreSQL query to fetch the data.")
    chart_type: Literal['bar', 'line', 'pie', 'doughnut'] = Field(description="The recommended chart type.")
    title: str = Field(description="A descriptive title for the chart (e.g., 'Revenue by Month').")
    label_column: str = Field(description="The name of the column in the SQL query that should be used for the chart's labels (e.g., 'month', 'status').")
    data_columns: List[str] = Field(description="A list of column names in the SQL query to be used as data (e.g., ['total_revenue', 'lead_count']).")

class ChartData(BaseModel):
    """The final, Chart.js-compatible JSON structure."""
    type: Literal['bar', 'line', 'pie', 'doughnut'] = Field(description="The type of chart to render.")
    data: Dict[str, Any] = Field(description="The Chart.js data object, containing 'labels' and 'datasets'.")
    options: Dict[str, Any] = Field(description="The Chart.js options object, including a title.")

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
    lead_score: Literal['Hot', 'Warm', 'Cold'] = Field('Cold', description="Score ('Hot', 'Warm', 'Cold'). Defaults to 'Cold'.")
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

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.png')

###############################################
# Lead Resource Endpoints: Notes, Documents, Tickets, Calls
###############################################

# Notes Endpoint
@app.route("/api/leads/<int:lead_id>/notes", methods=["GET", "POST"])
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
    except Exception as e:
        print(f"Error in manage_lead_notes: {type(e).__name__}: {e}")
        if isinstance(e, psycopg2.errors.UndefinedTable):
            # Table doesn't exist, create it
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS crm_schema.lead_notes (
                            id SERIAL PRIMARY KEY,
                            lead_id INTEGER NOT NULL,
                            user_id INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                conn.commit()
                # Retry the operation
                return manage_lead_notes(current_user, lead_id)
            except Exception as create_e:
                print(f"Failed to create lead_notes table: {create_e}")
                return jsonify({"message": "Database schema error: Could not create notes table.", "error": str(create_e)}), 500
        else:
            return jsonify({"message": "Database error managing notes.", "error": str(e)}), 500

# Documents Endpoint
@app.route("/api/leads/<int:lead_id>/documents", methods=["GET", "POST"])
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
                        SELECT id, filename, url, uploaded_at, user_id, verified FROM crm_schema.lead_documents
                        WHERE lead_id = %s
                        ORDER BY uploaded_at DESC
                    """, (lead_id,))
                    documents = []
                    for row in cur.fetchall():
                        documents.append({
                            "id": row[0], "filename": row[1], "url": row[2], "uploaded_at": row[3], "user_id": row[4], "verified": row[5]
                        })
                    return jsonify(documents)
                elif request.method == "POST":
                    data = request.json
                    filename = data.get("filename", "")
                    url = data.get("url", "")
                    verified = data.get("verified", False)
                    if not filename or not url:
                        return jsonify({"message": "Filename and URL are required."}), 400
                    cur.execute("""
                        INSERT INTO crm_schema.lead_documents (lead_id, user_id, filename, url, verified)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (lead_id, user_id, filename, url, verified))
                    doc_id = cur.fetchone()[0]
                    return jsonify({"message": "Document added successfully", "id": doc_id}), 201
    except Exception as e:
        print(f"Error in manage_lead_documents: {type(e).__name__}: {e}")
        if isinstance(e, psycopg2.errors.UndefinedTable):
            # Table doesn't exist, create it
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS crm_schema.lead_documents (
                            id SERIAL PRIMARY KEY,
                            lead_id INTEGER NOT NULL,
                            user_id INTEGER NOT NULL,
                            filename TEXT NOT NULL,
                            url TEXT NOT NULL,
                            verified BOOLEAN DEFAULT FALSE,
                            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                conn.commit()
                # Retry the operation
                return manage_lead_documents(current_user, lead_id)
            except Exception as create_e:
                print(f"Failed to create lead_documents table: {create_e}")
                return jsonify({"message": "Database schema error: Could not create documents table.", "error": str(create_e)}), 500
        else:
            return jsonify({"message": "Database error managing documents.", "error": str(e)}), 500

@app.route("/api/leads/<int:lead_id>/documents/<int:doc_id>", methods=["PUT"])
@app.route("/leads/<int:lead_id>/documents/<int:doc_id>", methods=["PUT"])
@token_required
def update_lead_document(current_user, lead_id, doc_id):
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
                # Verify document belongs to lead
                cur.execute("SELECT id FROM crm_schema.lead_documents WHERE id = %s AND lead_id = %s", (doc_id, lead_id))
                if not cur.fetchone():
                    return jsonify({"message": "Document not found or access denied."}), 404
                data = request.json
                verified = data.get("verified")
                if verified is None:
                    return jsonify({"message": "Verified status is required."}), 400
                cur.execute("""
                    UPDATE crm_schema.lead_documents
                    SET verified = %s
                    WHERE id = %s
                """, (verified, doc_id))
                return jsonify({"message": "Document updated successfully"}), 200
    except psycopg2.Error as e:
        return jsonify({"message": "Database error updating document.", "error": str(e)}), 500

def generate_lead_insight(lead_id: int, user_id: int) -> Dict[str, Any]:
    """
    Fetches lead data, calls LLM for analysis, updates DB, and returns insight.
    """
    global llm # Ensure access to the global LLM
    if llm is None:
        return {"error": "LLM not available for insight generation."}

    conn = None
    try:
        conn = get_db()
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # --- 1. Fetch Lead Data (Expand as needed) ---
            # Get basic lead info
            cur.execute("""
                SELECT l.*, u.first_name as owner_name
                FROM crm_schema.leads l
                LEFT JOIN crm_schema.users u ON l.user_id = u.id
                WHERE l.id = %s
            """, (lead_id,))
            lead_data = cur.fetchone()
            if not lead_data:
                return {"error": "Lead not found."}
            lead_dict = dict(lead_data) # Convert to mutable dict

            # Get recent activities (Example - adjust query as needed)
            cur.execute("""
                SELECT activity_type, title, description, created_at
                FROM crm_schema.activities
                WHERE lead_id = %s ORDER BY created_at DESC LIMIT 5
            """, (lead_id,))
            activities = [dict(row) for row in cur.fetchall()]

            # Get recent communications (Example - adjust query as needed)
            cur.execute("""
                SELECT communication_type, subject, direction, created_at
                FROM crm_schema.communication_logs
                WHERE lead_id = %s ORDER BY created_at DESC LIMIT 5
            """, (lead_id,))
            communications = [dict(row) for row in cur.fetchall()]

            # --- 2. Construct LLM Prompt ---
            # Serialize complex data for the prompt
            lead_details_str = json.dumps({k: str(v) for k, v in lead_dict.items() if k not in ['ai_tasks', 'enrichment_data']}, indent=2) # Basic details
            activities_str = json.dumps([dict(a) for a in activities], default=str, indent=2)
            communications_str = json.dumps([dict(c) for c in communications], default=str, indent=2)

            system_prompt = f"""
            You are an expert Sales AI assistant. Analyze the provided lead information and generate a concise insight, a lead score (Hot, Warm, or Cold), and a specific next best action for the sales representative.

            **Output Format:** Your response MUST be a single, valid JSON object matching this structure:
            {{
              "score": "Hot" | "Warm" | "Cold",
              "insight": "One-sentence analysis of the lead's potential and status.",
              "next_best_action": "A clear, actionable next step for the sales rep."
            }}

            **Lead Data:**
            ```json
            {lead_details_str}
            ```

            **Recent Activities:**
            ```json
            {activities_str}
            ```

            **Recent Communications:**
            ```json
            {communications_str}
            ```

            **Analysis Guidelines:**
            - **Score:**
                - 'Hot': Strong engagement, recent positive interactions, high value, good fit. Urgency required.
                - 'Warm': Some engagement, potential fit, needs nurturing or follow-up.
                - 'Cold': Little/no engagement, low value, poor fit, or old data. Deprioritize or re-evaluate.
            - **Insight:** Summarize the *reason* for the score and the lead's current situation in ONE sentence.
            - **Next Best Action:** Be specific. Instead of "Follow up", suggest *how* (e.g., "Send personalized email referencing recent activity X", "Schedule demo focusing on feature Y", "Call to discuss specific pain point Z").

            Generate the JSON output now based on the provided data.
            """
            human_prompt = "Analyze the lead data provided in the system prompt and generate the required JSON output."

            # --- 3. Call LLM ---
            print(f"--- Calling LLM for Lead Insight (Lead ID: {lead_id}) ---")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = llm.invoke(messages) # Use invoke for non-streaming structured output
            llm_output_raw = response.content if hasattr(response, 'content') else str(response)
            print(f"LLM Output (raw): {llm_output_raw}")

            # --- 4. Parse & Validate Response ---
            try:
                # Clean potential markdown/formatting
                if llm_output_raw.strip().startswith("```json"):
                     llm_output_raw = llm_output_raw.strip()[7:-3].strip()
                elif llm_output_raw.strip().startswith("```"):
                     llm_output_raw = llm_output_raw.strip()[3:-3].strip()

                repaired_json_str = json_repair.repair_json(llm_output_raw)
                insight_result = LeadInsight.model_validate_json(repaired_json_str)

            except (ValidationError, json.JSONDecodeError) as e:
                logging.error(f"LLM Insight Validation Error for lead {lead_id}: {e}\nRaw Output: {llm_output_raw}")
                # Update DB with error state? Or just return error? Let's return error for now.
                # You could update ai_insight to "Error generating insight" here.
                return {"error": f"AI failed to generate valid insight format. Error: {e}"}

            # --- 5. Map Score and Update Database ---
            score_text = insight_result.score
            # Map text score back to a numerical value if needed, or store text directly
            # For this example, let's keep lead_score as text 'Hot'/'Warm'/'Cold'
            # If your DB 'lead_score' is numeric, map here:
            numeric_score = None
            if score_text == 'Hot': numeric_score = random.randint(75, 100)
            elif score_text == 'Warm': numeric_score = random.randint(40, 74)
            else: numeric_score = random.randint(0, 39)

            with conn.cursor() as cur_update:
                # Define the query cleanly
                update_query = """
                    UPDATE crm_schema.leads
                    SET ai_insight = %s,
                        ai_next_action = %s,
                        lead_score = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """
                # Define the parameters cleanly
                update_params = (
                    insight_result.insight,
                    insight_result.next_best_action,
                    numeric_score,
                    lead_id
                )
                # Execute
                cur_update.execute(update_query, update_params)
            conn.commit()
            print(f"--- Successfully generated insight and updated DB for Lead ID: {lead_id} ---")

            # Return the generated data (as validated by Pydantic)
            return insight_result.model_dump()

    except psycopg2.Error as db_err:
        logging.error(f"Database error during insight generation for lead {lead_id}: {db_err}", exc_info=True)
        if conn: conn.rollback()
        return {"error": f"Database error: {db_err}"}
    except Exception as e:
        logging.error(f"Unexpected error during insight generation for lead {lead_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        return {"error": f"An unexpected error occurred: {e}"}
    # Ensure connection is handled by endpoint's teardown or context manager

@app.route("/leads/<int:lead_id>/generate_insight", methods=["POST"])
@token_required
def generate_lead_insight_endpoint(current_user, lead_id):
    user_id = current_user['user_id']
    conn = None # Connection managed within generate_lead_insight now

    try:
        # --- VERIFY ACCESS (Keep this part) ---
        conn = get_db() # Get connection for verification
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 1 FROM crm_schema.leads l
                JOIN crm_schema.user_organizations uo ON l.organization_id = uo.organization_id
                WHERE l.id = %s AND uo.user_id = %s
            """, (lead_id, user_id))
            if not cur.fetchone():
                 return jsonify({"error": "Lead not found or access denied."}), 403
        # --- End Verification ---

        # --- Call the new insight generation function ---
        insight_data = generate_lead_insight(lead_id, user_id=user_id)

        if "error" in insight_data:
             # Log the specific error from generate_lead_insight
             logging.error(f"Insight generation failed for lead {lead_id}: {insight_data['error']}")
             # Return a generic error to the user for security, but use 500 status
             return jsonify({"error": "Failed to generate AI insight."}), 500

        # --- Fetch the *entire updated lead* to return to frontend ---
        with conn.cursor(cursor_factory=DictCursor) as cur:
             # Fetch all columns needed by the frontend, using lead_score but mapping to lead_score
             cur.execute("""
                 SELECT
                     id, name, email, phone, status, value, created_at, user_id,
                     organization_name, notes, stage,
                     lead_score, -- Select lead_score from DB
                     ai_insight, ai_next_action, ai_tasks
                 FROM crm_schema.leads
                 WHERE id = %s
             """, (lead_id,))
             updated_lead_raw = cur.fetchone()
             if not updated_lead_raw:
                  # This shouldn't happen if the update succeeded, but handle defensively
                  return jsonify({"error": "Lead data inconsistency after insight update."}), 500

             updated_lead = dict(updated_lead_raw)

             # --- Consistent Serialization and Score Mapping ---
             if updated_lead.get('created_at'):
                 updated_lead['created_at'] = updated_lead['created_at'].isoformat()
             if updated_lead.get('value') is not None:
                 try: updated_lead['value'] = float(updated_lead['value'])
                 except (ValueError, TypeError): updated_lead['value'] = None

             # Map numeric lead_score back to Hot/Warm/Cold lead_score text for frontend
             if updated_lead.get('lead_score') is not None:
                  try:
                      score_value = int(updated_lead['lead_score'])
                      if score_value >= 75: updated_lead['lead_score'] = 'Hot'
                      elif score_value >= 40: updated_lead['lead_score'] = 'Warm'
                      else: updated_lead['lead_score'] = 'Cold'
                  except (ValueError, TypeError):
                      updated_lead['lead_score'] = 'Cold' # Default
             else:
                 updated_lead['lead_score'] = 'Cold' # Default

             # Remove original lead_score if frontend doesn't need the number
             # --- End Serialization and Mapping ---

             return jsonify(updated_lead), 200 # Return the full lead object

    # --- Keep Broad Error Handling ---
    except psycopg2.Error as db_err:
        logging.error(f"Database error in generate_lead_insight_endpoint for lead {lead_id}: {db_err}", exc_info=True)
        # Don't rollback here if generate_lead_insight handled its own transaction
        return jsonify({"error": f"Database error: {db_err}"}), 500
    except Exception as e:
        logging.error(f"Error in generate_lead_insight_endpoint for lead {lead_id}: {e}", exc_info=True)
        # Don't rollback here if generate_lead_insight handled its own transaction
        return jsonify({"error": "Failed to generate insight or fetch updated lead."}), 500


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
@app.route("/api/leads/<int:lead_id>/calls", methods=["GET", "POST"])
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

# Calendar Events Endpoint
@app.route("/api/leads/<int:lead_id>/calendar", methods=["GET", "POST"])
@app.route("/leads/<int:lead_id>/calendar", methods=["GET", "POST"])
@token_required
def manage_lead_calendar(current_user, lead_id):
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
                        SELECT id, title, start_time, end_time, extended_props 
                        FROM crm_schema.calendar_events 
                        WHERE org_id = %s AND extended_props->>'leadId' = %s
                        ORDER BY start_time ASC
                    """, (user_org_id, str(lead_id)))
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
                elif request.method == "POST":
                    data = request.json
                    title = data.get("title", "")
                    start_time = data.get("start", "")
                    end_time = data.get("end", "")
                    extended_props = data.get("extendedProps", {})
                    extended_props["leadId"] = str(lead_id)
                    
                    if not title or not start_time:
                        return jsonify({"message": "Title and start time are required."}), 400
                    
                    # Get lead email for Google Calendar attendee
                    cur.execute("SELECT email, name FROM crm_schema.leads WHERE id = %s", (lead_id,))
                    lead_info = cur.fetchone()
                    lead_email = lead_info[0] if lead_info else None
                    lead_name = lead_info[1] if lead_info else "Lead"
                    
                    cur.execute("""
                        INSERT INTO crm_schema.calendar_events (title, start_time, end_time, user_id, org_id, extended_props)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (title, start_time, end_time, user_id, user_org_id, json.dumps(extended_props)))
                    event_id = cur.fetchone()[0]
                    
                    # Try to sync with Google Calendar if lead has email
                    gcal_sync_success = False
                    if lead_email:
                        try:
                            gcal_data = {
                                "summary": title,
                                "description": f"Meeting with {lead_name} ({lead_email})",
                                "start_time": start_time,
                                "end_time": end_time,
                                "attendees": [lead_email]
                            }
                            gcal_response, status_code = proxy_to_mcp("actions/create_meeting", current_user, method="POST", data=gcal_data, port=GMEET_MCP_PORT)
                            if status_code == 200:
                                gcal_sync_success = True
                                # Update extended_props with Google Calendar info
                                extended_props["gcal_event_id"] = gcal_response.get_json().get("event_id")
                                extended_props["meet_link"] = gcal_response.get_json().get("meet_link")
                                cur.execute("""
                                    UPDATE crm_schema.calendar_events 
                                    SET extended_props = %s 
                                    WHERE id = %s
                                """, (json.dumps(extended_props), event_id))
                        except Exception as e:
                            logging.warning(f"Failed to sync calendar event with Google Calendar: {e}")
                    
                    response_data = {"message": "Event created successfully", "id": event_id}
                    if gcal_sync_success:
                        response_data["gcal_synced"] = True
                        response_data["meet_link"] = extended_props.get("meet_link")
                    
                    return jsonify(response_data), 201
    except psycopg2.Error as e:
        return jsonify({"message": "Database error managing calendar.", "error": str(e)}), 500

# Meeting Endpoint
@app.route("/api/leads/<int:lead_id>/meeting", methods=["POST"])
@app.route("/leads/<int:lead_id>/meeting", methods=["POST"])
@token_required
def create_lead_meeting(current_user, lead_id):
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
                data = request.json
                title = data.get("title", "")
                description = data.get("description", "")
                start_time = data.get("start", "")
                end_time = data.get("end", "")
                if not title or not start_time or not end_time:
                    return jsonify({"message": "Title, start and end time are required."}), 400
                cur.execute("""
                    INSERT INTO crm_schema.lead_events (lead_id, title, description, start_time, end_time, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (lead_id, title, description, start_time, end_time, user_id))
                event_id = cur.fetchone()[0]
                # Also log in communications
                cur.execute("""
                    INSERT INTO crm_schema.lead_communications (lead_id, type, message, created_by)
                    VALUES (%s, %s, %s, %s)
                """, (lead_id, "meeting", f"Meeting scheduled: {title}", user_id))
                return jsonify({"message": "Meeting created successfully", "id": event_id}), 201
    except psycopg2.Error as e:
        return jsonify({"message": "Database error creating meeting.", "error": str(e)}), 500

# Mail Endpoint
@app.route("/api/leads/<int:lead_id>/mail", methods=["POST"])
@app.route("/leads/<int:lead_id>/mail", methods=["POST"])
@token_required
def send_lead_mail(current_user, lead_id):
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
                cur.execute("SELECT organization_id, email FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_row = cur.fetchone()
                if not lead_row or user_org_id != lead_row[0]:
                    return jsonify({"message": "Lead not found or access denied."}), 404
                lead_email = lead_row[1]
                data = request.json
                subject = data.get("subject", "")
                body = data.get("body", "")
                if not subject or not body:
                    return jsonify({"message": "Subject and body are required."}), 400
                # Send email via MCP
                email_data = {
                    "to": lead_email,
                    "subject": subject,
                    "body": body
                }
                response = proxy_to_mcp("send_email", current_user, method="POST", data=email_data)
                if response.status_code == 200:
                    # Log in communications
                    cur.execute("""
                        INSERT INTO crm_schema.communication_logs (lead_id, user_id, communication_type, subject, content, direction)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (lead_id, user_id, "email", subject, body, "outbound"))
                    return jsonify({"message": "Email sent successfully"}), 200
                else:
                    return jsonify({"message": "Failed to send email"}), 500
    except psycopg2.Error as e:
        return jsonify({"message": "Database error sending mail.", "error": str(e)}), 500

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

# THIS IS THE NEW, FIXED CODE
try:
    llm = ChatOpenAI(
        model="./DeepSeek-R1-Distill-Qwen-32B",
        openai_api_base="http://13.202.235.154:8000/v1",
        openai_api_key="not-used",
        temperature=0.1,
        streaming=True  # <-- THIS IS THE FIX
    )
    print("--- ReAct Agent LLM Initialized (Streaming Enabled) ---")
except Exception as e:
    print(f"--- FAILED TO INITIALIZE LLM --- Error: {e}")
    llm = None

class VisualizeDataInput(BaseModel):
    user_query: str = Field(description="The user's full, natural language query for the data. For example: 'show me a chart of leads by status' or 'plot revenue by month'.")

# ==============================================================================
# --- ALL OTHER FLASK ENDPOINTS (UNCHANGED) ---
# ==============================================================================

@app.route("/agent/invoke", methods=["POST"])
@token_required
def agent_invoke(current_user):
    """
    (V8 - Final Fix using langchain.agents.create_tool_calling_agent and AgentExecutor)
    This is the standard, supported way to build agents in LangChain v1.0+.
    """
    if llm is None:
        return jsonify({"error": "Agent LLM not initialized"}), 500
        
    user_id = current_user["user_id"]
    user_input = request.json.get("command", "")
    print(f"\n--- User {user_id} Input: {user_input} ---") # DEBUG

    tools = [
        Tool(
            name="visualize_crm_data",
            func=functools.partial(visualize_data_from_database, current_user),
            description="Generates a chart visualization from the CRM database based on a user's natural language query. Use this for requests like 'show me a chart of...', 'visualize...', 'plot revenue by month', etc.",
            args_schema=VisualizeDataInput
        ),
        Tool(
            name="send_email",
            func=functools.partial(agent_tools.send_email_logic, current_user),
            description="Sends a new email. Requires 'recipient' (email) and 'subject' (string). The 'body' (string) is optional."
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
    
    # --- PROMPT SETUP ---
    # --- PROMPT SETUP ---
    # --- PROMPT SETUP ---
    # --- PROMPT SETUP (ReAct w/ Few-Shot Example) ---
    
    # 1. Define your custom system instructions
    # --- PROMPT SETUP (ReAct w/ Manual ChatPromptTemplate) ---
    
    # 1. Define your custom system instructions
    # 1. Define your custom system instructions
    # --- PROMPT SETUP (ReAct w/ STRING PromptTemplate) ---
    
    # 1. Define your custom system instructions (THIS IS UNCHANGED)
    # 1. Define your custom system instructions
    system_instructions = """
    You are a helpful sales assistant. You have access to tools.

    **TOOLS:**
    Here is a list of tools you can use:
    {tools}

    **YOUR AVAILABLE TOOL NAMES ARE: {tool_names}**

    **--- MANDATORY RULES ---**

    **1.  If the user is just making a comment, asking a follow-up question, or having a
        normal conversation (like "sales seems low" or "thanks!"),
        DO NOT USE A TOOL. Just respond politely as a helpful assistant.
        Only use a tool if they ask for a *new* action (like "show me last month" or "send an email").**
    
    **2.  CRITICAL RULE: DO NOT MAKE UP DATA.**
        If the user asks for any data, chart, or visualization (e.g., "show me leads", "plot revenue"),
        you MUST use the `visualize_crm_data` tool.
        This tool has only ONE argument: `user_query`.
        You must pass the user's *entire* natural language question as the `user_query` argument.
        Do NOT answer data questions from your own knowledge. Use the tool.

    **3.  FORMAT INSTRUCTIONS:**
        You MUST respond using the following format:

        Thought: The user's request is... I need to use a tool. I will use the `[tool_name]` tool from the list of tool names.
        Action: `[tool_name]`
        Action Input: `[tool_input as a JSON-compatible string]`

        *OR* if you are not using a tool (see rule #1):

        Thought: The user is making a comment. I should respond conversationally.
        Final Answer: [Your polite, conversational response]

    **4.  EXAMPLE OF YOUR REQUIRED OUTPUT (Tool Call):**
        Thought: The user is asking for a chart of leads. I must use the `visualize_crm_data` tool and pass their full query.
        Action: visualize_crm_data
        Action Input: {{"user_query": "show me a chart of leads by status"}}

    **5.  CRITICAL RULE FOR TOOL OUTPUT:**
When the `visualize_crm_data` tool runs, you will get an "Observation" that is a JSON string.
Your "Final Answer" MUST be the `Final Answer:` prefix followed by **ONLY** that JSON string.
Do NOT add any other conversational text.

Example after tool run:
Observation: {{"type": "bar", "data": ..., "options": ...}}
Thought: The tool returned the chart JSON. I will now provide this as the Final Answer.
Final Answer: {{"type": "bar", "data": ..., "options": ...}}
    """

    # 2. Manually create the PromptTemplate (string-based)
    # This will correctly accept the string-based 'agent_scratchpad'
    prompt = PromptTemplate.from_template(f"""
{system_instructions}

{{chat_history}}

Human: {{input}}

{{agent_scratchpad}}
""")
            
    # 3. Create the agent runnable using the ReAct agent
    agent_runnable = create_react_agent(llm, tools, prompt)

    # 4. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True, handle_parsing_errors=True)
    # 5. Get and format chat history AS A STRING
    history = chat_histories.get(user_id, [])
    # This MUST be a string to match the new PromptTemplate
    chat_history_for_input = "\n".join(
        [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in history if isinstance(msg, (HumanMessage, AIMessage))]
    )

    def generate_stream():
    
        full_response_content = ""
        loop_count = 0
        
        # 1. Input format for AgentExecutor
        stream_input = {
            "input": user_input,
            "chat_history": chat_history_for_input,
        }

        try:
            # 2. Stream the execution
            for chunk in agent_executor.stream(stream_input, config={"recursion_limit": 30}):
                loop_count += 1
                
                # --- Handle AgentExecutor Stream Output ---

                # A. Agent Action (Decides to call a tool)
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        tool_name = action.tool
                        tool_args = action.tool_input
                        args_str = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
                        
                        # Yield Thought (from action.log)
                        if action.log:
                             thought_data = json.dumps({"type": "thought", "content": action.log})
                             yield f"data: {thought_data}\n\n"

                        # Yield Tool Call
                        tool_call_data = json.dumps({
                            "type": "tool_call",
                            "content": f"**Calling Tool:**\n`{tool_name}({args_str})`"
                        })
                        yield f"data: {tool_call_data}\n\n"

                # B. Final Output (Agent's text or final answer)
                elif "output" in chunk:
                    output_content = chunk["output"]
                    if output_content:
                        # AgentExecutor streaming sends the full final output string once in the 'output' key
                        # This could be text or the raw tool result (JSON)
                        
                        is_visualization = False
                        # --- NEW FIX: Check for 'Final Answer:' prefix ---
                        json_string = output_content
                        if output_content.strip().startswith("Final Answer:"):
                            json_string = output_content.strip()[len("Final Answer:"):].strip()

                        

                        try:
                        # Check if the output is the chart JSON
                            chart_json = json.loads(json_string) # <-- Use the cleaned json_string
                            if isinstance(chart_json, dict) and 'type' in chart_json and 'data' in chart_json and 'options' in chart_json:
                                print(f"[Loop {loop_count}] Detected VISUALIZATION payload in final output.")
                                viz_data = json.dumps({"type": "visualization", "content": chart_json})
                                yield f"data: {viz_data}\n\n"
                                is_visualization = True
                        except (json.JSONDecodeError, TypeError):
                            pass # Not chart JSON
                        
                        if not is_visualization:
                            # If it's not JSON, stream the final text as a token
                            print(f"[Loop {loop_count}] Yielding FINAL TEXT: '{output_content}'")
                            full_response_content = output_content
                            token_data = json.dumps({"type": "token", "content": output_content})
                            yield f"data: {token_data}\n\n"

            # After the loop, save the final history
            print(f"--- Stream finished. Saving history. ---")
            
            # Since AgentExecutor doesn't manage history directly, we append the final user message and output
            chat_histories[user_id] = history + [
                HumanMessage(content=user_input),
                AIMessage(content=full_response_content)
            ]

        except Exception as e:
            logging.error(f"Agent stream error for user {user_id}: {e}", exc_info=True)
            print(f"--- ERROR during stream: {e} ---")
            error_msg = f"An error occurred: {e}"
            error_data = json.dumps({"type": "error", "content": error_msg})
            yield f"data: {error_data}\n\n"

    # Return the generator as a streaming response
    return Response(stream_with_context(generate_stream()), mimetype="text/event-stream")
# ==============================================================================
# --- THETA AI ENDPOINTS FOR LEAD ASSISTANCE ---
# ==============================================================================

def get_database_schema(conn):
    """
    Fetches the schema (table names and columns) for 'crm_schema'.
    This is crucial for the LLM to write accurate queries.
    """
    schema_info = {}
    try:
        with conn.cursor() as cur:
            # Get table names
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'crm_schema'
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            # For each table, get column names and types
            for table_name in tables:
                # Skip sensitive tables
                if table_name in ['users', 'user_integrations', 'user_organizations', 'invitations']:
                    continue

                cur.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'crm_schema' AND table_name = '{table_name}'
                """)
                schema_info[table_name] = [f"{col_name} ({data_type})" for col_name, data_type in cur.fetchall()]
        
        # Format for the prompt
        schema_string = "Database Schema (crm_schema):\n"
        for table, columns in schema_info.items():
            schema_string += f"Table '{table}':\n  " + "\n  ".join(columns) + "\n"
        return schema_string
    except Exception as e:
        logging.error(f"Error fetching DB schema: {e}")
        return "Error: Could not fetch database schema."

def visualize_data_from_database(current_user: Dict[str, Any], user_query: str) -> str:
    """
    User-facing tool. Takes a natural language query (e.g., "show me revenue by month"),
    uses an LLM to generate SQL, executes it, and formats the result as
    Chart.js JSON.
    """
    global llm # Access the global LLM instance
    print(f"--- Data Analyst Tool Called by User {current_user.get('user_id')} ---")
    print(f"User Query: {user_query}")
    
    conn = None
    try:
        user_id = current_user["user_id"]
        
        # Connect to DB and get schema
        conn = get_db()
        schema = get_database_schema(conn)
        
        # Get user's org ID to filter all queries
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return json.dumps({"error": "You are not in an organization."})
            user_org_id = org_id_tuple[0]

        # --- Step 1: Use LLM to generate SQL and Chart Config ---
        # --- Step 1: Use LLM to generate SQL and Chart Config ---
        sql_generation_prompt = f"""
        You are a secure, read-only data analyst. Your ONLY task is to convert a
        user's query into a single, valid JSON object that matches the
        ChartGenerationQuery Pydantic model.

        **CRITICAL: YOUR OUTPUT MUST BE THE JSON OBJECT AND NOTHING ELSE.
        Do NOT wrap it in "properties", "description", or any other keys.**

        DATABASE SCHEMA:
        {schema}

        USER CONTEXT:
        - The user's ID is: {user_id}
        - The user's Organization ID is: {user_org_id}
        
        SECURITY RULES (MANDATORY):
        1.  ALL queries MUST be read-only (SELECT only).
        2.  ALL queries MUST be filtered by the user's organization (e.g., `organization_id = {user_org_id}` or `org_id = {user_org_id}`).
        3.  Do NOT query sensitive tables. Focus on `leads`, `deals`, `tasks`, `activities`.
        
        **SQL QUERY RULES (MANDATORY):**
        1.  If the query is vague (e.g., "improve graph"), create a useful default chart,
            like "Revenue by Month".
        2.  For "Revenue", you MUST sum `value` from `crm_schema.deals`
            where `stage = 'Closed-Won'`.
        3.  **TIME-BASED QUERIES (CRITICAL):**
            When grouping by month, the `SELECT`, `GROUP BY`, and `ORDER BY` clauses
            MUST use the *exact same* `date_trunc` expression.
            DO NOT use `to_char` in the `SELECT` clause if you `GROUP BY date_trunc`.

            **CORRECT SQL EXAMPLE:**
            SELECT
                date_trunc('month', created_at) AS month_label,
                SUM(value)
            FROM crm_schema.deals
            WHERE org_id = {user_org_id}
            GROUP BY date_trunc('month', created_at)
            ORDER BY date_trunc('month', created_at)

            **INCORRECT SQL EXAMPLE (DO NOT DO THIS):**
            SELECT to_char(created_at, 'YYYY-MM') ... GROUP BY date_trunc('month', created_at)
        
        USER REQUEST:
        "{user_query}"
        
        Your output MUST be a single, valid JSON object matching this Pydantic model:
        {ChartGenerationQuery.model_json_schema()}
        """
        
        print("--- Calling LLM for SQL Generation ---")
        if llm is None:
            return json.dumps({"error": "LLM not available for query generation."})

        messages = [
            {"role": "system", "content": sql_generation_prompt},
            {"role": "user", "content": user_query}
        ]
        
        response = llm.invoke(messages)
        llm_output_json = response.content if hasattr(response, 'content') else str(response)
        
        print(f"LLM Output (raw): {llm_output_json}")
        
        # --- Step 2: Parse and Validate LLM Response ---
        try:
            if llm_output_json.startswith("```json"):
                llm_output_json = llm_output_json[7:-3].strip()
            repaired_json_str = json_repair.repair_json(llm_output_json)
            chart_query = ChartGenerationQuery.model_validate_json(repaired_json_str)
        
        except (ValidationError, json.JSONDecodeError) as e:
            logging.error(f"Data Analyst LLM Validation Error: {e}")
            return json.dumps({"error": f"The AI failed to generate a valid query. Error: {e}"})

        # --- Step 3: Security Validation ---
        sql_to_run = chart_query.sql_query.strip()
        if not sql_to_run.lower().startswith("select"):
            logging.error(f"Data Analyst SECURITY VIOLATION: Non-SELECT query attempted by user {user_id}")
            return json.dumps({"error": "Query failed security validation (not a SELECT query)."})
        
        # Check for org_id filter
        org_id_str = str(user_org_id)
        if org_id_str not in sql_to_run and f"organization_id = {org_id_str}" not in sql_to_run and f"org_id = {org_id_str}" not in sql_to_run:
            logging.error(f"Data Analyst SECURITY VIOLATION: Query missing org_id filter by user {user_id}")
            return json.dumps({"error": "Query failed security validation (missing organization filter)."})

        print(f"Validated SQL to run: {sql_to_run}")

        # --- Step 4: Execute SQL Query ---
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql_to_run)
            results = cur.fetchall()

        if not results:
            return json.dumps({"error": "Your query returned no data."})

        # --- Step 5: Format Data for Chart.js ---
        labels = []
        datasets = []
        
        # Initialize datasets based on data_columns
        dataset_map = {col_name: ChartDataset(label=col_name.replace('_', ' ').title(), data=[]) for col_name in chart_query.data_columns}
        
        for row in results:
            labels.append(row[chart_query.label_column])
            for col_name in chart_query.data_columns:
                # Convert decimal to float if necessary
                data_point = row[col_name]
                if isinstance(data_point, decimal.Decimal):
                    data_point = float(data_point)
                dataset_map[col_name].data.append(data_point)
        
        datasets = list(dataset_map.values())

        # Final Chart.js object
        chart_data = ChartData(
            type=chart_query.chart_type,
            data={
                "labels": labels,
                "datasets": [ds.model_dump() for ds in datasets]
            },
            options={
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {"display": True},
                    "title": {
                        "display": True,
                        "text": chart_query.title,
                        "font": {"size": 16}
                    }
                },
                "scales": {
                    "y": {
                        "TbeginAtZero": True
                    }
                }
            }
        )
        
        # Return the Chart.js JSON as a string
        return chart_data.model_dump_json()

    except psycopg2.Error as e:
        logging.error(f"Data Analyst DB Error: {e}")
        conn.rollback()
        return json.dumps({"error": f"Database error: {e}"})
    except Exception as e:
        logging.error(f"Data Analyst Unknown Error: {e}", exc_info=True)
        if conn: conn.rollback()
        return json.dumps({"error": f"An unexpected error occurred: {e}"})
    finally:
        if conn:
            close_db()



@app.route("/theta/command", methods=["POST"])
@token_required
def theta_command(current_user):
    """
    AI endpoint for drafting emails and other lead communications.
    """
    if llm is None:
        return jsonify({"error": "AI service not available"}), 503
    
    user_id = current_user["user_id"]
    data = request.json
    command = data.get("command", "")
    context = data.get("context", {})
    
    if not command:
        return jsonify({"error": "Command is required"}), 400
    
    # Build context-aware prompt
    recipient = context.get("recipient", {})
    lead_id = context.get("leadId")
    
    system_prompt = f"""You are an AI assistant helping a sales professional draft professional communications.

Context:
- Lead: {recipient.get('name', 'Unknown')}
- Email: {recipient.get('email', 'Unknown')}
- Lead ID: {lead_id}

Instructions:
- Draft professional, personalized communications
- Keep responses concise but complete
- Format emails with Subject: and Body: sections
- Be helpful and professional

User request: {command}"""
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": command}
        ]
        
        response = llm.invoke(messages)
        ai_response = response.content if hasattr(response, 'content') else str(response)
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        logging.error(f"Theta command error for user {user_id}: {e}")
        return jsonify({"error": f"AI processing failed: {str(e)}"}), 500

@app.route("/theta/message", methods=["POST"])
@token_required
def theta_message(current_user):
    """
    AI endpoint for drafting SMS/WhatsApp messages.
    """
    if llm is None:
        return jsonify({"error": "AI service not available"}), 503
    
    user_id = current_user["user_id"]
    data = request.json
    command = data.get("command", "")
    context = data.get("context", {})
    
    if not command:
        return jsonify({"error": "Command is required"}), 400
    
    # Build context-aware prompt for messages
    recipient = context.get("recipient", {})
    lead_id = context.get("leadId")
    
    system_prompt = f"""You are an AI assistant helping a sales professional draft concise, effective messages for SMS or WhatsApp.

Context:
- Lead: {recipient.get('name', 'Unknown')}
- Phone: {recipient.get('phone', 'Unknown')}
- Lead ID: {lead_id}

Instructions:
- Draft concise, friendly messages (under 160 characters when possible)
- Be professional but conversational
- Personalize when possible
- Keep it brief and actionable
- No need for subject lines - just the message content

User request: {command}"""
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": command}
        ]
        
        response = llm.invoke(messages)
        ai_response = response.content if hasattr(response, 'content') else str(response)
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        logging.error(f"Theta message error for user {user_id}: {e}")
        return jsonify({"error": f"AI processing failed: {str(e)}"}), 500

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
    return redirect("/mini-crm/leads")

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

@app.route("/mini-crm/finance")
@app.route("/mini-crm/finance/")
def mini_mini_finance():
    return render_template("lead_finance.html")
    
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
# --- THIS IS THE FIXED LINE ---
                return jsonify({"user_id": user_id, "message": "Profile updated. Proceed to organization setup."}), 200                # --- END NEW CODE ---
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

@app.route("/api/settings/details", methods=["GET"])
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
                SELECT username, email, first_name, primary_integration_key , primary_meeting_key
                FROM crm_schema.users WHERE id = %s
            """, (user_id,))
            profile = cur.fetchone()
            if profile:
                settings_data["profile"] = {"username": profile[0], "email": profile[1], "name": profile[2]}
                settings_data["primary_mail_key"] = profile[3]
                settings_data["primary_meeting_key"] = profile[4] # <-- ADD THIS LINE

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

@app.route("/api/settings/primary_mail", methods=["PUT"])
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
        return get_user_settings(current_user)

    except psycopg2.Error as e:
        logging.error(f"!!! DATABASE ERROR in set_primary_mail: {e} !!!")
        if conn:
            conn.rollback()
        return jsonify({"message": "Database error updating setting.", "error": str(e)}), 500
@app.route("/api/settings/profile", methods=["PUT"])
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

@app.route("/api/finance-data", methods=["GET"])
@token_required
def get_finance_data(current_user):
    user_id = current_user["user_id"]

    # Get filters from query params
    date_range = request.args.get('dateRange', 'month')
    owner_id = request.args.get('owner', '')

    try:
        conn = get_db()
        # Use DictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- 1. Get Org ID ---
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You are not in an organization."}), 400
            org_id = org_id_tuple['organization_id']

            # --- 2. Build Base Query String Segment with Filters ---
            filter_conditions = " WHERE d.org_id = %s "
            params = [org_id]

            # Add date filter
            if date_range == 'month':
                filter_conditions += " AND d.created_at >= NOW() - INTERVAL '1 month' "
            elif date_range == 'quarter':
                filter_conditions += " AND d.created_at >= NOW() - INTERVAL '3 months' "
            elif date_range == 'year':
                filter_conditions += " AND d.created_at >= NOW() - INTERVAL '1 year' "
            # 'all' has no date filter

            # Add owner filter
            if owner_id:
                filter_conditions += " AND d.user_id = %s "
                params.append(int(owner_id))

            # --- 3. Calculate KPI Summary ---

            # Total Pipeline
            cur.execute(f"""
                SELECT COALESCE(SUM(d.value), 0) AS total
                FROM crm_schema.deals d
                {filter_conditions} AND d.stage NOT IN ('Closed-Won', 'Closed-Lost')
            """, tuple(params))
            total_pipeline_decimal = cur.fetchone()['total'] or 0
            total_pipeline = float(total_pipeline_decimal) # <-- Convert to float

            # Total Won Value
            cur.execute(f"""
                SELECT COALESCE(SUM(d.value), 0) AS total
                FROM crm_schema.deals d
                {filter_conditions} AND d.stage = 'Closed-Won'
            """, tuple(params))
            won_value_decimal = cur.fetchone()['total'] or 0
            won_value = float(won_value_decimal) # <-- Convert to float

            # Won/Lost Counts
            cur.execute(f"SELECT COUNT(*) AS total FROM crm_schema.deals d {filter_conditions} AND d.stage = 'Closed-Won'", tuple(params))
            won_count = cur.fetchone()['total'] or 0
            cur.execute(f"SELECT COUNT(*) AS total FROM crm_schema.deals d {filter_conditions} AND d.stage = 'Closed-Lost'", tuple(params))
            lost_count = cur.fetchone()['total'] or 0

            total_closed_deals = won_count + lost_count
            conversion_rate = (won_count / total_closed_deals * 100) if total_closed_deals > 0 else 0

            # Avg Deal Size
            avg_deal_size = (won_value / won_count) if won_count > 0 else 0 # Already float

            # Avg Sales Cycle
            cur.execute(f"""
                SELECT COALESCE(AVG(EXTRACT(DAY FROM d.created_at - l.created_at)), 0) AS avg_cycle
                FROM crm_schema.deals d
                LEFT JOIN crm_schema.leads l ON d.lead_id = l.id
                {filter_conditions.replace('WHERE','WHERE l.id IS NOT NULL AND')} AND d.stage = 'Closed-Won'
            """, tuple(params))
            # AVG returns numeric/Decimal, but EXTRACT returns double precision (float) in PG,
            # so this result should already be float-compatible or float itself.
            # Convert explicitly for safety.
            avg_sales_cycle = float(cur.fetchone()['avg_cycle'] or 0)

            # --- MODIFICATION: Convert total_pipeline to float before multiplying ---
            projected_revenue = total_pipeline * 0.30

            # Sales Velocity
            sales_velocity = (won_value / avg_sales_cycle) if avg_sales_cycle > 0 else 0 # Already float

            # Est. CLV
            est_clv = avg_deal_size * 1.5 # Already float

            kpi_summary = {
                "totalPipeline": total_pipeline, # Already float
                "projectedRevenue": projected_revenue, # Already float
                "wonValue": won_value, # Already float
                "conversionRate": conversion_rate, # Already float
                "avgDealSize": avg_deal_size, # Already float
                "salesVelocity": sales_velocity, # Already float
                "avgSalesCycle": avg_sales_cycle, # Already float
                "estClv": est_clv # Already float
            }

            # --- 4. Prepare Chart Data ---

            # Revenue Chart (Last 6 Months)
            cur.execute("""
                SELECT
                    to_char(date_trunc('month', d.created_at), 'Mon') as month_name,
                    COALESCE(SUM(CASE WHEN d.stage = 'Closed-Won' THEN d.value ELSE 0 END), 0) as won,
                    COALESCE(SUM(CASE WHEN d.stage NOT IN ('Closed-Won', 'Closed-Lost') THEN d.value ELSE 0 END) * 0.3, 0) as projected
                FROM crm_schema.deals d
                WHERE d.org_id = %s AND d.created_at >= NOW() - INTERVAL '6 months'
                GROUP BY date_trunc('month', d.created_at)
                ORDER BY date_trunc('month', d.created_at)
            """, (org_id,))

            revenue_data = cur.fetchall()
            revenue_chart = {
                "labels": [row['month_name'].strip() for row in revenue_data],
                 # --- MODIFICATION: Convert Decimal results to float ---
                "projected": [float(row['projected']) for row in revenue_data],
                "won": [float(row['won']) for row in revenue_data]
            }

            # Pipeline Chart (by Stage)
            cur.execute("""
                SELECT stage, COALESCE(SUM(d.value), 0) as total_value
                FROM crm_schema.deals d
                WHERE d.org_id = %s AND stage NOT IN ('Closed-Won', 'Closed-Lost')
                GROUP BY stage
            """, (org_id,))
            pipeline_data = cur.fetchall()
            pipeline_chart = {
                "labels": [row['stage'] for row in pipeline_data],
                 # --- MODIFICATION: Convert Decimal results to float ---
                "values": [float(row['total_value']) for row in pipeline_data]
            }

            charts = {
                "revenue": revenue_chart,
                "pipeline": pipeline_chart
            }

            # --- 5. Prepare Rep Summary Table ---
            cur.execute("""
                SELECT
                    u.id,
                    COALESCE(u.first_name, '') || ' ' || COALESCE(u.last_name, '') as name,
                    u.email,
                    COALESCE(SUM(CASE WHEN d.stage NOT IN ('Closed-Won', 'Closed-Lost') THEN d.value ELSE 0 END), 0) as pipeline_value,
                    COALESCE(SUM(CASE WHEN d.stage = 'Closed-Won' THEN d.value ELSE 0 END), 0) as won_value,
                    COALESCE(COUNT(CASE WHEN d.stage = 'Closed-Won' THEN 1 END), 0) as won_count,
                    COALESCE(COUNT(CASE WHEN d.stage = 'Closed-Lost' THEN 1 END), 0) as lost_count,
                    COALESCE(AVG(CASE WHEN d.stage = 'Closed-Won' THEN EXTRACT(DAY FROM d.created_at - l.created_at) ELSE NULL END), 0) as avg_cycle
                FROM crm_schema.users u
                JOIN crm_schema.deals d ON u.id = d.user_id
                LEFT JOIN crm_schema.leads l ON d.lead_id = l.id
                WHERE d.org_id = %s
                GROUP BY u.id, u.first_name, u.last_name, u.email
            """, (org_id,)) # Note: filter_conditions not applied here intentionally for full org view

            rep_summary = []
            for row in cur.fetchall():
                won_value_rep = float(row['won_value']) # <-- Convert
                won_count_rep = row['won_count']
                lost_count_rep = row['lost_count']
                avg_cycle_rep = float(row['avg_cycle']) # <-- Convert

                total_closed_rep = won_count_rep + lost_count_rep
                conv_rate_rep = (won_count_rep / total_closed_rep * 100) if total_closed_rep > 0 else 0
                avg_size_rep = (won_value_rep / won_count_rep) if won_count_rep > 0 else 0

                rep_summary.append({
                    "id": row['id'],
                    "name": row['name'].strip(),
                    "email": row['email'],
                    "pipelineValue": float(row['pipeline_value']), # <-- Convert
                    "wonValue": won_value_rep, # Already float
                    "avgDealSize": avg_size_rep, # Already float
                    "conversionRate": conv_rate_rep, # Already float
                    "avgSalesCycle": avg_cycle_rep, # Already float
                    "aiWinProb": random.uniform(40.0, 75.0) # Mock AI prob
                })

            # --- 6. Assemble Final Response ---
            response_data = {
                "kpiSummary": kpi_summary,
                "charts": charts,
                "repSummary": rep_summary
            }

            return jsonify(response_data)

    except psycopg2.Error as e:
        import traceback
        traceback.print_exc()
        return jsonify({"message": "Database error fetching finance data.", "error": str(e)}), 500
    # --- Add broader exception catch for type errors ---
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Specifically catch TypeErrors which might arise from Decimal/float issues
        if isinstance(e, TypeError):
             return jsonify({"message": "Data type error during calculation.", "error": str(e)}), 500
        return jsonify({"message": "An unexpected error occurred.", "error": str(e)}), 500

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



@app.route("/presales/leads", methods=["GET"]) # Make sure this route exists or add it
@app.route("/api/leads", methods=["GET", "POST"]) # Keep the POST for creation if needed elsewhere
@token_required
def manage_leads_endpoint(current_user): # Renamed for clarity
    user_id = current_user["user_id"]

    # --- HANDLE POST FOR CREATION (Keep existing POST logic if needed) ---
    if request.method == "POST":
        # ... (Keep your existing lead creation logic here) ...
        # Example start:
        data = request.json
        name, email = data.get("name"), data.get("email")
        # ... rest of your POST logic ...
        # Make sure it returns appropriately
        # return jsonify({"message": "Lead created successfully", "lead_id": lead_id}), 201
        pass # Placeholder if you separate GET/POST routes later

    # --- MODIFIED GET LOGIC FOR PAGINATION ---
    elif request.method == "GET":
        # Get pagination and filtering parameters
        page = int(request.args.get('page', 1))
        # --- NEW: Default limit increased for faster initial load perception ---
        limit = int(request.args.get('limit', 50)) # Default to 50 leads per page
        sort = request.args.get('sort', 'created_at')
        order = request.args.get('order', 'desc')
        search = request.args.get('search', '').strip()
        status_filter = request.args.get('status', '')
        stage_filter = request.args.get('stage', '')
        owner_filter = request.args.get('owner', '')
        date_range = request.args.get('dateRange', '')

        try:
            conn = get_db()
            # --- NEW: Use DictCursor for easier column access by name ---
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
                org_id_tuple = cur.fetchone()
                if not org_id_tuple:
                    return jsonify({"message": "You must be in an organization to view leads."}), 400
                org_id = org_id_tuple['organization_id'] # Access by key with DictCursor

                # Base query selecting ALL necessary fields for display AND background processing
                query_base = """
                    FROM crm_schema.leads
                    WHERE organization_id = %(org_id)s
                """
                params = {'org_id': org_id}
                filter_conditions = []

                # Add search filter
                if search:
                    filter_conditions.append("(name ILIKE %(search)s OR email ILIKE %(search)s OR phone ILIKE %(search)s OR organization_name ILIKE %(search)s)")
                    params['search'] = f'%{search}%'

                # Add status filter
                if status_filter:
                    filter_conditions.append("status = %(status)s")
                    params['status'] = status_filter

                # Add stage filter
                if stage_filter:
                    filter_conditions.append("stage = %(stage)s")
                    params['stage'] = stage_filter

                # Add owner filter
                if owner_filter:
                    filter_conditions.append("user_id = %(owner)s")
                    params['owner'] = int(owner_filter)

                # Add date range filter
                if date_range:
                    if date_range == 'today': filter_conditions.append("DATE(created_at) = CURRENT_DATE")
                    elif date_range == 'week': filter_conditions.append("created_at >= NOW() - INTERVAL '7 days'")
                    elif date_range == 'month': filter_conditions.append("created_at >= NOW() - INTERVAL '1 month'")
                    elif date_range == 'quarter': filter_conditions.append("created_at >= NOW() - INTERVAL '3 months'")

                if filter_conditions:
                    query_base += " AND " + " AND ".join(filter_conditions)

                # Get total count for pagination
                count_query = f"SELECT COUNT(*) as total {query_base}"
                cur.execute(count_query, params)
                total_count = cur.fetchone()['total']

                # Build the final data query
                valid_sort_fields = ['name', 'email', 'status', 'created_at', 'value', 'stage', 'lead_score', 'lead_score'] # Added lead_score
                sort_column = sort if sort in valid_sort_fields else 'created_at'
                sort_order = 'DESC' if order.lower() == 'desc' else 'ASC'

                offset = (page - 1) * limit
                params['limit'] = limit
                params['offset'] = offset

                data_query = f"""
                    SELECT
                        id, name, email, phone, status, value, created_at, user_id,
                        organization_name, notes, stage, lead_score,
                        lead_score, ai_insight, ai_next_action, ai_tasks  -- Include AI fields
                    {query_base}
                    ORDER BY {sort_column} {sort_order}
                    LIMIT %(limit)s OFFSET %(offset)s
                """

                cur.execute(data_query, params)
                leads_raw = cur.fetchall()

                # Convert DictRow to plain dict and handle potential None values gracefully
                leads = []
                for row in leads_raw:
                    lead_dict = dict(row)
                    # Ensure datetime is serializable
                    if lead_dict.get('created_at'):
                        lead_dict['created_at'] = lead_dict['created_at'].isoformat()
                    # Ensure numeric types are floats or ints
                    if lead_dict.get('value') is not None:
                        try:
                           lead_dict['value'] = float(lead_dict['value'])
                        except (ValueError, TypeError):
                           lead_dict['value'] = None # or 0
                    if lead_dict.get('lead_score') is not None:
                         try:
                             lead_dict['lead_score'] = int(lead_dict['lead_score'])
                         except (ValueError, TypeError):
                             lead_dict['lead_score'] = None # or 0
                    leads.append(lead_dict)

                return jsonify({
                    "data": leads,
                    "total": total_count,
                    "page": page,
                    "limit": limit
                })
        except psycopg2.Error as e:
            logging.error(f"Database error in GET /presales/leads: {e}", exc_info=True)
            return jsonify({"message": "Database error fetching leads.", "error": str(e)}), 500
        except Exception as e:
             logging.error(f"Unexpected error in GET /presales/leads: {e}", exc_info=True)
             return jsonify({"message": "An unexpected error occurred."}), 500
    else:
        # Handle unsupported methods if necessary
        return jsonify({"message": f"Method {request.method} not allowed for this endpoint"}), 405

@app.route("/api/stats/leads", methods=["GET"])
@token_required
def get_leads_stats(current_user):
    user_id = current_user["user_id"]
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You must be in an organization to view stats."}), 400
            org_id = org_id_tuple[0]
            
            # Get stats by status
            cur.execute("""
                SELECT status, COUNT(*) 
                FROM crm_schema.leads 
                WHERE organization_id = %s 
                GROUP BY status
            """, (org_id,))
            
            stats = {"new": 0, "contacted": 0, "qualified": 0, "converted": 0}
            for row in cur.fetchall():
                status, count = row
                if status and status.lower() in stats:
                    stats[status.lower()] = count
            
            return jsonify(stats)
    except psycopg2.Error as e:
        return jsonify({"message": "Database error fetching stats.", "error": str(e)}), 500

@app.route("/api/users", methods=["GET"])
@token_required
def get_users(current_user):
    user_id = current_user["user_id"]
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You must be in an organization to view users."}), 400
            org_id = org_id_tuple[0]
            
            cur.execute("""
                SELECT u.id, u.first_name, u.email
                FROM crm_schema.users u
                JOIN crm_schema.user_organizations uo ON u.id = uo.user_id
                WHERE uo.organization_id = %s
            """, (org_id,))
            
            users = []
            for row in cur.fetchall():
                users.append({
                    "id": row[0],
                    "name": row[1] or row[2],  # Use first_name if available, otherwise email
                    "email": row[2]
                })
            
            return jsonify(users)
    except psycopg2.Error as e:
        return jsonify({"message": "Database error fetching users.", "error": str(e)}), 500

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
                    return jsonify({"message": "You are not in an organization."}), 400 # Added status code
                user_org_id = org_id_tuple[0]
                
                cur.execute("SELECT organization_id FROM crm_schema.leads WHERE id = %s", (lead_id,))
                lead_org_tuple = cur.fetchone()
                if not lead_org_tuple:
                    return jsonify({"message": "Lead not found"}), 404
                
                if user_org_id != lead_org_tuple[0]:
                    return jsonify({"message": "Forbidden: You do not have access to this lead."}), 403 # Added status code

                if request.method == "GET":
                    # --- MODIFICATION: Add stage, lead_score, enrichment_data ---
                    cur.execute("""
                        SELECT id, name, email, phone, status, value, created_at, user_id, organization_name, notes, 
                               title, source, last_contact_date, tags,
                               stage, lead_score, enrichment_data
                        FROM crm_schema.leads WHERE id = %s
                    """, (lead_id,))
                    row = cur.fetchone()
                    if not row: return jsonify({"message": "Lead not found"}), 404
                    
                    # --- MODIFICATION: Add new fields to dictionary ---
                    lead = {
                        "id": row[0], "name": row[1], "email": row[2], "phone": row[3],
                        "status": row[4], "value": row[5], "created_at": row[6],
                        "user_id": row[7], "organization": row[8], "notes": row[9],
                        "title": row[10], "source": row[11], "last_contact_date": row[12], "tags": row[13],
                        "stage": row[14],                 # <-- ADDED
                        "lead_score": row[15],            # <-- ADDED
                        "enrichment_data": row[16]        # <-- ADDED
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
                    # --- (UNCHANGED) PRE-FETCH ---
                    cur.execute("SELECT status FROM crm_schema.leads WHERE id = %s", (lead_id,))
                    old_status_tuple = cur.fetchone()
                    if not old_status_tuple:
                        return jsonify({"message": "Cannot update a non-existent lead."}), 404
                    old_status = old_status_tuple[0]
                    # --- END OF PRE-FETCH ---

                    data = request.json
                    
                    # --- (UNCHANGED) VALIDATION ---
                    if "status" in data:
                        valid_statuses = ['New', 'Contacted', 'Qualified', 'Won', 'Lost']
                        if data["status"] not in valid_statuses:
                            return jsonify({"message": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"}), 400
                    
                    if "email" in data and data["email"]:
                        import re
                        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        if not re.match(email_pattern, data["email"]):
                            return jsonify({"message": "Invalid email format"}), 400
                    
                    if "value" in data and data["value"] is not None:
                        try:
                            if isinstance(data["value"], str):
                                data["value"] = float(data["value"])
                            if data["value"] < 0:
                                return jsonify({"message": "Value cannot be negative"}), 400
                        except (ValueError, TypeError):
                            return jsonify({"message": "Invalid value format"}), 400
                    
                    fields, params = [], []
                    # --- MODIFICATION: Add 'stage' to list of updatable fields ---
                    for key in ["name", "email", "phone", "status", "stage", "value", "organization", "notes", "title", "source", "tags"]:
                        if key in data:
                            db_key = "organization_name" if key == "organization" else key
                            fields.append(f"{db_key} = %s")
                            params.append(data[key])
                    
                    if fields:
                        params.append(lead_id)
                        # --- MODIFICATION: Add updated_at to query ---
                        query = f"UPDATE crm_schema.leads SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
                        cur.execute(query, tuple(params))
                        
                        # (UNCHANGED) Log activity for lead update
                        cur.execute("""
                            INSERT INTO crm_schema.activities (lead_id, user_id, activity_type, title, description, created_by, metadata, organization_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (lead_id, user_id, 'field_updated', 'Lead updated', 
                              f'Lead information was updated', user_id,
                              json.dumps({'updated_fields': list(data.keys())}), user_org_id))

                    # --- (UNCHANGED) DEAL CREATION/DELETION LOGIC ---
                    new_status = data.get("status")

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
                    
                    elif old_status == 'Won' and new_status and new_status != 'Won':
                        cur.execute("DELETE FROM crm_schema.deals WHERE lead_id = %s", (lead_id,))
                        
                    return jsonify({"message": "Lead updated successfully"})

                if request.method == "DELETE":
                    cur.execute("DELETE FROM crm_schema.deals WHERE lead_id = %s", (lead_id,))
                    cur.execute("DELETE FROM crm_schema.leads WHERE id = %s", (lead_id,))
                    return jsonify({"message": "Lead and associated deal deleted successfully"})
    except psycopg2.Error as e:
        import traceback
        print("[Lead Update DB Error]", str(e))
        traceback.print_exc()
        
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

@app.route("/api/leads/bulk-assign", methods=["POST"])
@token_required
def bulk_assign_leads(current_user):
    user_id = current_user["user_id"]
    data = request.get_json()
    
    if not data or 'lead_ids' not in data or 'owner_id' not in data:
        return jsonify({"message": "Missing required fields: lead_ids and owner_id"}), 400
    
    lead_ids = data['lead_ids']
    owner_id = data['owner_id']
    
    if not isinstance(lead_ids, list) or len(lead_ids) == 0:
        return jsonify({"message": "lead_ids must be a non-empty array"}), 400
    
    conn = get_db()
    try:
        with conn:
            cur = conn.cursor()
            
            # Get user's org
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You are not in an organization."}), 400
            user_org_id = org_id_tuple[0]

            # Verify the new owner is in the same organization
            cur.execute("SELECT 1 FROM crm_schema.user_organizations WHERE user_id = %s AND organization_id = %s", (owner_id, user_org_id))
            if not cur.fetchone():
                return jsonify({"message": "The assigned user is not in your organization."}), 403

            # Update all specified leads that belong to the user's organization
            cur.execute("""
                UPDATE crm_schema.leads 
                SET user_id = %s, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ANY(%s) AND organization_id = %s
            """, (owner_id, lead_ids, user_org_id))
            
            updated_count = cur.rowcount
            
            return jsonify({
                "message": f"Successfully assigned {updated_count} leads",
                "updated_count": updated_count
            }), 200
            
    except Exception as e:
        print(f"Database error in bulk assign: {e}")
        return jsonify({"message": "Database error during bulk assign"}), 500
@app.route("/api/leads/bulk-update-status", methods=["POST"])
@token_required
def bulk_update_lead_status(current_user):
    user_id = current_user["user_id"]
    data = request.get_json()
    
    if not data or 'lead_ids' not in data or 'status' not in data:
        return jsonify({"message": "Missing required fields: lead_ids and status"}), 400
    
    lead_ids = data['lead_ids']
    new_status = data['status']
    
    # Validate status
    valid_statuses = ['New', 'Contacted', 'Qualified', 'Won', 'Lost']
    if new_status not in valid_statuses:
        return jsonify({"message": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"}), 400
    
    if not isinstance(lead_ids, list) or len(lead_ids) == 0:
        return jsonify({"message": "lead_ids must be a non-empty array"}), 400
    
    conn = get_db()
    try:
        with conn:
            cur = conn.cursor()
            
            # --- MODIFICATION: Get org_id to update all leads in the organization ---
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You are not in an organization."}), 400
            user_org_id = org_id_tuple[0]

            # Update all specified leads that belong to the user's ORGANIZATION
            cur.execute("""
                UPDATE crm_schema.leads 
                SET status = %s, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ANY(%s) AND organization_id = %s
            """, (new_status, lead_ids, user_org_id))
            # --- END MODIFICATION ---
            
            updated_count = cur.rowcount
            
            # --- (Rest of the function is unchanged) ---
            return jsonify({
                "message": f"Successfully updated {updated_count} leads to {new_status} status",
                "updated_count": updated_count
            }), 200
            
    except Exception as e:
        print(f"Database error in bulk status update: {e}")
        return jsonify({"message": "Database error during bulk status update"}), 500
    
@app.route("/api/leads/bulk-delete", methods=["POST"])
@token_required
def bulk_delete_leads(current_user):
    user_id = current_user["user_id"]
    data = request.get_json()
    
    if not data or 'lead_ids' not in data:
        return jsonify({"message": "Missing required field: lead_ids"}), 400
    
    lead_ids = data['lead_ids']
    
    if not isinstance(lead_ids, list) or len(lead_ids) == 0:
        return jsonify({"message": "lead_ids must be a non-empty array"}), 400
    
    conn = get_db()
    try:
        with conn:
            cur = conn.cursor()
            
            # --- MODIFICATION: Get org_id to delete all leads in the organization ---
            cur.execute("SELECT organization_id FROM crm_schema.user_organizations WHERE user_id = %s", (user_id,))
            org_id_tuple = cur.fetchone()
            if not org_id_tuple:
                return jsonify({"message": "You are not in an organization."}), 400
            user_org_id = org_id_tuple[0]

            # Delete all specified leads that belong to the user's ORGANIZATION
            cur.execute("""
                DELETE FROM crm_schema.leads 
                WHERE id = ANY(%s) AND organization_id = %s
            """, (lead_ids, user_org_id))
            # --- END MODIFICATION ---
            
            deleted_count = cur.rowcount
            
            return jsonify({
                "message": f"Successfully deleted {deleted_count} leads",
                "deleted_count": deleted_count
            }), 200
            
    except Exception as e:
        print(f"Database error in bulk delete: {e}")
        return jsonify({"message": "Database error during bulk delete"}), 500
    
@app.route("/api/leads/<int:lead_id>/communications", methods=["GET", "POST"])
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
@app.route("/api/leads/<int:lead_id>/activity", methods=["GET"])
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

@app.route("/api/settings/primary_meeting", methods=["PUT"])
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
        return get_user_settings(current_user)

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