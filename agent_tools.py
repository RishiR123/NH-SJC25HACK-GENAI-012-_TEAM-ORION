import logging
# from app import proxy_to_mcp, llm  # Moved to lazy imports inside functions
from flask import g
from langchain_core.messages import SystemMessage
from typing import Optional, Dict, Any
import json

# --- (No changes to this helper function) ---
EMAIL_PARTS_PROMPT = """
You are an assistant. A user wants to send an email with the following details.
Your *only* job is to generate the `subject` and `body` for this email based on the user's intent.
You MUST respond in a single, valid JSON object with keys "subject" and "body".

Recipient: {recipient}
User's Intent: {intent}
---
JSON:
"""

def _generate_email_parts(recipient: str, intent: str) -> Dict[str, str]:
# ... (This function is unchanged from the previous version) ...
# ... (It calls llm.invoke and returns {"subject": "...", "body": "..."} or {"error": "..."}) ...
    from app import llm  # Lazy import to avoid circular dependency
    
    if not llm:
        logging.error("LLM not available for email parts generation")
        return {"error": "Could not generate email content."}
    
    prompt_messages = [
        SystemMessage(content=EMAIL_PARTS_PROMPT.format(recipient=recipient, intent=intent))
    ]
    try:
        ai_response = llm.invoke(prompt_messages)
        content = ai_response.content.strip()
        
        # Clean potential markdown fences
        if content.startswith("```json"):
            content = content[7:-3].strip()
        
        parts = json.loads(content)
        if "subject" in parts and "body" in parts:
            return parts
        else:
            return {"error": "LLM failed to return valid JSON with 'subject' and 'body'."}
            
    except json.JSONDecodeError as e:
        logging.error(f"LLM call failed to return valid JSON in _generate_email_parts: {e}. Raw content: {content}")
        return {"error": f"Error parsing LLM response: {e}"}
    except Exception as e:
        logging.error(f"LLM call failed in _generate_email_parts: {e}")
        return {"error": f"Error generating email parts: {e}"}

# --- (NEW) Tool 1: The "Dumb" Sender ---
# This tool is for when the agent *already* has all the info.
def send_email_logic(
    current_user: dict, 
    recipient: str, 
    subject: str, 
    body: str
) -> dict:
    """
    A simple wrapper that sends an email.
    It requires recipient, subject, AND body.
    """
    from app import proxy_to_mcp  # Lazy import to avoid circular dependency
    
    try:
        endpoint_name = "actions/send_email"
        data = {"recipient": recipient, "subject": subject, "body": body}
        
        response, status_code = proxy_to_mcp(
            endpoint_name, 
            current_user, 
            method="POST", 
            data=data
        )
        
        if status_code >= 400:
            logging.warning(f"send_email_logic failed: {response.get_data(as_text=True)}")
            return {"status": "error", "message": response.get_json()}
        
        return {"status": "success", "result": response.get_json()}

    except Exception as e:
        logging.error(f"Error in send_email_logic: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# --- (NEW) Tool 2: The "Smart" Generator/Sender ---
# This is the tool the agent will use for your specific prompt.
def generate_and_send_email_logic(
    current_user: dict,
    recipient: str,
    user_intent: str
) -> dict:
    """
    A smart wrapper that generates and sends an email.
    It requires only the recipient and the user's intent.
    """
    from app import proxy_to_mcp  # Lazy import to avoid circular dependency
    
    try:
        logging.info(f"Generating email parts for intent: {user_intent}")
        generated_parts = _generate_email_parts(recipient, user_intent)
        
        if "error" in generated_parts:
            return {"status": "error", "message": generated_parts["error"]}
        
        final_subject = generated_parts.get("subject")
        final_body = generated_parts.get("body")

        # Now call the *same* proxy logic as the simple tool
        endpoint_name = "actions/send_email"
        data = {"recipient": recipient, "subject": final_subject, "body": final_body}
        
        response, status_code = proxy_to_mcp(
            endpoint_name, 
            current_user, 
            method="POST", 
            data=data
        )
        
        if status_code >= 400:
            logging.warning(f"generate_and_send_email_logic failed: {response.get_data(as_text=True)}")
            return {"status": "error", "message": response.get_json()}
        
        # Return generated parts so the agent can show the user
        success_result = response.get_json()
        success_result['generated_subject'] = final_subject
        success_result['generated_body'] = final_body
        return {"status": "success", "result": success_result}

    except Exception as e:
        logging.error(f"Error in generate_and_send_email_logic: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# --- (No changes to get_emails_logic or reply_email_logic) ---
def get_emails_logic(current_user: dict, query: str = None, folder: str = None, top: int = 5) -> dict:
# ... (rest of get_emails_logic) ...
    """
    A wrapper to get emails from either Gmail or Outlook.
    If 'query' is provided, it searches Gmail.
    If 'folder' is provided, it gets from Outlook.
    """
    from app import proxy_to_mcp  # Lazy import to avoid circular dependency
    
    try:
        if query:
            # --- Call Gmail ---
            endpoint_name = "actions/get_gmail_emails" # [cite: app.py, tools.py]
            params = {"query": query}
            method = "GET"
            response, status_code = proxy_to_mcp(
                endpoint_name, current_user, method=method, params=params
            )
        elif folder:
            # --- Call Outlook ---
            endpoint_name = "actions/get_outlook_emails" # [cite: app.py, tools.py]
            params = {"folder": folder, "top": top}
            method = "GET"
            response, status_code = proxy_to_mcp(
                endpoint_name, current_user, method=method, params=params
            )
        else:
            return {"status": "error", "message": "You must provide either a 'query' for Gmail or a 'folder' for Outlook."}

        if status_code >= 400:
            logging.warning(f"get_emails_logic failed: {response.get_data(as_text=True)}")
            return {"status": "error", "message": response.get_json()}
        return {"status": "success", "result": response.get_json()}

    except Exception as e:
        logging.error(f"Error in get_emails_logic: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

def reply_email_logic(current_user: dict, message_id: str, body: str, tool_choice: str = "gmail") -> dict:
# ... (rest of reply_email_logic) ...
    """
    A wrapper to reply to an email in either Gmail or Outlook.
    """
    from app import proxy_to_mcp  # Lazy import to avoid circular dependency
    
    try:
        data = {"message_id": message_id, "body": body}
        
        if "outlook" in tool_choice.lower():
            endpoint_name = "actions/reply_outlook_email" # [cite: app.py, tools.py]
        else:
            endpoint_name = "actions/reply_gmail_email" # [cite: app.py, tools.py]
            
        response, status_code = proxy_to_mcp(
            endpoint_name, 
            current_user, 
            method="POST", 
            data=data
        )

        if status_code >= 400:
            logging.warning(f"reply_email_logic failed: {response.get_data(as_text=True)}")
            return {"status": "error", "message": response.get_json()}
        return {"status": "success", "result": response.get_json()}

    except Exception as e:
        logging.error(f"Error in reply_email_logic: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

