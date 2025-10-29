"""
Agent tools for email operations.
Implements email management tools: read, search, reply, forward, and compose.
Each tool logs trace events for observability.
"""

import json
from pathlib import Path
from typing import List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from utils import generate_id, get_timestamp, append_trace_event, USER_EMAIL


def _find_best_email_match(search_query: str, config, folder: str = "inbox", limit: int = 1):
    """
    Shared helper to find the best matching email(s) based on search query.
    
    Args:
        search_query: Search terms to match against
        config: EmailToolsConfig instance
        folder: 'inbox', 'outbox', or 'drafts'
        limit: Maximum number of matches to return (1 for single best match)
    
    Returns:
        List of matching emails sorted by relevance and recency
    """
    # Determine which folder to search
    if folder.lower() == "inbox":
        search_dir = config.mailbox_dir
    elif folder.lower() == "outbox":
        search_dir = config.outbox_dir
    elif folder.lower() == "drafts":
        search_dir = config.drafts_dir
    else:
        return []
    
    matches = []
    query_lower = search_query.lower()
    query_words = query_lower.split()
    
    # Search across all emails in the folder
    for file_path in search_dir.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            email = json.load(f)
            
            # Search in from, to, subject, and body
            searchable_text = (
                f"{email.get('from', '')} "
                f"{email.get('to', '')} "
                f"{email.get('subject', '')} "
                f"{email.get('body', email.get('body_plain', ''))}"
            ).lower()
            
            # Calculate relevance score (number of matching words)
            relevance = sum(1 for word in query_words if word in searchable_text)
            
            # If query is empty, include all emails (relevance = 1)
            if not query_words:
                relevance = 1
            
            if relevance > 0:
                email['_relevance'] = relevance
                email['_file_path'] = str(file_path)
                matches.append(email)
    
    # Sort by relevance (most relevant first), then by date (newest first)
    def parse_timestamp(ts: str) -> float:
        """Convert ISO timestamp to float for sorting. Returns 0 if invalid."""
        try:
            from datetime import datetime
            return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
        except:
            return 0.0
    
    matches.sort(key=lambda x: (
        -x.get('_relevance', 0),
        -parse_timestamp(x.get('received_ts', x.get('created_ts', '')))
    ))
    
    return matches[:limit]


class EmailToolsConfig:
    """Configuration for email tools."""
    def __init__(self, mailbox_dir: str, drafts_dir: str, outbox_dir: str, trace_file: str):
        self.mailbox_dir = Path(mailbox_dir)
        self.drafts_dir = Path(drafts_dir)
        self.outbox_dir = Path(outbox_dir)
        self.trace_file = trace_file
        self.session_id: Optional[str] = None


class ReadAllEmailsInput(BaseModel):
    """Input schema for read_all_emails tool - no input required."""


class SearchEmailsInput(BaseModel):
    """Input schema for search_emails tool."""
    query: str = Field(description="Search query to find emails (sender name, subject keywords, content)")
    folder: str = Field(
        default="inbox",
        description="Folder to search in: 'inbox', 'outbox', or 'drafts'"
    )


class ReplyToEmailInput(BaseModel):
    """Input schema for reply_to_email tool."""
    search_query: str = Field(
        description="Keywords to identify the email to reply to (sender name, subject keywords). If multiple matches, the most recent email will be used."
    )
    reply_body: str = Field(description="Your reply message content")


class ForwardEmailInput(BaseModel):
    """Input schema for forward_email tool."""
    search_query: str = Field(
        description="Keywords to identify the email to forward (sender name, subject keywords). If multiple matches, the most recent email will be used."
    )
    forward_to: str = Field(description="Email address to forward to")
    message: str = Field(
        default="",
        description="Optional message to include with the forwarded email"
    )


class DraftEmailInput(BaseModel):
    """Input schema for draft_email tool."""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")


class ComposeEmailInput(BaseModel):
    """Input schema for compose_email tool."""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email message body")


class ReadAllEmailsTool(BaseTool):
    """Tool for reading all emails in the inbox."""
    name: str = "read_all_emails"
    description: str = """Read all emails in the inbox. Returns complete information about each email including sender, recipient, subject, body, date, and status. Emails are sorted from newest to oldest. Use this to get an overview of all emails."""
    args_schema: type[BaseModel] = ReadAllEmailsInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self) -> str:
        """Read and return all inbox emails."""
        call_id = generate_id("tcall")
        
        # Log tool call
        if self.config and self.config.session_id:
            append_trace_event(
                self.config.trace_file,
                "tool_call",
                self.config.session_id,
                {
                    "tool_name": self.name,
                    "inputs": {},
                    "call_id": call_id
                }
            )
        
        try:
            # Use shared helper to get all emails (empty query returns all)
            emails = _find_best_email_match("", self.config, "inbox", limit=1000)
            
            if not emails:
                result = "No emails found in inbox."
            else:
                result_lines = [f"Found {len(emails)} email(s) in inbox:\n"]
                for i, email in enumerate(emails, 1):
                    read_status = "Read" if email.get("metadata", {}).get("read") else "Unread"
                    result_lines.append(
                        f"\n--- Email {i} ---\n"
                        f"From: {email['from']}\n"
                        f"To: {email['to']}\n"
                        f"Subject: {email['subject']}\n"
                        f"Date: {email['received_ts']}\n"
                        f"Status: {read_status}\n"
                        f"Body:\n{email['body_plain']}\n"
                    )
                result = "\n".join(result_lines)
            
            # Log tool result
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"result": result, "count": len(emails)},
                        "call_id": call_id
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error reading emails: {str(e)}"
            
            # Log error
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg},
                        "call_id": call_id
                    }
                )
            
            return error_msg


class SearchEmailsTool(BaseTool):
    """Tool for searching emails with smart ranking."""
    name: str = "search_emails"
    description: str = """Search for emails matching a query in inbox, outbox, or drafts. Searches across sender, recipient, subject, and body. Returns most relevant matches sorted by relevance and recency. Use this to find specific emails when you need more details than read_all_emails provides."""
    args_schema: type[BaseModel] = SearchEmailsInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, query: str, folder: str = "inbox") -> str:
        """Search emails and return matching results."""
        call_id = generate_id("tcall")
        
        # Log tool call
        if self.config and self.config.session_id:
            append_trace_event(
                self.config.trace_file,
                "tool_call",
                self.config.session_id,
                {
                    "tool_name": self.name,
                    "inputs": {"query": query, "folder": folder},
                    "call_id": call_id
                }
            )
        
        try:
            # Use shared helper to find matches
            matches = _find_best_email_match(query, self.config, folder, limit=10)
            
            if not matches:
                result = f"No emails found matching '{query}' in {folder}."
            else:
                result_lines = [f"Found {len(matches)} email(s) matching '{query}' in {folder} (showing most relevant):\n"]
                for i, email in enumerate(matches, 1):
                    # Remove internal fields from display
                    relevance_score = email.pop('_relevance', 0)
                    email.pop('_file_path', None)
                    
                    status_info = ""
                    if folder == "inbox":
                        read_status = "Read" if email.get("metadata", {}).get("read") else "Unread"
                        status_info = f"Status: {read_status}\n"
                    elif folder == "outbox":
                        status_info = "Status: Sent\n"
                    elif folder == "drafts":
                        status_info = "Status: Draft\n"
                    
                    result_lines.append(
                        f"\n--- Match {i} (Relevance: {relevance_score}) ---\n"
                        f"From: {email.get('from', 'N/A')}\n"
                        f"To: {email.get('to', 'N/A')}\n"
                        f"Subject: {email.get('subject', 'N/A')}\n"
                        f"Date: {email.get('received_ts', email.get('created_ts', 'N/A'))}\n"
                        f"{status_info}"
                        f"Body:\n{email.get('body', email.get('body_plain', 'N/A'))}\n"
                    )
                result = "\n".join(result_lines)
            
            # Log tool result
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"result": result, "count": len(matches), "folder": folder},
                        "call_id": call_id
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error searching emails: {str(e)}"
            
            # Log error
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg},
                        "call_id": call_id
                    }
                )
            
            return error_msg


class ReplyToEmailTool(BaseTool):
    """Tool for replying to an email."""
    name: str = "reply_to_email"
    description: str = """Reply to an email. Automatically finds the email using your search query, extracts the sender's email address, constructs a 'Re: [subject]' reply, and sends it. If multiple emails match, the MOST RECENT one is used. This is a one-step operation - no need to manually construct the reply email."""
    args_schema: type[BaseModel] = ReplyToEmailInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, search_query: str, reply_body: str) -> str:
        """Find an email and send a reply."""
        call_id = generate_id("tcall")
        
        # Log tool call
        if self.config and self.config.session_id:
            append_trace_event(
                self.config.trace_file,
                "tool_call",
                self.config.session_id,
                {
                    "tool_name": self.name,
                    "inputs": {"search_query": search_query, "reply_body": reply_body},
                    "call_id": call_id
                }
            )
        
        try:
            # Use shared helper to find the best matching email
            matches = _find_best_email_match(search_query, self.config, "inbox", limit=1)
            
            if not matches:
                result = f"No email found matching '{search_query}'. Please check your search terms and try again."
            else:
                original_email = matches[0]
                
                # Extract details for reply
                reply_to = original_email.get('from', '')
                original_subject = original_email.get('subject', '')
                original_body = original_email.get('body_plain', original_email.get('body', ''))
                original_date = original_email.get('received_ts', '')
                
                # Construct reply subject
                if original_subject.lower().startswith('re:'):
                    reply_subject = original_subject
                else:
                    reply_subject = f"Re: {original_subject}"
                
                # Construct full reply body with original email quoted
                full_reply_body = f"{reply_body}\n\n"
                full_reply_body += f"On {original_date}, {reply_to} wrote:\n"
                # Quote original message (prefix each line with >)
                quoted_lines = [f"> {line}" for line in original_body.split('\n')]
                full_reply_body += '\n'.join(quoted_lines)
                
                # Create and send reply email
                email_id = generate_id("sent")
                email_file = self.config.outbox_dir / f"{email_id}.json"
                
                reply_email = {
                    "from": USER_EMAIL,
                    "to": reply_to,
                    "subject": reply_subject,
                    "body": full_reply_body,
                    "created_ts": get_timestamp(),
                    "sent_ts": get_timestamp(),
                    "status": "sent",
                    "in_reply_to": original_email.get('id', 'unknown')
                }
                
                with open(email_file, "w", encoding="utf-8") as f:
                    json.dump(reply_email, f, indent=2)
                
                match_info = ""
                if len(matches) > 1:
                    match_info = f" (Found {len(matches)} matches, replied to most recent)"
                
                result = (
                    f"Reply sent successfully!{match_info}\n\n"
                    f"Original from: {reply_to}\n"
                    f"Original subject: {original_subject}\n"
                    f"Reply subject: {reply_subject}\n\n"
                    f"Your reply has been sent."
                )
            
            # Log tool result
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"result": result, "matches_found": len(matches)},
                        "call_id": call_id
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error replying to email: {str(e)}"
            
            # Log error
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg},
                        "call_id": call_id
                    }
                )
            
            return error_msg


class ForwardEmailTool(BaseTool):
    """Tool for forwarding an email."""
    name: str = "forward_to_email"
    description: str = """Forward an email to someone else. Automatically finds the email using your search query, constructs a 'Fwd: [subject]' forward, and sends it with your optional message. If multiple emails match, the MOST RECENT one is forwarded."""
    args_schema: type[BaseModel] = ForwardEmailInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, search_query: str, forward_to: str, message: str = "") -> str:
        """Find an email and forward it."""
        call_id = generate_id("tcall")
        
        # Log tool call
        if self.config and self.config.session_id:
            append_trace_event(
                self.config.trace_file,
                "tool_call",
                self.config.session_id,
                {
                    "tool_name": self.name,
                    "inputs": {"search_query": search_query, "forward_to": forward_to, "message": message},
                    "call_id": call_id
                }
            )
        
        try:
            # Use shared helper to find the best matching email
            matches = _find_best_email_match(search_query, self.config, "inbox", limit=1)
            
            if not matches:
                result = f"No email found matching '{search_query}'. Please check your search terms and try again."
            else:
                original_email = matches[0]
                
                # Extract details for forward
                original_from = original_email.get('from', '')
                original_to = original_email.get('to', '')
                original_subject = original_email.get('subject', '')
                original_body = original_email.get('body_plain', original_email.get('body', ''))
                original_date = original_email.get('received_ts', '')
                
                # Construct forward subject
                if original_subject.lower().startswith('fwd:'):
                    forward_subject = original_subject
                else:
                    forward_subject = f"Fwd: {original_subject}"
                
                # Construct forward body with original message (Gmail/Outlook style)
                forward_body = ""
                if message:
                    forward_body += f"{message}\n\n"
                
                forward_body += (
                    f"---------- Forwarded message ---------\n"
                    f"From: {original_from}\n"
                    f"Date: {original_date}\n"
                    f"Subject: {original_subject}\n"
                    f"To: {original_to}\n\n"
                    f"{original_body}"
                )
                
                # Create and send forward email
                email_id = generate_id("sent")
                email_file = self.config.outbox_dir / f"{email_id}.json"
                
                forward_email = {
                    "from": USER_EMAIL,
                    "to": forward_to,
                    "subject": forward_subject,
                    "body": forward_body,
                    "created_ts": get_timestamp(),
                    "sent_ts": get_timestamp(),
                    "status": "sent",
                    "forwarded_from": original_email.get('id', 'unknown')
                }
                
                with open(email_file, "w", encoding="utf-8") as f:
                    json.dump(forward_email, f, indent=2)
                
                match_info = ""
                if len(matches) > 1:
                    match_info = f" (Found {len(matches)} matches, forwarded most recent)"
                
                result = (
                    f"Email forwarded successfully!{match_info}\n\n"
                    f"Original from: {original_from}\n"
                    f"Original subject: {original_subject}\n"
                    f"Forwarded to: {forward_to}\n"
                    f"Forward subject: {forward_subject}\n\n"
                    f"Your forward has been sent."
                )
            
            # Log tool result
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"result": result, "matches_found": len(matches)},
                        "call_id": call_id
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error forwarding email: {str(e)}"
            
            # Log error
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg},
                        "call_id": call_id
                    }
                )
            
            return error_msg


class ComposeEmailTool(BaseTool):
    """Tool for composing and sending a new email."""
    name: str = "compose_email"
    description: str = """Compose and send a new email. Use this for creating original emails (not replies or forwards). The email is sent immediately."""
    args_schema: type[BaseModel] = ComposeEmailInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, to: str, subject: str, body: str) -> str:
        """Create and immediately send an email."""
        call_id = generate_id("tcall")
        
        # Log tool call
        if self.config and self.config.session_id:
            append_trace_event(
                self.config.trace_file,
                "tool_call",
                self.config.session_id,
                {
                    "tool_name": self.name,
                    "inputs": {"to": to, "subject": subject, "body": body},
                    "call_id": call_id
                }
            )
        
        try:
            # Generate email ID and filename
            email_id = generate_id("sent")
            email_file = self.config.outbox_dir / f"{email_id}.json"
            
            # Create sent email object
            email = {
                "from": USER_EMAIL,
                "to": to,
                "subject": subject,
                "body": body,
                "created_ts": get_timestamp(),
                "sent_ts": get_timestamp(),
                "status": "sent"
            }
            
            # Save directly to outbox
            with open(email_file, "w", encoding="utf-8") as f:
                json.dump(email, f, indent=2)
            
            result = (
                f"Email sent successfully!\n\n"
                f"To: {to}\n"
                f"Subject: {subject}\n\n"
                f"Your email has been delivered."
            )
            
            # Log tool result
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"result": result, "email_id": email_id},
                        "call_id": call_id
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error sending email: {str(e)}"
            
            # Log error
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg},
                        "call_id": call_id
                    }
                )
            
            return error_msg


class DraftEmailTool(BaseTool):
    """Tool for creating email drafts without sending."""
    name: str = "draft_email"
    description: str = """Create an email draft without sending it. The draft will be saved for later review or sending. Use this when you want to prepare an email but not send it immediately."""
    args_schema: type[BaseModel] = DraftEmailInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, to: str, subject: str, body: str) -> str:
        """Create and save an email draft."""
        call_id = generate_id("tcall")
        
        # Log tool call
        if self.config and self.config.session_id:
            append_trace_event(
                self.config.trace_file,
                "tool_call",
                self.config.session_id,
                {
                    "tool_name": self.name,
                    "inputs": {"to": to, "subject": subject, "body": body},
                    "call_id": call_id
                }
            )
        
        try:
            # Generate draft ID and filename
            draft_id = generate_id("draft")
            draft_file = self.config.drafts_dir / f"{draft_id}.json"
            
            # Create draft object
            draft = {
                "from": USER_EMAIL,
                "to": to,
                "subject": subject,
                "body": body,
                "created_ts": get_timestamp(),
                "status": "draft"
            }
            
            # Save draft to file
            with open(draft_file, "w", encoding="utf-8") as f:
                json.dump(draft, f, indent=2)
            
            result = (
                f"Draft created successfully!\n\n"
                f"To: {to}\n"
                f"Subject: {subject}\n\n"
                f"Draft saved. You can review or send it later."
            )
            
            # Log tool result
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"result": result, "draft_id": draft_id},
                        "call_id": call_id
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error creating draft: {str(e)}"
            
            # Log error
            if self.config and self.config.session_id:
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg},
                        "call_id": call_id
                    }
                )
            
            return error_msg


def create_tools(config: EmailToolsConfig) -> List[BaseTool]:
    """
    Create and configure all email tools.
    
    Args:
        config: EmailToolsConfig instance with paths and settings
    
    Returns:
        List of configured tools
    """
    read_all_tool = ReadAllEmailsTool()
    read_all_tool.config = config
    
    search_tool = SearchEmailsTool()
    search_tool.config = config
    
    reply_tool = ReplyToEmailTool()
    reply_tool.config = config
    
    forward_tool = ForwardEmailTool()
    forward_tool.config = config
    
    compose_tool = ComposeEmailTool()
    compose_tool.config = config
    
    draft_tool = DraftEmailTool()
    draft_tool.config = config
    
    return [read_all_tool, search_tool, reply_tool, forward_tool, compose_tool, draft_tool]
