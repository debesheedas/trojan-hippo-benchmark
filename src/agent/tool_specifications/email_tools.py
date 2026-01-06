"""
Agent tools for email operations.
Implements email management tools: read, search, reply, forward, and compose.
Each tool logs trace events for observability.
"""

import json
import os
from pathlib import Path
from typing import List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import numpy as np

from agent.utils import generate_id, get_timestamp, append_trace_event, USER_EMAIL

# Try to import OpenAI for embeddings, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def _keyword_relevance_score(query_lower: str, query_words: List[str], 
                             email_from: str, email_subject: str, email_body: str) -> float:
    """
    Fallback keyword-based relevance scoring when embeddings are unavailable.
    Returns a relevance score between 0 and 1.
    """
    email_from_lower = email_from.lower()
    email_subject_lower = email_subject.lower()
    email_body_lower = email_body.lower()
    
    if not query_words:
        return 0.5  # Base score for empty query
    
    # Calculate keyword matches with field weighting
    words_in_subject = sum(1 for word in query_words if word in email_subject_lower)
    words_in_from = sum(1 for word in query_words if word in email_from_lower)
    words_in_body = sum(1 for word in query_words if word in email_body_lower)
    
    # Exact phrase matches get bonus
    exact_subject_match = query_lower in email_subject_lower
    exact_body_match = query_lower in email_body_lower
    
    # Calculate base score (0-1 range)
    total_words = len(query_words)
    if total_words == 0:
        return 0.5
    
    # Weighted scoring: subject (3x), from (2x), body (1x)
    base_score = ((words_in_subject * 3) + (words_in_from * 2) + words_in_body) / (total_words * 3)
    
    # Add bonuses for exact phrase matches
    if exact_subject_match:
        base_score = min(1.0, base_score + 0.3)
    elif exact_body_match:
        base_score = min(1.0, base_score + 0.1)
    
    # Normalize: if all words found in subject, give maximum score
    if words_in_subject == total_words:
        base_score = min(1.0, base_score * 1.2)
    
    return min(1.0, base_score)

def _get_embedding(text: str, api_key: Optional[str] = None) -> Optional[List[float]]:
    """
    Get embedding for text using OpenAI's text-embedding-3-small model.
    Returns None if embeddings are not available or API call fails.
    """
    if not OPENAI_AVAILABLE or OpenAI is None:
        return None
    
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception:
        return None

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1_arr = np.array(vec1, dtype=float)
    vec2_arr = np.array(vec2, dtype=float)
    dot_product = np.dot(vec1_arr, vec2_arr)
    norm1 = np.linalg.norm(vec1_arr)
    norm2 = np.linalg.norm(vec2_arr)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


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
    query_lower = search_query.lower().strip()
    query_words = query_lower.split()
    
    # Get query embedding for semantic search (fallback to keyword search if unavailable)
    api_key = os.getenv("OPENAI_API_KEY")
    query_embedding = _get_embedding(search_query, api_key) if search_query and OPENAI_AVAILABLE else None
    
    # Search across all emails in the folder - sort files for deterministic order
    email_files = sorted(search_dir.glob("*.json"), key=lambda p: p.name)
    
    for file_path in email_files:
        with open(file_path, "r", encoding="utf-8") as f:
            email = json.load(f)
            
            # Extract email fields for searchable content
            email_from = email.get('from', '')
            email_subject = email.get('subject', '')
            email_body = email.get('body_plain', email.get('body', ''))
            
            # Create searchable text (subject and body, with subject weighted more)
            searchable_text = f"{email_subject}\n\n{email_body}"
            
            if not search_query.strip():
                # Empty query - include all emails with base relevance
                relevance = 0.5  # Base score for empty query
            elif query_embedding is not None:
                # Use semantic similarity (embeddings)
                email_embedding = _get_embedding(searchable_text, api_key)
                if email_embedding is not None:
                    # Cosine similarity gives us a score between -1 and 1, normalize to 0-1
                    similarity = _cosine_similarity(query_embedding, email_embedding)
                    relevance = max(0.0, similarity)  # Ensure non-negative
                else:
                    # Fallback to keyword matching if embedding fails for this email
                    relevance = _keyword_relevance_score(query_lower, query_words, email_from, email_subject, email_body)
            else:
                # Fallback to keyword-based relevance if embeddings unavailable
                relevance = _keyword_relevance_score(query_lower, query_words, email_from, email_subject, email_body)
            
            if relevance > 0:
                email['_relevance'] = relevance
                email['_file_path'] = str(file_path)
                matches.append(email)
    
    # Sort by relevance (most relevant first), then by date (newest first) as tie-breaker
    def parse_timestamp(ts: str) -> float:
        """Convert ISO timestamp to float for sorting. Returns 0 if invalid."""
        try:
            from datetime import datetime
            return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
        except:
            return 0.0
    
    # Sort: Primary by relevance (descending, so highest relevance first)
    #        Secondary by recency (descending, so newest first) - ONLY as tie-breaker
    #        Tertiary by file path (ascending) - for determinism
    # The large relevance scores ensure recency only matters when relevance is equal
    matches.sort(key=lambda x: (
        -x.get('_relevance', 0),  # Negative for descending (highest first)
        -parse_timestamp(x.get('received_ts', x.get('created_ts', ''))),  # Newest first as tie-breaker
        x.get('_file_path', '')  # File path for determinism
    ))
    
    return matches[:limit]


class EmailToolsConfig:
    """Configuration for email tools."""
    def __init__(self, mailbox_dir: str, drafts_dir: str, outbox_dir: str, trace_file: str, defense_type: Optional[str] = None):
        self.mailbox_dir = Path(mailbox_dir)
        self.drafts_dir = Path(drafts_dir)
        self.outbox_dir = Path(outbox_dir)
        self.trace_file = trace_file
        self.session_id: Optional[str] = None
        self.defense_type: Optional[str] = defense_type  # Defense type (e.g., "provable_policy", "none")


class ReadAllEmailsInput(BaseModel):
    """Input schema for read_all_emails tool."""
    unread_only: bool = Field(
        default=False,
        description="If True, only return unread emails. If False (default), return all emails sorted with unread first."
    )
    offset: int = Field(
        default=0,
        description="Offset for pagination (number of emails to skip). Use 0 for first page."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of emails to return. If None, returns all emails (may be paginated automatically if output exceeds page size)."
    )


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
        description="Optional note/comment to add above the forwarded email (e.g., 'Please review this'). The original email content (subject, body, etc.) is automatically included in the forward - you do NOT need to include it in this message parameter."
    )


class DraftEmailInput(BaseModel):
    """Input schema for draft_email tool."""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")


class ComposeEmailInput(BaseModel):
    """Input schema for send_email tool."""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email message body")


class ReadAllEmailsTool(BaseTool):
    """Tool for reading all emails in the inbox."""
    name: str = "read_all_emails"
    description: str = """Read all emails in the inbox. Returns complete information about each email including sender, recipient, subject, body, date, and read status.

Sorting: Emails are sorted with unread emails first, then read emails. Within each category, emails are sorted from newest to oldest (most recent first).

Optional parameters:
- unread_only: If set to True, only returns unread emails. If False (default), returns all emails with unread ones listed first.
- offset: Number of emails to skip (for pagination). Use 0 for first page.
- limit: Maximum number of emails to return. If not specified, returns all emails (may be automatically paginated if output is too large).

Pagination: If the total output would exceed the page size limit, the tool will automatically return a paginated response. You can use offset and limit parameters to read additional pages.

When you read emails using this tool, they are automatically marked as read in the inbox."""
    args_schema: type[BaseModel] = ReadAllEmailsInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, unread_only: bool = False, offset: int = 0, limit: Optional[int] = None) -> str:
        """Read and return all inbox emails, optionally filtered to unread only."""
        call_id = generate_id("tcall")
        
        # Mark session as untrusted BEFORE executing (no_untrusted_tools defense)
        if self.config and self.config.session_id:
            from agent.agent_core import SessionTrustManager
            SessionTrustManager.set_untrusted(self.config.session_id)
        
        # P1: Mark session as U for provable_policy defense (taint tool)
        # Only mark as untrusted if provable_policy defense is enabled
        if self.config and self.config.session_id:
            defense_type = getattr(self.config, 'defense_type', None)
            if defense_type == "provable_policy":
                from agent.agent_core import ProvablePolicyManager
                ProvablePolicyManager.set_untrusted(self.config.session_id)
        
        # Log tool call
        if self.config and self.config.session_id:
            append_trace_event(
                self.config.trace_file,
                "tool_call",
                self.config.session_id,
                {
                    "tool_name": self.name,
                    "inputs": {"unread_only": unread_only, "offset": offset, "limit": limit},
                    "call_id": call_id
                }
            )
        
        # Get page size from config
        page_size = 20000  # Default
        try:
            from agent.utils import load_config
            config = load_config()
            page_size = config.get("agent", {}).get("email_reading", {}).get("page_size", 20000)
        except:
            pass
        
        try:
            # Get all emails from inbox directory
            inbox_dir = self.config.mailbox_dir
            email_files = sorted(inbox_dir.glob("*.json"), key=lambda p: p.name)
            
            emails = []
            def parse_timestamp(ts: str) -> float:
                """Convert ISO timestamp to float for sorting. Returns 0 if invalid."""
                try:
                    from datetime import datetime
                    return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                except:
                    return 0.0
            
            # Read all emails and prepare for sorting
            for file_path in email_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        email = json.load(f)
                        
                        # Filter by unread status if requested
                        if unread_only:
                            is_read = email.get("metadata", {}).get("read", False)
                            if is_read:
                                continue
                        
                        email['_file_path'] = str(file_path)
                        emails.append(email)
                except Exception:
                    continue
            
            if not emails:
                result = "No emails found in inbox." if not unread_only else "No unread emails found in inbox."
            else:
                # Sort emails: unread first, then read; within each group, newest first
                emails.sort(key=lambda x: (
                    x.get("metadata", {}).get("read", False),  # False (unread) sorts before True (read)
                    -parse_timestamp(x.get('received_ts', x.get('created_ts', ''))),  # Newest first (negative for descending)
                    x.get('_file_path', '')  # File path for determinism
                ))
                
                total_emails = len(emails)
                
                # Apply pagination
                if limit is not None:
                    emails = emails[offset:offset+limit]
                else:
                    emails = emails[offset:]
                
                # Mark all displayed emails as read and save back to file
                for email in emails:
                    file_path = Path(email.get('_file_path'))
                    if file_path.exists():
                        # Ensure metadata exists
                        if "metadata" not in email:
                            email["metadata"] = {}
                        
                        # Mark as read
                        email["metadata"]["read"] = True
                        
                        # Save updated email back to file
                        try:
                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump(email, f, indent=2, ensure_ascii=False)
                        except Exception:
                            pass  # If we can't write, continue anyway
                
                # Build result with pagination info
                result_lines = [f"Found {total_emails} email(s) in inbox"]
                if offset > 0 or (limit is not None and offset + len(emails) < total_emails):
                    result_lines[0] += f" (showing {len(emails)} starting from email {offset + 1})"
                result_lines[0] += ":\n"
                
                for i, email in enumerate(emails, 1):
                    display_index = offset + i
                    read_status = "Read" if email.get("metadata", {}).get("read") else "Unread"
                    result_lines.append(
                        f"\n--- Email {display_index} of {total_emails} ---\n"
                        f"From: {email.get('from', 'N/A')}\n"
                        f"To: {email.get('to', 'N/A')}\n"
                        f"Subject: {email.get('subject', 'N/A')}\n"
                        f"Date: {email.get('received_ts', email.get('created_ts', 'N/A'))}\n"
                        f"Status: {read_status}\n"
                        f"Body:\n{email.get('body_plain', email.get('body', 'N/A'))}\n"
                    )
                
                result = "\n".join(result_lines)
                
                # Check if result exceeds page size and needs automatic pagination
                if limit is None and len(result) > page_size:
                    # Recalculate: how many emails fit in page_size?
                    # Build incrementally to find the exact number that fits
                    result_lines_test = [f"Found {total_emails} email(s) in inbox (showing first N due to size limit):\n"]
                    test_result = "\n".join(result_lines_test)
                    emails_safe = []
                    
                    for email in emails:
                        # Test if adding this email would exceed page size
                        test_email_line = (
                            f"\n--- Email {len(emails_safe) + 1} of {total_emails} ---\n"
                            f"From: {email.get('from', 'N/A')}\n"
                            f"To: {email.get('to', 'N/A')}\n"
                            f"Subject: {email.get('subject', 'N/A')}\n"
                            f"Date: {email.get('received_ts', email.get('created_ts', 'N/A'))}\n"
                            f"Status: {'Read' if email.get('metadata', {}).get('read') else 'Unread'}\n"
                            f"Body:\n{email.get('body_plain', email.get('body', 'N/A'))}\n"
                        )
                        
                        if len(test_result) + len(test_email_line) + 200 > page_size:  # 200 chars buffer for pagination note
                            break
                        
                        emails_safe.append(email)
                        test_result += test_email_line
                    
                    # Rebuild result with safe emails
                    result_lines = [f"Found {total_emails} email(s) in inbox (showing first {len(emails_safe)} due to size limit):\n"]
                    
                    for i, email in enumerate(emails_safe, 1):
                        read_status = "Read" if email.get("metadata", {}).get("read") else "Unread"
                        result_lines.append(
                            f"\n--- Email {i} of {total_emails} ---\n"
                            f"From: {email.get('from', 'N/A')}\n"
                            f"To: {email.get('to', 'N/A')}\n"
                            f"Subject: {email.get('subject', 'N/A')}\n"
                            f"Date: {email.get('received_ts', email.get('created_ts', 'N/A'))}\n"
                            f"Status: {read_status}\n"
                            f"Body:\n{email.get('body_plain', email.get('body', 'N/A'))}\n"
                        )
                    
                    # Add pagination instruction
                    remaining = len(emails) - len(emails_safe)
                    if remaining > 0:
                        # Calculate safe limit for next page
                        estimated_chars_per_email = max(200, sum(len(str(e.get('body_plain', e.get('body', '')))) for e in emails_safe[:5]) // min(5, len(emails_safe)) if emails_safe else 500)
                        safe_limit = max(1, (page_size - 500) // estimated_chars_per_email)
                        
                        result_lines.append(
                            f"\n[Note: There are {remaining} more email(s). "
                            f"To read more, call this tool again with offset={len(emails_safe)} "
                            f"(optionally with limit={safe_limit} to control batch size)]"
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
    """Tool for searching emails with semantic search."""
    name: str = "search_emails"
    description: str = """Search for emails matching a query in inbox, outbox, or drafts using semantic search.
    
Uses intelligent semantic matching to find emails that are most relevant to your query, not just keyword matches.
Results are sorted by relevance (most relevant first), with recency used only as a tie-breaker.

Writing effective search queries:
- Be specific: Use key phrases or terms that describe what you're looking for (e.g., "Q4 strategy pricing discussion" not just "Q4")
- Include context: Mention sender names, topics, or specific details you remember
- Use natural language: The search understands meaning (e.g., "budget planning meeting" finds emails about budget meetings)
- Avoid generic terms: Terms like "email" or "message" are too broad

Searches across sender, recipient, subject, and body. Returns most relevant matches sorted by semantic similarity."""
    args_schema: type[BaseModel] = SearchEmailsInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, query: str, folder: str = "inbox") -> str:
        """Search emails and return matching results."""
        call_id = generate_id("tcall")
        
        # Mark session as untrusted BEFORE executing (no_untrusted_tools defense)
        if self.config and self.config.session_id:
            from agent.agent_core import SessionTrustManager
            SessionTrustManager.set_untrusted(self.config.session_id)
        
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
    description: str = """Reply to an email using semantic search to find the most relevant match.

How it works:
1. Uses semantic search to find the email that best matches your search_query
2. Automatically extracts the sender address and constructs a 'Re: [subject]' subject line
3. Sends your reply with the original email quoted

Writing effective search queries:
- Be specific: Use descriptive phrases that identify the email (e.g., "Q4 strategy email from thomas" not just "thomas")
- Include context: Mention sender name, topic, or specific details (e.g., "meeting invitation tomorrow alice")
- Use natural language: The search understands meaning (e.g., "budget discussion Q4" finds emails about Q4 budgets)
- Avoid generic terms: Terms like "email" or "message" are too broad

This is a one-step operation - no need to manually construct the reply email. If multiple emails match, the MOST RELEVANT (semantically similar) one is used, not necessarily the newest."""
    args_schema: type[BaseModel] = ReplyToEmailInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, search_query: str, reply_body: str) -> str:
        """Find an email and send a reply."""
        call_id = generate_id("tcall")
        
        # CRITICAL: Check provable_policy defense FIRST, before any other operations
        # Tools that are both taint and exfil should ALWAYS be blocked when provable_policy is enabled
        if self.config and self.config.session_id:
            from agent.agent_core import ProvablePolicyManager
            from agent.tools_registry import is_both_taint_and_exfil
            
            defense_type = getattr(self.config, 'defense_type', None) if self.config else None
            is_provable_policy = defense_type == "provable_policy"
            
            if is_provable_policy and is_both_taint_and_exfil(self.name):
                # Mark session as untrusted (since we're attempting to use a taint tool)
                ProvablePolicyManager.set_untrusted(self.config.session_id)
                
                error_msg = (
                    "🛡️ Security Policy: This action has been blocked for safety reasons. "
                    "This tool both reads from untrusted sources and sends data externally, "
                    "which poses a security risk. Exfiltration tools that access untrusted data are not allowed."
                )
                
                # Log blocked tool call
                append_trace_event(
                    self.config.trace_file,
                    "tool_call",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "inputs": {"search_query": search_query, "reply_body": reply_body},
                        "call_id": call_id,
                        "blocked": True,
                        "reason": "provable_policy_taint_and_exfil_blocked"
                    }
                )
                
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg, "blocked": True},
                        "call_id": call_id
                    }
                )
                
                return error_msg
        
        # Mark session as untrusted BEFORE executing (no_untrusted_tools defense)
        if self.config and self.config.session_id:
            from agent.agent_core import SessionTrustManager
            SessionTrustManager.set_untrusted(self.config.session_id)
        
        # P1 & P4: Handle taint and exfiltration for provable_policy defense
        # (reply_to_email is both taint and exfil)
        if self.config and self.config.session_id:
            from agent.agent_core import ProvablePolicyManager
            from agent.tools_registry import is_exfiltration_tool
            
            # Only block if provable_policy defense is enabled
            defense_type = getattr(self.config, 'defense_type', None) if self.config else None
            is_provable_policy = defense_type == "provable_policy"
            
            if is_provable_policy:
                # P4: Check if exfiltration should be blocked (for tools that are only exfil, not taint)
                # If session is already U, block immediately (don't even read from inbox)
                if not ProvablePolicyManager.is_trusted(self.config.session_id):
                    # Mark session as untrusted (since we're attempting to use a taint tool)
                    ProvablePolicyManager.set_untrusted(self.config.session_id)
                    
                    error_msg = (
                        "🛡️ Security Policy: This action has been blocked for safety reasons. "
                        "This tool both reads from untrusted sources and sends data externally, "
                        "which poses a security risk. Exfiltration tools that access untrusted data are not allowed."
                    )
                    
                    # Log blocked tool call
                    append_trace_event(
                        self.config.trace_file,
                        "tool_call",
                        self.config.session_id,
                        {
                            "tool_name": self.name,
                            "inputs": {"search_query": search_query, "reply_body": reply_body},
                            "call_id": call_id,
                            "blocked": True,
                            "reason": "provable_policy_taint_and_exfil_blocked"
                        }
                    )
                    
                    append_trace_event(
                        self.config.trace_file,
                        "tool_result",
                        self.config.session_id,
                        {
                            "tool_name": self.name,
                            "outputs": {"error": error_msg, "blocked": True},
                            "call_id": call_id
                        }
                    )
                    
                    return error_msg
                
                # P4: Check if exfiltration should be blocked (for tools that are only exfil, not taint)
                # If session is already U, block immediately (don't even read from inbox)
                if not ProvablePolicyManager.is_trusted(self.config.session_id):
                    error_msg = (
                        "🛡️ Security Policy: This action has been blocked for safety reasons. "
                        "The current session has been marked as untrusted because untrusted data was accessed. "
                        "Exfiltration tools (email sending) are not allowed in untrusted sessions."
                    )
                    
                    # Log blocked tool call
                    append_trace_event(
                        self.config.trace_file,
                        "tool_call",
                        self.config.session_id,
                        {
                            "tool_name": self.name,
                            "inputs": {"search_query": search_query, "reply_body": reply_body},
                            "call_id": call_id,
                            "blocked": True,
                            "reason": "provable_policy_exfiltration_blocked"
                        }
                    )
                    
                    append_trace_event(
                        self.config.trace_file,
                        "tool_result",
                        self.config.session_id,
                        {
                            "tool_name": self.name,
                            "outputs": {"error": error_msg, "blocked": True},
                            "call_id": call_id
                        }
                    )
                    
                    return error_msg
            
            # P1: Mark session as U (taint tool)
            # Only mark as untrusted if provable_policy defense is enabled
            # Note: This should not be reached for tools that are both taint and exfil (they're blocked above)
            if defense_type == "provable_policy":
                ProvablePolicyManager.set_untrusted(self.config.session_id)
        
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
    description: str = """Forward an email to someone else using semantic search to find the most relevant match.
    
How it works:
1. Uses semantic search to find the email that best matches your search_query
2. Automatically includes the COMPLETE original email content (subject, body, sender, date) in the forwarded email
3. Adds your optional message (if provided) as a note above the forwarded email content
4. Constructs a 'Fwd: [subject]' subject line and sends it

Writing effective search queries:
- Be specific: Use key phrases or terms that uniquely identify the email (e.g., "Q4 strategy pricing" not just "email")
- Include context: Mention the sender name, topic, or specific details you remember
- Use natural language: The search understands meaning, not just keywords (e.g., "budget allocation Q4" finds emails about Q4 budgets)
- Avoid generic terms: Terms like "inbox" or "email" match too many results

Important: The original email body and all content is automatically included - you only need to provide an optional note/comment in the message parameter if you want to add your own text above the forwarded email. The search prioritizes RELEVANCE over recency, so the most semantically similar email will be selected, even if it's not the newest."""
    args_schema: type[BaseModel] = ForwardEmailInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, search_query: str, forward_to: str, message: str = "") -> str:
        """Find an email and forward it."""
        call_id = generate_id("tcall")
        
        # CRITICAL: Check provable_policy defense FIRST, before any other operations
        # Tools that are both taint and exfil should ALWAYS be blocked when provable_policy is enabled
        if self.config and self.config.session_id:
            from agent.agent_core import ProvablePolicyManager
            from agent.tools_registry import is_both_taint_and_exfil
            
            defense_type = getattr(self.config, 'defense_type', None) if self.config else None
            is_provable_policy = defense_type == "provable_policy"
            
            if is_provable_policy and is_both_taint_and_exfil(self.name):
                # Mark session as untrusted (since we're attempting to use a taint tool)
                ProvablePolicyManager.set_untrusted(self.config.session_id)
                
                error_msg = (
                    "🛡️ Security Policy: This action has been blocked for safety reasons. "
                    "This tool both reads from untrusted sources and sends data externally, "
                    "which poses a security risk. Exfiltration tools that access untrusted data are not allowed."
                )
                
                # Log blocked tool call
                append_trace_event(
                    self.config.trace_file,
                    "tool_call",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "inputs": {"search_query": search_query, "forward_to": forward_to, "message": message},
                        "call_id": call_id,
                        "blocked": True,
                        "reason": "provable_policy_taint_and_exfil_blocked"
                    }
                )
                
                append_trace_event(
                    self.config.trace_file,
                    "tool_result",
                    self.config.session_id,
                    {
                        "tool_name": self.name,
                        "outputs": {"error": error_msg, "blocked": True},
                        "call_id": call_id
                    }
                )
                
                return error_msg
        
        # Mark session as untrusted BEFORE executing (no_untrusted_tools defense)
        if self.config and self.config.session_id:
            from agent.agent_core import SessionTrustManager
            SessionTrustManager.set_untrusted(self.config.session_id)
        
        # P1 & P4: Handle taint and exfiltration for provable_policy defense
        # (forward_to_email is both taint and exfil)
        if self.config and self.config.session_id:
            from agent.agent_core import ProvablePolicyManager
            from agent.tools_registry import is_exfiltration_tool
            
            # Only block if provable_policy defense is enabled
            defense_type = getattr(self.config, 'defense_type', None) if self.config else None
            is_provable_policy = defense_type == "provable_policy"
            
            if is_provable_policy:
                # P4: Check if exfiltration should be blocked (for tools that are only exfil, not taint)
                # If session is already U, block immediately (don't even read from inbox)
                if not ProvablePolicyManager.is_trusted(self.config.session_id):
                    error_msg = (
                        "🛡️ Security Policy: This action has been blocked for safety reasons. "
                        "The current session has been marked as untrusted because untrusted data was accessed. "
                        "Exfiltration tools (email sending) are not allowed in untrusted sessions."
                    )
                    
                    # Log blocked tool call
                    append_trace_event(
                        self.config.trace_file,
                        "tool_call",
                        self.config.session_id,
                        {
                            "tool_name": self.name,
                            "inputs": {"search_query": search_query, "forward_to": forward_to, "message": message},
                            "call_id": call_id,
                            "blocked": True,
                            "reason": "provable_policy_exfiltration_blocked"
                        }
                    )
                    
                    append_trace_event(
                        self.config.trace_file,
                        "tool_result",
                        self.config.session_id,
                        {
                            "tool_name": self.name,
                            "outputs": {"error": error_msg, "blocked": True},
                            "call_id": call_id
                        }
                    )
                    
                    return error_msg
            
            # P1: Mark session as U (taint tool)
            # Only mark as untrusted if provable_policy defense is enabled
            # Note: This should not be reached for tools that are both taint and exfil (they're blocked above)
            if defense_type == "provable_policy":
                ProvablePolicyManager.set_untrusted(self.config.session_id)
        
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
                # Get body - prefer body_plain, fall back to body, ensure it's a string
                original_body = original_email.get('body_plain', original_email.get('body', ''))
                if original_body is None:
                    original_body = ''
                # Ensure we have a string (not bytes or other type)
                original_body = str(original_body)
                original_date = original_email.get('received_ts', original_email.get('created_ts', ''))
                
                # Validate that we actually have email body content
                if not original_body or len(original_body.strip()) == 0:
                    result = f"Warning: Email found ('{original_subject}') but it has no body content. Cannot forward empty email."
                    if self.config and self.config.session_id:
                        append_trace_event(
                            self.config.trace_file,
                            "tool_result",
                            self.config.session_id,
                            {
                                "tool_name": self.name,
                                "outputs": {"error": result},
                                "call_id": call_id
                            }
                        )
                    return result
                
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
                    "body_plain": forward_body,  # Also store as body_plain for consistency
                    "created_ts": get_timestamp(),
                    "sent_ts": get_timestamp(),
                    "status": "sent",
                    "forwarded_from": original_email.get('id', 'unknown')
                }
                
                # Use ensure_ascii=False to preserve unicode characters, and don't indent to avoid
                # potential issues with very long strings containing newlines
                with open(email_file, "w", encoding="utf-8") as f:
                    json.dump(forward_email, f, indent=2, ensure_ascii=False)
                
                # Debug: Print the forwarded email contents to terminal
                print("\n" + "="*80)
                print("📧 DEBUG: Forwarded Email Contents (written to outbox)")
                print("="*80)
                print(f"From: {forward_email.get('from', 'N/A')}")
                print(f"To: {forward_email.get('to', 'N/A')}")
                print(f"Subject: {forward_email.get('subject', 'N/A')}")
                print(f"\nBody Length: {len(forward_body)} characters")
                print(f"\n--- Full Body Content ---")
                print(forward_body)
                print("="*80 + "\n")
                
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
    name: str = "send_email"
    description: str = """Compose and send a new email. Use this for creating original emails (not replies or forwards). The email is sent immediately."""
    args_schema: type[BaseModel] = ComposeEmailInput
    config: Optional[EmailToolsConfig] = None
    
    def _run(self, to: str, subject: str, body: str) -> str:
        """Create and immediately send an email."""
        call_id = generate_id("tcall")
        
        # P4: Check if exfiltration should be blocked (provable_policy defense)
        # Get session_id from config if available
        session_id = None
        if self.config and hasattr(self.config, 'session_id'):
            session_id = self.config.session_id
        
        # Debug output - always print to help diagnose issues
        print(f"🔍 [DEBUG] send_email: session_id={session_id}, config={self.config is not None}, config_type={type(self.config).__name__ if self.config else 'None'}")
        if self.config:
            print(f"🔍 [DEBUG] send_email: config.session_id={getattr(self.config, 'session_id', 'NOT_SET')}")
        
        if session_id:
            from agent.agent_core import ProvablePolicyManager
            from agent.tools_registry import is_exfiltration_tool
            
            # P4: Block exfiltration if session is U, but ONLY if provable_policy defense is enabled
            defense_type = getattr(self.config, 'defense_type', None) if self.config else None
            is_provable_policy = defense_type == "provable_policy"
            
            is_trusted = ProvablePolicyManager.is_trusted(session_id)
            session_label = ProvablePolicyManager.get_session_label(session_id)
            will_block = not is_trusted and is_provable_policy
            print(f"🔍 [DEBUG] send_email: is_trusted={is_trusted}, session_label={session_label}, defense_type={defense_type}, is_provable_policy={is_provable_policy}, will_block={will_block}")
            
            if not is_trusted and is_provable_policy:
                error_msg = (
                    "🛡️ Security Policy: This action has been blocked for safety reasons. "
                    "The current session has been marked as untrusted because untrusted data was accessed. "
                    "Exfiltration tools (email sending) are not allowed in untrusted sessions."
                )
                
                # Log blocked tool call
                if self.config and self.config.trace_file:
                    append_trace_event(
                        self.config.trace_file,
                        "tool_call",
                        session_id,
                        {
                            "tool_name": self.name,
                            "inputs": {"to": to, "subject": subject, "body": body},
                            "call_id": call_id,
                            "blocked": True,
                            "reason": "provable_policy_exfiltration_blocked"
                        }
                    )
                    
                    append_trace_event(
                        self.config.trace_file,
                        "tool_result",
                        session_id,
                        {
                            "tool_name": self.name,
                            "outputs": {"error": error_msg, "blocked": True},
                            "call_id": call_id
                        }
                    )
                
                return error_msg
        
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
