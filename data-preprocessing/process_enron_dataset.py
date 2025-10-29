#!/usr/bin/env python3
"""
Script to process Enron email dataset and extract Vince Kaminski's inbox/outbox emails.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_email_headers(email_text: str) -> Dict[str, Optional[str]]:
    """
    Extract email headers (From, To, Subject, Date) from raw email text.
    """
    headers = {
        'from': None,
        'to': None,
        'subject': None,
        'date': None
    }
    
    lines = email_text.split('\n')
    
    # Look for header patterns in first 20 lines
    for i, line in enumerate(lines[:20]):
        # From field
        if re.match(r'^From:', line, re.IGNORECASE):
            headers['from'] = line.split(':', 1)[1].strip()
        # To field
        elif re.match(r'^To:', line, re.IGNORECASE):
            headers['to'] = line.split(':', 1)[1].strip()
        # Subject field
        elif re.match(r'^Subject:', line, re.IGNORECASE):
            headers['subject'] = line.split(':', 1)[1].strip()
        # Date field
        elif re.match(r'^Date:', line, re.IGNORECASE):
            headers['date'] = line.split(':', 1)[1].strip()
    
    return headers

def extract_vince_email(text: str) -> Optional[str]:
    """
    Try to extract Vince's email address from the text.
    """
    # Common Vince Kaminski email patterns
    vince_patterns = [
        r'vince\.kaminski@enron\.com',
        r'vkaminski@enron\.com',
        r'kaminski@enron\.com'
    ]
    
    for pattern in vince_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).lower()
    
    return None

def is_vince_sender(email_text: str, headers: Dict) -> bool:
    """
    Determine if Vince is the sender of this email.
    """
    if not headers['from']:
        return False
    
    from_field = headers['from'].lower()
    
    # Check if Vince is in the From field
    vince_indicators = ['vince', 'kaminski', 'vkaminski', 'vince.kaminski']
    
    return any(indicator in from_field for indicator in vince_indicators)

def is_vince_recipient(email_text: str, headers: Dict) -> bool:
    """
    Determine if Vince is a recipient of this email.
    """
    if not headers['to']:
        return False
    
    to_field = headers['to'].lower()
    
    # Check if Vince is in the To field
    vince_indicators = ['vince', 'kaminski', 'vkaminski', 'vince.kaminski']
    
    return any(indicator in to_field for indicator in vince_indicators)

def extract_body(email_text: str) -> str:
    """
    Extract the body from the email (skip headers).
    """
    lines = email_text.split('\n')
    
    # Find where headers end (usually blank line after headers)
    body_start = 0
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == '' and i < 30:
            # Check if previous lines had headers
            prev_lines = '\n'.join(lines[:i])
            if re.search(r'(From:|To:|Subject:)', prev_lines, re.IGNORECASE):
                body_start = i + 1
                break
    
    if body_start > 0:
        return '\n'.join(lines[body_start:]).strip()
    
    return email_text.strip()

def use_llm_to_complete_email(raw_text: str, headers: Dict, is_inbox: bool) -> Dict:
    """
    Use LLM to fill in missing fields and create a complete email JSON.
    """
    vince_email = "vince.kaminski@enron.com"
    
    prompt = f"""You are helping to parse an email from the Enron dataset. 

Raw email text:
{raw_text[:1500]}

Extracted headers:
- From: {headers.get('from', 'NOT FOUND')}
- To: {headers.get('to', 'NOT FOUND')}
- Subject: {headers.get('subject', 'NOT FOUND')}
- Date: {headers.get('date', 'NOT FOUND')}

This email is in Vince Kaminski's {"INBOX" if is_inbox else "OUTBOX"}.

Please provide the missing information in JSON format with these fields:
- from: email address of sender (if inbox, this is someone else; if outbox, this is {vince_email})
- to: email address of recipient (if inbox, this is {vince_email}; if outbox, this is someone else)
- subject: email subject line
- received_ts: timestamp in ISO format (YYYY-MM-DDTHH:MM:SSZ), infer from date or use a reasonable 2001 date

Only output valid JSON, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from emails. Only respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return json.loads(result)
    
    except Exception as e:
        print(f"LLM error: {e}")
        # Return defaults
        return {
            'from': vince_email if not is_inbox else "unknown@enron.com",
            'to': vince_email if is_inbox else "unknown@enron.com",
            'subject': headers.get('subject', 'No Subject'),
            'received_ts': '2001-01-01T00:00:00Z'
        }

def create_email_json(email_id: str, raw_text: str, headers: Dict, is_inbox: bool) -> Dict:
    """
    Create the final email JSON structure.
    """
    vince_email = "vince.kaminski@enron.com"
    
    # Get body
    body = extract_body(raw_text)
    
    # Use LLM to fill missing fields
    llm_data = use_llm_to_complete_email(raw_text, headers, is_inbox)
    
    # Construct final JSON
    email_json = {
        'id': email_id,
        'from': headers.get('from') or llm_data.get('from', vince_email if not is_inbox else "unknown@enron.com"),
        'to': headers.get('to') or llm_data.get('to', vince_email if is_inbox else "unknown@enron.com"),
        'subject': headers.get('subject') or llm_data.get('subject', 'No Subject'),
        'body_plain': body,
        'received_ts': llm_data.get('received_ts', '2001-01-01T00:00:00Z'),
        'metadata': {
            'folder': 'inbox' if is_inbox else 'sent',
            'read': False,
            'source': 'enron_dataset'
        }
    }
    
    return email_json

def process_enron_dataset():
    """
    Main function to process the Enron dataset.
    """
    print("Loading Enron email dataset...")
    dataset = load_dataset("LLM-PBE/enron-email", split="train")
    
    print(f"Total emails in dataset: {len(dataset)}")
    
    inbox_emails = []
    outbox_emails = []
    
    print("\nProcessing emails...")
    for idx, item in enumerate(tqdm(dataset)):
        email_text = item['text']
        
        # Filter: must mention "Vince"
        if 'vince' not in email_text.lower():
            continue
        
        # Parse headers
        headers = parse_email_headers(email_text)
        
        # Determine if inbox or outbox
        is_sender = is_vince_sender(email_text, headers)
        is_recipient = is_vince_recipient(email_text, headers)
        
        # Only process if we're confident about the direction
        if is_sender and not is_recipient:
            # Vince sent this - OUTBOX
            if len(outbox_emails) < 150:  # Get extra to allow for variety selection
                outbox_emails.append({
                    'idx': idx,
                    'text': email_text,
                    'headers': headers
                })
        elif is_recipient and not is_sender:
            # Vince received this - INBOX
            if len(inbox_emails) < 150:  # Get extra to allow for variety selection
                inbox_emails.append({
                    'idx': idx,
                    'text': email_text,
                    'headers': headers
                })
        
        # Stop if we have enough
        if len(inbox_emails) >= 150 and len(outbox_emails) >= 150:
            break
    
    print(f"\nFound {len(inbox_emails)} potential inbox emails")
    print(f"Found {len(outbox_emails)} potential outbox emails")
    
    # Select 100 diverse emails from each category
    # Simple diversity: take emails at regular intervals
    def select_diverse(emails, count=100):
        if len(emails) <= count:
            return emails
        step = len(emails) // count
        return [emails[i * step] for i in range(count)]
    
    inbox_selected = select_diverse(inbox_emails, 100)
    outbox_selected = select_diverse(outbox_emails, 100)
    
    print(f"\nSelected {len(inbox_selected)} inbox emails")
    print(f"Selected {len(outbox_selected)} outbox emails")
    
    # Create output directories
    inbox_dir = "data/vince_inbox"
    outbox_dir = "data/vince_outbox"
    os.makedirs(inbox_dir, exist_ok=True)
    os.makedirs(outbox_dir, exist_ok=True)
    
    # Process and save inbox emails
    print("\nProcessing inbox emails...")
    for i, email_data in enumerate(tqdm(inbox_selected)):
        email_id = f"vince_inbox_{i+1:03d}"
        email_json = create_email_json(
            email_id,
            email_data['text'],
            email_data['headers'],
            is_inbox=True
        )
        
        with open(f"{inbox_dir}/{email_id}.json", 'w') as f:
            json.dump(email_json, f, indent=2)
    
    # Process and save outbox emails
    print("\nProcessing outbox emails...")
    for i, email_data in enumerate(tqdm(outbox_selected)):
        email_id = f"vince_outbox_{i+1:03d}"
        email_json = create_email_json(
            email_id,
            email_data['text'],
            email_data['headers'],
            is_inbox=False
        )
        
        with open(f"{outbox_dir}/{email_id}.json", 'w') as f:
            json.dump(email_json, f, indent=2)
    
    print(f"\n✅ Done! Created:")
    print(f"  - {len(inbox_selected)} files in {inbox_dir}/")
    print(f"  - {len(outbox_selected)} files in {outbox_dir}/")

if __name__ == "__main__":
    process_enron_dataset()

