---
name: email_compose
version: 1.0.0
agent: email
risk_level: high
description: "Compose and send emails via SMTP. Support for plain text, HTML, attachments, templates, and mail merge. Includes draft review before sending."
---

# Email Compose Skill

## Purpose

Compose, draft, and send emails on behalf of the Gorgon system. This skill enables automated email workflows including notifications, reports, outreach, and responses. Due to the irreversible nature of sending emails, this skill requires elevated consensus for send operations.

## Safety Rules

### CRITICAL - EMAIL IS IRREVERSIBLE
1. **ALWAYS create draft first, send only after approval**
2. **NEVER send without Triumvirate UNANIMOUS consensus**
3. **ALWAYS confirm recipient addresses before sending**
4. **NEVER send to large recipient lists without explicit approval**
5. **NEVER include sensitive data in email body**

### ANTI-SPAM REQUIREMENTS
1. **Respect unsubscribe requests**
2. **Include clear sender identification**
3. **Don't spoof headers or sender addresses**
4. **Rate limit outbound messages**
5. **Keep mailing lists opt-in only**

### CONSENSUS REQUIREMENTS
| Operation | Risk Level | Consensus Required |
|-----------|------------|-------------------|
| create_draft | low | any |
| review_draft | low | any |
| send_email | critical | unanimous + user confirmation |
| send_bulk | critical | unanimous + user confirmation |
| add_attachment | medium | majority |

## Configuration

Email credentials should be stored securely, never in code:

```yaml
# ~/.gorgon/config/email.yaml (chmod 600)
smtp:
  host: smtp.gmail.com
  port: 587
  use_tls: true
  username: ${GORGON_EMAIL_USER}
  password: ${GORGON_EMAIL_PASS}  # Use app password for Gmail

defaults:
  from_name: "Gorgon System"
  from_address: "gorgon@yourdomain.com"
  reply_to: "you@yourdomain.com"
  signature: |
    --
    Sent by Gorgon Automation System
    This is an automated message.
```

## Capabilities

### create_draft
Create an email draft for review before sending.

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import json
from pathlib import Path

@dataclass
class EmailDraft:
    id: str
    to: list[str]
    cc: list[str]
    bcc: list[str]
    subject: str
    body_text: str
    body_html: Optional[str]
    attachments: list[str]
    created_at: str
    status: str  # "draft", "approved", "sent", "cancelled"

DRAFTS_DIR = Path.home() / ".gorgon" / "email_drafts"

def create_draft(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = None,
    bcc: list[str] = None,
    html_body: str = None,
    attachments: list[str] = None
) -> dict:
    """Create an email draft for review."""
    
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    
    draft_id = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    draft = EmailDraft(
        id=draft_id,
        to=to,
        cc=cc or [],
        bcc=bcc or [],
        subject=subject,
        body_text=body,
        body_html=html_body,
        attachments=attachments or [],
        created_at=datetime.now().isoformat(),
        status="draft"
    )
    
    # Save draft
    draft_file = DRAFTS_DIR / f"{draft_id}.json"
    draft_file.write_text(json.dumps(draft.__dict__, indent=2))
    
    return {
        "success": True,
        "draft_id": draft_id,
        "draft_path": str(draft_file),
        "preview": {
            "to": to,
            "subject": subject,
            "body_preview": body[:200] + "..." if len(body) > 200 else body
        }
    }
```

---

### review_draft
Display draft for review and approval.

```python
def review_draft(draft_id: str) -> dict:
    """Load and display a draft for review."""
    
    draft_file = DRAFTS_DIR / f"{draft_id}.json"
    
    if not draft_file.exists():
        return {"success": False, "error": f"Draft not found: {draft_id}"}
    
    draft_data = json.loads(draft_file.read_text())
    
    # Format for display
    review = f"""
═══════════════════════════════════════════════════════════════
                     EMAIL DRAFT REVIEW
═══════════════════════════════════════════════════════════════
Draft ID: {draft_data['id']}
Created:  {draft_data['created_at']}
Status:   {draft_data['status']}
═══════════════════════════════════════════════════════════════

TO:      {', '.join(draft_data['to'])}
CC:      {', '.join(draft_data['cc']) if draft_data['cc'] else '(none)'}
BCC:     {', '.join(draft_data['bcc']) if draft_data['bcc'] else '(none)'}
SUBJECT: {draft_data['subject']}

───────────────────────────────────────────────────────────────
BODY:
───────────────────────────────────────────────────────────────
{draft_data['body_text']}
───────────────────────────────────────────────────────────────

ATTACHMENTS: {', '.join(draft_data['attachments']) if draft_data['attachments'] else '(none)'}

═══════════════════════════════════════════════════════════════
    """
    
    return {
        "success": True,
        "draft_id": draft_id,
        "status": draft_data['status'],
        "review_text": review,
        "draft_data": draft_data
    }
```

---

### approve_draft
Mark a draft as approved for sending.

```python
def approve_draft(draft_id: str) -> dict:
    """Mark draft as approved (requires Triumvirate unanimous consensus)."""
    
    draft_file = DRAFTS_DIR / f"{draft_id}.json"
    
    if not draft_file.exists():
        return {"success": False, "error": f"Draft not found: {draft_id}"}
    
    draft_data = json.loads(draft_file.read_text())
    
    if draft_data['status'] != 'draft':
        return {"success": False, "error": f"Draft status is {draft_data['status']}, cannot approve"}
    
    draft_data['status'] = 'approved'
    draft_data['approved_at'] = datetime.now().isoformat()
    
    draft_file.write_text(json.dumps(draft_data, indent=2))
    
    return {
        "success": True,
        "draft_id": draft_id,
        "status": "approved",
        "message": "Draft approved. Ready to send with send_email."
    }
```

---

### send_email
Send an approved draft. REQUIRES UNANIMOUS CONSENSUS.

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import yaml
from pathlib import Path

def load_email_config() -> dict:
    """Load email configuration."""
    config_path = Path.home() / ".gorgon" / "config" / "email.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError("Email config not found. Create ~/.gorgon/config/email.yaml")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def send_email(draft_id: str, confirm: bool = False) -> dict:
    """
    Send an approved email draft.
    
    REQUIRES:
    - Triumvirate UNANIMOUS consensus
    - User confirmation (confirm=True)
    """
    
    # Load draft
    draft_file = DRAFTS_DIR / f"{draft_id}.json"
    
    if not draft_file.exists():
        return {"success": False, "error": f"Draft not found: {draft_id}"}
    
    draft_data = json.loads(draft_file.read_text())
    
    # Verify approved
    if draft_data['status'] != 'approved':
        return {
            "success": False, 
            "error": f"Draft must be approved first. Current status: {draft_data['status']}"
        }
    
    # Require explicit confirmation
    if not confirm:
        return {
            "success": False,
            "error": "Confirmation required. Call with confirm=True after Triumvirate approval.",
            "draft_id": draft_id,
            "recipients": draft_data['to'] + draft_data['cc'] + draft_data['bcc']
        }
    
    # Load config
    config = load_email_config()
    
    # Build message
    msg = MIMEMultipart('alternative')
    msg['From'] = f"{config['defaults']['from_name']} <{config['defaults']['from_address']}>"
    msg['To'] = ', '.join(draft_data['to'])
    msg['Subject'] = draft_data['subject']
    
    if draft_data['cc']:
        msg['Cc'] = ', '.join(draft_data['cc'])
    
    if config['defaults'].get('reply_to'):
        msg['Reply-To'] = config['defaults']['reply_to']
    
    # Add body
    body_with_sig = draft_data['body_text'] + config['defaults'].get('signature', '')
    msg.attach(MIMEText(body_with_sig, 'plain'))
    
    if draft_data['body_html']:
        msg.attach(MIMEText(draft_data['body_html'], 'html'))
    
    # Add attachments
    for attachment_path in draft_data['attachments']:
        path = Path(attachment_path)
        if path.exists():
            with open(path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{path.name}"'
                )
                msg.attach(part)
    
    # Send
    try:
        all_recipients = draft_data['to'] + draft_data['cc'] + draft_data['bcc']
        
        with smtplib.SMTP(config['smtp']['host'], config['smtp']['port']) as server:
            if config['smtp'].get('use_tls', True):
                server.starttls()
            
            server.login(config['smtp']['username'], config['smtp']['password'])
            server.sendmail(
                config['defaults']['from_address'],
                all_recipients,
                msg.as_string()
            )
        
        # Update draft status
        draft_data['status'] = 'sent'
        draft_data['sent_at'] = datetime.now().isoformat()
        draft_file.write_text(json.dumps(draft_data, indent=2))
        
        return {
            "success": True,
            "draft_id": draft_id,
            "status": "sent",
            "recipients": all_recipients,
            "sent_at": draft_data['sent_at']
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "draft_id": draft_id
        }
```

---

### add_attachment
Add an attachment to a draft.

```python
def add_attachment(draft_id: str, file_path: str) -> dict:
    """Add an attachment to an existing draft."""
    
    draft_file = DRAFTS_DIR / f"{draft_id}.json"
    
    if not draft_file.exists():
        return {"success": False, "error": f"Draft not found: {draft_id}"}
    
    # Verify file exists
    attachment = Path(file_path)
    if not attachment.exists():
        return {"success": False, "error": f"Attachment not found: {file_path}"}
    
    # Check file size (limit 25MB)
    if attachment.stat().st_size > 25 * 1024 * 1024:
        return {"success": False, "error": "Attachment too large (max 25MB)"}
    
    draft_data = json.loads(draft_file.read_text())
    
    if draft_data['status'] != 'draft':
        return {"success": False, "error": "Cannot modify approved/sent draft"}
    
    draft_data['attachments'].append(str(attachment.absolute()))
    draft_file.write_text(json.dumps(draft_data, indent=2))
    
    return {
        "success": True,
        "draft_id": draft_id,
        "attachment_added": str(attachment),
        "total_attachments": len(draft_data['attachments'])
    }
```

---

### use_template
Create a draft from a template with variable substitution.

```python
TEMPLATES_DIR = Path.home() / ".gorgon" / "email_templates"

def use_template(
    template_name: str,
    to: list[str],
    variables: dict,
    cc: list[str] = None
) -> dict:
    """Create a draft from a template."""
    
    template_file = TEMPLATES_DIR / f"{template_name}.yaml"
    
    if not template_file.exists():
        return {"success": False, "error": f"Template not found: {template_name}"}
    
    with open(template_file) as f:
        template = yaml.safe_load(f)
    
    # Substitute variables
    subject = template['subject']
    body = template['body']
    
    for key, value in variables.items():
        subject = subject.replace(f"{{{{{key}}}}}", str(value))
        body = body.replace(f"{{{{{key}}}}}", str(value))
    
    # Check for unsubstituted variables
    import re
    remaining = re.findall(r'\{\{(\w+)\}\}', subject + body)
    if remaining:
        return {
            "success": False,
            "error": f"Missing template variables: {remaining}"
        }
    
    # Create draft
    return create_draft(
        to=to,
        subject=subject,
        body=body,
        cc=cc,
        html_body=template.get('body_html')
    )
```

**Example template file:**
```yaml
# ~/.gorgon/email_templates/job_followup.yaml
name: job_followup
description: Follow-up email after job application

subject: "Following up on {{position}} application - {{applicant_name}}"

body: |
  Dear {{hiring_manager}},
  
  I hope this email finds you well. I wanted to follow up on my application 
  for the {{position}} position at {{company}} that I submitted on {{apply_date}}.
  
  I remain very interested in this opportunity and would welcome the chance 
  to discuss how my experience in {{relevant_skill}} could benefit your team.
  
  Please let me know if you need any additional information from me.
  
  Best regards,
  {{applicant_name}}
```

---

## Examples

### Example 1: Send a status report email
**Intent:** "Send a daily status report to the team"

**Execution:**
```python
# Step 1: Create draft (any agent can do this)
draft = create_draft(
    to=["team@company.com"],
    cc=["manager@company.com"],
    subject="Daily Status Report - 2026-01-27",
    body="""
Team,

Here's today's status update:

COMPLETED:
- Deployed v2.3.1 to production
- Fixed authentication bug (#1234)
- Updated documentation

IN PROGRESS:
- API rate limiting implementation (80% complete)
- Performance optimization review

BLOCKED:
- Waiting on AWS credentials for staging environment

Let me know if you have questions.

Best,
Gorgon System
"""
)

print(f"Draft created: {draft['draft_id']}")

# Step 2: Review draft
review = review_draft(draft['draft_id'])
print(review['review_text'])

# Step 3: Approve (requires Triumvirate unanimous consensus)
# This would be called by the Triumvirate after review
approval = approve_draft(draft['draft_id'])
print(f"Approval status: {approval['status']}")

# Step 4: Send (requires user confirmation)
# Only after Triumvirate approval
result = send_email(draft['draft_id'], confirm=True)
print(f"Sent: {result['success']}")
```

---

### Example 2: Use a template for job follow-up
**Intent:** "Send a follow-up email for my Palantir application"

**Execution:**
```python
# Create draft from template
draft = use_template(
    template_name="job_followup",
    to=["recruiter@palantir.com"],
    variables={
        "position": "AI Solutions Engineer",
        "applicant_name": "ARETE",
        "hiring_manager": "Hiring Team",
        "company": "Palantir",
        "apply_date": "January 15, 2026",
        "relevant_skill": "AI enablement and enterprise operations"
    }
)

# Review
review = review_draft(draft['draft_id'])
print(review['review_text'])

# Continue with approval/send process...
```

---

## Security Considerations

### Credential Storage
```bash
# Never store credentials in plain text
# Use environment variables or encrypted secrets

# Option 1: Environment variables
export GORGON_EMAIL_USER="your-email@gmail.com"
export GORGON_EMAIL_PASS="your-app-password"

# Option 2: Use a secrets manager
# Reference in config: ${GORGON_EMAIL_USER}
```

### Gmail App Passwords
For Gmail, you must use an App Password:
1. Enable 2FA on your Google account
2. Go to Google Account → Security → 2-Step Verification → App passwords
3. Generate a password for "Gorgon"
4. Use this instead of your regular password

## Error Handling

| Error | Response |
|-------|----------|
| SMTP auth failed | Check credentials, verify app password |
| Connection refused | Check host/port, firewall settings |
| Recipient rejected | Verify email address format |
| Attachment too large | Compress or use file sharing link |
| Rate limited | Wait and retry, reduce send frequency |

## Output Format

```json
{
  "success": true,
  "operation": "send_email",
  "draft_id": "draft_20260127_103000",
  "status": "sent",
  "recipients": ["team@company.com", "manager@company.com"],
  "sent_at": "2026-01-27T10:35:00Z",
  "message_id": "<unique-id@smtp.gmail.com>"
}
```
