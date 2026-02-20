---
name: github_operations
version: 1.0.0
agent: system
risk_level: variable
description: "Interact with GitHub repositories. Clone, pull, push, create issues, manage PRs, and automate repository workflows. Supports both CLI and API operations."
---

# GitHub Operations Skill

## Purpose

Interact with GitHub repositories for code management, issue tracking, and automation. This skill enables Gorgon to manage code repositories, create issues, submit pull requests, and automate development workflows.

## Safety Rules

### CRITICAL - CODE CHANGES ARE VISIBLE
1. **Never push to main/master without explicit approval**
2. **Always create feature branches for changes**
3. **Never force push to shared branches**
4. **Always review diffs before committing**
5. **Never commit secrets or credentials**

### CREDENTIAL SECURITY
1. **Use Personal Access Tokens (PAT), never passwords**
2. **Store tokens in environment variables or secrets manager**
3. **Use minimum required scopes for tokens**
4. **Rotate tokens regularly**

### CONSENSUS REQUIREMENTS
| Operation | Risk Level | Consensus Required |
|-----------|------------|-------------------|
| clone/pull | low | any |
| create_branch | low | any |
| commit | medium | majority |
| push | high | unanimous |
| create_pr | medium | majority |
| merge_pr | critical | unanimous |
| create_issue | low | any |
| delete_branch | medium | majority |

## Configuration

```yaml
# ~/.gorgon/config/github.yaml (chmod 600)
github:
  token: ${GITHUB_TOKEN}
  username: your-username
  default_branch: main
  
  # Workspace for cloned repos
  workspace: ~/gorgon-repos
  
  # Commit settings
  commit_author_name: "Gorgon Bot"
  commit_author_email: "gorgon@yourdomain.com"
```

## Capabilities

### clone_repo
Clone a GitHub repository.

```bash
# Using git CLI
git clone https://github.com/owner/repo.git ~/gorgon-repos/repo

# With token for private repos
git clone https://${GITHUB_TOKEN}@github.com/owner/repo.git ~/gorgon-repos/repo

# Shallow clone (faster for large repos)
git clone --depth 1 https://github.com/owner/repo.git ~/gorgon-repos/repo
```

```python
import subprocess
from pathlib import Path

def clone_repo(
    owner: str,
    repo: str,
    workspace: Path = Path.home() / "gorgon-repos",
    shallow: bool = False,
    branch: str = None
) -> dict:
    """Clone a GitHub repository."""
    
    workspace.mkdir(parents=True, exist_ok=True)
    target_path = workspace / repo
    
    if target_path.exists():
        return {
            "success": False,
            "error": f"Directory already exists: {target_path}",
            "suggestion": "Use pull_repo to update, or remove directory first"
        }
    
    # Build clone command
    url = f"https://github.com/{owner}/{repo}.git"
    cmd = ["git", "clone"]
    
    if shallow:
        cmd.extend(["--depth", "1"])
    
    if branch:
        cmd.extend(["--branch", branch])
    
    cmd.extend([url, str(target_path)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return {
            "success": True,
            "path": str(target_path),
            "url": url,
            "branch": branch or "default"
        }
    
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr
        }
```

---

### pull_repo
Pull latest changes from remote.

```python
def pull_repo(repo_path: Path) -> dict:
    """Pull latest changes from remote."""
    
    if not (repo_path / ".git").exists():
        return {"success": False, "error": "Not a git repository"}
    
    try:
        # Fetch first
        subprocess.run(
            ["git", "fetch", "--all"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        current_branch = result.stdout.strip()
        
        # Pull
        result = subprocess.run(
            ["git", "pull", "origin", current_branch],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        return {
            "success": True,
            "branch": current_branch,
            "output": result.stdout
        }
    
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr
        }
```

---

### create_branch
Create a new branch.

```python
def create_branch(repo_path: Path, branch_name: str, from_branch: str = None) -> dict:
    """Create and checkout a new branch."""
    
    try:
        # Optionally checkout source branch first
        if from_branch:
            subprocess.run(
                ["git", "checkout", from_branch],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["git", "pull", "origin", from_branch],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
        
        # Create and checkout new branch
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        return {
            "success": True,
            "branch": branch_name,
            "from_branch": from_branch or "current HEAD"
        }
    
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
        }
```

---

### commit_changes
Stage and commit changes. REQUIRES MAJORITY CONSENSUS.

```python
def commit_changes(
    repo_path: Path,
    message: str,
    files: list[str] = None,
    all_changes: bool = False
) -> dict:
    """Stage and commit changes."""
    
    config = load_github_config()
    
    try:
        # Configure commit author
        subprocess.run(
            ["git", "config", "user.name", config['commit_author_name']],
            cwd=repo_path,
            check=True
        )
        subprocess.run(
            ["git", "config", "user.email", config['commit_author_email']],
            cwd=repo_path,
            check=True
        )
        
        # Stage files
        if all_changes:
            subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
        elif files:
            for f in files:
                subprocess.run(["git", "add", f], cwd=repo_path, check=True)
        else:
            return {"success": False, "error": "Specify files or use all_changes=True"}
        
        # Show what will be committed
        diff_result = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        # Commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "success": True,
            "commit_hash": result.stdout.strip()[:8],
            "message": message,
            "changes": diff_result.stdout
        }
    
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr.decode() if isinstance(e.stderr, bytes) else str(e)
        }
```

---

### push_branch
Push branch to remote. REQUIRES UNANIMOUS CONSENSUS.

```python
def push_branch(repo_path: Path, branch: str = None, set_upstream: bool = True) -> dict:
    """Push branch to remote. REQUIRES UNANIMOUS CONSENSUS."""
    
    try:
        # Get current branch if not specified
        if not branch:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            branch = result.stdout.strip()
        
        # Prevent pushing to protected branches
        protected = ["main", "master", "production", "release"]
        if branch in protected:
            return {
                "success": False,
                "error": f"Cannot push directly to protected branch: {branch}",
                "suggestion": "Create a feature branch and submit a PR"
            }
        
        # Push
        cmd = ["git", "push"]
        if set_upstream:
            cmd.extend(["--set-upstream", "origin", branch])
        else:
            cmd.extend(["origin", branch])
        
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        return {
            "success": True,
            "branch": branch,
            "output": result.stdout or result.stderr
        }
    
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr
        }
```

---

### create_issue
Create a GitHub issue via API.

```python
import requests

def create_issue(
    owner: str,
    repo: str,
    title: str,
    body: str,
    labels: list[str] = None,
    assignees: list[str] = None
) -> dict:
    """Create a GitHub issue."""
    
    config = load_github_config()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {
        "Authorization": f"token {config['token']}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "title": title,
        "body": body
    }
    
    if labels:
        data["labels"] = labels
    if assignees:
        data["assignees"] = assignees
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 201:
        issue = response.json()
        return {
            "success": True,
            "issue_number": issue["number"],
            "url": issue["html_url"],
            "title": title
        }
    else:
        return {
            "success": False,
            "error": response.json().get("message", "Unknown error"),
            "status_code": response.status_code
        }
```

---

### create_pull_request
Create a pull request. REQUIRES MAJORITY CONSENSUS.

```python
def create_pull_request(
    owner: str,
    repo: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str = "main",
    draft: bool = False
) -> dict:
    """Create a pull request."""
    
    config = load_github_config()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = {
        "Authorization": f"token {config['token']}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch,
        "draft": draft
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 201:
        pr = response.json()
        return {
            "success": True,
            "pr_number": pr["number"],
            "url": pr["html_url"],
            "title": title,
            "head": head_branch,
            "base": base_branch
        }
    else:
        return {
            "success": False,
            "error": response.json().get("message", "Unknown error"),
            "status_code": response.status_code
        }
```

---

### list_repos
List repositories for a user or organization.

```python
def list_repos(owner: str, repo_type: str = "all") -> dict:
    """List repositories for a user or organization."""
    
    config = load_github_config()
    
    # Try user endpoint first
    url = f"https://api.github.com/users/{owner}/repos"
    headers = {
        "Authorization": f"token {config['token']}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "type": repo_type,
        "sort": "updated",
        "per_page": 100
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        repos = response.json()
        return {
            "success": True,
            "count": len(repos),
            "repos": [
                {
                    "name": r["name"],
                    "full_name": r["full_name"],
                    "description": r["description"],
                    "url": r["html_url"],
                    "private": r["private"],
                    "updated_at": r["updated_at"]
                }
                for r in repos
            ]
        }
    else:
        return {
            "success": False,
            "error": response.json().get("message", "Unknown error")
        }
```

---

### get_diff
Show uncommitted changes or diff between branches.

```python
def get_diff(repo_path: Path, staged: bool = False, branch: str = None) -> dict:
    """Show diff of changes."""
    
    try:
        if branch:
            # Diff against branch
            cmd = ["git", "diff", branch]
        elif staged:
            # Staged changes
            cmd = ["git", "diff", "--cached"]
        else:
            # Unstaged changes
            cmd = ["git", "diff"]
        
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "success": True,
            "diff": result.stdout,
            "has_changes": len(result.stdout) > 0
        }
    
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": str(e)
        }
```

---

## Examples

### Example 1: Clone and create feature branch
**Intent:** "Clone my gorgon-skills repo and create a branch for new email templates"

**Execution:**
```python
# Clone repository
result = clone_repo(
    owner="yourusername",
    repo="gorgon-skills",
    shallow=False
)
print(f"Cloned to: {result['path']}")

repo_path = Path(result['path'])

# Create feature branch
branch = create_branch(
    repo_path=repo_path,
    branch_name="feature/email-templates",
    from_branch="main"
)
print(f"Created branch: {branch['branch']}")
```

---

### Example 2: Commit and push changes
**Intent:** "Commit the new skill files and push"

**Execution:**
```python
repo_path = Path.home() / "gorgon-repos" / "gorgon-skills"

# Review changes first
diff = get_diff(repo_path)
print(f"Changes:\n{diff['diff']}")

# Commit (REQUIRES MAJORITY CONSENSUS)
commit = commit_changes(
    repo_path=repo_path,
    message="feat: add email template skill with job followup template",
    all_changes=True
)
print(f"Committed: {commit['commit_hash']}")

# Push (REQUIRES UNANIMOUS CONSENSUS)
push = push_branch(repo_path=repo_path)
print(f"Pushed: {push['branch']}")
```

---

### Example 3: Create issue and PR
**Intent:** "Create an issue for the bug and submit a fix"

**Execution:**
```python
# Create issue
issue = create_issue(
    owner="yourusername",
    repo="gorgon",
    title="Bug: Email skill fails with unicode characters",
    body="""
## Description
The email compose skill fails when the body contains unicode characters.

## Steps to Reproduce
1. Create draft with emoji in body
2. Attempt to send

## Expected Behavior
Email should send successfully.

## Actual Behavior
UnicodeEncodeError raised.
""",
    labels=["bug", "email-skill"]
)
print(f"Created issue #{issue['issue_number']}: {issue['url']}")

# After fixing...
# Create PR (REQUIRES MAJORITY CONSENSUS)
pr = create_pull_request(
    owner="yourusername",
    repo="gorgon",
    title=f"Fix: Handle unicode in email body (closes #{issue['issue_number']})",
    body=f"""
## Summary
Fixed unicode handling in email compose skill.

## Changes
- Added explicit UTF-8 encoding
- Added test cases for unicode content

Closes #{issue['issue_number']}
""",
    head_branch="fix/email-unicode",
    base_branch="main"
)
print(f"Created PR #{pr['pr_number']}: {pr['url']}")
```

---

## Security Considerations

### Token Scopes
Use minimum required scopes:
- `repo` - Full repository access (private repos)
- `public_repo` - Public repos only
- `read:org` - Read org membership
- `workflow` - If managing GitHub Actions

### .gitignore
Ensure secrets never get committed:
```gitignore
# Gorgon config with secrets
.gorgon/config/
*.env
.env.*
credentials.yaml
```

## Error Handling

| Error | Response |
|-------|----------|
| Authentication failed | Check token, verify scopes |
| Repository not found | Check owner/repo spelling, verify access |
| Push rejected | Pull first, or check branch protection |
| Merge conflict | Report, require manual resolution |
| Rate limited | Wait and retry (check X-RateLimit-Reset header) |

## Output Format

```json
{
  "success": true,
  "operation": "push_branch",
  "branch": "feature/email-templates",
  "remote": "origin",
  "commit_range": "abc1234..def5678",
  "timestamp": "2026-01-27T10:30:00Z"
}
```
