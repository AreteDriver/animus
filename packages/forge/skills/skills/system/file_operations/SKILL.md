---
name: file_operations
version: 1.0.0
agent: system
risk_level: variable
description: "Safe file system operations including create, read, update, delete, move, copy, search, and permission management. Core skill for any task involving filesystem interaction."
---

# File Operations Skill

## Purpose

Perform safe, auditable file system operations on the Gorgon host machine. This skill provides the foundation for all filesystem interactions and enforces safety guardrails to prevent accidental system damage.

## Safety Rules

### PROTECTED PATHS - NEVER MODIFY
```
/boot
/etc/fstab
/etc/passwd
/etc/shadow
/etc/sudoers
/usr/bin
/usr/lib
/usr/sbin
/lib
/lib64
/sbin
/bin
/proc
/sys
/dev
~/.ssh (except creating new keys with explicit approval)
```

### MANDATORY PRACTICES
1. **Always use absolute paths** - relative paths cause ambiguity
2. **Verify path exists before destructive operations**
3. **Create backups before modifying existing files** (unless explicitly told not to)
4. **Never follow symlinks outside the working directory without verification**
5. **Check available disk space before large write operations**
6. **Preserve file permissions when copying/moving**

### CONSENSUS REQUIREMENTS
| Operation | Risk Level | Consensus Required |
|-----------|------------|-------------------|
| read_file | low | any |
| search_files | low | any |
| create_file (new) | low | any |
| create_file (overwrite) | medium | majority |
| update_file | medium | majority |
| copy_file | low | any |
| move_file | medium | majority |
| delete_file | high | unanimous |
| delete_directory | critical | unanimous + confirmation |
| chmod/chown | high | unanimous |

## Capabilities

### read_file
Read contents of a file. Supports text and binary files.

**Usage:**
```bash
# Text file
cat /path/to/file.txt

# Binary file (hex dump)
xxd /path/to/file.bin | head -100

# Large file (paginated)
less /path/to/large.log

# Specific lines
sed -n '10,20p' /path/to/file.txt
```

**Safety:** Verify file exists and is readable before attempting.

---

### create_file
Create a new file with specified content.

**Usage:**
```bash
# Create with content
cat << 'EOF' > /path/to/newfile.txt
File content here
Multiple lines supported
EOF

# Create empty file
touch /path/to/empty.txt

# Create with specific permissions
install -m 644 /dev/null /path/to/file.txt
echo "content" > /path/to/file.txt
```

**Safety:** 
- Check if file already exists
- If exists, require majority consensus for overwrite
- Verify parent directory exists and is writable

---

### update_file
Modify existing file content.

**Usage:**
```bash
# Replace text (sed)
sed -i 's/old_text/new_text/g' /path/to/file.txt

# Insert line after pattern
sed -i '/pattern/a new line content' /path/to/file.txt

# Delete lines matching pattern
sed -i '/pattern_to_delete/d' /path/to/file.txt

# Append to file
echo "new content" >> /path/to/file.txt

# Prepend to file
echo "new first line" | cat - /path/to/file.txt > temp && mv temp /path/to/file.txt
```

**Safety:**
- Create backup before modification: `cp file.txt file.txt.bak.$(date +%s)`
- Verify file exists and is writable
- For config files, validate syntax after edit if validator available

---

### delete_file
Permanently remove a file.

**Usage:**
```bash
# Delete with backup
cp /path/to/file.txt /path/to/file.txt.deleted.$(date +%s)
rm /path/to/file.txt

# Delete without backup (requires explicit approval)
rm /path/to/file.txt
```

**Safety:**
- REQUIRES UNANIMOUS CONSENSUS
- Create backup by default
- Never use `rm -rf` on directories through this capability
- Verify path is not in PROTECTED PATHS
- Confirm file is not a symlink to protected location

---

### delete_directory
Remove a directory and its contents.

**Usage:**
```bash
# Safe deletion with backup
tar -czf /backup/dirname.$(date +%s).tar.gz /path/to/dir
rm -rf /path/to/dir
```

**Safety:**
- REQUIRES UNANIMOUS CONSENSUS + USER CONFIRMATION
- Always create backup archive first
- Maximum depth check: refuse if > 1000 files without explicit approval
- Never delete directories matching: `/home`, `/var`, `/tmp` root level
- List contents and get approval before deletion

---

### move_file
Move or rename a file.

**Usage:**
```bash
# Move file
mv /source/path/file.txt /dest/path/file.txt

# Rename file
mv /path/to/oldname.txt /path/to/newname.txt

# Move with backup of destination if exists
[ -f /dest/file.txt ] && cp /dest/file.txt /dest/file.txt.bak.$(date +%s)
mv /source/file.txt /dest/file.txt
```

**Safety:**
- Verify source exists
- Verify destination directory exists and is writable
- Check if destination file exists (backup if so)
- Preserve permissions

---

### copy_file
Duplicate a file.

**Usage:**
```bash
# Simple copy
cp /source/file.txt /dest/file.txt

# Copy preserving attributes
cp -p /source/file.txt /dest/file.txt

# Copy directory recursively
cp -rp /source/dir /dest/dir
```

**Safety:**
- Verify source exists and is readable
- Check destination disk space
- Preserve permissions with `-p` flag

---

### search_files
Find files matching criteria.

**Usage:**
```bash
# By name pattern
find /search/path -name "*.log" -type f

# By content
grep -rl "search_term" /search/path

# By modification time (last 24 hours)
find /search/path -type f -mtime -1

# By size (larger than 100MB)
find /search/path -type f -size +100M

# Combined criteria
find /search/path -name "*.py" -type f -mtime -7 -size +1k

# With detailed output
find /search/path -name "*.txt" -type f -exec ls -lh {} \;
```

**Safety:** 
- Limit search depth for broad searches: `-maxdepth 5`
- Avoid searching `/proc`, `/sys`, `/dev`

---

### set_permissions
Change file permissions or ownership.

**Usage:**
```bash
# Change permissions
chmod 644 /path/to/file.txt
chmod 755 /path/to/script.sh

# Change ownership
chown user:group /path/to/file.txt

# Recursive (use with caution)
chmod -R 755 /path/to/dir
```

**Safety:**
- REQUIRES UNANIMOUS CONSENSUS
- Never chmod 777 anything
- Never recursive chmod/chown on system directories
- Verify current permissions before change
- Log original permissions for potential rollback

---

## Examples

### Example 1: Create a Python script
**Intent:** "Create a Python script that prints system info"

**Execution:**
```bash
# Verify target directory exists
ls -la /home/gorgon/scripts/

# Create the file
cat << 'EOF' > /home/gorgon/scripts/sysinfo.py
#!/usr/bin/env python3
"""Print basic system information."""

import platform
import psutil

def main():
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}%")
    print(f"Disk: {psutil.disk_usage('/').percent}%")

if __name__ == "__main__":
    main()
EOF

# Make executable
chmod 755 /home/gorgon/scripts/sysinfo.py

# Verify
ls -la /home/gorgon/scripts/sysinfo.py
```

---

### Example 2: Find and clean old log files
**Intent:** "Find log files older than 30 days and delete them"

**Execution:**
```bash
# First, find and list (no deletion yet)
find /var/log/gorgon -name "*.log" -type f -mtime +30

# Count files
find /var/log/gorgon -name "*.log" -type f -mtime +30 | wc -l

# REQUIRES UNANIMOUS CONSENSUS FOR DELETION
# Create backup archive first
tar -czf /backup/old_logs.$(date +%s).tar.gz $(find /var/log/gorgon -name "*.log" -type f -mtime +30)

# Delete after approval
find /var/log/gorgon -name "*.log" -type f -mtime +30 -delete
```

---

### Example 3: Safely update a config file
**Intent:** "Change the port number from 8080 to 9090 in the config"

**Execution:**
```bash
# Verify file exists and show current content
cat /home/gorgon/app/config.yaml

# Create timestamped backup
cp /home/gorgon/app/config.yaml /home/gorgon/app/config.yaml.bak.$(date +%s)

# Make the change
sed -i 's/port: 8080/port: 9090/g' /home/gorgon/app/config.yaml

# Verify change
grep -n "port:" /home/gorgon/app/config.yaml

# Validate YAML syntax if possible
python3 -c "import yaml; yaml.safe_load(open('/home/gorgon/app/config.yaml'))" && echo "Valid YAML"
```

---

## Error Handling

| Error | Response |
|-------|----------|
| File not found | Report clearly, suggest alternatives if path looks like typo |
| Permission denied | Report, suggest checking ownership/permissions |
| Disk full | Report available space, suggest cleanup |
| Path is symlink to protected | Refuse operation, report target |
| File locked by process | Report which process, suggest resolution |

## Output Format

All operations should return structured results:

```json
{
  "success": true,
  "operation": "create_file",
  "path": "/home/gorgon/scripts/test.py",
  "details": {
    "bytes_written": 1234,
    "permissions": "755",
    "backup_created": null
  },
  "timestamp": "2026-01-27T10:30:00Z"
}
```

For failures:
```json
{
  "success": false,
  "operation": "delete_file",
  "path": "/etc/passwd",
  "error": "Path is protected - operation refused",
  "error_code": "PROTECTED_PATH",
  "timestamp": "2026-01-27T10:30:00Z"
}
```
