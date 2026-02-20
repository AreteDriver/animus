"""Data models for filesystem tools.

Pydantic models for tool inputs, outputs, and proposals.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProposalStatus(str, Enum):
    """Status of an edit proposal."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


class FileContent(BaseModel):
    """Content of a file with metadata."""

    path: str = Field(..., description="Relative path from project root")
    content: str = Field(..., description="File content")
    line_count: int = Field(..., description="Number of lines")
    size_bytes: int = Field(..., description="File size in bytes")
    encoding: str = Field(default="utf-8", description="File encoding")
    truncated: bool = Field(default=False, description="Whether content was truncated")


class FileEntry(BaseModel):
    """Entry in a directory listing."""

    name: str = Field(..., description="File or directory name")
    path: str = Field(..., description="Relative path from project root")
    is_dir: bool = Field(..., description="Whether this is a directory")
    size_bytes: int | None = Field(default=None, description="File size (None for directories)")


class DirectoryListing(BaseModel):
    """Contents of a directory."""

    path: str = Field(..., description="Relative path of directory")
    entries: list[FileEntry] = Field(default_factory=list)
    total_files: int = Field(default=0, description="Total file count")
    total_dirs: int = Field(default=0, description="Total directory count")
    truncated: bool = Field(default=False, description="Whether listing was truncated")


class SearchMatch(BaseModel):
    """A single search match."""

    path: str = Field(..., description="Relative path to file")
    line_number: int = Field(..., description="Line number (1-indexed)")
    line_content: str = Field(..., description="Content of matching line")
    match_start: int = Field(..., description="Start position of match in line")
    match_end: int = Field(..., description="End position of match in line")


class SearchResult(BaseModel):
    """Results from a code search."""

    pattern: str = Field(..., description="Search pattern used")
    matches: list[SearchMatch] = Field(default_factory=list)
    total_matches: int = Field(default=0, description="Total match count")
    files_searched: int = Field(default=0, description="Number of files searched")
    truncated: bool = Field(default=False, description="Whether results were truncated")


class ProjectStructure(BaseModel):
    """Overview of project structure."""

    root_path: str = Field(..., description="Project root path")
    total_files: int = Field(default=0, description="Total file count")
    total_dirs: int = Field(default=0, description="Total directory count")
    tree: str = Field(..., description="Tree-style representation")
    file_types: dict[str, int] = Field(default_factory=dict, description="Count by file extension")


class EditProposal(BaseModel):
    """A proposed edit to a file."""

    id: str = Field(..., description="Unique proposal ID")
    session_id: str = Field(..., description="Chat session ID")
    file_path: str = Field(..., description="Relative path to file")
    old_content: str | None = Field(
        default=None, description="Original content (None for new files)"
    )
    new_content: str = Field(..., description="Proposed new content")
    description: str = Field(default="", description="Description of the change")
    status: ProposalStatus = Field(default=ProposalStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    applied_at: datetime | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ToolCallRequest(BaseModel):
    """Request to execute a filesystem tool."""

    tool: str = Field(..., description="Tool name")
    path: str | None = Field(default=None, description="File or directory path")
    pattern: str | None = Field(default=None, description="Search pattern or glob")
    old_content: str | None = Field(default=None, description="Content to replace")
    new_content: str | None = Field(default=None, description="Replacement content")
    description: str | None = Field(default=None, description="Edit description")
    max_results: int | None = Field(default=None, description="Maximum results")


class ToolCallResult(BaseModel):
    """Result from a filesystem tool execution."""

    tool: str = Field(..., description="Tool that was called")
    success: bool = Field(..., description="Whether the call succeeded")
    data: dict[str, Any] | None = Field(default=None, description="Result data (tool-specific)")
    error: str | None = Field(default=None, description="Error message if failed")


class FileAccessLog(BaseModel):
    """Audit log entry for file access."""

    id: str = Field(..., description="Unique log ID")
    session_id: str = Field(..., description="Chat session ID")
    tool: str = Field(..., description="Tool used")
    file_path: str = Field(..., description="File accessed")
    operation: str = Field(..., description="Operation type (read, list, search)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)
