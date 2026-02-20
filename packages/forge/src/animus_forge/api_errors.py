"""Structured API error responses for Gorgon.

Provides consistent error formatting across all API endpoints with:
- Machine-readable error codes
- Human-readable messages
- Contextual details for debugging
- OpenAPI documentation support
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from animus_forge.errors import (
    GorgonError,
)

# =============================================================================
# Response Models
# =============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the error",
    )
    request_id: str | None = Field(None, description="Request ID for tracing (if available)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error_code": "WORKFLOW_NOT_FOUND",
                    "message": "Workflow 'my-workflow' not found",
                    "details": {"workflow_id": "my-workflow"},
                    "request_id": "abc12345",
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response format.

    All API errors return this consistent format for easy client handling.
    """

    error: ErrorDetail = Field(..., description="Error details")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "error_code": "AUTH_FAILED",
                        "message": "Invalid or expired token",
                        "details": {},
                        "request_id": "abc12345",
                    }
                }
            ]
        }
    }


class ValidationErrorItem(BaseModel):
    """Single validation error."""

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any | None = Field(None, description="The invalid value (if safe to show)")


class ValidationErrorResponse(BaseModel):
    """Validation error response with field-level details.

    Used for 422 Unprocessable Entity responses.
    """

    error: ErrorDetail = Field(..., description="Error summary")
    validation_errors: list[ValidationErrorItem] = Field(
        default_factory=list,
        description="List of field-level validation errors",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "error_code": "VALIDATION",
                        "message": "Request validation failed",
                        "details": {"field_count": 2},
                        "request_id": "abc12345",
                    },
                    "validation_errors": [
                        {
                            "field": "workflow_id",
                            "message": "Field required",
                            "value": None,
                        },
                        {
                            "field": "variables.count",
                            "message": "Must be a positive integer",
                            "value": -1,
                        },
                    ],
                }
            ]
        }
    }


class RateLimitErrorResponse(BaseModel):
    """Rate limit exceeded response."""

    error: ErrorDetail = Field(..., description="Error details")
    retry_after: int = Field(..., description="Seconds until rate limit resets")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "error_code": "RATE_LIMITED",
                        "message": "Rate limit exceeded",
                        "details": {"limit": "10/minute"},
                        "request_id": "abc12345",
                    },
                    "retry_after": 45,
                }
            ]
        }
    }


# =============================================================================
# Error Code Mapping
# =============================================================================


# Map GorgonError codes to HTTP status codes
ERROR_STATUS_MAP: dict[str, int] = {
    # Authentication/Authorization (401, 403)
    "AUTH_FAILED": 401,
    "TOKEN_EXPIRED": 401,
    "UNAUTHORIZED": 403,
    # Validation (400, 422)
    "VALIDATION": 400,
    "CONTRACT_VIOLATION": 400,
    # Not Found (404)
    "WORKFLOW_NOT_FOUND": 404,
    "NOT_FOUND": 404,
    # Rate Limiting (429)
    "RATE_LIMITED": 429,
    "BUDGET_EXCEEDED": 429,
    "TOKEN_LIMIT": 429,
    # Timeout (408, 504)
    "TIMEOUT": 504,
    # Conflict (409)
    "CONFLICT": 409,
    "CHECKPOINT_ERROR": 409,
    # Server Errors (500, 502, 503)
    "GORGON_ERROR": 500,
    "AGENT_ERROR": 500,
    "WORKFLOW_ERROR": 500,
    "STATE_ERROR": 500,
    "API_ERROR": 502,
    "STAGE_FAILED": 500,
    "MAX_RETRIES": 500,
    "RESUME_ERROR": 500,
}


def get_status_code(error_code: str) -> int:
    """Get HTTP status code for an error code."""
    return ERROR_STATUS_MAP.get(error_code, 500)


# =============================================================================
# Exception Handlers
# =============================================================================


def gorgon_error_to_response(
    error: GorgonError,
    request_id: str | None = None,
) -> JSONResponse:
    """Convert a GorgonError to a structured JSON response."""
    status_code = get_status_code(error.code)

    content = ErrorResponse(
        error=ErrorDetail(
            error_code=error.code,
            message=error.message,
            details=error.details,
            request_id=request_id,
        )
    )

    return JSONResponse(
        status_code=status_code,
        content=content.model_dump(),
    )


async def gorgon_exception_handler(
    request: Request,
    exc: GorgonError,
) -> JSONResponse:
    """FastAPI exception handler for GorgonError."""
    request_id = request.headers.get("X-Request-ID")
    return gorgon_error_to_response(exc, request_id)


def http_error_to_gorgon(
    status_code: int,
    detail: str,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> ErrorResponse:
    """Convert HTTP status code and detail to structured error response."""
    # Infer error code from status if not provided
    if error_code is None:
        error_code = {
            400: "VALIDATION",
            401: "AUTH_FAILED",
            403: "UNAUTHORIZED",
            404: "NOT_FOUND",
            408: "TIMEOUT",
            409: "CONFLICT",
            422: "VALIDATION",
            429: "RATE_LIMITED",
            500: "INTERNAL_ERROR",
            502: "API_ERROR",
            503: "SERVICE_UNAVAILABLE",
            504: "TIMEOUT",
        }.get(status_code, "UNKNOWN_ERROR")

    return ErrorResponse(
        error=ErrorDetail(
            error_code=error_code,
            message=detail,
            details=details or {},
        )
    )


# =============================================================================
# API Exception Classes
# =============================================================================


class APIException(HTTPException):
    """Enhanced HTTPException with structured error response.

    Use this instead of HTTPException for consistent error formatting.
    """

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        self.error_code = error_code
        self.message = message
        self.error_details = details or {}
        super().__init__(status_code=status_code, detail=message)

    def to_response(self, request_id: str | None = None) -> ErrorResponse:
        """Convert to ErrorResponse model."""
        return ErrorResponse(
            error=ErrorDetail(
                error_code=self.error_code,
                message=self.message,
                details=self.error_details,
                request_id=request_id,
            )
        )


async def api_exception_handler(
    request: Request,
    exc: APIException,
) -> JSONResponse:
    """FastAPI exception handler for APIException."""
    request_id = request.headers.get("X-Request-ID")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response(request_id).model_dump(),
    )


# =============================================================================
# Common Response Definitions
# =============================================================================


# Pre-defined response schemas for OpenAPI documentation
COMMON_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Bad Request - Invalid input or parameters",
    },
    401: {
        "model": ErrorResponse,
        "description": "Unauthorized - Missing or invalid authentication",
    },
    403: {
        "model": ErrorResponse,
        "description": "Forbidden - Insufficient permissions",
    },
    404: {
        "model": ErrorResponse,
        "description": "Not Found - Resource does not exist",
    },
    422: {
        "model": ValidationErrorResponse,
        "description": "Validation Error - Request body validation failed",
    },
    429: {
        "model": RateLimitErrorResponse,
        "description": "Rate Limited - Too many requests",
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal Server Error",
    },
    502: {
        "model": ErrorResponse,
        "description": "Bad Gateway - External service error",
    },
    503: {
        "model": ErrorResponse,
        "description": "Service Unavailable - Server is shutting down or overloaded",
    },
    504: {
        "model": ErrorResponse,
        "description": "Gateway Timeout - External service timeout",
    },
}


def responses(*status_codes: int) -> dict:
    """Generate responses dict for specific status codes.

    Usage:
        @router.get("/items/{id}", responses=responses(404, 500))
        def get_item(id: str): ...
    """
    return {code: COMMON_RESPONSES[code] for code in status_codes if code in COMMON_RESPONSES}


# Convenience response sets
AUTH_RESPONSES = responses(401, 403)
CRUD_RESPONSES = responses(400, 401, 404, 500)
WORKFLOW_RESPONSES = responses(400, 401, 404, 422, 429, 500, 502, 504)


# =============================================================================
# Helper Functions
# =============================================================================


def not_found(resource: str, identifier: str) -> APIException:
    """Create a not found exception."""
    return APIException(
        status_code=404,
        error_code="NOT_FOUND",
        message=f"{resource} '{identifier}' not found",
        details={"resource": resource, "identifier": identifier},
    )


def unauthorized(message: str = "Authentication required") -> APIException:
    """Create an unauthorized exception."""
    return APIException(
        status_code=401,
        error_code="AUTH_FAILED",
        message=message,
    )


def bad_request(message: str, details: dict[str, Any] | None = None) -> APIException:
    """Create a bad request exception."""
    return APIException(
        status_code=400,
        error_code="VALIDATION",
        message=message,
        details=details,
    )


def internal_error(
    message: str = "An internal error occurred",
    details: dict[str, Any] | None = None,
) -> APIException:
    """Create an internal server error exception."""
    return APIException(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message=message,
        details=details,
    )


def conflict(message: str, details: dict[str, Any] | None = None) -> APIException:
    """Create a conflict exception."""
    return APIException(
        status_code=409,
        error_code="CONFLICT",
        message=message,
        details=details,
    )
