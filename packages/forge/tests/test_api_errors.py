"""Tests for api_errors module - structured API error responses."""

from fastapi.responses import JSONResponse

from animus_forge.api_errors import (
    AUTH_RESPONSES,
    COMMON_RESPONSES,
    CRUD_RESPONSES,
    WORKFLOW_RESPONSES,
    APIException,
    ErrorDetail,
    ErrorResponse,
    RateLimitErrorResponse,
    ValidationErrorItem,
    ValidationErrorResponse,
    bad_request,
    conflict,
    get_status_code,
    gorgon_error_to_response,
    http_error_to_gorgon,
    internal_error,
    not_found,
    responses,
    unauthorized,
)
from animus_forge.errors import (
    BudgetExceededError,
    GorgonError,
    ValidationError,
    WorkflowNotFoundError,
)

# =============================================================================
# ErrorDetail Tests
# =============================================================================


class TestErrorDetail:
    """Tests for ErrorDetail model."""

    def test_create_error_detail(self):
        """Test creating an ErrorDetail."""
        detail = ErrorDetail(
            error_code="TEST_ERROR",
            message="Test error message",
        )
        assert detail.error_code == "TEST_ERROR"
        assert detail.message == "Test error message"
        assert detail.details == {}
        assert detail.request_id is None

    def test_create_with_all_fields(self):
        """Test creating ErrorDetail with all fields."""
        detail = ErrorDetail(
            error_code="VALIDATION",
            message="Validation failed",
            details={"field": "email", "reason": "invalid format"},
            request_id="abc123",
        )
        assert detail.details == {"field": "email", "reason": "invalid format"}
        assert detail.request_id == "abc123"

    def test_to_dict(self):
        """Test ErrorDetail serialization."""
        detail = ErrorDetail(
            error_code="TEST",
            message="Test",
            details={"key": "value"},
        )
        data = detail.model_dump()
        assert data["error_code"] == "TEST"
        assert data["message"] == "Test"
        assert data["details"] == {"key": "value"}


# =============================================================================
# ErrorResponse Tests
# =============================================================================


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_create_error_response(self):
        """Test creating an ErrorResponse."""
        response = ErrorResponse(
            error=ErrorDetail(
                error_code="NOT_FOUND",
                message="Resource not found",
            )
        )
        assert response.error.error_code == "NOT_FOUND"

    def test_model_dump(self):
        """Test ErrorResponse serialization."""
        response = ErrorResponse(
            error=ErrorDetail(
                error_code="AUTH_FAILED",
                message="Invalid token",
                request_id="xyz789",
            )
        )
        data = response.model_dump()
        assert data["error"]["error_code"] == "AUTH_FAILED"
        assert data["error"]["message"] == "Invalid token"
        assert data["error"]["request_id"] == "xyz789"


# =============================================================================
# ValidationErrorResponse Tests
# =============================================================================


class TestValidationErrorResponse:
    """Tests for ValidationErrorResponse model."""

    def test_create_validation_error_response(self):
        """Test creating a ValidationErrorResponse."""
        response = ValidationErrorResponse(
            error=ErrorDetail(
                error_code="VALIDATION",
                message="Request validation failed",
            ),
            validation_errors=[
                ValidationErrorItem(
                    field="email",
                    message="Invalid email format",
                    value="not-an-email",
                ),
                ValidationErrorItem(
                    field="age",
                    message="Must be positive",
                    value=-5,
                ),
            ],
        )
        assert len(response.validation_errors) == 2
        assert response.validation_errors[0].field == "email"
        assert response.validation_errors[1].value == -5


# =============================================================================
# RateLimitErrorResponse Tests
# =============================================================================


class TestRateLimitErrorResponse:
    """Tests for RateLimitErrorResponse model."""

    def test_create_rate_limit_response(self):
        """Test creating a RateLimitErrorResponse."""
        response = RateLimitErrorResponse(
            error=ErrorDetail(
                error_code="RATE_LIMITED",
                message="Rate limit exceeded",
            ),
            retry_after=60,
        )
        assert response.retry_after == 60


# =============================================================================
# Error Status Mapping Tests
# =============================================================================


class TestErrorStatusMapping:
    """Tests for error code to HTTP status mapping."""

    def test_auth_errors_map_to_401(self):
        """Test that auth errors map to 401."""
        assert get_status_code("AUTH_FAILED") == 401
        assert get_status_code("TOKEN_EXPIRED") == 401

    def test_not_found_maps_to_404(self):
        """Test that not found errors map to 404."""
        assert get_status_code("NOT_FOUND") == 404
        assert get_status_code("WORKFLOW_NOT_FOUND") == 404

    def test_validation_maps_to_400(self):
        """Test that validation errors map to 400."""
        assert get_status_code("VALIDATION") == 400
        assert get_status_code("CONTRACT_VIOLATION") == 400

    def test_rate_limit_maps_to_429(self):
        """Test that rate limit errors map to 429."""
        assert get_status_code("RATE_LIMITED") == 429
        assert get_status_code("BUDGET_EXCEEDED") == 429
        assert get_status_code("TOKEN_LIMIT") == 429

    def test_timeout_maps_to_504(self):
        """Test that timeout errors map to 504."""
        assert get_status_code("TIMEOUT") == 504

    def test_unknown_error_maps_to_500(self):
        """Test that unknown errors map to 500."""
        assert get_status_code("UNKNOWN_ERROR_CODE") == 500


# =============================================================================
# Gorgon Error Conversion Tests
# =============================================================================


class TestGorgonErrorConversion:
    """Tests for converting GorgonError to response."""

    def test_convert_gorgon_error(self):
        """Test converting a GorgonError to JSON response."""
        error = GorgonError("Something went wrong", {"context": "test"})
        response = gorgon_error_to_response(error)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

    def test_convert_validation_error(self):
        """Test converting ValidationError."""
        error = ValidationError("Invalid input")
        response = gorgon_error_to_response(error)

        assert response.status_code == 400

    def test_convert_workflow_not_found(self):
        """Test converting WorkflowNotFoundError."""
        error = WorkflowNotFoundError("Workflow 'xyz' not found")
        response = gorgon_error_to_response(error)

        assert response.status_code == 404

    def test_convert_budget_exceeded(self):
        """Test converting BudgetExceededError."""
        error = BudgetExceededError("Budget exceeded", budget=1000, used=1500)
        response = gorgon_error_to_response(error)

        assert response.status_code == 429

    def test_convert_with_request_id(self):
        """Test conversion includes request ID."""
        error = GorgonError("Error")
        response = gorgon_error_to_response(error, request_id="req123")

        # The response body should include the request_id
        assert response.status_code == 500


# =============================================================================
# HTTP Error Conversion Tests
# =============================================================================


class TestHttpErrorConversion:
    """Tests for converting HTTP errors to structured format."""

    def test_convert_401(self):
        """Test converting 401 error."""
        response = http_error_to_gorgon(401, "Unauthorized")
        assert response.error.error_code == "AUTH_FAILED"

    def test_convert_404(self):
        """Test converting 404 error."""
        response = http_error_to_gorgon(404, "Not found")
        assert response.error.error_code == "NOT_FOUND"

    def test_convert_500(self):
        """Test converting 500 error."""
        response = http_error_to_gorgon(500, "Internal error")
        assert response.error.error_code == "INTERNAL_ERROR"

    def test_convert_with_custom_code(self):
        """Test conversion with custom error code."""
        response = http_error_to_gorgon(
            400,
            "Invalid request",
            error_code="CUSTOM_ERROR",
        )
        assert response.error.error_code == "CUSTOM_ERROR"

    def test_convert_with_details(self):
        """Test conversion with details."""
        response = http_error_to_gorgon(
            400,
            "Invalid request",
            details={"field": "email"},
        )
        assert response.error.details == {"field": "email"}


# =============================================================================
# APIException Tests
# =============================================================================


class TestAPIException:
    """Tests for APIException class."""

    def test_create_api_exception(self):
        """Test creating an APIException."""
        exc = APIException(
            status_code=404,
            error_code="NOT_FOUND",
            message="Resource not found",
        )
        assert exc.status_code == 404
        assert exc.error_code == "NOT_FOUND"
        assert exc.message == "Resource not found"

    def test_api_exception_with_details(self):
        """Test APIException with details."""
        exc = APIException(
            status_code=400,
            error_code="VALIDATION",
            message="Invalid input",
            details={"field": "name", "reason": "required"},
        )
        assert exc.error_details == {"field": "name", "reason": "required"}

    def test_api_exception_to_response(self):
        """Test converting APIException to response."""
        exc = APIException(
            status_code=404,
            error_code="NOT_FOUND",
            message="Not found",
        )
        response = exc.to_response(request_id="abc")

        assert response.error.error_code == "NOT_FOUND"
        assert response.error.request_id == "abc"


# =============================================================================
# Response Helper Tests
# =============================================================================


class TestResponseHelpers:
    """Tests for response helper functions."""

    def test_responses_single(self):
        """Test generating responses for single status."""
        result = responses(404)
        assert 404 in result
        assert result[404]["model"] == ErrorResponse

    def test_responses_multiple(self):
        """Test generating responses for multiple statuses."""
        result = responses(400, 401, 404, 500)
        assert len(result) == 4
        assert 400 in result
        assert 401 in result
        assert 404 in result
        assert 500 in result

    def test_responses_invalid_status_ignored(self):
        """Test that invalid status codes are ignored."""
        result = responses(404, 999)  # 999 is not in COMMON_RESPONSES
        assert 404 in result
        assert 999 not in result

    def test_auth_responses(self):
        """Test AUTH_RESPONSES preset."""
        assert 401 in AUTH_RESPONSES
        assert 403 in AUTH_RESPONSES

    def test_crud_responses(self):
        """Test CRUD_RESPONSES preset."""
        assert 400 in CRUD_RESPONSES
        assert 401 in CRUD_RESPONSES
        assert 404 in CRUD_RESPONSES
        assert 500 in CRUD_RESPONSES

    def test_workflow_responses(self):
        """Test WORKFLOW_RESPONSES preset."""
        assert 400 in WORKFLOW_RESPONSES
        assert 401 in WORKFLOW_RESPONSES
        assert 404 in WORKFLOW_RESPONSES
        assert 422 in WORKFLOW_RESPONSES
        assert 429 in WORKFLOW_RESPONSES
        assert 500 in WORKFLOW_RESPONSES
        assert 502 in WORKFLOW_RESPONSES
        assert 504 in WORKFLOW_RESPONSES


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for convenience helper functions."""

    def test_not_found(self):
        """Test not_found helper."""
        exc = not_found("Workflow", "my-workflow")

        assert exc.status_code == 404
        assert exc.error_code == "NOT_FOUND"
        assert "Workflow" in exc.message
        assert "my-workflow" in exc.message
        assert exc.error_details["resource"] == "Workflow"
        assert exc.error_details["identifier"] == "my-workflow"

    def test_unauthorized(self):
        """Test unauthorized helper."""
        exc = unauthorized("Token expired")

        assert exc.status_code == 401
        assert exc.error_code == "AUTH_FAILED"
        assert exc.message == "Token expired"

    def test_unauthorized_default_message(self):
        """Test unauthorized with default message."""
        exc = unauthorized()

        assert exc.message == "Authentication required"

    def test_bad_request(self):
        """Test bad_request helper."""
        exc = bad_request("Invalid email format", {"field": "email"})

        assert exc.status_code == 400
        assert exc.error_code == "VALIDATION"
        assert exc.error_details == {"field": "email"}

    def test_internal_error(self):
        """Test internal_error helper."""
        exc = internal_error("Database connection failed")

        assert exc.status_code == 500
        assert exc.error_code == "INTERNAL_ERROR"

    def test_internal_error_default_message(self):
        """Test internal_error with default message."""
        exc = internal_error()

        assert exc.message == "An internal error occurred"

    def test_conflict(self):
        """Test conflict helper."""
        exc = conflict("Resource already exists", {"id": "123"})

        assert exc.status_code == 409
        assert exc.error_code == "CONFLICT"
        assert exc.error_details == {"id": "123"}


# =============================================================================
# Integration Tests
# =============================================================================


class TestErrorResponseIntegration:
    """Integration tests for error responses."""

    def test_error_response_json_schema(self):
        """Test that ErrorResponse produces valid JSON schema."""
        schema = ErrorResponse.model_json_schema()

        assert "properties" in schema
        assert "error" in schema["properties"]

    def test_validation_error_response_json_schema(self):
        """Test that ValidationErrorResponse produces valid JSON schema."""
        schema = ValidationErrorResponse.model_json_schema()

        assert "properties" in schema
        assert "error" in schema["properties"]
        assert "validation_errors" in schema["properties"]

    def test_rate_limit_response_json_schema(self):
        """Test that RateLimitErrorResponse produces valid JSON schema."""
        schema = RateLimitErrorResponse.model_json_schema()

        assert "properties" in schema
        assert "retry_after" in schema["properties"]

    def test_common_responses_have_models(self):
        """Test that all common responses have models defined."""
        for status_code, config in COMMON_RESPONSES.items():
            assert "model" in config
            assert "description" in config
