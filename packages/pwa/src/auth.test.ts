import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { getToken, setToken, clearToken, AuthError } from "./auth";

describe("auth", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe("getToken", () => {
    it("returns null when no token is stored", () => {
      expect(getToken()).toBeNull();
    });

    it("returns the stored token", () => {
      localStorage.setItem("animus_token", "test-token-123");
      expect(getToken()).toBe("test-token-123");
    });
  });

  describe("setToken", () => {
    it("stores the token in localStorage", () => {
      setToken("my-token");
      expect(localStorage.getItem("animus_token")).toBe("my-token");
    });
  });

  describe("clearToken", () => {
    it("removes the token from localStorage", () => {
      localStorage.setItem("animus_token", "to-clear");
      clearToken();
      expect(localStorage.getItem("animus_token")).toBeNull();
    });
  });

  describe("AuthError", () => {
    it("has the correct name and default message", () => {
      const err = new AuthError();
      expect(err.name).toBe("AuthError");
      expect(err.message).toBe("Authentication required");
    });

    it("accepts a custom message", () => {
      const err = new AuthError("Session expired");
      expect(err.message).toBe("Session expired");
    });
  });
});
