package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
)

// copilotHTTPUpstreamStub is a test double for HTTPUpstream that returns
// configurable responses. It is safe for concurrent use.
type copilotHTTPUpstreamStub struct {
	mu      sync.Mutex
	handler func(req *http.Request) (*http.Response, error)
	calls   int
	lastReq *http.Request
}

func (s *copilotHTTPUpstreamStub) Do(req *http.Request, proxyURL string, accountID int64, accountConcurrency int) (*http.Response, error) {
	s.mu.Lock()
	s.calls++
	s.lastReq = req
	handler := s.handler
	s.mu.Unlock()
	if handler == nil {
		return nil, fmt.Errorf("no handler configured")
	}
	return handler(req)
}

func (s *copilotHTTPUpstreamStub) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, profile *tlsfingerprint.Profile) (*http.Response, error) {
	return s.Do(req, proxyURL, accountID, accountConcurrency)
}

// makeCopilotTokenResponse creates a valid token exchange HTTP response.
func makeCopilotTokenResponse(token string, expiresAt int64, apiEndpoint string) *http.Response {
	body := copilotAPITokenResponse{
		Token:     token,
		ExpiresAt: expiresAt,
	}
	body.Endpoints.API = apiEndpoint
	data, _ := json.Marshal(body)
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader(data)),
		Header:     make(http.Header),
	}
}

// ---------- CopilotTokenProvider Tests ----------

func TestCopilotTokenProvider_BasicExchange(t *testing.T) {
	upstream := &copilotHTTPUpstreamStub{
		handler: func(req *http.Request) (*http.Response, error) {
			return makeCopilotTokenResponse("jwt-abc", time.Now().Add(30*time.Minute).Unix(), "https://custom.copilot.api"), nil
		},
	}

	provider := NewCopilotTokenProvider(upstream)
	token, endpoint, err := provider.GetCopilotAPIToken(context.Background(), "ghp_testtoken", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if token != "jwt-abc" {
		t.Errorf("expected token 'jwt-abc', got %q", token)
	}
	if endpoint != "https://custom.copilot.api" {
		t.Errorf("expected endpoint 'https://custom.copilot.api', got %q", endpoint)
	}
}

func TestCopilotTokenProvider_CacheHit(t *testing.T) {
	var callCount atomic.Int32
	upstream := &copilotHTTPUpstreamStub{
		handler: func(req *http.Request) (*http.Response, error) {
			callCount.Add(1)
			return makeCopilotTokenResponse("jwt-cached", time.Now().Add(30*time.Minute).Unix(), ""), nil
		},
	}

	provider := NewCopilotTokenProvider(upstream)

	// First call — cache miss.
	token1, _, err := provider.GetCopilotAPIToken(context.Background(), "ghp_test", "")
	if err != nil {
		t.Fatalf("first call error: %v", err)
	}

	// Second call — should hit cache (no new HTTP request).
	token2, _, err := provider.GetCopilotAPIToken(context.Background(), "ghp_test", "")
	if err != nil {
		t.Fatalf("second call error: %v", err)
	}

	if token1 != token2 {
		t.Errorf("tokens should match: %q vs %q", token1, token2)
	}
	if callCount.Load() != 1 {
		t.Errorf("expected 1 HTTP call (cache hit), got %d", callCount.Load())
	}
}

func TestCopilotTokenProvider_CacheExpiry(t *testing.T) {
	var callCount atomic.Int32
	upstream := &copilotHTTPUpstreamStub{
		handler: func(req *http.Request) (*http.Response, error) {
			callCount.Add(1)
			// Token expires in 2 seconds (well within the 5-minute buffer).
			return makeCopilotTokenResponse("jwt-expiring", time.Now().Add(2*time.Second).Unix(), ""), nil
		},
	}

	provider := NewCopilotTokenProvider(upstream)

	// First call — cache miss.
	_, _, err := provider.GetCopilotAPIToken(context.Background(), "ghp_test", "")
	if err != nil {
		t.Fatalf("first call error: %v", err)
	}

	// Second call — token is within expiry buffer, should trigger refresh.
	_, _, err = provider.GetCopilotAPIToken(context.Background(), "ghp_test", "")
	if err != nil {
		t.Fatalf("second call error: %v", err)
	}

	if callCount.Load() != 2 {
		t.Errorf("expected 2 HTTP calls (expired cache), got %d", callCount.Load())
	}
}

func TestCopilotTokenProvider_Singleflight(t *testing.T) {
	var callCount atomic.Int32
	upstream := &copilotHTTPUpstreamStub{
		handler: func(req *http.Request) (*http.Response, error) {
			callCount.Add(1)
			// Simulate some latency to ensure concurrent requests overlap.
			time.Sleep(50 * time.Millisecond)
			return makeCopilotTokenResponse("jwt-sf", time.Now().Add(30*time.Minute).Unix(), ""), nil
		},
	}

	provider := NewCopilotTokenProvider(upstream)

	// Fire 10 concurrent requests — only 1 should hit upstream.
	var wg sync.WaitGroup
	errors := make([]error, 10)
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, _, errors[idx] = provider.GetCopilotAPIToken(context.Background(), "ghp_concurrent", "")
		}(i)
	}
	wg.Wait()

	for i, err := range errors {
		if err != nil {
			t.Errorf("goroutine %d returned error: %v", i, err)
		}
	}

	if callCount.Load() != 1 {
		t.Errorf("expected 1 HTTP call (singleflight), got %d", callCount.Load())
	}
}

func TestCopilotTokenProvider_EmptyToken(t *testing.T) {
	provider := NewCopilotTokenProvider(nil)
	_, _, err := provider.GetCopilotAPIToken(context.Background(), "", "")
	if err == nil {
		t.Fatal("expected error for empty token")
	}
}

func TestCopilotTokenProvider_WhitespaceToken(t *testing.T) {
	provider := NewCopilotTokenProvider(nil)
	_, _, err := provider.GetCopilotAPIToken(context.Background(), "   ", "")
	if err == nil {
		t.Fatal("expected error for whitespace-only token")
	}
}

func TestCopilotTokenProvider_ExchangeFailure(t *testing.T) {
	upstream := &copilotHTTPUpstreamStub{
		handler: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 401,
				Body:       io.NopCloser(bytes.NewBufferString(`{"message":"Bad credentials"}`)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider := NewCopilotTokenProvider(upstream)
	_, _, err := provider.GetCopilotAPIToken(context.Background(), "ghp_invalid", "")
	if err == nil {
		t.Fatal("expected error for failed exchange")
	}
}

func TestCopilotTokenProvider_DefaultEndpoint(t *testing.T) {
	upstream := &copilotHTTPUpstreamStub{
		handler: func(req *http.Request) (*http.Response, error) {
			// No endpoint in response — should use default.
			return makeCopilotTokenResponse("jwt-default", time.Now().Add(30*time.Minute).Unix(), ""), nil
		},
	}

	provider := NewCopilotTokenProvider(upstream)
	_, endpoint, err := provider.GetCopilotAPIToken(context.Background(), "ghp_test", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if endpoint != copilotDefaultBaseURL {
		t.Errorf("expected default endpoint %q, got %q", copilotDefaultBaseURL, endpoint)
	}
}

func TestCopilotTokenProvider_EvictsExpiredEntries(t *testing.T) {
	upstream := &copilotHTTPUpstreamStub{
		handler: func(req *http.Request) (*http.Response, error) {
			return makeCopilotTokenResponse("jwt-evict", time.Now().Add(30*time.Minute).Unix(), ""), nil
		},
	}

	provider := NewCopilotTokenProvider(upstream)

	// Manually insert an expired entry.
	expiredKey := tokenFingerprint("ghp_expired")
	provider.mu.Lock()
	provider.cache[expiredKey] = &cachedCopilotToken{
		token:     "old-jwt",
		expiresAt: time.Now().Add(-1 * time.Hour),
	}
	provider.mu.Unlock()

	// Trigger a new exchange for a different token (causes eviction sweep).
	_, _, err := provider.GetCopilotAPIToken(context.Background(), "ghp_new", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The expired entry should have been cleaned up.
	provider.mu.RLock()
	_, exists := provider.cache[expiredKey]
	provider.mu.RUnlock()
	if exists {
		t.Error("expired cache entry was not evicted")
	}
}

// ---------- ShouldFailoverCopilotUpstreamError Tests ----------

func TestShouldFailoverCopilotUpstreamError(t *testing.T) {
	tests := []struct {
		statusCode int
		expected   bool
	}{
		{200, false},
		{400, false},
		{401, true},
		{402, true},
		{403, true},
		{404, false},
		{429, true},
		{500, true},
		{502, true},
		{503, true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("status_%d", tc.statusCode), func(t *testing.T) {
			result := ShouldFailoverCopilotUpstreamError(tc.statusCode)
			if result != tc.expected {
				t.Errorf("ShouldFailoverCopilotUpstreamError(%d) = %v, want %v", tc.statusCode, result, tc.expected)
			}
		})
	}
}

// ---------- Helper Function Tests ----------

func TestIsCopilotAgentInitiated(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected bool
	}{
		{
			name:     "empty body",
			body:     "",
			expected: false,
		},
		{
			name:     "user message",
			body:     `{"messages":[{"role":"user","content":"hello"}]}`,
			expected: false,
		},
		{
			name:     "tool message last",
			body:     `{"messages":[{"role":"user","content":"hello"},{"role":"tool","content":"result"}]}`,
			expected: true,
		},
		{
			name:     "assistant message last",
			body:     `{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"thinking..."}]}`,
			expected: true,
		},
		{
			name:     "responses API function_call_output",
			body:     `{"input":[{"type":"function_call_output","output":"done"}]}`,
			expected: true,
		},
		{
			name:     "responses API user input",
			body:     `{"input":[{"role":"user","content":"hello"}]}`,
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := isCopilotAgentInitiated([]byte(tc.body))
			if result != tc.expected {
				t.Errorf("isCopilotAgentInitiated(%q) = %v, want %v", tc.body, result, tc.expected)
			}
		})
	}
}

func TestDetectCopilotVisionContent(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected bool
	}{
		{
			name:     "no vision",
			body:     `{"messages":[{"role":"user","content":"hello"}]}`,
			expected: false,
		},
		{
			name:     "with image_url",
			body:     `{"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.com/img.png"}}]}]}`,
			expected: true,
		},
		{
			name:     "with image",
			body:     `{"messages":[{"role":"user","content":[{"type":"image","data":"base64..."}]}]}`,
			expected: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := detectCopilotVisionContent([]byte(tc.body))
			if result != tc.expected {
				t.Errorf("detectCopilotVisionContent(%q) = %v, want %v", tc.body, result, tc.expected)
			}
		})
	}
}

func TestCopilotTruncateString(t *testing.T) {
	if r := copilotTruncateString("short", 10); r != "short" {
		t.Errorf("expected 'short', got %q", r)
	}
	if r := copilotTruncateString("a very long string", 5); r != "a ver..." {
		t.Errorf("expected 'a ver...', got %q", r)
	}
}
