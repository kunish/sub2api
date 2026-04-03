package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

const (
	// Copilot API request header constants.
	// These mimic the VS Code Copilot extension to ensure the upstream
	// API accepts the request without flagging it as an unknown client.
	copilotUserAgent     = "GitHubCopilotChat/0.35.0"
	copilotEditorVersion = "vscode/1.107.0"
	copilotPluginVersion = "copilot-chat/0.35.0"
	copilotIntegrationID = "vscode-chat"
	copilotOpenAIIntent  = "conversation-edits"
	copilotGitHubAPIVer  = "2025-04-01"

	// copilotChatPath and copilotResponsesPath are the upstream endpoint paths.
	copilotChatPath      = "/chat/completions"
	copilotResponsesPath = "/responses"

	// copilotMaxUpstreamResponseSize caps the non-streaming response body
	// size to prevent unbounded memory allocation (50 MB).
	copilotMaxUpstreamResponseSize = 50 << 20

	// copilotMaxSSELineSize is the maximum buffer for a single SSE line (20 MB).
	copilotMaxSSELineSize = 20 << 20
)

// CopilotGatewayService handles forwarding requests to the GitHub Copilot API.
//
// Copilot uses the OpenAI-compatible API format (/chat/completions, /responses)
// but requires a two-phase authentication flow:
//  1. Exchange a GitHub OAuth access_token for a short-lived Copilot API JWT
//  2. Send requests to api.githubcopilot.com with the JWT and special headers
type CopilotGatewayService struct {
	httpUpstream     HTTPUpstream
	copilotTokenProv *CopilotTokenProvider
	rateLimitService *RateLimitService
	cfg              *config.Config
}

// NewCopilotGatewayService creates a new CopilotGatewayService.
func NewCopilotGatewayService(
	httpUpstream HTTPUpstream,
	copilotTokenProv *CopilotTokenProvider,
	rateLimitService *RateLimitService,
	cfg *config.Config,
) *CopilotGatewayService {
	return &CopilotGatewayService{
		httpUpstream:     httpUpstream,
		copilotTokenProv: copilotTokenProv,
		rateLimitService: rateLimitService,
		cfg:              cfg,
	}
}

// CopilotForwardResult is the result of a successful forward to Copilot upstream.
type CopilotForwardResult struct {
	StatusCode    int
	UpstreamModel string
}

// ForwardChatCompletions forwards a request to the Copilot /chat/completions endpoint.
func (s *CopilotGatewayService) ForwardChatCompletions(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
) (*CopilotForwardResult, error) {
	return s.forward(ctx, c, account, body, copilotChatPath)
}

// ForwardResponses forwards a request to the Copilot /responses endpoint.
func (s *CopilotGatewayService) ForwardResponses(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
) (*CopilotForwardResult, error) {
	return s.forward(ctx, c, account, body, copilotResponsesPath)
}

// forward is the core forwarding logic shared by all Copilot endpoints.
func (s *CopilotGatewayService) forward(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	upstreamPath string,
) (*CopilotForwardResult, error) {
	reqLog := logger.L().With(
		zap.String("service", "copilot_gateway"),
		zap.Int64("account_id", account.ID),
		zap.String("account_name", account.Name),
	)

	// 1. Extract the GitHub access_token from account credentials.
	githubToken := account.GetGitHubToken()
	if githubToken == "" {
		return nil, fmt.Errorf("copilot: account %d has no github_token in credentials", account.ID)
	}

	// 2. Exchange GitHub token for Copilot API token (cached).
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	apiToken, apiEndpoint, err := s.copilotTokenProv.GetCopilotAPIToken(ctx, githubToken, proxyURL)
	if err != nil {
		reqLog.Error("copilot: token exchange failed", zap.Error(err))
		return nil, fmt.Errorf("copilot: token exchange failed: %w", err)
	}

	// 3. Extract request metadata.
	reqModel := gjson.GetBytes(body, "model").String()
	reqStream := gjson.GetBytes(body, "stream").Bool()
	reqLog = reqLog.With(zap.String("model", reqModel), zap.Bool("stream", reqStream))

	// 4. Build upstream URL.
	// Honor account-level base_url override; fall back to the endpoint
	// returned by the token exchange (or its default).
	baseURL := strings.TrimSpace(account.GetCopilotBaseURL())
	if baseURL == "" || baseURL == copilotDefaultBaseURL {
		baseURL = apiEndpoint
	}
	targetURL := strings.TrimRight(baseURL, "/") + upstreamPath

	// 5. Build upstream HTTP request.
	upstreamReq, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create upstream request: %w", err)
	}
	applyCopilotHeaders(upstreamReq, apiToken, body)

	// 6. Send request to upstream.
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return nil, fmt.Errorf("copilot: upstream request failed: %w", err)
	}

	// 7. Handle error responses.
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		_ = resp.Body.Close()

		// Let rate limit service handle upstream errors for this account.
		if s.rateLimitService != nil {
			resp.Body = io.NopCloser(bytes.NewReader(respBody))
			s.rateLimitService.HandleUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
		}

		reqLog.Warn("copilot: upstream error",
			zap.Int("status", resp.StatusCode),
			zap.String("body_preview", copilotTruncateString(string(respBody), 500)),
		)

		// Return a failover-compatible error for 401/429/5xx.
		return &CopilotForwardResult{StatusCode: resp.StatusCode}, &CopilotUpstreamError{
			StatusCode: resp.StatusCode,
			Body:       respBody,
		}
	}

	// 8. Forward the successful response to the client.
	if reqStream {
		s.streamResponse(c, resp, reqLog)
	} else {
		if err := s.nonStreamResponse(c, resp, reqLog); err != nil {
			return nil, err
		}
	}

	return &CopilotForwardResult{
		StatusCode:    resp.StatusCode,
		UpstreamModel: reqModel,
	}, nil
}

// streamResponse streams SSE data from the upstream response to the client.
func (s *CopilotGatewayService) streamResponse(c *gin.Context, resp *http.Response, reqLog *zap.Logger) {
	defer func() {
		if err := resp.Body.Close(); err != nil {
			reqLog.Warn("copilot: close upstream response body error", zap.Error(err))
		}
	}()

	// Copy relevant response headers.
	for key, values := range resp.Header {
		lk := strings.ToLower(key)
		if lk == "content-length" || lk == "transfer-encoding" || lk == "connection" {
			continue
		}
		for _, v := range values {
			c.Header(key, v)
		}
	}
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Status(resp.StatusCode)

	flusher, _ := c.Writer.(http.Flusher)
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), copilotMaxSSELineSize)

	for scanner.Scan() {
		line := scanner.Bytes()
		_, _ = c.Writer.Write(line)
		_, _ = c.Writer.Write([]byte("\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		reqLog.Warn("copilot: stream scanner error", zap.Error(err))
	}
}

// nonStreamResponse reads the full upstream response and writes it to the client.
func (s *CopilotGatewayService) nonStreamResponse(c *gin.Context, resp *http.Response, reqLog *zap.Logger) error {
	defer func() {
		if err := resp.Body.Close(); err != nil {
			reqLog.Warn("copilot: close upstream response body error", zap.Error(err))
		}
	}()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, copilotMaxUpstreamResponseSize))
	if err != nil {
		reqLog.Error("copilot: failed to read upstream response", zap.Error(err))
		return fmt.Errorf("copilot: failed to read upstream response: %w", err)
	}

	for key, values := range resp.Header {
		lk := strings.ToLower(key)
		if lk == "content-length" || lk == "transfer-encoding" || lk == "connection" {
			continue
		}
		for _, v := range values {
			c.Header(key, v)
		}
	}
	c.Data(resp.StatusCode, "application/json", respBody)
	return nil
}

// ShouldFailoverCopilotUpstreamError returns true if the upstream error
// should trigger account failover (try the next available Copilot account).
func ShouldFailoverCopilotUpstreamError(statusCode int) bool {
	switch statusCode {
	case 401, 402, 403, 429:
		return true
	default:
		return statusCode >= 500
	}
}

// ListModels fetches the available models from the Copilot /models endpoint.
func (s *CopilotGatewayService) ListModels(ctx context.Context, account *Account) (json.RawMessage, error) {
	githubToken := account.GetGitHubToken()
	if githubToken == "" {
		return nil, fmt.Errorf("copilot: account has no github_token")
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	apiToken, apiEndpoint, err := s.copilotTokenProv.GetCopilotAPIToken(ctx, githubToken, proxyURL)
	if err != nil {
		return nil, fmt.Errorf("copilot: token exchange failed: %w", err)
	}

	// Honor account-level base_url override.
	baseURL := strings.TrimSpace(account.GetCopilotBaseURL())
	if baseURL == "" || baseURL == copilotDefaultBaseURL {
		baseURL = apiEndpoint
	}
	modelsURL := strings.TrimRight(baseURL, "/") + "/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create models request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiToken)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", copilotUserAgent)
	req.Header.Set("Editor-Version", copilotEditorVersion)
	req.Header.Set("Editor-Plugin-Version", copilotPluginVersion)

	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return nil, fmt.Errorf("copilot: models request failed: %w", err)
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			logger.L().Warn("copilot: close models response body error", zap.Error(errClose))
		}
	}()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to read models response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("copilot: models request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	return json.RawMessage(respBody), nil
}

// applyCopilotHeaders sets all required headers for Copilot API requests.
func applyCopilotHeaders(r *http.Request, apiToken string, body []byte) {
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+apiToken)
	r.Header.Set("Accept", "application/json")
	r.Header.Set("User-Agent", copilotUserAgent)
	r.Header.Set("Editor-Version", copilotEditorVersion)
	r.Header.Set("Editor-Plugin-Version", copilotPluginVersion)
	r.Header.Set("Openai-Intent", copilotOpenAIIntent)
	r.Header.Set("Copilot-Integration-Id", copilotIntegrationID)
	r.Header.Set("X-Github-Api-Version", copilotGitHubAPIVer)
	r.Header.Set("X-Request-Id", uuid.NewString())

	// Determine X-Initiator header for Copilot billing:
	//   "user"  → consumes premium request quota
	//   "agent" → free (tool loops, continuations)
	initiator := "user"
	if isCopilotAgentInitiated(body) {
		initiator = "agent"
	}
	r.Header.Set("X-Initiator", initiator)

	// Vision detection
	if detectCopilotVisionContent(body) {
		r.Header.Set("Copilot-Vision-Request", "true")
	}
}

// isCopilotAgentInitiated detects whether the request is agent-initiated
// (tool callback / continuation) rather than user-initiated. Copilot uses
// the X-Initiator header for billing purposes.
func isCopilotAgentInitiated(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	// Chat Completions API: check messages array.
	if messages := gjson.GetBytes(body, "messages"); messages.Exists() && messages.IsArray() {
		arr := messages.Array()
		if len(arr) == 0 {
			return false
		}
		// Find the last message role.
		for i := len(arr) - 1; i >= 0; i-- {
			role := arr[i].Get("role").String()
			if role == "assistant" || role == "tool" {
				return true
			}
			if role == "user" {
				// Check if it contains tool_result content (indicating a tool loop).
				content := arr[i].Get("content")
				if content.IsArray() {
					for _, part := range content.Array() {
						if part.Get("type").String() == "tool_result" {
							return true
						}
					}
				}
				return false
			}
		}
	}

	// Responses API: check input array.
	if inputs := gjson.GetBytes(body, "input"); inputs.Exists() && inputs.IsArray() {
		arr := inputs.Array()
		if len(arr) == 0 {
			return false
		}
		last := arr[len(arr)-1]
		if last.Get("role").String() == "assistant" {
			return true
		}
		switch last.Get("type").String() {
		case "function_call", "function_call_output", "function_call_response":
			return true
		}
	}

	return false
}

// detectCopilotVisionContent checks if the request body contains image content.
func detectCopilotVisionContent(body []byte) bool {
	messages := gjson.GetBytes(body, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return false
	}
	for _, msg := range messages.Array() {
		content := msg.Get("content")
		if content.IsArray() {
			for _, block := range content.Array() {
				blockType := block.Get("type").String()
				if blockType == "image_url" || blockType == "image" {
					return true
				}
			}
		}
	}
	return false
}

// CopilotUpstreamError represents an error response from the Copilot upstream.
type CopilotUpstreamError struct {
	StatusCode int
	Body       []byte
}

func (e *CopilotUpstreamError) Error() string {
	return fmt.Sprintf("copilot upstream error: status %d", e.StatusCode)
}

// copilotTruncateString truncates s to maxLen, appending "..." if truncated.
func copilotTruncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
