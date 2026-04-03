package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

const (
	// copilotTokenExchangeURL is the GitHub API endpoint for exchanging
	// a GitHub OAuth access_token into a short-lived Copilot API JWT.
	copilotTokenExchangeURL = "https://api.github.com/copilot_internal/v2/token"

	// copilotDefaultBaseURL is the default Copilot API endpoint.
	copilotDefaultBaseURL = "https://api.githubcopilot.com"

	// copilotTokenCacheTTL is the default cache duration when the token
	// response does not include an explicit expires_at field.
	copilotTokenCacheTTL = 25 * time.Minute

	// copilotTokenExpiryBuffer is the time before expiry at which we proactively
	// refresh the cached token. This avoids sending requests with a token that
	// is about to expire within seconds.
	copilotTokenExpiryBuffer = 5 * time.Minute

	// copilotTokenExchangeUserAgent mimics the VS Code Copilot extension.
	copilotTokenExchangeUserAgent = "GithubCopilot/1.0"
	copilotTokenEditorVersion     = "vscode/1.100.0"
	copilotTokenPluginVersion     = "copilot/1.300.0"

	// maxCopilotTokenResponseSize caps the response body from the token
	// exchange endpoint to prevent unbounded memory allocation (1 MB).
	maxCopilotTokenResponseSize = 1 << 20
)

// copilotAPITokenResponse represents the JSON response returned by
// the Copilot token exchange endpoint.
type copilotAPITokenResponse struct {
	Token     string `json:"token"`
	ExpiresAt int64  `json:"expires_at"`
	Endpoints struct {
		API string `json:"api"`
	} `json:"endpoints,omitempty"`
	ErrorDetails *struct {
		Message string `json:"message"`
	} `json:"error_details,omitempty"`
}

// cachedCopilotToken stores a Copilot API token together with its expiry
// and the resolved API endpoint.
type cachedCopilotToken struct {
	token       string
	apiEndpoint string
	expiresAt   time.Time
}

// CopilotTokenProvider manages the exchange of GitHub OAuth access_tokens
// for short-lived Copilot API JWTs. Tokens are cached per access_token
// (keyed by SHA-256 fingerprint) and automatically refreshed before expiry.
//
// The exchange flow follows the same protocol that the VS Code Copilot
// extension uses:
//
//	GET https://api.github.com/copilot_internal/v2/token
//	Authorization: token <github_access_token>
//	→ { "token": "<jwt>", "expires_at": <unix>, "endpoints": { "api": "..." } }
type CopilotTokenProvider struct {
	httpUpstream HTTPUpstream
	mu           sync.RWMutex
	cache        map[string]*cachedCopilotToken // keyed by SHA-256 of github token
	sf           singleflight.Group
}

// NewCopilotTokenProvider creates a new provider instance.
func NewCopilotTokenProvider(httpUpstream HTTPUpstream) *CopilotTokenProvider {
	return &CopilotTokenProvider{
		httpUpstream: httpUpstream,
		cache:        make(map[string]*cachedCopilotToken),
	}
}

// tokenFingerprint returns a SHA-256 hex digest of the token, used as
// the cache key to avoid storing raw credentials in memory.
func tokenFingerprint(token string) string {
	h := sha256.Sum256([]byte(token))
	return hex.EncodeToString(h[:])
}

// GetCopilotAPIToken returns a valid Copilot API token and the resolved
// API endpoint for the given GitHub access_token. Cached tokens are reused
// until they are within copilotTokenExpiryBuffer of expiry.
//
// Uses singleflight to deduplicate concurrent exchange requests for the
// same access_token, preventing thundering herd on cache miss/expiry.
func (p *CopilotTokenProvider) GetCopilotAPIToken(ctx context.Context, githubAccessToken string, proxyURL string) (token string, apiEndpoint string, err error) {
	githubAccessToken = strings.TrimSpace(githubAccessToken)
	if githubAccessToken == "" {
		return "", "", fmt.Errorf("copilot: github access token is empty")
	}

	cacheKey := tokenFingerprint(githubAccessToken)

	// Fast path: read-lock check for cached token.
	p.mu.RLock()
	if cached, ok := p.cache[cacheKey]; ok && cached.expiresAt.After(time.Now().Add(copilotTokenExpiryBuffer)) {
		p.mu.RUnlock()
		return cached.token, cached.apiEndpoint, nil
	}
	p.mu.RUnlock()

	// Slow path: use singleflight to deduplicate concurrent exchanges
	// for the same token.
	type tokenResult struct {
		token       string
		apiEndpoint string
	}

	val, err, _ := p.sf.Do(cacheKey, func() (any, error) {
		// Double-check under singleflight: another goroutine may have
		// already refreshed the cache while we were waiting.
		p.mu.RLock()
		if cached, ok := p.cache[cacheKey]; ok && cached.expiresAt.After(time.Now().Add(copilotTokenExpiryBuffer)) {
			p.mu.RUnlock()
			return &tokenResult{token: cached.token, apiEndpoint: cached.apiEndpoint}, nil
		}
		p.mu.RUnlock()

		// Exchange token.
		apiToken, exchangeErr := p.exchangeToken(ctx, githubAccessToken, proxyURL)
		if exchangeErr != nil {
			return nil, exchangeErr
		}

		resolvedEndpoint := copilotDefaultBaseURL
		if ep := strings.TrimRight(apiToken.Endpoints.API, "/"); ep != "" {
			resolvedEndpoint = ep
		}

		expiresAt := time.Now().Add(copilotTokenCacheTTL)
		if apiToken.ExpiresAt > 0 {
			expiresAt = time.Unix(apiToken.ExpiresAt, 0)
		}

		// Update cache and evict expired entries.
		p.mu.Lock()
		p.cache[cacheKey] = &cachedCopilotToken{
			token:       apiToken.Token,
			apiEndpoint: resolvedEndpoint,
			expiresAt:   expiresAt,
		}
		// Opportunistic eviction of expired entries to prevent unbounded growth.
		now := time.Now()
		for k, v := range p.cache {
			if v.expiresAt.Before(now) {
				delete(p.cache, k)
			}
		}
		p.mu.Unlock()

		return &tokenResult{token: apiToken.Token, apiEndpoint: resolvedEndpoint}, nil
	})

	if err != nil {
		return "", "", err
	}

	result, ok := val.(*tokenResult)
	if !ok {
		return "", "", fmt.Errorf("copilot: unexpected token result type")
	}
	return result.token, result.apiEndpoint, nil
}

// exchangeToken performs the actual HTTP request to exchange a GitHub
// access_token for a Copilot API token.
func (p *CopilotTokenProvider) exchangeToken(ctx context.Context, githubAccessToken string, proxyURL string) (*copilotAPITokenResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, copilotTokenExchangeURL, nil)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create token exchange request: %w", err)
	}

	req.Header.Set("Authorization", "token "+githubAccessToken)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", copilotTokenExchangeUserAgent)
	req.Header.Set("Editor-Version", copilotTokenEditorVersion)
	req.Header.Set("Editor-Plugin-Version", copilotTokenPluginVersion)

	resp, err := p.httpUpstream.Do(req, proxyURL, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("copilot: token exchange request failed: %w", err)
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			logger.L().Warn("copilot: close token exchange response body error", zap.Error(errClose))
		}
	}()

	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, maxCopilotTokenResponseSize))
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to read token exchange response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("copilot: token exchange failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var apiToken copilotAPITokenResponse
	if err = json.Unmarshal(bodyBytes, &apiToken); err != nil {
		return nil, fmt.Errorf("copilot: failed to parse token exchange response: %w", err)
	}

	if apiToken.Token == "" {
		errMsg := "empty copilot api token"
		if apiToken.ErrorDetails != nil && apiToken.ErrorDetails.Message != "" {
			errMsg = apiToken.ErrorDetails.Message
		}
		return nil, fmt.Errorf("copilot: token exchange returned error: %s", errMsg)
	}

	return &apiToken, nil
}
