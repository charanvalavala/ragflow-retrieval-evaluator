package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestRetrieveUsesBearerAuthAndTopFiveRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/v1/retrieval" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization header missing")
		}
		var payload retrievePayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if payload.PageSize != resultLimit || payload.DatasetIDs[0] != "dataset-1" {
			t.Errorf("unexpected retrieval parameters: %+v", payload)
		}
		chunks := make([]Chunk, 7)
		for index := range chunks {
			chunks[index] = Chunk{Content: "evidence"}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"code": 0, "data": map[string]any{"chunks": chunks}})
	}))
	defer server.Close()

	client, err := newEvaluator(server.URL, "test-key", "dataset-1")
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := client.retrieve(context.Background(), "What is the interface?")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != resultLimit {
		t.Fatalf("got %d chunks, want %d", len(chunks), resultLimit)
	}
}

func TestNewEvaluatorRejectsCredentialsInURLAndMissingDataset(t *testing.T) {
	if _, err := newEvaluator("https://user:pass@example.com", "key", "dataset"); err == nil {
		t.Fatal("expected URL credentials to be rejected")
	}
	if _, err := newEvaluator("http://localhost:9380", "key", ""); err == nil {
		t.Fatal("expected missing dataset id to be rejected")
	}
}

func TestMatchingRankRequiresEveryEvidenceGroup(t *testing.T) {
	groups := [][]string{{"2.4 ghz", "2.4ghz"}, {"cortex-a76"}}
	chunks := []Chunk{{Content: "A Cortex-A76 processor"}, {Content: "2.4 GHz quad-core Cortex-A76"}}
	if got := matchingRank(chunks, groups); got != 2 {
		t.Fatalf("rank = %d, want 2", got)
	}
	if evidenceMatch("2.4 GHz processor", groups) {
		t.Fatal("partial evidence unexpectedly matched")
	}
}

func TestEvidenceMatchIsCaseInsensitive(t *testing.T) {
	if !evidenceMatch("Dual micro HDMI outputs support 4Kp60", [][]string{{"micro HDMI"}, {"4Kp60"}}) {
		t.Fatal("expected case-insensitive evidence match")
	}
}

func TestLoadCasesRejectsDuplicateIDs(t *testing.T) {
	temp := t.TempDir() + "/cases.jsonl"
	content := "{\"id\":\"same\",\"question\":\"q\",\"evidence_groups\":[[\"a\"]]}\n{\"id\":\"same\",\"question\":\"q2\",\"evidence_groups\":[[\"b\"]]}\n"
	if err := os.WriteFile(temp, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadCases(temp); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("got error %v, expected duplicate id error", err)
	}
}
