package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const resultLimit = 5

type Case struct {
	ID      string     `json:"id"`
	Query   string     `json:"question"`
	Anchors [][]string `json:"evidence_groups"`
}

type Chunk struct {
	Content         string  `json:"content"`
	DocumentKeyword string  `json:"document_keyword"`
	Similarity      float64 `json:"similarity"`
}

type retrievePayload struct {
	Question                 string   `json:"question"`
	DatasetIDs               []string `json:"dataset_ids"`
	Page                     int      `json:"page"`
	PageSize                 int      `json:"page_size"`
	SimilarityThreshold      float64  `json:"similarity_threshold"`
	KeywordsSimilarityWeight float64  `json:"keywords_similarity_weight"`
	KNNTopK                  int      `json:"knn_top_k"`
	KNNNumCandidates         int      `json:"knn_num_candidates"`
	Highlight                bool     `json:"highlight"`
}

type retrieveEnvelope struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    struct {
		Chunks []Chunk `json:"chunks"`
	} `json:"data"`
}

type evaluator struct {
	baseURL   string
	apiKey    string
	datasetID string
	http      *http.Client
}

func newEvaluator(baseURL, apiKey, datasetID string) (*evaluator, error) {
	parsed, err := url.Parse(strings.TrimRight(baseURL, "/"))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, errors.New("RAGFLOW_BASE_URL must be an absolute http or https URL")
	}
	if parsed.User != nil {
		return nil, errors.New("credentials must not be embedded in RAGFLOW_BASE_URL")
	}
	if strings.TrimSpace(apiKey) == "" || strings.TrimSpace(datasetID) == "" {
		return nil, errors.New("RAGFLOW_API_KEY and RAGFLOW_DATASET_ID are required")
	}
	return &evaluator{
		baseURL: parsed.String(), apiKey: apiKey, datasetID: datasetID,
		http: &http.Client{Timeout: 45 * time.Second},
	}, nil
}

func (e *evaluator) retrieve(ctx context.Context, query string) ([]Chunk, error) {
	payload := retrievePayload{
		Question: query, DatasetIDs: []string{e.datasetID}, Page: 1, PageSize: resultLimit,
		SimilarityThreshold: 0, KeywordsSimilarityWeight: 0.5,
		KNNTopK: 64, KNNNumCandidates: 128, Highlight: false,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/api/v1/retrieval", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("retrieval request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("RAGFlow returned HTTP %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}
	var envelope retrieveEnvelope
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&envelope); err != nil {
		return nil, fmt.Errorf("decode retrieval response: %w", err)
	}
	if envelope.Code != 0 {
		return nil, fmt.Errorf("RAGFlow API error %d: %s", envelope.Code, envelope.Message)
	}
	if len(envelope.Data.Chunks) > resultLimit {
		envelope.Data.Chunks = envelope.Data.Chunks[:resultLimit]
	}
	return envelope.Data.Chunks, nil
}

func loadCases(path string) ([]Case, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open benchmark file: %w", err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	var cases []Case
	seen := map[string]struct{}{}
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var item Case
		if err := json.Unmarshal(line, &item); err != nil {
			return nil, fmt.Errorf("decode benchmark row %d: %w", len(cases)+1, err)
		}
		if item.ID == "" || item.Query == "" || len(item.Anchors) == 0 {
			return nil, fmt.Errorf("benchmark row %d needs id, question, and evidence groups", len(cases)+1)
		}
		if _, exists := seen[item.ID]; exists {
			return nil, fmt.Errorf("duplicate benchmark id %q", item.ID)
		}
		seen[item.ID] = struct{}{}
		cases = append(cases, item)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read benchmark file: %w", err)
	}
	if len(cases) == 0 {
		return nil, errors.New("benchmark file has no cases")
	}
	return cases, nil
}

func evidenceMatch(content string, groups [][]string) bool {
	content = strings.ToLower(content)
	for _, group := range groups {
		matched := false
		for _, alternative := range group {
			if alternative != "" && strings.Contains(content, strings.ToLower(alternative)) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return len(groups) > 0
}

func matchingRank(chunks []Chunk, groups [][]string) int {
	for index, chunk := range chunks {
		if evidenceMatch(chunk.Content, groups) {
			return index + 1
		}
	}
	return 0
}

func run(ctx context.Context, output io.Writer) error {
	client, err := newEvaluator(os.Getenv("RAGFLOW_BASE_URL"), os.Getenv("RAGFLOW_API_KEY"), os.Getenv("RAGFLOW_DATASET_ID"))
	if err != nil {
		return err
	}
	cases, err := loadCases("testdata/rpi5-evidence.jsonl")
	if err != nil {
		return err
	}
	hits := 0
	var reciprocalRank float64
	for _, item := range cases {
		chunks, err := client.retrieve(ctx, item.Query)
		if err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
		rank := matchingRank(chunks, item.Anchors)
		if rank == 0 {
			fmt.Fprintf(output, "MISS %-22s no evidence-matching chunk in top %d\n", item.ID, resultLimit)
			continue
		}
		hits++
		reciprocalRank += 1 / float64(rank)
		fmt.Fprintf(output, "HIT  %-22s rank=%d\n", item.ID, rank)
	}
	fmt.Fprintf(output, "\nCases: %d\nEvidence Hit@%d: %.3f (%d/%d)\n", len(cases), resultLimit, float64(hits)/float64(len(cases)), hits, len(cases))
	fmt.Fprintf(output, "Evidence MRR@%d: %.3f\n", resultLimit, reciprocalRank/float64(len(cases)))
	fmt.Fprintln(output, "Scope: retrieved evidence anchors, not generated-answer correctness.")
	return nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	if err := run(ctx, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "ragflow-eval:", err)
		os.Exit(1)
	}
}
