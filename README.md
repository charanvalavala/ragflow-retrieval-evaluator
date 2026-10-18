# RAGFlow Retrieval Evaluator

A standard-library Go command-line tool that measures whether a running
RAGFlow dataset returns chunks containing manually labeled evidence for six
questions about the Raspberry Pi 5 product brief. It reports evidence Hit@5
and MRR@5. It does not grade generated answers.

## Prepare a corpus

1. Run a RAGFlow instance and configure an embedding model.
2. Create a dataset and index the official [Raspberry Pi 5 product brief](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf).
3. Copy the dataset ID and create an API key.

The PDF is not included here. The benchmark labels refer to the facts in that
official document and should be reviewed if the document changes.

## Run

Requires Go 1.22 or later.

```powershell
$env:RAGFLOW_BASE_URL = "http://localhost:9380"
$env:RAGFLOW_API_KEY = "your-api-key"
$env:RAGFLOW_DATASET_ID = "your-dataset-id"
go test ./...
go vet ./...
go run .
```

The evaluator calls the documented retrieval endpoint at
`POST /api/v1/retrieval`. It requests five chunks with a fixed configuration.
The API key is read only from the environment and is not printed.

## Metrics and limits

A retrieved chunk counts as an evidence match only if it contains at least one
anchor from every labeled evidence group for that case. Hit@5 measures the
fraction of questions with a matching chunk in the first five results. MRR@5
rewards the rank of the first match. These lexical anchor checks are narrow;
they do not establish answer truth, citation quality, or semantic equivalence.
Keep the dataset, chunking, embedding model, and retrieval settings alongside
any score. The benchmark does not publish a live result until run against a
configured indexed dataset.

## API and license

The client follows the [RAGFlow HTTP API reference](https://ragflow.io/docs/v1.0.0-rc2/http_api_reference).
The product brief is published by Raspberry Pi. This evaluator is MIT licensed.
