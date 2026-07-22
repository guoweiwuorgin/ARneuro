# GLM-4.5-Air Batch test

This folder tests Methods information extraction on 100 valid papers using the
BigModel Batch API. It uses the same prompt, `typical_human_*` schema, short
Methods supplementation, and normalization as the serial extraction pipeline.

Official documentation:

https://docs.bigmodel.cn/cn/guide/tools/batch

## Install the Batch SDK

```powershell
conda run -n reviewer pip install -U zai-sdk
```

The scripts use `BIGMODEL_API_KEY` when it is set. Otherwise they use the test
key configured in `common.py`.

```powershell
$env:BIGMODEL_API_KEY="your-api-key"
```

## 1. Prepare 100 requests

This step is offline:

```powershell
conda run --no-capture-output -n reviewer python -B `
  .\ARneuro\examples\glm45_air_batch_test\01_prepare_batch_input.py
```

Outputs:

- `output/batch_requests_100.jsonl`
- `output/batch_manifest_100.json`

Inspect the JSONL before submission. Every line has a unique
`custom_id=paper-{PMID}`.

## 2. Upload and create the Batch

```powershell
conda run --no-capture-output -n reviewer python -B `
  .\ARneuro\examples\glm45_air_batch_test\02_submit_batch.py
```

The returned file ID and Batch ID are stored in `output/batch_job.json`.
The script refuses to submit another task while this state file exists.

## 3. Monitor and download

Check once:

```powershell
conda run --no-capture-output -n reviewer python -B `
  .\ARneuro\examples\glm45_air_batch_test\03_monitor_and_download.py
```

Wait until completion and download automatically:

```powershell
conda run --no-capture-output -n reviewer python -B `
  .\ARneuro\examples\glm45_air_batch_test\03_monitor_and_download.py --wait
```

The official terminal statuses are `completed`, `failed`, `expired`, and
`cancelled`. Successful and failed request files are downloaded separately.

## 4. Parse results

```powershell
conda run --no-capture-output -n reviewer python -B `
  .\ARneuro\examples\glm45_air_batch_test\04_parse_batch_results.py
```

Outputs:

- `output/parsed_json/paper_{PMID}_method_info.json`
- `output/method_info_table.csv`
- `output/parse_report.json`
- `output/retry_requests.jsonl`

`retry_requests.jsonl` contains only API failures, missing responses, and model
outputs that could not be parsed as JSON. It can be submitted as a separate
repair Batch after review.

## Notes

- The default model identifier is `glm-4.5-air`.
- The Batch input endpoint is `/v4/chat/completions`.
- This test does not modify the serial full-run output directory.
- Batch result order is not assumed; all matching uses `custom_id`.
- Do not delete the local input JSONL or manifest until parsing is complete.
