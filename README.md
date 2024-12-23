# espai (Enumerate, Search, Parse, and Iterate)

A powerful tool for structured data extraction from search results using Google Search and Gemini AI.

## Features

- Parse natural language queries into structured search parameters
- Automatically discover and enumerate search spaces
- Perform intelligent Google searches
- Extract structured data from search results using Gemini AI
- Store results in efficient Polars DataFrames
- Real-time progress tracking
- Multiple output formats (CSV, JSON, Parquet)

## Installation

```bash
pip install espai
```

## Usage

Basic usage:
```bash
espai "Athletic center names and addresses in all California zip codes"
```

With options:
```bash
espai "Athletic center names and addresses in all California zip codes" \
  --max-results=10 \
  --output-format=csv
```

## Configuration

Set your API keys as environment variables:
```bash
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CSE_ID="your_google_cse_id"
export GEMINI_API_KEY="your_gemini_api_key"
```

## License

MIT License
