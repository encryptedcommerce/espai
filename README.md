# espai (Enumerate, Search, Parse, and Iterate)

A powerful tool for structured data extraction from search results using Google Search or Exa.ai with Gemini AI.

## Features

- Parse natural language queries into structured search parameters
- Automatically discover and enumerate search spaces
- Multiple search providers (Google Custom Search, Exa.ai)
- Extract structured data from search results using Gemini AI
- Store results in efficient Polars DataFrames
- Real-time progress tracking
- Multiple output formats (CSV, JSON, Parquet)

## Installation

### From PyPI
```bash
pip install espai
```

### From Source
```bash
git clone https://github.com/yourusername/espai.git
cd espai
poetry install
```

## API Key Setup

### Google Custom Search
1. Create a Google Custom Search Engine:
   - Go to [Google Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/all)
   - Click "Add" to create a new search engine
   - Under "Sites to search", select "Search the entire web" for unrestricted search
   - Click "Create"
   - Copy your "Search engine ID" (this will be your GOOGLE_CSE_ID)

2. Get a Google API Key:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the "Custom Search API" for your project
   - Go to "Credentials" and create an API key (this will be your GOOGLE_API_KEY)

3. Set environment variables:
```bash
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CSE_ID="your_search_engine_id"
```

### Exa.ai (Optional)
1. Sign up at [Exa.ai](https://exa.ai)
2. Get your API key from the dashboard
3. Set environment variable:
```bash
export EXA_API_KEY="your_exa_api_key"
```

### Gemini AI
1. Go to [Google AI Studio API Keys](https://aistudio.google.com/apikey)
2. Create an API key (no credit card required)
3. Set environment variable:
```bash
export GEMINI_API_KEY="your_gemini_api_key"
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
  --output-format=csv \
  --provider=exa
```

Available options:
- `--max-results`, `-n`: Maximum number of results per search (default: 10)
- `--output-format`, `-f`: Output format: csv, json, or parquet (default: csv)
- `--output-file`, `-o`: Output file path (default: results.[format])
- `--verbose`, `-v`: Show verbose output
- `--temperature`, `-t`: Temperature for LLM generation (0.0 to 1.0)
- `--provider`, `-p`: Search provider: google or exa (default: google)

## Search Provider Comparison

### Google Custom Search
- Pros:
  - More precise keyword matching
  - Better for finding specific attributes
  - Reliable and stable API
- Cons:
  - Limited to 100 queries per day on free tier
  - Maximum 10 results per query
  - Requires setting up a custom search engine

### Exa.ai
- Pros:
  - Better semantic understanding
  - Returns more context in results
  - Up to 100 results per query
  - No need to set up a search engine
- Cons:
  - May be less precise for specific attribute searches
  - API is newer and may change
  - Requires paid subscription for production use

## Example Queries

Find businesses:
```bash
espai "Coffee shop names and addresses in Seattle neighborhoods"
```

Research organizations:
```bash
espai "Research labs working on AI in Massachusetts universities"
```

Gather event information:
```bash
espai "Tech conferences and their dates in European cities in 2024"
```

## License

MIT License
