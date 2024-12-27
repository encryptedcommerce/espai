# espai

Enumerate, Search, Parse, and Iterate - A tool for structured data extraction from search results

## Overview

Espai is a command-line tool that combines search engines, web scraping, and LLMs to extract structured data from the web. It can find entities (like companies, schools, or government agencies) and extract specific attributes about them (like websites, phone numbers, or addresses).

## Architecture

Espai is built with a modular architecture that combines several components:

### 1. Query Understanding
- Uses Google's Gemini LLM to parse natural language queries into:
  - Entity type (what we're looking for)
  - Attributes (what information to extract)
  - Search space (geographic or domain constraints)
- Example: "find tech companies in california with their websites and phone numbers"
  - Entity: company
  - Attributes: website, phone
  - Search space: california

### 2. Search Providers
- Pluggable search provider interface supporting multiple backends:
  - Google Custom Search API
  - Exa.ai API (with neural search)
- Each provider returns normalized SearchResult objects containing:
  - Title
  - Snippet
  - URL
  - Domain
  - Published date

### 3. Entity Extraction
- Two-pass approach:
  1. First pass: Find entities matching the query
     - Uses search results to identify entity names
     - Deduplicates entities based on normalized names
  2. Second pass: Extract requested attributes
     - Uses targeted searches for each attribute
     - Scrapes web pages when needed
     - Updates existing entities with new information

### 4. Web Scraping
- Asynchronous web scraping with httpx
- Robust text extraction:
  - Handles multiple character encodings
  - Removes irrelevant HTML elements
  - Cleans and normalizes text
  - Truncates long content for LLM processing

### 5. LLM Processing
- Uses Gemini for multiple tasks:
  - Query parsing
  - Entity name extraction
  - Attribute extraction
  - Search space enumeration
- Custom prompts ensure consistent and clean output

### 6. Data Management
- Stores results in EntityResult objects with fields for:
  - Name
  - Search space
  - Website
  - Phone
  - Email
  - Address components
- Supports multiple output formats:
  - CSV
  - JSON
  - Parquet

## How It Works

1. **Query Processing**:
   - User inputs a natural language query
   - Gemini parses it into structured components
   - Search space is enumerated if needed (e.g., "all New England states")

2. **Entity Discovery**:
   - Primary search provider finds potential entities
   - Results are processed to extract entity names
   - Entities are deduplicated and normalized

3. **Attribute Enrichment**:
   - For each entity, missing attributes are identified
   - Targeted searches find attribute information
   - Web pages are scraped when needed
   - LLM extracts structured data from text

4. **Result Management**:
   - Results are continuously updated and deduplicated
   - Progress is shown in real-time
   - Results can be saved even if interrupted
   - Output is formatted according to user preference

## Features

- Natural language query interface
- Multiple search provider support (Google and Exa.ai)
- Asynchronous operation for better performance
- Robust error handling and recovery
- Clean shutdown with result saving
- Multiple output formats (CSV, JSON, Parquet)
- Progress tracking with rich console output
- Verbose mode for debugging

## Installation

```bash
pip install espai
```

Or with Poetry:
```bash
poetry add espai
```

## Configuration

You'll need to set up the following environment variables:

```bash
# Required for Google Custom Search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Required for Exa.ai search
EXA_API_KEY=your_exa_api_key

# Required for Gemini AI
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

Basic usage:
```bash
espai "tech companies in San Francisco with their websites"
```

With options:
```bash
espai "department of education websites for all US states" \
  --max-results 20 \
  --output-format json \
  --output-file results.json \
  --verbose
```

Available options:
- `--max-results, -n`: Maximum results per search (default: 10)
- `--output-format, -f`: Output format: csv, json, or parquet (default: csv)
- `--output-file, -o`: Output file (default: results.[format])
- `--verbose, -v`: Show verbose output
- `--provider, -p`: Search provider: google or exa (default: google)
- `--temperature, -t`: Temperature for LLM generation (default: 0.7)

## Example Queries

Find companies:
```bash
espai "tech startups in Boston with websites and phone numbers"
```

Find organizations:
```bash
espai "environmental nonprofits in California with email addresses"
```

Find people:
```bash
espai "state governors with their official websites"
```

Find locations:
```bash
espai "national parks in Utah with visitor center addresses"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
