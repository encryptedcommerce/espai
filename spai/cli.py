"""Command line interface for SPAI."""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import polars as pl
import rich
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .gemini_client import GeminiClient
from .models import SearchQuery, EntityResult
from .search_client import GoogleSearchClient

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="spai",
    help="""
    SPAI (Search, Parse, and Iterate) - A tool for structured data extraction from search results.
    
    This tool uses Google Search and Gemini AI to:
    1. Parse natural language queries into structured components
    2. Search for relevant information
    3. Extract structured data from search results
    
    Environment Variables Required:
    - GOOGLE_API_KEY: API key for Google Custom Search
    - GOOGLE_CSE_ID: Custom Search Engine ID
    - GEMINI_API_KEY: API key for Gemini AI
    
    Example Usage:
        $ spai search "Find coffee shops with good ratings in Seattle"
        $ spai search "List gyms with their hours in New York" --max-results 5 --format csv
        $ spai search "Show me restaurants in San Francisco" --format json
    
    Note: Always wrap your query in quotes when it contains spaces:
        ✓ spai search "athletic center in arizona"
        ✗ spai search athletic center in arizona
    """,
    no_args_is_help=True,
)

console = Console()

def version_callback(value: bool):
    if value:
        console.print("SPAI version 0.1.0")
        raise typer.Exit()

@app.callback()
def main_callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    SPAI - Search, Parse, and Iterate.
    A tool for structured data extraction from search results.
    """
    pass

@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help='Natural language query (wrap in quotes: "your query here")',
        metavar="QUERY",
    ),
    max_results: int = typer.Option(
        10,
        "--max-results", "-n",
        help="Maximum number of search results to process",
        min=1,
        max=50,
    ),
    output_format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, json, or csv",
        case_sensitive=False,
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path. If not specified, output is written to stdout.",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature", "-t",
        help="Temperature for Gemini AI (0.0 to 1.0). Higher values make output more creative",
        min=0.0,
        max=1.0,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed progress and debug information",
    ),
) -> None:
    """
    Search and extract structured data based on a natural language query.
    
    The query should be wrapped in quotes if it contains spaces:
        $ spai search "athletic center in arizona zip codes"
    
    Examples:
        $ spai search "Find coffee shops with good ratings in Seattle"
        $ spai search "List gyms with their hours in New York" --max-results 5 --format csv
        $ spai search "Show me restaurants in San Francisco" --format json
    """
    try:
        # Run the async main function
        asyncio.run(async_main(
            query=query,
            max_results=max_results,
            output_format=output_format,
            output_file=output_file,
            temperature=temperature,
            verbose=verbose,
        ))
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

async def async_main(
    query: str,
    max_results: int,
    output_format: str,
    output_file: Optional[str],
    temperature: float,
    verbose: bool,
) -> None:
    """Main async function to handle the search and extraction process."""
    
    if verbose:
        console.print("[yellow]Initializing clients...[/yellow]")
    
    # Initialize clients
    gemini_client = GeminiClient(temperature=temperature, verbose=verbose)
    search_client = GoogleSearchClient(max_results=max_results)
    
    # Parse the query
    if verbose:
        console.print("[yellow]Parsing query...[/yellow]")
    query_struct: SearchQuery = await gemini_client.parse_query(query)
    
    if verbose:
        console.print(f"[green]Parsed query:[/green]")
        console.print(f"- Entities: {query_struct.entities}")
        console.print(f"- Attributes: {query_struct.entity_attributes}")
        console.print(f"- Search Space: {query_struct.search_space}")
    
    # Get search results
    if verbose:
        console.print("[yellow]Searching...[/yellow]")
    search_results = await search_client.search(
        f"{query_struct.entities} in {query_struct.search_space}"
    )
    
    if not search_results:
        console.print("[red]No search results found[/red]")
        return
    
    # Process each result
    if verbose:
        console.print("[yellow]Processing search results...[/yellow]")
    
    results = []
    with console.status("[bold green]Extracting data...") as status:
        for result in search_results:
            if verbose:
                console.print(f"[blue]Processing: {result['title']}[/blue]")
            
            try:
                extracted = await gemini_client.parse_search_result(
                    result["snippet"],
                    query_struct.entity_attributes
                )
                results.append(extracted.to_flat_dict(set(query_struct.entity_attributes)))
            except Exception as e:
                if verbose:
                    console.print(f"[red]Failed to process result: {str(e)}[/red]")
    
    # Convert results to DataFrame
    if results:
        df = pl.DataFrame(results)
        
        # Preview results in table format (limited to 20 rows)
        if len(df) > 0:
            table = Table(show_header=True, header_style="bold magenta")
            
            # Add columns
            for col in df.columns:
                table.add_column(col)
            
            # Add rows (limited to 20)
            for row in df.head(20).rows():
                table.add_row(*[str(cell) for cell in row])
            
            console.print("\n[bold]Preview of results:[/bold]")
            console.print(table)
            
            if len(df) > 20:
                console.print(f"\n[dim]... and {len(df) - 20} more rows[/dim]")
        
        # Write output in requested format
        output_format = output_format.lower()
        if output_file:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format == "json":
                with open(output_file, "w") as f:
                    json.dump(df.to_dict(as_series=False), f, indent=2)
            elif output_format == "csv":
                df.write_csv(output_file)
            else:
                console.print(f"[red]Unsupported output format for file: {output_format}[/red]")
                return
            
            console.print(f"[green]Results written to: {output_file}[/green]")
        else:
            if output_format == "json":
                console.print_json(df.to_dict(as_series=False))
            elif output_format == "csv":
                console.print(df.write_csv())
            # Table format is already shown in preview
    else:
        console.print("[red]No structured data could be extracted[/red]")

# Expose the Typer app as main for the Poetry script
main = app

if __name__ == "__main__":
    app()
