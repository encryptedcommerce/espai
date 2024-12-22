"""Command line interface for SPAI."""

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import polars as pl
import rich
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID, TaskProgressColumn
from rich.table import Table

from .gemini_client import GeminiClient
from .models import EntityResult
from .search_client import GoogleSearchClient
from .scraper import Scraper  # Import Scraper class

# Load environment variables
load_dotenv()

console = Console()
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
        $ spai search "athletic center in arizona"
    """,
    no_args_is_help=True,
)

# Global variables for graceful shutdown
_current_results = []
_global_output_format = "csv"
_global_output_file = "results.csv"
should_shutdown = False

def signal_handler(signum, frame):
    """Handle Ctrl-C by setting shutdown flag and writing current results."""
    global should_shutdown
    if not should_shutdown:  # Only handle first Ctrl-C
        console.print("\n[yellow]Received interrupt signal. Writing current results and shutting down...[/yellow]")
        should_shutdown = True
        # Write current results only if we have any
        if _current_results:
            write_results(_current_results, fmt=_global_output_format, file=_global_output_file)
        else:
            console.print("[yellow]No results to write, exiting...[/yellow]")
            sys.exit(1)  # Exit immediately if no results

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """Flatten nested dictionaries for CSV output."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif v is not None:  # Only include non-None values
            items.append((new_key, v))
    return dict(items)

def write_results(results: List[Dict[str, Any]], fmt: str = "csv", file: Optional[str] = None) -> None:
    """Write results to file or stdout."""
    if not results:
        return
    
    if fmt == "json":
        output = json.dumps(results, indent=2)
        if file:
            Path(file).write_text(output)
        else:
            console.print_json(output)
    else:  # csv
        # Flatten nested dictionaries and remove empty values
        flat_results = []
        for result in results:
            if result:  # Skip empty results
                flat_dict = flatten_dict(result)
                if flat_dict:  # Only include non-empty results
                    flat_results.append(flat_dict)
        
        if flat_results:
            # Create DataFrame with all available columns
            df = pl.DataFrame(flat_results)
            
            if file:
                df.write_csv(file)
            else:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                    df.write_csv(tmp.name)
                    tmp.seek(0)
                    print(tmp.read())
                os.unlink(tmp.name)

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
        10,  # Reset to 10
        "--max-results", "-n",
        help="Maximum number of search results to process per item",
        min=1,
        max=50,
    ),
    output_format: str = typer.Option(
        "csv",  # Changed default
        "--format", "-f",
        help="Output format: table, json, or csv",
        case_sensitive=False,
    ),
    output_file: Optional[str] = typer.Option(
        "results.csv",  # Default output file
        "--output", "-o",
        help="Output file path. If not specified, writes to results.csv for CSV format.",
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
    global _current_results, _global_output_format, _global_output_file, should_shutdown
    
    # Store output settings for signal handler
    _global_output_format = output_format
    _global_output_file = output_file or "results.csv"
    
    results = []
    
    try:
        gemini = GeminiClient(verbose=verbose, temperature=temperature)
        search = GoogleSearchClient(max_results=max_results)
        scraper = Scraper()  # Initialize Scraper
        
        # Parse the query
        if verbose:
            console.print("[yellow]Parsing query...[/yellow]")
        try:
            if should_shutdown:  # Check for early shutdown
                console.print("\n[yellow]Received interrupt signal. Shutting down...[/yellow]")
                return
                
            entity_type, attributes, search_space = await gemini.parse_query(query)
            if verbose:
                console.print(f"Entity Type: {entity_type}")
                console.print(f"Attributes: {attributes}")
                console.print(f"Search Space: {search_space}")
        except Exception as e:
            console.print(f"[red]Error parsing query: {str(e)}[/red]")
            return
            
        if should_shutdown:  # Check for shutdown after query parsing
            console.print("\n[yellow]Received interrupt signal. Shutting down...[/yellow]")
            return
        
        # Get search terms
        try:
            search_terms = await gemini.enumerate_search_space(search_space)
        except Exception as e:
            console.print(f"[red]Error enumerating search space: {str(e)}[/red]")
            return
        
        # Create progress bars
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            expand=True,
            transient=False  # Keep the progress bar visible
        )
        
        status_console = Console()  # Separate console for status updates
        
        all_search_results = []
        
        with progress:
            # First task: searching
            search_task = progress.add_task(
                description="[cyan]Searching...",
                total=len(search_terms)
            )
            
            # Search for each term
            for term in search_terms:
                if should_shutdown:
                    break
                    
                search_query = f"{entity_type} in {term}"
                progress.update(search_task, description=f"[cyan]Searching: {term}")
                
                if verbose:
                    # Use print instead of console.print to avoid interfering with progress bar
                    print(f"\033[34mSearching: {search_query}\033[0m")
                
                try:
                    results = await search.search(search_query)
                    all_search_results.extend(results)
                except Exception as e:
                    if verbose:
                        print(f"\033[31mError searching for {term}: {str(e)}\033[0m")
                
                progress.update(search_task, advance=1)
                
                # Update global results in case of interrupt
                _current_results = all_search_results
        
        if not all_search_results:
            status_console.print("[red]No search results found[/red]")
            return
        
        if verbose:
            status_console.print(f"[green]Found {len(all_search_results)} total results[/green]")
        
        # Process results with Gemini
        results = []
        
        # Single progress bar for all processing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            expand=True,
            transient=False
        ) as process_progress:
            # First pass: Extract names from search results
            first_pass = process_progress.add_task(
                description="[cyan]Starting name extraction...",
                total=len(all_search_results)
            )

            # Use a set to track unique names we've seen
            seen_names = set()
            
            for idx, result in enumerate(all_search_results):
                if should_shutdown:
                    break
                    
                title = result.get('title', 'Untitled')
                # Truncate and pad title to fixed width of 50 chars
                display_title = f"{title[:47]}..." if len(title) > 50 else title.ljust(50)
                
                process_progress.update(
                    first_pass,
                    description=f"[cyan]Extracting from {idx+1}/{len(all_search_results)}: {display_title}"
                )
                
                if verbose:
                    print(f"\033[34mExtracting from: {title}\033[0m")
                
                try:
                    # First pass: extract name from title
                    extracted = await gemini.parse_search_result(title, entity_type, ["name"])
                    if extracted and "name" in extracted:
                        name = extracted["name"]
                        # Only add if we haven't seen this name before
                        if name not in seen_names:
                            seen_names.add(name)
                            results.append({"name": name})
                            # Update global results in case of interrupt
                            _current_results = results
                except Exception as e:
                    if verbose:
                        print(f"\033[31mError extracting name: {str(e)}\033[0m\n")
                finally:
                    # Always update progress, even if no name was extracted
                    process_progress.update(first_pass, advance=1)

            # Second pass: Search for and extract attributes
            remaining_attributes = [attr for attr in attributes if attr != "name"]
            if remaining_attributes:
                second_pass = process_progress.add_task(
                    description="[cyan]Starting attribute extraction...",
                    total=len(results)
                )

                for idx, result in enumerate(results):
                    if should_shutdown:
                        break
                        
                    name = result["name"]
                    # Truncate name for display
                    display_name = f"{name[:47]}..." if len(name) > 50 else name.ljust(50)
                    
                    process_progress.update(
                        second_pass,
                        description=f"[cyan]Gathering attributes for {idx+1}/{len(results)}: {display_name}"
                    )

                    for attr in remaining_attributes:
                        if should_shutdown:
                            break
                            
                        # Build search query for specific attribute
                        query = f'"{name}" {attr}'
                        if verbose:
                            print(f"\033[34mobtaining attributes by searching: {query}\033[0m")
                        
                        try:
                            # Search specifically for this entity's attributes
                            attr_results = await search.search(query)
                            
                            # Extract attributes from each search result until we find one
                            for attr_result in attr_results:
                                if should_shutdown:
                                    break
                                    
                                # Build full text from search result
                                text_parts = []
                                if attr_result.get("title"):
                                    text_parts.append(attr_result["title"])
                                if attr_result.get("snippet"):
                                    text_parts.append(attr_result["snippet"])
                                if attr_result.get("displayLink"):
                                    text_parts.append(attr_result["displayLink"])
                                
                                text = "\n".join(text_parts)
                                
                                if verbose:
                                    print(f"\033[34mExtracting from text:\n{text}\033[0m\n")
                                
                                extracted = await gemini.parse_search_result(
                                    text,
                                    entity_type,
                                    [attr]
                                )
                                if extracted and attr in extracted:
                                    result[attr] = extracted[attr]
                                    # Update global results in case of interrupt
                                    _current_results = results
                                    break  # Found the attribute, move to next one
                                
                                # If we didn't find the attribute in the snippet, try scraping the page
                                if attr_result.get("link") and not (extracted and attr in extracted):
                                    if verbose:
                                        print(f"\033[34mTrying to scrape page: {attr_result['link']}\033[0m")
                                    
                                    page_text = await scraper.scrape_page(attr_result["link"])
                                    if page_text:
                                        if verbose:
                                            print(f"\033[34mExtracted text from page:\n{page_text[:500]}...\033[0m\n")
                                        
                                        extracted = await gemini.parse_search_result(
                                            page_text,
                                            entity_type,
                                            [attr]
                                        )
                                        if extracted and attr in extracted:
                                            result[attr] = extracted[attr]
                                            # Update global results in case of interrupt
                                            _current_results = results
                                            break  # Found the attribute, move to next one
                                            
                        except Exception as e:
                            if verbose:
                                print(f"\033[31mError searching for {attr}: {str(e)}\033[0m\n")
                
                process_progress.update(second_pass, advance=1)
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
    finally:
        # Write results if we have any
        if results:
            write_results(results, fmt=output_format, file=output_file)
            status_console.print(f"[green]Wrote {len(results)} results[/green]")

# Expose the Typer app as main for the Poetry script
main = app

if __name__ == "__main__":
    app()
