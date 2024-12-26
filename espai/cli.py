"""Command-line interface for espai."""

import asyncio
import signal
from enum import Enum
from typing import List, Optional

import polars as pl
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .gemini_client import GeminiClient
from .models import EntityResult
from .scraper import Scraper
from .search_providers import SearchProvider
from .search_providers.exa import ExaSearchProvider
from .search_providers.google import GoogleSearchProvider

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create the CLI app
app = typer.Typer()

# Global state for signal handling
should_shutdown = False
_current_results = []

# Console for status messages
console = Console()
status_console = Console(stderr=True)

# Output format options
class OutputFormat(str, Enum):
    """Output format options."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"

# Global output format for signal handler
_global_output_format = OutputFormat.CSV
_global_output_file = "results.csv"

def write_results(results: List[EntityResult], fmt: OutputFormat, file: str):
    """Write results to file in specified format."""
    if not results:
        return
        
    # Convert to DataFrame
    df = pl.DataFrame([vars(r) for r in results])
    
    # Write to file
    if fmt == OutputFormat.CSV:
        df.write_csv(file)
    elif fmt == OutputFormat.JSON:
        df.write_json(file)
    elif fmt == OutputFormat.PARQUET:
        df.write_parquet(file)

def signal_handler(signum, frame):
    """Handle interrupt signal."""
    global should_shutdown
    should_shutdown = True
    
    # Write any results we have
    if _current_results:
        write_results(_current_results, _global_output_format, _global_output_file)
        status_console.print(f"\n[yellow]Received interrupt signal. Wrote {len(_current_results)} results.[/yellow]")
    else:
        status_console.print("\n[yellow]Received interrupt signal. No results to write.[/yellow]")
    
    # Exit immediately if no results
    if not _current_results:
        raise typer.Exit()

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

class SearchProvider(str, Enum):
    """Available search providers."""
    GOOGLE = "google"
    EXA = "exa"

def get_search_provider(provider: SearchProvider) -> SearchProvider:
    """Get the search provider instance."""
    if provider == SearchProvider.GOOGLE:
        return GoogleSearchProvider()
    elif provider == SearchProvider.EXA:
        return ExaSearchProvider()
    else:
        raise ValueError(f"Unknown search provider: {provider}")

async def search(
    query: str,
    max_results: int = typer.Option(
        10,
        "--max-results",
        "-n",
        help="Maximum number of results to return per search"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.CSV,
        "--output-format",
        "-f",
        help="Output format (csv, json, or parquet)"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Output file (default: results.[format])"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output"
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Temperature for LLM generation (0.0 to 1.0)"
    ),
    provider: SearchProvider = typer.Option(
        SearchProvider.GOOGLE,
        "--provider",
        "-p",
        help="Search provider to use"
    )
):
    """Search and extract structured data from the web."""
    global _global_output_format, _global_output_file, should_shutdown, _current_results
    
    # Set globals for signal handler
    _global_output_format = output_format
    _global_output_file = output_file or f"results.{output_format.value}"
    
    results = []
    
    try:
        gemini = GeminiClient(verbose=verbose, temperature=temperature)
        search = get_search_provider(provider)
        google_search = GoogleSearchProvider()  # For attribute searches
        scraper = Scraper()
        
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
            
        # Get search space items if needed
        search_items = []
        if search_space:
            try:
                if verbose:
                    console.print("[yellow]Enumerating search space...[/yellow]")
                search_items = await gemini.enumerate_search_space(search_space)
                if verbose:
                    console.print("Enumerated search space:")
                    console.print(search_items)
            except Exception as e:
                console.print(f"[red]Error enumerating search space: {str(e)}[/red]")
                return
        else:
            search_items = [""]  # Single empty item if no search space
            
        # Create progress bars
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=status_console
        )
        
        try:
            # First pass - get entities
            with progress:
                first_pass = progress.add_task(
                    "[cyan]Finding entities...",
                    total=len(search_items)
                )
                
                for space_item in search_items:
                    if should_shutdown:
                        break
                        
                    # Build search query
                    if space_item:
                        search_query = f"{entity_type} in {space_item}"
                    else:
                        search_query = entity_type
                        
                    try:
                        # Search for entities
                        entity_results = await search.search(
                            search_query,
                            max_results=max_results
                        )
                        
                        # Extract entity names from results
                        for result in entity_results:
                            if should_shutdown:
                                break
                                
                            try:
                                extracted = await gemini.parse_search_result(
                                    f"{result.title}\n{result.snippet}",
                                    entity_type,
                                    ["name"]
                                )
                                
                                if extracted and "name" in extracted:
                                    result = EntityResult(
                                        name=extracted["name"],
                                        search_space=space_item
                                    )
                                    results.append(result)
                                    # Update global results in case of interrupt
                                    _current_results = results
                                    
                            except Exception as e:
                                if verbose:
                                    print(f"\033[31mError extracting name: {str(e)}\033[0m\n")
                                continue
                                
                    except Exception as e:
                        if verbose:
                            print(f"\033[31mError searching for entities: {str(e)}\033[0m\n")
                        continue
                        
                    progress.update(first_pass, advance=1)
            
            # Second pass - get attributes
            with progress:
                second_pass = progress.add_task(
                    "[cyan]Getting attributes...",
                    total=len(results)
                )
                
                for result in results:
                    if should_shutdown:
                        break
                        
                    try:
                        # Get remaining attributes
                        remaining_attrs = [
                            attr for attr in attributes
                            if attr != "name" and not getattr(result, attr, None)
                        ]
                        
                        if not remaining_attrs:
                            continue
                            
                        # Build attribute search query
                        attr_query = f"{result.name} {' '.join(remaining_attrs)}"
                        if result.search_space:
                            attr_query += f" in {result.search_space}"
                            
                        if verbose:
                            print(f"\033[34mSearching for attributes: {attr_query}\033[0m\n")
                            
                        # Search for attributes using Google
                        attr_results = await google_search.search(
                            attr_query,
                            max_results=max_results
                        )
                        
                        # Extract attributes from each result
                        for attr in remaining_attrs:
                            if should_shutdown:
                                break
                                
                            # Extract attributes from each search result until we find one
                            for attr_result in attr_results:
                                if should_shutdown:
                                    break
                                    
                                # Build full text from search result
                                text_parts = []
                                if attr_result.title:
                                    text_parts.append(attr_result.title)
                                if attr_result.snippet:
                                    text_parts.append(attr_result.snippet)
                                if attr_result.domain:
                                    text_parts.append(attr_result.domain)
                                
                                text = "\n".join(text_parts)
                                
                                if verbose:
                                    print(f"\033[34mExtracting from text:\n{text}\033[0m\n")
                                
                                extracted = await gemini.parse_search_result(
                                    text,
                                    entity_type,
                                    [attr]
                                )
                                if extracted and attr in extracted:
                                    setattr(result, attr, extracted[attr])
                                    # Update global results in case of interrupt
                                    _current_results = results
                                    break  # Found the attribute, move to next one
                                    
                                # If we didn't find the attribute in the snippet, try scraping the page
                                if attr_result.url and not (extracted and attr in extracted):
                                    if verbose:
                                        print(f"\033[34mTrying to scrape page: {attr_result.url}\033[0m")
                                    
                                    page_text = await scraper.scrape_page(attr_result.url)
                                    if page_text:
                                        if verbose:
                                            print(f"\033[34mExtracted text from page:\n{page_text[:500]}...\033[0m\n")
                                        
                                        extracted = await gemini.parse_search_result(
                                            page_text,
                                            entity_type,
                                            [attr]
                                        )
                                        if extracted and attr in extracted:
                                            setattr(result, attr, extracted[attr])
                                            # Update global results in case of interrupt
                                            _current_results = results
                                            break  # Found the attribute, move to next one
                                            
                    except Exception as e:
                        if verbose:
                            print(f"\033[31mError searching for {attr}: {str(e)}\033[0m\n")
                        continue
                        
                    progress.update(second_pass, advance=1)
                    
        finally:
            # Clean up resources
            if isinstance(search, ExaSearchProvider):
                await search.close()
            if isinstance(google_search, GoogleSearchProvider):
                await google_search.close()
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
    finally:
        # Write results if we have any
        if results:
            write_results(results, fmt=output_format, file=output_file)
            status_console.print(f"[green]Wrote {len(results)} results[/green]")

@app.command()
def search_wrapper(
    query: str,
    max_results: int = typer.Option(
        10,
        "--max-results",
        "-n",
        help="Maximum number of results to return per search"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.CSV,
        "--output-format",
        "-f",
        help="Output format (csv, json, or parquet)"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Output file (default: results.[format])"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output"
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Temperature for LLM generation (0.0 to 1.0)"
    ),
    provider: SearchProvider = typer.Option(
        SearchProvider.GOOGLE,
        "--provider",
        "-p",
        help="Search provider to use"
    )
):
    """Search and extract structured data from the web."""
    asyncio.run(search(
        query=query,
        max_results=max_results,
        output_format=output_format,
        output_file=output_file,
        verbose=verbose,
        temperature=temperature,
        provider=provider
    ))

# Expose the Typer app as main for the Poetry script
def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
