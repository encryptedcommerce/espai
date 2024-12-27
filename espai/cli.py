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

# Create the CLI app
app = typer.Typer()

# Global state for signal handling
should_shutdown = False
_current_results = []
_current_attributes = []  # Add global for tracking requested attributes

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

class SearchProvider(str, Enum):
    """Available search providers."""
    GOOGLE = "google"
    EXA = "exa"

def write_results(results: List[EntityResult], fmt: OutputFormat, file: str, requested_attrs: List[str]):
    """Write results to file in specified format."""
    if not results:
        return
        
    # Deduplicate results while preserving order
    seen = set()
    unique_results = []
    for result in results:
        if result.name.lower() not in seen:
            seen.add(result.name.lower())
            unique_results.append(result)
    
    # Convert to DataFrame with only requested attributes
    result_dicts = []
    for r in unique_results:
        # Always include name and search_space
        result_dict = {
            "name": r.name,
            "search_space": r.search_space
        }
        # Add only requested attributes
        for attr in requested_attrs:
            if attr != "name":  # name is already included
                result_dict[attr] = getattr(r, attr, None)
        result_dicts.append(result_dict)
    
    df = pl.DataFrame(result_dicts)
    
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
        status_console.print("\n[yellow]Received interrupt signal. Please wait a few moments while writing results...[/yellow]")
        write_results(_current_results, _global_output_format, _global_output_file, _current_attributes)
        status_console.print(f"[green]Successfully wrote {len(_current_results)} results.[/green]")
    else:
        status_console.print("\n[yellow]Received interrupt signal. No results to write.[/yellow]")
    
    # Force exit after writing results
    import sys
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

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
    global _global_output_format, _global_output_file, should_shutdown, _current_results, _current_attributes
    
    # Set globals for signal handler
    _global_output_format = output_format
    _global_output_file = output_file or f"results.{output_format.value}"
    
    results = []
    entity_type = None
    attributes = []
    search_space = None
    
    try:
        gemini = GeminiClient(verbose=verbose, temperature=temperature)
        search = get_search_provider(provider)
        google_search = GoogleSearchProvider()  # For attribute searches
        scraper = Scraper()
        
        # Parse the query
        if verbose:
            console.print("[yellow]Parsing query...[/yellow]")
            
        entity_type, attributes, search_space = await gemini.parse_query(query)
        # Update global attributes right after parsing
        _current_attributes = attributes
        if verbose:
            console.print(f"Entity Type: {entity_type}")
            console.print(f"Attributes: {attributes}")
            console.print(f"Search Space: {search_space}")
            
        if should_shutdown:
            return
            
        # Get search space items
        search_items = []
        if search_space:
            if verbose:
                console.print("[yellow]Enumerating search space...[/yellow]")
                
            search_items = await gemini.enumerate_search_space(search_space)
            if verbose:
                console.print("Enumerated search space:")
                console.print(search_items)
        else:
            search_items = [""]  # Single empty item if no search space
            
        if should_shutdown:
            return
            
        # Create progress bars
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=status_console
        )
        
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
                            # Include URL in text for better context
                            text_parts = []
                            if result.title:
                                text_parts.append(result.title)
                            if result.snippet:
                                text_parts.append(result.snippet)
                            if result.url:  # Add URL for better context
                                text_parts.append(f"URL: {result.url}")
                            
                            text = "\n".join(text_parts)
                            
                            extracted = await gemini.parse_search_result(
                                text,
                                entity_type,
                                ["name"]
                            )
                            
                            if extracted and "name" in extracted:
                                new_result = EntityResult.from_dict(extracted, space_item)
                                if new_result:
                                    results.append(new_result)
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
        
        if should_shutdown:
            return
            
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
                                new_result = EntityResult.from_dict(extracted, result.search_space)
                                if new_result:
                                    # Update attributes on existing result
                                    if new_result.website:
                                        result.website = new_result.website
                                    if new_result.phone:
                                        result.phone = new_result.phone
                                    if new_result.email:
                                        result.email = new_result.email
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
                                        new_result = EntityResult.from_dict(extracted, result.search_space)
                                        if new_result:
                                            # Update attributes on existing result
                                            if new_result.website:
                                                result.website = new_result.website
                                            if new_result.phone:
                                                result.phone = new_result.phone
                                            if new_result.email:
                                                result.email = new_result.email
                                            # Update global results in case of interrupt
                                            _current_results = results
                                        break  # Found the attribute, move to next one
                                            
                except Exception as e:
                    if verbose:
                        print(f"\033[31mError searching for {attr}: {str(e)}\033[0m\n")
                    continue
                    
                progress.update(second_pass, advance=1)
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        
    finally:
        # Clean up resources
        if isinstance(search, ExaSearchProvider):
            await search.close()
        if isinstance(google_search, GoogleSearchProvider):
            await google_search.close()
            
        # Write results if we have any and we haven't already written them via signal handler
        if results and not should_shutdown:
            write_results(results, fmt=output_format, file=_global_output_file, requested_attrs=attributes)
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
