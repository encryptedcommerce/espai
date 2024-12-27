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
    base_columns = ['name', 'search_space']
    attribute_columns = []
    for attr in requested_attrs:
        if attr == 'address':
            # Add address components
            attribute_columns.extend(['address', 'street_address', 'city', 'state', 'zip'])
        else:
            attribute_columns.append(attr)
            
    # Remove duplicates while preserving order
    columns = base_columns + list(dict.fromkeys(attribute_columns))
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
            
        # Create progress bars
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=status_console
        )
        
        # Initialize empty DataFrame with all columns
        columns = ['name', 'search_space'] + attributes
        if 'address' in attributes:
            # Add address component columns
            columns.extend(['street_address', 'city', 'state', 'zip'])
        
        # Create empty DataFrame with specified schema
        results_df = pl.DataFrame(schema={col: pl.Utf8 for col in columns})
        
        # Enumerate search space if needed
        enumerated_space = None
        if search_space:
            try:
                enumerated_space = await gemini.enumerate_search_space(search_space)
                if verbose and enumerated_space:
                    print("\nEnumerated search space:")
                    for item in enumerated_space:
                        print(f"- {item}")
                    print()
            except Exception as e:
                if verbose:
                    print(f"\033[38;5;209mError enumerating search space: {str(e)}\033[0m\n")
        
        # First pass - get entities from search space
        with progress:
            first_pass = progress.add_task(
                "[cyan]Finding entities...",
                total=len(enumerated_space) if enumerated_space else 1
            )
            
            # Track found entities to avoid duplicates
            found_entities = set()
            
            # Search within each enumerated item or the general search space
            search_spaces_to_try = enumerated_space if enumerated_space else [search_space]
            for space_item in search_spaces_to_try:
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
                            if verbose:
                                print(f"\033[38;5;33mExtracting from text:\n{result.title}\n{result.snippet}\n{result.url}\033[0m\n")
                                
                            extracted = await gemini.extract_attributes(
                                f"{result.title}\n{result.snippet}",
                                result.url,
                                entity_type,
                                ["name"]  # Only look for name in first pass
                            )
                            
                            if extracted and "name" in extracted:
                                entity_name = extracted["name"]
                                if entity_name not in found_entities:
                                    found_entities.add(entity_name)
                                    
                                    # Initialize row with name and search space
                                    row_data = {col: None for col in columns}
                                    row_data.update({
                                        "name": entity_name,
                                        "search_space": space_item  # Use the enumerated item instead of original search space
                                    })
                                    
                                    # Add to DataFrame
                                    new_row = pl.DataFrame([row_data])
                                    results_df = pl.concat([results_df, new_row], how="vertical")
                                    
                                    if verbose:
                                        print(f"\nFound {entity_type}: {entity_name}")
                                        for attr in attributes:
                                            if attr == 'address' and row_data.get('address'):
                                                print(f"  Address: {row_data['address']}")
                                                # Only show address components that exist
                                                if row_data.get('street_address'):
                                                    print(f"    Street: {row_data.get('street_address')}")
                                                if row_data.get('city'):
                                                    print(f"    City: {row_data.get('city')}")
                                                if row_data.get('state'):
                                                    print(f"    State: {row_data.get('state')}")
                                                if row_data.get('zip'):
                                                    print(f"    ZIP: {row_data.get('zip')}")
                                            elif row_data.get(attr):
                                                print(f"  {attr.title()}: {row_data.get(attr)}")
                                        
                                # Stop if we found all entities
                                if len(found_entities) == max_results:
                                    break
                                    
                        except Exception as e:
                            if verbose:
                                print(f"\033[38;5;209mError extracting entity: {str(e)}\033[0m\n")
                            continue
                            
                except Exception as e:
                    if verbose:
                        print(f"\033[38;5;209mError searching for entities: {str(e)}\033[0m\n")
                    continue
                    
                progress.update(first_pass, advance=1)
                
        if should_shutdown:
            return
            
        # Second pass - get attributes for each entity
        with progress:
            second_pass = progress.add_task(
                "[cyan]Getting attributes...",
                total=len(results_df)
            )
            
            # Process each found entity
            for i in range(len(results_df)):
                if should_shutdown:
                    break
                    
                row = results_df.row(i, named=True)
                entity_name = row['name']
                
                # Get the specific enumerated item for this row
                space_item = None
                if enumerated_space and len(enumerated_space) > 0:
                    space_item = enumerated_space[i % len(enumerated_space)]
                else:
                    space_item = search_space
                        
                try:
                    # Build attribute search query
                    attr_query = f"{entity_name} {' '.join(attributes)}"
                    if space_item:
                        attr_query += f" in {space_item}"
                        
                    if verbose:
                        print(f"\033[38;5;12mSearching for attributes: {attr_query}\033[0m")
                        
                    # Search for attributes
                    attr_results = await search.search(
                        attr_query,
                        max_results=max_results
                    )
                    
                    # Track which attributes we've found
                    found_attributes = set()
                    
                    # Extract attributes from results
                    for result in attr_results:
                        if should_shutdown:
                            break
                            
                        try:
                            if verbose:
                                print(f"\033[38;5;33mExtracting from text:\n{result.title}\n{result.snippet}\n{result.url}\033[0m\n")
                                
                            extracted = await gemini.extract_attributes(
                                f"{result.title}\n{result.snippet}",
                                result.url,
                                entity_type,
                                [attr for attr in attributes if attr not in found_attributes]
                            )
                            
                            if extracted:
                                if verbose:
                                    print(f"\033[38;5;33mExtracted attributes: {extracted}\033[0m")
                                    
                                # Create new row data starting with existing row
                                new_row = dict(results_df.row(i, named=True))
                                
                                # Set the enumerated search space item
                                if space_item:
                                    new_row['search_space'] = space_item
                                
                                # First handle address if present
                                if 'address' in extracted:
                                    address = extracted['address']
                                    parts = address.split(',')
                                    if len(parts) >= 1:
                                        new_row['street_address'] = parts[0].strip()
                                    if len(parts) >= 2:
                                        city_state = parts[1].strip().split()
                                        if len(city_state) > 0:
                                            new_row['city'] = ' '.join(city_state[:-1]) if len(city_state) > 1 else city_state[0]
                                        if len(city_state) > 1:
                                            new_row['state'] = city_state[-1]
                                    if len(parts) >= 3:
                                        new_row['zip'] = parts[2].strip()
                                    # Remove the full address after decomposing
                                    extracted.pop('address')
                                
                                # Then update with any other extracted values
                                for col in results_df.columns:
                                    if col in extracted:
                                        new_row[col] = extracted[col]
                                
                                # Update DataFrame
                                results_df = results_df.with_row_count('index').with_columns([
                                    pl.when(pl.col('index') == i)
                                    .then(pl.lit(new_row.get(col)))
                                    .otherwise(pl.col(col))
                                    .alias(col)
                                    for col in results_df.columns
                                ]).drop('index')
                                
                                if verbose:
                                    print(f"\033[38;5;33mUpdated row {i}: {new_row}\033[0m")
                                    
                                # Update found attributes
                                found_attributes.update(extracted.keys())
                                found_attributes.update(['street_address', 'city', 'state', 'zip'])
                                    
                        except Exception as e:
                            if verbose:
                                print(f"\033[38;5;209mError extracting attributes: {str(e)}\033[0m\n")
                            continue
                            
                except Exception as e:
                    if verbose:
                        print(f"\033[38;5;209mError searching for attributes: {str(e)}\033[0m\n")
                    continue
                    
                progress.update(second_pass, advance=1)
        
        if should_shutdown:
            return
        
        # Save results
        if not output_file:
            output_file = f"results.{output_format.value}"
            
        if output_format == OutputFormat.CSV:
            results_df.write_csv(output_file)
        elif output_format == OutputFormat.JSON:
            results_df.write_json(output_file)
        elif output_format == OutputFormat.PARQUET:
            results_df.write_parquet(output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format.value}")
            
        status_console.print(f"[green]Wrote {len(results_df)} results to {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        
    finally:
        # Clean up resources
        if isinstance(search, ExaSearchProvider):
            await search.close()
        if isinstance(google_search, GoogleSearchProvider):
            await google_search.close()

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
