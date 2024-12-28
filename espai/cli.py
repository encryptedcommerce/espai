"""Command-line interface for espai."""

import asyncio
import signal
import json
import shutil
import textwrap
from typing import Dict, List, Optional
from enum import Enum
import re

import polars as pl
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.console import Console

from .search_providers import SearchProvider
from .search_providers.google import GoogleSearchProvider
from .search_providers.exa import ExaSearchProvider
from .gemini_client import GeminiClient
from .scraper import Scraper
from .models import EntityResult

# Create console for status messages
status_console = Console(stderr=True)
console = Console()

# Global variables for signal handler
should_shutdown = False
_current_results = []
_current_attributes = []
_global_output_format = None
_global_output_file = None

gray = "\033[38;5;240m"
green = "\033[38;5;34m"
blue = "\033[38;5;33m"
purple = "\033[38;5;12m"
red = "\033[38;5;209m"
end_color = "\033[0m"

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
    ),
    scrape: bool = typer.Option(
        False,
        "--scrape",
        "-s",
        help="Scrape URLs for additional content"
    )
):
    """Search and extract structured data from the web."""

    # Initialize global variables for signal handler
    global _current_results, _current_attributes, _global_output_format, _global_output_file
    _current_results = []
    _current_attributes = []
    _global_output_format = output_format
    _global_output_file = output_file or f"results.{output_format.lower()}"

    results = []
    entity_type = None
    attributes = []
    search_space = None

    try:
        gemini = GeminiClient(verbose=verbose, temperature=temperature)
        search = get_search_provider(provider)
        google_search = GoogleSearchProvider()  # For attribute searches
        scraper = Scraper() if scrape else None

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

        # Initialize empty DataFrame
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
                                left_text = f"{gray}Found entity in: {green}{result.url}{end_color}\n" + \
                                          f"{green}{result.title}{end_color}\n" + \
                                          f"{green}{result.snippet}{end_color}" 

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

                                    if verbose:
                                        right_text = f"{gray}LLM Entity Extraction:{end_color}\n" + \
                                                  f"{blue}    {entity_name}{end_color}"
                                        print(format_two_columns(left_text, right_text))

                                    # Initialize row with name and search space
                                    row_data = {col: None for col in columns}
                                    row_data.update({
                                        "name": entity_name,
                                        "search_space": space_item  # Use the enumerated item instead of original search space
                                    })

                                    # Add to DataFrame
                                    results_df = pl.concat([
                                        results_df,
                                        pl.DataFrame([row_data], schema=results_df.schema)
                                    ], how="vertical")

                                    # Update current results for signal handler
                                    _current_results.append(EntityResult(
                                        name=row_data['name'],
                                        search_space=row_data.get('search_space'),
                                        website=row_data.get('website'),
                                        phone=row_data.get('phone'),
                                        email=row_data.get('email'),
                                        address=row_data.get('address'),
                                        street_address=row_data.get('street_address'),
                                        city=row_data.get('city'),
                                        state=row_data.get('state'),
                                        zip=row_data.get('zip')
                                    ))

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
                        left_text = f"{purple}Searching for attributes:{end_color}"
                        right_text = f"{purple}{attr_query}{end_color}"
                        print(format_two_columns(left_text, right_text))

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
                                left_text = f"{gray}Extracting from text:{end_color}\n" + \
                                          f"{green}{result.title}{end_color}\n" + \
                                          f"{green}{result.snippet}{end_color}\n" + \
                                          f"{green}{result.url}{end_color}"

                            # Only scrape if enabled
                            content = ""
                            if scraper:
                                try:
                                    content = await scraper.scrape(result.url)
                                except Exception as e:
                                    if verbose:
                                        print(f"{red}Error scraping {result.url}: {str(e)}{end_color}\n")
                                    content = ""

                            text = f"{result.title}\n{result.snippet}"
                            if content:
                                text += f"\n{content}"

                            extracted = await gemini.extract_attributes(
                                text,
                                result.url,
                                entity_type,
                                [attr for attr in attributes if attr not in found_attributes]
                            )

                            if extracted:
                                if verbose:
                                    right_text = f"{gray}LLM Attribute Extraction:{end_color}\n" + \
                                        blue + \
                                        json.dumps(extracted, indent=2) + \
                                        end_color
                                    print(format_two_columns(left_text, right_text))

                            if extracted:
                                # Create new row data starting with existing row
                                new_row = dict(results_df.row(i, named=True))

                                # Set the enumerated search space item
                                if space_item:
                                    new_row['search_space'] = space_item

                                # First handle address if present
                                if 'address' in extracted:
                                    address = extracted['address']
                                    # Handle case where address is already a dictionary
                                    if isinstance(address, dict):
                                        for field in ['street_address', 'city', 'state', 'zip']:
                                            if field in address:
                                                new_row[field] = address[field]
                                    else:
                                        # Parse address string
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

                                # Update the DataFrame with the new row
                                results_df = pl.concat([
                                    results_df,
                                    pl.DataFrame([new_row], schema=results_df.schema)
                                ], how="vertical")

                                # Update current results for signal handler
                                _current_results = [
                                    EntityResult(
                                        name=row['name'],
                                        search_space=row.get('search_space'),
                                        website=row.get('website'),
                                        phone=row.get('phone'),
                                        email=row.get('email'),
                                        address=row.get('address'),
                                        street_address=row.get('street_address'),
                                        city=row.get('city'),
                                        state=row.get('state'),
                                        zip=row.get('zip')
                                    )
                                    for row in results_df.to_dicts()
                                ]

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

def format_two_columns(left_text: str, right_text: str, color: str = "\033[38;5;33m") -> str:
    """Format text in two columns, each taking 50% of terminal width."""
    term_width = shutil.get_terminal_size().columns
    col_width = term_width // 2 - 2  # -2 for padding

    # Split texts into lines
    left_lines = left_text.split('\n')
    right_lines = right_text.split('\n')

    # Wrap each line to fit column width
    wrapped_left = []
    for line in left_lines:
        wrapped_left.extend(textwrap.wrap(line, width=col_width) or [''])

    wrapped_right = []
    for line in right_lines:
        wrapped_right.extend(textwrap.wrap(line, width=col_width) or [''])

    # Make both columns same height
    max_lines = max(len(wrapped_left), len(wrapped_right))
    wrapped_left.extend([''] * (max_lines - len(wrapped_left)))
    wrapped_right.extend([''] * (max_lines - len(wrapped_right)))

    # Combine lines
    result = []
    for left, right in zip(wrapped_left, wrapped_right):
        # Account for ANSI color codes in width calculation
        left_content = re.sub(r'\033\[[0-9;]*m', '', left)
        right_content = re.sub(r'\033\[[0-9;]*m', '', right)
        left_padding = ' ' * (col_width - len(left_content))
        right_padding = ' ' * (col_width - len(right_content))
        result.append(f"{left}{left_padding} │ {right}{right_padding}")

    # Add horizontal separator with same width as content
    separator = f"{color}{'─' * col_width}─┼{'─' * col_width}─\033[0m"

    return '\n'.join(result) + '\n' + separator

# Create the CLI app
app = typer.Typer()

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
    ),
    scrape: bool = typer.Option(
        False,
        "--scrape",
        "-s",
        help="Scrape URLs for additional content"
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
        provider=provider,
        scrape=scrape
    ))

# Expose the Typer app as main for the Poetry script
def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
