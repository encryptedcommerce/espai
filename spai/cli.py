"""Command-line interface for the spai tool."""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .gemini_client import GeminiClient
from .models import EntityResult
from .search_client import GoogleSearchClient

app = typer.Typer()
console = Console()

# Load environment variables
load_dotenv()

# Global state for graceful shutdown
should_exit = False


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    global should_exit
    should_exit = True
    console.print("\n[yellow]Received interrupt signal. Saving partial results...[/yellow]")


async def process_search_space_item(
    item: str,
    entities: str,
    search_client: GoogleSearchClient,
    gemini_client: GeminiClient,
) -> list[EntityResult]:
    """Process a single search space item."""
    query = f"{entities} in {item}"
    results = await search_client.search(query)
    
    processed_results = []
    for result in results:
        if should_exit:
            break
            
        parsed = await gemini_client.parse_search_result(result.get("snippet", ""))
        processed_results.append(
            EntityResult(
                entity_id=result["link"],
                name=parsed["name"],
                address=parsed.get("address"),
                website_url=result["link"],
                additional_details=parsed.get("additional_details", {}),
                search_space_item=item
            )
        )
    
    return processed_results


@app.command()
def main(
    query: str = typer.Argument(..., help="Natural language query to process"),
    max_results: int = typer.Option(10, help="Maximum number of results per search"),
    output_format: str = typer.Option("csv", help="Output format (csv, json, parquet)"),
    output_file: Optional[Path] = typer.Option(None, help="Output file path"),
):
    """Search, Parse, and Iterate - Extract structured data from search results."""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize clients
        gemini_client = GeminiClient()
        search_client = GoogleSearchClient(max_results=max_results)
        
        # Create event loop for async operations
        loop = asyncio.get_event_loop()
        
        # Parse the query
        with console.status("[bold green]Parsing query..."):
            parsed_query = loop.run_until_complete(gemini_client.parse_query(query))
        
        # Enumerate search space
        with console.status("[bold green]Enumerating search space..."):
            search_space = loop.run_until_complete(
                gemini_client.enumerate_search_space(parsed_query.search_space)
            )
        
        # Process results
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing {len(search_space)} items...",
                total=len(search_space)
            )
            
            for item in search_space:
                if should_exit:
                    break
                    
                item_results = loop.run_until_complete(
                    process_search_space_item(
                        item,
                        parsed_query.entities,
                        search_client,
                        gemini_client
                    )
                )
                results.extend(item_results)
                progress.advance(task)
        
        # Convert results to DataFrame
        df = pl.DataFrame([result.model_dump() for result in results])
        
        # Determine output path
        if output_file is None:
            output_file = Path(f"spai_results.{output_format}")
        
        # Save results
        if output_format == "csv":
            df.write_csv(output_file)
        elif output_format == "json":
            df.write_json(output_file)
        elif output_format == "parquet":
            df.write_parquet(output_file)
        else:
            console.print(f"[red]Unsupported output format: {output_format}[/red]")
            sys.exit(1)
        
        console.print(f"[green]Results saved to {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)
    finally:
        if 'loop' in locals():
            loop.close()


if __name__ == "__main__":
    app()
