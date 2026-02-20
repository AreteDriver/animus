"""Calendar commands — Google Calendar integration."""

from __future__ import annotations

from datetime import UTC

import typer

from ..helpers import console

calendar_app = typer.Typer(help="Google Calendar integration")


@calendar_app.command("list")
def calendar_list(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to show"),
    max_events: int = typer.Option(20, "--max", "-m", help="Maximum events"),
    calendar_id: str = typer.Option("primary", "--calendar", "-c", help="Calendar ID"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List upcoming calendar events.

    Example:
        gorgon calendar list
        gorgon calendar list --days 30 --max 50
    """
    from datetime import datetime, timedelta

    try:
        from animus_forge.api_clients import CalendarClient
    except ImportError as e:
        console.print(f"[red]Missing dependencies:[/red] {e}")
        raise typer.Exit(1)

    client = CalendarClient()
    if not client.authenticate():
        console.print("[red]Failed to authenticate with Google Calendar.[/red]")
        console.print("\nMake sure you have:")
        console.print("  1. Set up OAuth credentials in Google Cloud Console")
        console.print("  2. Downloaded credentials.json")
        console.print("  3. Set GMAIL_CREDENTIALS_PATH in .env")
        raise typer.Exit(1)

    now = datetime.now(UTC)
    end = now + timedelta(days=days)

    events = client.list_events(
        calendar_id=calendar_id,
        max_results=max_events,
        time_min=now,
        time_max=end,
    )

    if json_output:
        import json as json_mod

        events_dict = [
            {
                "id": e.id,
                "summary": e.summary,
                "start": e.start.isoformat() if e.start else None,
                "end": e.end.isoformat() if e.end else None,
                "location": e.location,
                "all_day": e.all_day,
            }
            for e in events
        ]
        print(json_mod.dumps(events_dict, indent=2))
        return

    if not events:
        console.print(f"[yellow]No events in the next {days} days[/yellow]")
        return

    console.print(f"[bold]Upcoming Events ({len(events)}):[/bold]\n")

    current_date = None
    for event in events:
        if event.start:
            event_date = event.start.strftime("%A, %B %d")
            if event_date != current_date:
                current_date = event_date
                console.print(f"\n[cyan]{event_date}[/cyan]")

            if event.all_day:
                time_str = "All day"
            else:
                time_str = event.start.strftime("%I:%M %p")

            console.print(f"  {time_str} - [bold]{event.summary}[/bold]")
            if event.location:
                console.print(f"            [dim]{event.location}[/dim]")


@calendar_app.command("today")
def calendar_today(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show today's remaining events.

    Example:
        gorgon calendar today
    """
    try:
        from animus_forge.api_clients import CalendarClient
    except ImportError as e:
        console.print(f"[red]Missing dependencies:[/red] {e}")
        raise typer.Exit(1)

    client = CalendarClient()
    if not client.authenticate():
        console.print("[red]Authentication failed[/red]")
        raise typer.Exit(1)

    events = client.get_upcoming_today()

    if json_output:
        import json as json_mod

        events_dict = [
            {
                "summary": e.summary,
                "start": e.start.isoformat() if e.start else None,
                "end": e.end.isoformat() if e.end else None,
            }
            for e in events
        ]
        print(json_mod.dumps(events_dict, indent=2))
        return

    if not events:
        console.print("[green]No more events today![/green]")
        return

    console.print("[bold]Remaining Today:[/bold]\n")
    for event in events:
        if event.start:
            if event.all_day:
                time_str = "All day"
            else:
                time_str = event.start.strftime("%I:%M %p")
            console.print(f"  {time_str} - {event.summary}")


@calendar_app.command("add")
def calendar_add(
    summary: str = typer.Argument(..., help="Event title"),
    start_time: str = typer.Option(
        None, "--start", "-s", help="Start time (e.g., '2024-01-15 14:00')"
    ),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in minutes"),
    location: str = typer.Option(None, "--location", "-l", help="Event location"),
    description: str = typer.Option(None, "--desc", help="Event description"),
    all_day: bool = typer.Option(False, "--all-day", help="Create all-day event"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Use natural language parsing"),
):
    """Create a new calendar event.

    Example:
        gorgon calendar add "Team Meeting" --start "2024-01-15 14:00" --duration 60
        gorgon calendar add "Doctor Appointment" --start "tomorrow 10am" --quick
        gorgon calendar add "Vacation" --start "2024-01-20" --all-day
    """
    from datetime import datetime, timedelta

    try:
        from animus_forge.api_clients import CalendarClient, CalendarEvent
    except ImportError as e:
        console.print(f"[red]Missing dependencies:[/red] {e}")
        raise typer.Exit(1)

    client = CalendarClient()
    if not client.authenticate():
        console.print("[red]Authentication failed[/red]")
        raise typer.Exit(1)

    if quick:
        # Use Google's natural language parsing
        full_text = f"{summary}"
        if start_time:
            full_text += f" {start_time}"
        if location:
            full_text += f" at {location}"

        result = client.quick_add(full_text)
        if result:
            console.print(f"[green]Event created:[/green] {result.summary}")
            if result.start:
                console.print(f"  Start: {result.start.strftime('%Y-%m-%d %I:%M %p')}")
            if result.html_link:
                console.print(f"  Link: {result.html_link}")
        else:
            console.print("[red]Failed to create event[/red]")
        return

    # Parse start time
    if not start_time:
        console.print("[red]Start time required (use --start or --quick)[/red]")
        raise typer.Exit(1)

    try:
        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
        ]:
            try:
                start = datetime.strptime(start_time, fmt)
                start = start.replace(tzinfo=UTC)
                break
            except ValueError:
                continue
        else:
            console.print(f"[red]Could not parse date:[/red] {start_time}")
            console.print("Use format: YYYY-MM-DD HH:MM or YYYY-MM-DD")
            raise typer.Exit(1)
    except Exception:
        console.print(f"[red]Invalid date format:[/red] {start_time}")
        raise typer.Exit(1)

    # Calculate end time
    if all_day:
        end = start + timedelta(days=1)
    else:
        end = start + timedelta(minutes=duration)

    event = CalendarEvent(
        summary=summary,
        description=description or "",
        location=location or "",
        start=start,
        end=end,
        all_day=all_day,
    )

    result = client.create_event(event)
    if result:
        console.print(f"[green]Event created:[/green] {result.summary}")
        console.print(f"  ID: {result.id}")
        if result.html_link:
            console.print(f"  Link: {result.html_link}")
    else:
        console.print("[red]Failed to create event[/red]")
        raise typer.Exit(1)


@calendar_app.command("delete")
def calendar_delete(
    event_id: str = typer.Argument(..., help="Event ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a calendar event.

    Example:
        gorgon calendar delete abc123xyz
    """
    try:
        from animus_forge.api_clients import CalendarClient
    except ImportError:
        console.print("[red]Missing dependencies[/red]")
        raise typer.Exit(1)

    client = CalendarClient()
    if not client.authenticate():
        console.print("[red]Authentication failed[/red]")
        raise typer.Exit(1)

    # Get event first to show what we're deleting
    event = client.get_event(event_id)
    if not event:
        console.print(f"[red]Event not found:[/red] {event_id}")
        raise typer.Exit(1)

    if not force:
        console.print(f"[yellow]Event:[/yellow] {event.summary}")
        if event.start:
            console.print(f"  Start: {event.start.strftime('%Y-%m-%d %I:%M %p')}")
        if not typer.confirm("Delete this event?"):
            raise typer.Abort()

    if client.delete_event(event_id):
        console.print("[green]Event deleted[/green]")
    else:
        console.print("[red]Failed to delete event[/red]")
        raise typer.Exit(1)


@calendar_app.command("busy")
def calendar_busy(
    days: int = typer.Option(1, "--days", "-d", help="Number of days to check"),
):
    """Check calendar availability.

    Example:
        gorgon calendar busy
        gorgon calendar busy --days 7
    """
    from datetime import datetime, timedelta

    try:
        from animus_forge.api_clients import CalendarClient
    except ImportError:
        console.print("[red]Missing dependencies[/red]")
        raise typer.Exit(1)

    client = CalendarClient()
    if not client.authenticate():
        console.print("[red]Authentication failed[/red]")
        raise typer.Exit(1)

    now = datetime.now(UTC)
    end = now + timedelta(days=days)

    busy_periods = client.check_availability(now, end)

    if not busy_periods:
        console.print(f"[green]No busy periods in the next {days} day(s)[/green]")
        return

    console.print(f"[bold]Busy Periods ({len(busy_periods)}):[/bold]\n")
    for period in busy_periods:
        start = period.get("start", "")
        end_p = period.get("end", "")
        if start and end_p:
            # Parse and format times
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_p.replace("Z", "+00:00"))
                console.print(
                    f"  {start_dt.strftime('%a %m/%d %I:%M %p')} - {end_dt.strftime('%I:%M %p')}"
                )
            except ValueError:
                console.print(f"  {start} - {end_p}")
