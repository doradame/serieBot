# Logs Directory

This directory contains SerieBot application logs with detailed execution information.

## Log Files

Log files are automatically created with the naming pattern:
```
seriebot_YYYYMMDD_HHMMSS.log
```

Example: `seriebot_20251028_123336.log`

## Log Levels

- **DEBUG**: Detailed information for diagnosing problems (file only)
- **INFO**: Confirmation that things are working as expected (file only)
- **WARNING**: An indication that something unexpected happened (file + console)
- **ERROR**: A more serious problem (file + console)
- **CRITICAL**: A serious error that may prevent the program from continuing (file + console)

## Log Contents

Each log file includes:
- Session initialization details
- User input and normalized versions
- LLM preference extraction results
- TMDB API calls and responses
- Genre and freshness scoring calculations
- Error stack traces when issues occur
- Feedback system operations

## Configuration

Enable debug console output by setting `DEBUG=true` in `.env`.

Default log level: `DEBUG` (file), `WARNING` (console)

## Git Ignore

Log files are excluded from version control via `.gitignore`.
Only the `.gitkeep` file is tracked to preserve directory structure.
