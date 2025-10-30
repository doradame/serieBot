# Feedback Directory

This directory stores user feedback data collected by SerieBot.

## Feedback File

The feedback system automatically creates and maintains:
```
feedback.csv
```

## CSV Format

The feedback file contains the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | ISO 8601 timestamp of when feedback was given |
| `session_id` | Unique session identifier (YYYYMMDD_HHMMSS) |
| `rating` | User rating: `positive` or `negative` |
| `genre` | Requested genre (e.g., sci-fi, comedy, drama) |
| `mood` | Requested mood (e.g., intense, light-hearted) |
| `providers` | Streaming providers requested (e.g., Netflix, Prime Video) |
| `language` | Content language preference (e.g., en, it, any) |
| `country` | User's country code (detected via geolocation) |
| `suggestions_count` | Number of suggestions shown to user |
| `comment` | Optional user comment (for negative feedback) |

## Example Entry

```csv
timestamp,session_id,rating,genre,mood,providers,language,country,suggestions_count,comment
2025-10-28T12:54:33,20251028_125433,positive,sci-fi,intense,Netflix,en,IT,3,""
2025-10-28T13:15:22,20251028_131522,negative,comedy,light-hearted,Disney+,en,US,2,"Not funny enough"
```

## Usage

This data can be used for:

- **Quality Analysis**: Identify which preference combinations work best
- **Algorithm Improvement**: Detect gaps in the recommendation system
- **User Satisfaction Tracking**: Monitor overall user happiness over time
- **A/B Testing**: Compare different recommendation strategies
- **Personalization**: Build user preference profiles for future enhancements

## Privacy

Feedback files are excluded from version control via `.gitignore` to protect user privacy.
Only the `.gitkeep` file is tracked to preserve directory structure.

## Git Ignore

Feedback CSV files are excluded from version control.
Only the `.gitkeep` file is tracked to preserve directory structure.
