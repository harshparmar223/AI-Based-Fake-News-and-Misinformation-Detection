# Datasets and Preprocessing

## Dataset Sources

The primary training dataset consists of the **George McIntire Real & Fake News Dataset** alongside the structure of the **ISOT Fake News Dataset** (University of Victoria).

### Dataset Composition
- **Total Articles Processed**: 6,042 articles
- **Real News Articles**: 2,989 articles
- **Fake News Articles**: 3,053 articles
- **Class Balance**: 50.5% Fake / 49.5% Real (essentially a perfectly balanced benchmark)

### File Organization
```text
data/
├── raw/
│   ├── fake_or_real_news.csv    # Consolidated dataset with title, text, label
│   ├── True.csv                 # Reliable news articles (Reuters, AP style)
│   └── Fake.csv                 # Misinformation, conspiracy, and satire articles
└── processed/
    ├── train.csv                # 4,833 stratified training instances (80%)
    ├── test.csv                 # 1,209 stratified testing instances (20%)
    └── dataset_summary.json     # Preprocessing metadata and summary statistics
```

## Data Fields

| Column | Type | Description |
|---|---|---|
| `title` | String | Original article headline or title |
| `text` | String | Raw article body content |
| `clean_text` | String | HTML-stripped, lowercased, punctuation-cleaned text |
| `label` | String | Discrete ground truth (`REAL` or `FAKE`) |
| `target` | Integer | Binary target (`0` = REAL, `1` = FAKE) |
| `word_count` | Integer | Total words after text cleaning |
| `char_count` | Integer | Total characters after cleaning |

## Preprocessing Pipeline

1. **Deduplication**: Exact text duplicates removed.
2. **Text Normalization**:
   - HTML markup stripped using custom parser.
   - Non-ASCII/Unicode normalized to NFC form.
   - URLs and hyperlinks scrubbed.
   - Punctuation removed (apostrophes preserved for standard contractions).
   - Multi-whitespace collapsed into single space.
3. **Filtering**: Articles under 5 words or 30 characters discarded to prevent trivial uninformative samples.
4. **Stratified Splitting**: 80/20 train/test split maintaining exact class proportions across splits.
