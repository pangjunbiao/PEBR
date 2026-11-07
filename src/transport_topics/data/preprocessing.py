"""Conservative normalization matching the locked primary-corpus pilot."""
import html
import re

WORD = re.compile(r'(?u)\b[a-z][a-z]+\b')


def clean_text(text):
    text = html.unescape(text).lower()
    text = re.sub(r'@user|httpurl|https?://\S+', ' ', text)
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def deduplicate(posts):
    kept, texts, duplicate_map, seen = [], [], {}, {}
    for post in posts:
        text = clean_text(post.original_text)
        if text in seen:
            duplicate_map[post.source_row] = seen[text]
        else:
            seen[text] = post.source_row
            kept.append(post); texts.append(text)
    return kept, texts, duplicate_map


def query_without(text, hidden_words):
    return re.sub(r'\s+', ' ', WORD.sub(lambda match: ' ' if match[0] in hidden_words else match[0], text)).strip()
