import os
import re
import csv
import json
import time
import requests
import wikipediaapi
from langdetect import detect

def clean_filename(filename):
    """
    Sanitizes a filename by replacing illegal characters with underscores.

    Parameters
    ----------
    filename : str
        The original filename to sanitize.

    Returns
    -------
    str
        A sanitized filename safe for filesystem use.
    """
    return re.sub(r'[\\/*?<>|:"\n]', '_', filename)

def ensure_folder(path):
    """
    Ensures that a folder exists at the specified path. Creates it if it does not exist.

    Parameters
    ----------
    path : str
        The path to the folder.
    """
    os.makedirs(path, exist_ok=True)

def load_config(config_path="config.json"):
    """
    Loads a JSON configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to the config file. Defaults to "config.json".

    Returns
    -------
    dict
        Parsed JSON configuration as a dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    """
    Cleans raw text by removing URLs, HTML tags, anchor elements, and extra whitespace.

    Parameters
    ----------
    text : str
        The input text to clean.

    Returns
    -------
    str
        Cleaned and normalized text.
    """
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'<img [^>]*>', '', text)
    text = re.sub(r'<a [^>]*>', '', text)
    text = re.sub(r'</a>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def download_from_wikipedia(topic, language, user_agent):
        """
    Downloads a Wikipedia page and all its linked pages, cleans the text, and saves them as .txt files.

    Parameters
    ----------
    topic : str
        The main topic to search in Wikipedia.

    language : str
        Language code for Wikipedia (e.g., "en", "es").

    user_agent : str
        A custom user-agent string for the Wikipedia API request.

    Returns
    -------
    None
    """
    print(f"[Wikipedia] Downloading articles for '{topic}' in '{language}'...")
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=language)
    page = wiki_wiki.page(topic)

    if not page.exists():
        print(f"[Wikipedia] Page not found for '{topic}' in '{language}'")
        return

    folder_path = os.path.join("Data", topic, "Wikipedia", language)
    ensure_folder(folder_path)

    clean_title = clean_filename(page.title)
    with open(os.path.join(folder_path, f"{clean_title}.txt"), "w", encoding="utf-8") as f:
        f.write(clean_text(page.text))
    print(f"[Wikipedia] Saved: {clean_title}.txt")

    for link in page.links:
        related_page = wiki_wiki.page(link)
        if related_page.exists():
            title = clean_filename(related_page.title)
            with open(os.path.join(folder_path, f"{title}.txt"), "w", encoding="utf-8") as f:
                f.write(clean_text(related_page.text))
            print(f"[Wikipedia] Saved: {title}.txt")

def download_from_scopus(topic, languages, api_key, max_articles):
    """
    Downloads academic articles from the Scopus API, saves them as text and CSV files categorized by language.

    Parameters
    ----------
    topic : str
        Topic to search for in Scopus (used in the title, abstract, or keywords).

    languages : list of str
        List of language codes to filter and save articles (e.g., ["en", "es"]).

    api_key : str
        API key for authenticating with the Scopus API.

    max_articles : int
        Maximum number of articles to download across all languages.

    Returns
    -------
    None
    """
    print(f"[Scopus] Downloading articles for '{topic}'...")
    base_url = 'https://api.elsevier.com/content/search/scopus'
    query = f'TITLE-ABS-KEY("{topic}")'
    headers = {'X-ELS-APIKey': api_key, 'Accept': 'application/json'}

    start = 0
    count = 25
    downloaded = 0
    total_results = None

    csv_files = {}
    csv_writers = {}

    for lang in languages:
        folder = os.path.join("Data", topic, "Scopus", lang)
        ensure_folder(folder)
        csv_path = os.path.join(folder, f"{lang}_articles.csv")
        csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow(["Title", "Authors", "Keywords", "Abstract", "URL"])
        csv_files[lang] = csv_file
        csv_writers[lang] = writer

    while True:
        params = {'query': query, 'count': count, 'start': start, 'sort': 'relevance'}
        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"[Scopus] Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        if total_results is None:
            total_results = int(data.get('search-results', {}).get('opensearch:totalResults', 0))
            print(f"[Scopus] Total articles founded: {total_results}")

        entries = data.get('search-results', {}).get('entry', [])
        if not entries:
            break

        for entry in entries:
            if downloaded >= max_articles:
                print(f"[Scopus] Limit of {max_articles} articles have been reached.")
                for f in csv_files.values():
                    f.close()
                return

            try:
                title = entry.get('dc:title', 'Sin título')
                abstract = entry.get('dc:description', '')
                keywords = entry.get('authkeywords', '')
                authors = ', '.join([a.get('authname', '') for a in entry.get('author', [])]) if 'author' in entry else 'N/A'
                language = entry.get('dc:language', 'unknown')
                url = next((l.get('@href') for l in entry.get('link', []) if l.get('@ref') == 'scopus'), '')

                if language == 'unknown':
                    try:
                        language = detect(abstract or title)
                        print(f"[Scopus] Language automatically detected: {language}")
                    except:
                        continue

                if language not in languages:
                    continue

                folder = os.path.join("Data", topic, "Scopus", language)
                ensure_folder(folder)
                clean_title = clean_filename(title)
                file_path = os.path.join(folder, f"{clean_title}.txt")
                full_text = f"TITLE: {title}\n\nAUTHORS: {authors}\n\nKEYWORDS: {keywords}\n\nABSTRACT:\n{abstract}"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)

                csv_writers[language].writerow([title, authors, keywords, abstract, url])
                downloaded += 1
                print(f"[Scopus] Saving: {clean_title}.txt")
            except Exception as e:
                print(f"[Scopus] Error: {e}")
                continue

        start += count
        if start >= total_results:
            break
        time.sleep(0.5)

    for f in csv_files.values():
        f.close()

def main():
    config = load_config()
    topics = config["topics"]
    languages = config["languages"]
    api_key = config["api_key_scopus"]
    max_articles = config["max_articles_scopus"]
    user_agent = config["user_agent_wiki"]

    for topic in topics:
        for lang in languages:
            download_from_wikipedia(topic, lang, user_agent)
        download_from_scopus(topic, languages, api_key, max_articles)

if __name__ == "__main__":
    main()
