"""Prepare the CMU movie data and build a persistent ChromaDB collection."""

from __future__ import annotations

import argparse
import html
import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PLOTS_PATH = DATA_DIR / "plot_summaries.txt"
METADATA_PATH = DATA_DIR / "movie.metadata.tsv"
JSON_PATH = DATA_DIR / "movies.json"
CHROMA_PATH = PROJECT_DIR / "chroma_store"
MODEL_CACHE_PATH = PROJECT_DIR / ".model_cache"
COLLECTION_NAME = "movie_plots"
CHUNK_SIZE_WORDS = 90
CHUNK_OVERLAP_WORDS = 20
BATCH_SIZE = 100
TOP_K = 3
SEARCH_CANDIDATE_COUNT = 30


@dataclass(frozen=True)
class Movie:
    movie_id: str
    title: str
    genres: tuple[str, ...]
    plot: str


@dataclass(frozen=True)
class SearchResult:
    movie_id: str
    title: str
    genre: str
    matched_plot: str
    distance: float


def clean_plot(raw_plot: str) -> str:
    """Remove common Wikipedia markup while preserving readable plot text."""
    text = html.unescape(raw_plot)

    # Some summaries begin with {{plot}} or a malformed Plot template that
    # contains another template before its first closing braces.
    text = re.sub(
        r"^\s*\{\{\s*plot\b.*?\}\}\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # The corpus contains malformed references such as ``<ref ... }}``.
    # Remove these before stripping ordinary templates so the citation payload
    # does not remain after its closing braces disappear.
    text = re.sub(r"<ref\b.*?\}\}", " ", text, flags=re.IGNORECASE | re.DOTALL)

    # One malformed reference runs into a valid section labelled "Secondary
    # plot". Keep that section, then discard any other unterminated reference.
    text = re.sub(
        r"<ref\b.*?\b(Secondary plot\b)",
        r" \1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<ref\b.*$", " ", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove well-formed references, comments, and short Wikipedia templates.
    text = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<ref\b[^>]*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)

    # Preserve the readable label in [[article|label]] and [[article]] links.
    text = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Clean up malformed leftover brackets, HTML tags, URLs, and emphasis marks.
    text = text.replace("[[", "").replace("]]", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"={2,}", " ", text)

    # Remove any isolated template braces without discarding nearby prose.
    text = text.replace("{{", " ").replace("}}", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_genres(raw_genres: str) -> tuple[str, ...]:
    """Convert the CMU genre JSON object into a tuple of genre names."""
    if not raw_genres:
        return ()

    try:
        genre_map = json.loads(raw_genres)
    except json.JSONDecodeError:
        return ()

    if not isinstance(genre_map, dict):
        return ()

    return tuple(str(name) for name in genre_map.values())


def load_metadata(path: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return movie_id -> (title, genres) from movie.metadata.tsv."""
    metadata: dict[str, tuple[str, tuple[str, ...]]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 9:
                print(f"Skipping malformed metadata row {line_number}.")
                continue

            movie_id = columns[0]
            title = columns[2]
            genres = parse_genres(columns[8])
            metadata[movie_id] = (title, genres)

    return metadata


def load_movies(limit: int | None = None) -> list[Movie]:
    """Join plot summaries to metadata using the Wikipedia movie ID."""
    if not PLOTS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset files were not found. Expected:\n"
            f"  {PLOTS_PATH}\n"
            f"  {METADATA_PATH}"
        )

    metadata = load_metadata(METADATA_PATH)
    movies: list[Movie] = []

    with PLOTS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            movie_id, separator, raw_plot = line.rstrip("\n").partition("\t")
            plot = clean_plot(raw_plot)
            if not separator or not plot:
                print(f"Skipping malformed plot row {line_number}.")
                continue

            movie_metadata = metadata.get(movie_id)
            if movie_metadata is None:
                continue

            title, genres = movie_metadata
            movies.append(
                Movie(
                    movie_id=movie_id,
                    title=title,
                    genres=genres,
                    plot=plot.strip(),
                )
            )

            if limit is not None and len(movies) >= limit:
                break

    return movies


def preview_movies(movies: list[Movie]) -> None:
    """Print a readable preview so the joined data can be checked."""
    for index, movie in enumerate(movies, start=1):
        genre_text = ", ".join(movie.genres) if movie.genres else "Unknown"
        plot_preview = movie.plot[:150]
        if len(movie.plot) > 150:
            plot_preview += "..."

        print(f"\n{index}. {movie.title}")
        print(f"   Movie ID: {movie.movie_id}")
        print(f"   Genres: {genre_text}")
        print(f"   Plot: {plot_preview}")


def export_movies_json(movies: list[Movie], output_path: Path = JSON_PATH) -> None:
    """Write all cleaned movie records to a readable JSON file."""
    records = [
        {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "genres": list(movie.genres),
            "plot": movie.plot,
        }
        for movie in movies
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Exported {len(records)} cleaned movies to {output_path}.")


def load_movies_from_json(path: Path = JSON_PATH) -> list[Movie]:
    """Load cleaned movie records from the generated JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Movie JSON was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("movies.json must contain a JSON list")

    movies: list[Movie] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid movie record at position {index}")

        movie_id = str(record.get("movie_id", "")).strip()
        title = str(record.get("title", "")).strip()
        plot = clean_plot(str(record.get("plot", "")))
        raw_genres = record.get("genres", [])
        genres = (
            tuple(str(genre).strip() for genre in raw_genres if str(genre).strip())
            if isinstance(raw_genres, list)
            else ()
        )

        if not movie_id or not title or not plot:
            raise ValueError(f"Incomplete movie record at position {index}")

        movies.append(Movie(movie_id, title, genres, plot))

    return movies


def chunk_plot(
    plot: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split a long plot into overlapping chunks safe for local embeddings."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = plot.split()
    if len(words) <= chunk_size:
        return [plot]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


def get_chroma_collection():
    """Open the persistent collection with the project-local model cache."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )
    except ImportError as error:
        raise RuntimeError(
            "ChromaDB is not installed. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from error

    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = (
        MODEL_CACHE_PATH / "onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def import_movies_to_collection(movies: list[Movie], collection) -> None:
    """Upsert cleaned plot chunks into an open ChromaDB collection."""

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    imported_chunks = 0

    def flush_batch() -> None:
        nonlocal imported_chunks
        if not ids:
            return

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        imported_chunks += len(ids)
        print(f"Embedded {imported_chunks} plot chunks...")
        ids.clear()
        documents.clear()
        metadatas.clear()

    for movie in movies:
        chunks = chunk_plot(movie.plot)
        genre_text = ", ".join(movie.genres) if movie.genres else "Unknown"

        for chunk_index, chunk in enumerate(chunks):
            ids.append(f"{movie.movie_id}_chunk_{chunk_index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "movie_id": movie.movie_id,
                    "title": movie.title,
                    "genre": genre_text,
                    "chunk_index": chunk_index,
                    "source": (
                        "user" if movie.movie_id.startswith("user_") else "cmu"
                    ),
                }
            )

            if len(ids) >= BATCH_SIZE:
                flush_batch()

    flush_batch()
    print(
        f"ChromaDB import complete: {len(movies)} movies, "
        f"{collection.count()} stored chunks, path={CHROMA_PATH}."
    )


def import_movies_to_chroma(movies: list[Movie]) -> None:
    """Open the persistent database and import movie records."""
    import_movies_to_collection(movies, get_chroma_collection())


def ensure_movie_json() -> None:
    """Create movies.json from the raw subset only when it is missing."""
    if JSON_PATH.exists():
        return

    print("movies.json is missing; generating it from the source files...")
    export_movies_json(load_movies())


def initialize_database():
    """Load a persistent database, importing JSON only on the first run."""
    collection = get_chroma_collection()
    stored_chunks = collection.count()

    if stored_chunks > 0:
        print(
            f"Loaded persistent database: {stored_chunks} plot chunks "
            f"from {CHROMA_PATH}."
        )
        return collection

    ensure_movie_json()
    movies = load_movies_from_json()
    print(f"First run detected. Importing {len(movies)} movies from {JSON_PATH}...")
    import_movies_to_collection(movies, collection)
    return collection


def search_movies(collection, query: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Return the closest distinct movies, deduplicating matching chunks."""
    if not query.strip() or collection.count() == 0:
        return []

    candidate_count = min(collection.count(), max(SEARCH_CANDIDATE_COUNT, top_k))
    response = collection.query(
        query_texts=[query.strip()],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    results: list[SearchResult] = []
    seen_movie_ids: set[str] = set()

    for document, metadata, distance in zip(documents, metadatas, distances):
        if not metadata:
            continue

        movie_id = str(metadata.get("movie_id", ""))
        if not movie_id or movie_id in seen_movie_ids:
            continue

        seen_movie_ids.add(movie_id)
        results.append(
            SearchResult(
                movie_id=movie_id,
                title=str(metadata.get("title", "Unknown title")),
                genre=str(metadata.get("genre", "Unknown")),
                matched_plot=str(document or ""),
                distance=float(distance),
            )
        )
        if len(results) >= top_k:
            break

    return results


def core_situation(text: str, max_length: int = 230) -> str:
    """Extract a short readable situation from the matched plot chunk."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:2]).strip()
    if len(summary) <= max_length:
        return summary
    return summary[: max_length - 3].rsplit(" ", 1)[0] + "..."


def distance_assessment(distance: float) -> str:
    """Give a cautious interpretation of the default Chroma distance."""
    if distance < 0.90:
        return "strong semantic match"
    if distance < 1.30:
        return "moderate semantic match"
    return "exploratory match"


def split_genres(genre_text: str) -> list[str]:
    return [genre.strip() for genre in genre_text.split(",") if genre.strip()]


def display_search_results(query: str, results: list[SearchResult]) -> None:
    """Display Top-3 results and a deterministic viewing decision report."""
    if not results:
        print("No searchable movie results were found.")
        return

    print("\n=== Top matching movies ===")
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result.title}")
        print(f"   Genre: {result.genre}")
        print(f"   Distance: {result.distance:.4f} (lower is more similar)")
        print(f"   Core situation: {core_situation(result.matched_plot)}")

    genre_counts = Counter(
        genre for result in results for genre in split_genres(result.genre)
    )
    all_genres = list(genre_counts)
    shared_genres = [genre for genre, count in genre_counts.items() if count >= 2]
    best = results[0]
    distance_spread = results[-1].distance - best.distance

    print("\n=== Smart viewing decision report ===")
    print(f"Your request: {query}")
    print(f"Genres covered: {', '.join(all_genres) if all_genres else 'Unknown'}")
    if shared_genres:
        print(f"Shared genre signals: {', '.join(shared_genres)}")
    else:
        print("Shared genre signals: none; the results approach the idea from varied genres.")

    print("Story-angle comparison:")
    for result in results:
        result_genres = split_genres(result.genre)
        distinctive = [genre for genre in result_genres if genre not in shared_genres]
        angle = ", ".join(distinctive or result_genres) or "uncategorized story"
        print(f"- {result.title}: {angle}")

    print(
        f"Match assessment: the best result is a {distance_assessment(best.distance)} "
        f"at {best.distance:.4f}. The Top-{len(results)} distance spread is "
        f"{distance_spread:.4f}."
    )
    print(
        f"Recommendation: start with {best.title} because it is the closest semantic "
        "match to your description."
    )

    if len(results) > 1:
        alternatives = ", ".join(result.title for result in results[1:])
        print(
            f"Choose {alternatives} if you want to explore the same underlying idea "
            "through different story situations or genre combinations."
        )
    if best.distance >= 1.30:
        print(
            "Caution: all matches are relatively distant. Treat them as exploratory "
            "suggestions and try a more specific description if they do not fit."
        )


def append_movie_to_json(movie: Movie, path: Path = JSON_PATH) -> None:
    """Persist a user-added movie in JSON as well as in ChromaDB."""
    records = []
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("movies.json must contain a JSON list")

    records.append(
        {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "genres": list(movie.genres),
            "plot": movie.plot,
        }
    )
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")


def add_movie(collection, title: str, genres: tuple[str, ...], plot: str) -> Movie:
    """Embed a new movie immediately and persist it under a unique ID."""
    cleaned_title = title.strip()
    cleaned_plot = clean_plot(plot)
    if not cleaned_title or not cleaned_plot:
        raise ValueError("Title and plot cannot be empty")

    movie = Movie(
        movie_id=f"user_{uuid.uuid4().hex}",
        title=cleaned_title,
        genres=genres,
        plot=cleaned_plot,
    )
    chunks = chunk_plot(movie.plot)
    genre_text = ", ".join(movie.genres) if movie.genres else "Unknown"
    collection.upsert(
        ids=[f"{movie.movie_id}_chunk_{index}" for index in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genre": genre_text,
                "chunk_index": index,
                "source": "user",
            }
            for index in range(len(chunks))
        ],
    )
    append_movie_to_json(movie)
    return movie


def show_statistics(collection) -> None:
    """Show collection size, unique movies, sources, and common genres."""
    stored = collection.get(include=["metadatas"])
    metadatas = stored.get("metadatas") or []
    movie_ids: set[str] = set()
    movie_genres: dict[str, list[str]] = {}
    sources: dict[str, str] = {}

    for metadata in metadatas:
        if not metadata:
            continue
        movie_id = str(metadata.get("movie_id", ""))
        if not movie_id:
            continue
        movie_ids.add(movie_id)
        movie_genres[movie_id] = split_genres(str(metadata.get("genre", "")))
        sources[movie_id] = str(metadata.get("source", "cmu"))

    genre_counts = Counter(
        genre for genres in movie_genres.values() for genre in genres if genre != "Unknown"
    )
    source_counts = Counter(sources.values())
    average_chunks = collection.count() / len(movie_ids) if movie_ids else 0

    print("\n=== Database statistics ===")
    print(f"Movies: {len(movie_ids)}")
    print(f"Plot chunks: {collection.count()}")
    print(f"Average chunks per movie: {average_chunks:.2f}")
    print(f"CMU movies: {source_counts.get('cmu', 0)}")
    print(f"User-added movies: {source_counts.get('user', 0)}")
    if genre_counts:
        top_genres = ", ".join(
            f"{genre} ({count})" for genre, count in genre_counts.most_common(8)
        )
        print(f"Most common genres: {top_genres}")


def run_cli(collection) -> None:
    """Run the interactive menu used for testing and video demonstrations."""
    while True:
        print(
            "\n=== Movie Mood & Theme Recommender ===\n"
            "1. Search movies\n"
            "2. Add a movie\n"
            "3. Database statistics\n"
            "4. Exit"
        )
        choice = input("Choose an option: ").strip()

        if choice == "1":
            query = input("Describe what you want to watch (English recommended): ").strip()
            if not query:
                print("Please enter a description.")
                continue
            display_search_results(query, search_movies(collection, query))
        elif choice == "2":
            title = input("Movie title: ").strip()
            genre_input = input("Genres (comma-separated): ").strip()
            plot = input("Plot summary: ").strip()
            genres = tuple(
                dict.fromkeys(
                    genre.strip() for genre in genre_input.split(",") if genre.strip()
                )
            )
            try:
                movie = add_movie(collection, title, genres, plot)
            except ValueError as error:
                print(f"Could not add movie: {error}")
                continue
            print(f"Added {movie.title} with ID {movie.movie_id}.")
            print("It is searchable immediately and will persist after restart.")
        elif choice == "3":
            show_statistics(collection)
        elif choice == "4":
            print("Goodbye!")
            return
        else:
            print("Please choose 1, 2, 3, or 4.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    preview_parser = subparsers.add_parser("preview", help="Preview joined data.")
    preview_parser.add_argument("--limit", type=int, default=10)

    subparsers.add_parser("run", help="Start the interactive application.")
    subparsers.add_parser("export-json", help="Create data/movies.json.")
    subparsers.add_parser("import-db", help="Import movies into ChromaDB.")
    subparsers.add_parser("build", help="Create JSON and import ChromaDB.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "run"

    if command == "run":
        try:
            run_cli(initialize_database())
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
        return

    if command == "preview":
        movies = load_movies(limit=args.limit)
        print(f"Loaded {len(movies)} movies for preview.")
        preview_movies(movies)
        return

    if command == "export-json":
        movies = load_movies()
    else:
        ensure_movie_json()
        movies = load_movies_from_json()
    print(f"Loaded {len(movies)} cleaned movies.")

    if command in {"export-json", "build"}:
        export_movies_json(movies)
    if command in {"import-db", "build"}:
        import_movies_to_chroma(movies)


if __name__ == "__main__":
    main()
