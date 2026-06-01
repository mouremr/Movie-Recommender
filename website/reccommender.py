import pandas as pd
import tmdbsimple as tmdb
import time
import os
from dotenv import load_dotenv
import feedparser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import pickle
import threading
from upstash_redis import Redis
import json

load_dotenv()

CACHE_FILE = "data/popular_cache.pkl"
CACHE_TTL = 86400

r = Redis.from_env()


def init_movie_db():
    tmdb.API_KEY = os.getenv("TMDB_API_KEY")
    tmdb.REQUESTS_TIMEOUT = 5



def run_recommender(watched_df):
    
    init_movie_db()


    watched_df = watched_df.rename(columns={"Name": "entry_title", "Rating": "entry_rating"})    
    watched_df = watched_df[['entry_title', 'entry_rating', 'movie_id', 'tv_id']]
    watched_df = watched_df.drop_duplicates(subset=['entry_title'])
    # return watched_df

    # movie_ids, tv_ids = [], []
    
    # rows = [row for _, row in watched_df.iterrows()]
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     id_results = list(executor.map(search_tmdb, rows))

    # movie_ids, tv_ids = zip(*id_results)
    # watched_df["movie_id"] = movie_ids
    # watched_df["tv_id"] = tv_ids
    # watched_df["movie_id"] = pd.to_numeric(watched_df["movie_id"])
    # watched_df["tv_id"] = pd.to_numeric(watched_df["tv_id"])

    movie_df = watched_df.dropna(subset=["movie_id"]).copy()
    tv_df = watched_df.dropna(subset = ["tv_id"]).copy()



    movie_df['tv_id'] = pd.to_numeric(movie_df['tv_id'])
    movie_df['movie_id'] = pd.to_numeric(movie_df['movie_id'])
    movie_df['genres'] = None




    with ThreadPoolExecutor(max_workers=15) as executor:
        genre_results = list(executor.map(fetch_genres, movie_df["movie_id"]))
    movie_df["genres"] = genre_results
    movie_df.rename(columns={"movie_id" : "id"}, inplace = True)
    movie_df = get_crew(movie_df, "watched movies")





    
    tv_df = watched_df.dropna(subset = ["tv_id"])
    tv_df['movie_id'] = pd.to_numeric(tv_df['movie_id'])
    tv_df['tv_id'] = pd.to_numeric(tv_df['tv_id'])

    tv_df['genres'] = None  # resets the column to object dtype




    popular_df = get_top_movies_with_crew(watched_df)











    # popular_df['overview'] = popular_df['overview'].fillna('')


    
    
    

    features = ['actors', 'keywords', 'director', 'genres', 'original_title']

    genres = genre_mapping()

    popular_df['genres'] = popular_df['genre_ids'].apply(lambda ids: [genres[i] for i in ids if i in genres])

    for feature in features:
        popular_df[feature] = popular_df[feature].apply(clean_data)
    
    popular_df['soup'] = popular_df.apply(create_soup, axis=1)

    popular_df = popular_df.reset_index()


    features = ['actors', 'keywords', 'director', 'genres', 'entry_title']
    for feature in features:
        movie_df[feature] = movie_df[feature].apply(clean_data)
    
    movie_df['soup'] = movie_df.apply(create_soup, axis=1)

    all_soup = pd.concat([popular_df['soup'], movie_df['soup']], ignore_index=True)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(all_soup)

    n_popular = len(popular_df)
    popular_matrix = count_matrix[:n_popular]   # popular movies
    watched_matrix = count_matrix[n_popular:]   # your watched movies
    ratings = movie_df['entry_rating']
    weights = ratings / ratings.sum()
    taste_profile = weights @ watched_matrix.toarray()

    scores = cosine_similarity([taste_profile], popular_matrix)[0]

    # watched_ids = set(movie_df['id'].dropna().astype(int))
    results = []
    for idx, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True):
        results.append(popular_df.iloc[idx]['title'])
        if len(results) == 10:
            break

    results_df = pd.DataFrame(results, columns=['top movies'])
    return results_df




def search_tmdb(row):
    search = tmdb.Search()
    title = row["entry_title"]
    print("searching for: " + title)

    kwargs = {"query": title}

    for attempt in range(3):
        try:
            search.movie(**kwargs)
            time.sleep(0.25)
            if search.results:
                return float(search.results[0]["id"]), float("nan")

            search.tv(query=title)
            if search.results:
                return float("nan"), float(search.results[0]["id"])

            return float("nan"), float("nan")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for '{title}': {e}")
            time.sleep(1 * (attempt + 1))  # back off: 1s, 2s, 3s

    print(f"All retries failed for '{title}', skipping.")
    return float("nan"), float("nan")

def get_crew(df, name):
    print("Getting credits for " + name)
    def get_credits(movie_id):
        cache_key = f"crew:{int(movie_id)}"
        
        # Check cache first
        cached = r.get(cache_key)
        if cached:
            data = json.loads(cached)
            return data["director"], data["actors"], data["keywords"]
        
        # Otherwise fetch from TMDB
        try:
            movie = tmdb.Movies(movie_id)
            credits = movie.credits()
            director = next(
                (m["name"] for m in credits["crew"] if m["job"] == "Director"),
                ""
            )
            actors = [m["name"] for m in credits["cast"][:5]]
            keyworddict = movie.keywords()
            keywords = [kw["name"] for kw in keyworddict["keywords"][:3]]
            
            # Cache for 30 days (crew doesn't change)
            r.setex(cache_key, 2592000, json.dumps({
                "director": director,
                "actors": actors,
                "keywords": keywords
            }))
            return director, actors, keywords
        except Exception as e:
            print(f"Credits failed for {movie_id}: {e}")
            return "", [], []
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(get_credits, df["id"]))
    # return pd.DataFrame(results)
    df["director"], df["actors"], df["keywords"] = zip(*results)
    return df

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    director = x['director'] + ' ' + x['director']
    genres = ' '.join(x['genres']) + ' ' + ' '.join(x['genres'])
    actors = ' '.join(x['actors'][:3])
    keywords = ' '.join(x['keywords'])
    return f"{director} {genres} {actors} {keywords}"

def get_recommendations_weighted(df, watched_titles_with_ratings, cosine_sim, indices, top_n=10):
    """
    watched_titles_with_ratings: list of (title, rating) tuples
    e.g. [("The Avengers", 4.5), ("Parasite", 5.0)]
    """
    
    valid = [(indices[t], r) for t, r in watched_titles_with_ratings if t in indices]

    notmissing = [t for t, r in watched_titles_with_ratings if t in indices]
    if notmissing:
        print(f"{notmissing}")
    if not valid:
        return []

    watched_idx = {idx for idx, _ in valid}
    
    # Build a weighted sum of similarity rows
    total_weight = sum(r for _, r in valid)
    weighted_scores = sum(cosine_sim[idx] * (r / total_weight) for idx, r in valid)

    sim_scores = sorted(enumerate(weighted_scores), key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i not in watched_idx][:top_n]

    return df['title'].iloc[[i for i, _ in sim_scores]]













def genre_mapping():
    genres = tmdb.Genres()
    response = genres.movie_list()

    return {d['id']: d['name'] for d in response['genres']}


def get_top_movies(watched_df):
    _tmdb_semaphore = threading.Semaphore(5)
    seen_ids = set(watched_df['movie_id'].dropna().astype(int).tolist())
    os.makedirs("data", exist_ok=True)

    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        if age < CACHE_TTL:
            with open(CACHE_FILE, "rb") as f:
                print("Using cached popular movies.")
                return pickle.load(f)

    print("Fetching fresh popular movies from TMDB...")

    # TMDB returns 20 results per page so 50 pages = 1000 movies
    pages = range(1, 51)

    def fetch_page(page):
        with _tmdb_semaphore:
            try:
                movie = tmdb.Movies()
                response = movie.top_rated(page=page)
                time.sleep(0.1)
                return [item for item in response['results'] if item['id'] not in seen_ids] # filter out movies that have already been seen
            except Exception as e:
                print(f"Failed to fetch page {page}: {e}")
                return []

    with ThreadPoolExecutor(max_workers=10) as executor:
        page_results = list(executor.map(fetch_page, pages))

    # flatten results, preserving page order
    movies = [item for page in page_results for item in page]

    df = pd.DataFrame(movies)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(df, f)
    return df



def fetch_genres(movie_id):
    try:
        movie = tmdb.Movies(int(movie_id))
        info = movie.info()
        return ", ".join([g["name"] for g in movie.genres])
    except:
        return ""
    




def process_username(data):
    username = data
    feed = feedparser.parse(f"https://letterboxd.com/{username}/rss/")
    movieDict = {
        "entry_title" : [],
        "entry_published" : [],
        "entry_rating" : [],
        "movie_id": [],
        "tv_id" : [],
    }
    feed.entries[0].tmdb_movieid
    for entry in feed.entries:
        
        movieDict["entry_title"].append(entry.letterboxd_filmtitle)
        movieDict["entry_published"].append(entry.published)
        try:
            movieDict["movie_id"].append(entry.tmdb_movieid)
            movieDict["tv_id"].append(pd.NA)
        except:
            movieDict["movie_id"].append(pd.NA)
            movieDict["tv_id"].append(entry.tmdb_tvid)
        try:
            movieDict["entry_rating"].append(entry.letterboxd_memberrating)

        except:
            movieDict["entry_rating"].append(0)


    df = pd.DataFrame.from_dict(movieDict, orient='columns')

    df['movie_id'] = pd.to_numeric(df['movie_id'])
    df['tv_id'] = pd.to_numeric(df['tv_id'])
    df['entry_rating'] = pd.to_numeric(df['entry_rating']) 
    return df


def process_file(filepath):
    df = pd.read_csv(filepath)
    return df


CREW_CACHE_FILE = "data/popular_crew_cache.pkl"

def get_top_movies_with_crew(watched_df):
    print()
    if os.path.exists(CREW_CACHE_FILE):
        age = time.time() - os.path.getmtime(CREW_CACHE_FILE)
        if age < CACHE_TTL:
            with open(CREW_CACHE_FILE, "rb") as f:
                return pickle.load(f)
    
    df = get_top_movies(watched_df)
    df = get_crew(df, "top movies")
    
    with open(CREW_CACHE_FILE, "wb") as f:
        pickle.dump(df, f)
    return df