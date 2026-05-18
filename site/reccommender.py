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



def run_recommender(watched_df):

    tmdb.API_KEY = os.getenv("TMDB_API_KEY")
    tmdb.REQUESTS_TIMEOUT = 5 



    watched_df = watched_df.rename(columns={"Name": "entry_title", "Rating": "entry_rating"})
    watched_df["entry_published"] = pd.to_datetime(watched_df["Date"]).dt.strftime("%a, %-d %b %Y %H:%M:%S +0000")
    


    movie_ids, tv_ids = [], []
    
    for _, row in watched_df.iterrows():
        title = row["entry_title"]
        # Extract year from the Letterboxd "Year" column if present
        year = row.get("Year")
        year = int(year) if pd.notna(year) and str(year).isdigit() else None
    
        m_id, t_id = search_tmdb(title, year)
        movie_ids.append(m_id)
        tv_ids.append(t_id)
    
        time.sleep(0.25)   # stay well within TMDB rate limits (40 req/10 s)
    
    watched_df["movie_id"] = movie_ids
    watched_df["tv_id"]    = tv_ids




    movie_df = watched_df.dropna(subset = ["movie_id"])
    movie_df['tv_id'] = pd.to_numeric(movie_df['tv_id'])
    movie_df['movie_id'] = pd.to_numeric(movie_df['movie_id'])



    movie_df['genres'] = None  # resets the column to object dtype

    for movieId in movie_df['movie_id']:
        movie = tmdb.Movies(int(movieId))
        response = movie.info()
        idx = movie_df[movie_df['movie_id'] == movieId].index[0]
        movie_df.at[idx, 'genres'] = ', '.join([g['name'] for g in movie.genres])
        time.sleep(0.1)



    
    tv_df = watched_df.dropna(subset = ["tv_id"])
    tv_df['movie_id'] = pd.to_numeric(tv_df['movie_id'])
    tv_df['tv_id'] = pd.to_numeric(tv_df['tv_id'])

    tv_df['genres'] = None  # resets the column to object dtype



    movie = tmdb.Movies()
    movies = []
    page = 1
    seen_ids = set(watched_df['movie_id'].dropna().astype(int).tolist())

    while len(movies) < 1000:
        response = movie.top_rated(page=page)
        for item in response['results']:
            if item['id'] not in seen_ids:
                movies.append(item)
        page += 1
        time.sleep(.1)

    print(response["results"][0]["title"])

    len(movies)

    popular_df = pd.DataFrame(movies)
    popular_df = get_crew(popular_df)




    for tvId in tv_df['tv_id']:
        tv = tmdb.TV(int(tvId))
        response = tv.info()
        idx = tv_df[tv_df['tv_id'] == tvId].index[0]
        tv_df.at[idx, 'genres'] = ', '.join([g['name'] for g in tv.genres])
        time.sleep(0.1)
    
    tfidf = TfidfVectorizer(stop_words='english')
    popular_df['overview'] = popular_df['overview'].fillna('')


    get_crew(popular_df)
    
    
    

    features = ['actors', 'keywords', 'director', 'genres', "original_title"]

    genres = tmdb.Genres()
    response = genres.movie_list()

    merged = {d['id']: d['name'] for d in response['genres']}

    popular_df['genres'] = popular_df['genre_ids'].apply(lambda ids: [merged[i] for i in ids if i in merged])

    for feature in features:
        popular_df[feature] = popular_df[feature].apply(clean_data)
    
    popular_df['soup'] = popular_df.apply(create_soup, axis=1)

    idx = popular_df[popular_df['id'] == 497].index[0]
    idx
    popular_df.loc[idx, 'soup']


    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(popular_df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    popular_df = popular_df.reset_index()
    indices = pd.Series(popular_df.index, index=popular_df['title'])
    
    watched_pairs = list(zip(watched_df['entry_title'], watched_df['entry_rating']))

    for movieId in movie_df['movie_id']:
        movie = tmdb.Movies(int(movieId))
        response = movie.info()
        idx = movie_df[movie_df['movie_id'] == movieId].index[0]
        movie_df.at[idx, 'overview'] = response['overview']
        time.sleep(0.2)
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(get_credits, movie_df['movie_id']))

    movie_df["director"], movie_df["actors"], movie_df["keywords"] = zip(*results)
    features = ['actors', 'keywords', 'director', 'genres', 'entry_title']

    for feature in features:
        movie_df[feature] = movie_df[feature].apply(clean_data)
    
    movie_df['soup'] = movie_df.apply(create_soup, axis=1)

    all_soup = pd.concat([popular_df['soup'], movie_df['soup']], ignore_index=True)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(all_soup)

    n_popular = len(popular_df)
    popular_matrix = count_matrix[:n_popular]   # candidate pool
    watched_matrix = count_matrix[n_popular:]   # your watched movies
    ratings = movie_df['entry_rating']
    weights = ratings / ratings.sum()
    taste_profile = weights @ watched_matrix.toarray()

    scores = cosine_similarity([taste_profile], popular_matrix)[0]

    watched_ids = set(movie_df['movie_id'].dropna().astype(int))
    results = []
    for idx, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True):
        if popular_df.iloc[idx]['id'] not in watched_ids:
            results.append(popular_df.iloc[idx]['title'])
        if len(results) == 10:
            break

    results_df = pd.DataFrame(results, columns=['top movies'])
    return results_df




def search_tmdb(title, year=None):
    """
    Try a movie search first, then fall back to TV.
    Returns (movie_id, tv_id) — one will always be NaN.
    """
    search = tmdb.Search()
 
    # Movie search
    kwargs = {"query": title}
    if year:
        kwargs["year"] = year
    search.movie(**kwargs)
    if search.results:
        return float(search.results[0]["id"]), float("nan")
 
    # TV search (no year filter — TMDB TV search ignores it anyway)
    search.tv(query=title)
    if search.results:
        return float("nan"), float(search.results[0]["id"])
 
    return float("nan"), float("nan")

def get_crew(df):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from concurrent.futures import ThreadPoolExecutor

    def get_credits(movie_id):
        try:
            movie = tmdb.Movies(movie_id)
            credits = movie.credits()
            
            director = next(
                (member["name"] for member in credits["crew"] if member["job"] == "Director"),
                "Director not found"
            )
            actors = [member["name"] for member in credits["cast"][:5]]
            keyworddict = movie.keywords()
            keywords = [kw["name"] for kw in keyworddict["keywords"][:3]]
            
            return director, actors, keywords
        except Exception as e:
            return "Error", []

    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(get_credits, df["id"]))

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

def get_credits(movie_id):
    try:
        movie = tmdb.Movies(movie_id)
        credits = movie.credits()
        
        director = next(
            (member["name"] for member in credits["crew"] if member["job"] == "Director"),
            "Director not found"
        )
        actors = [member["name"] for member in credits["cast"][:5]]
        keyworddict = movie.keywords()
        keywords = [kw["name"] for kw in keyworddict["keywords"][:3]]
        
        return director, actors, keywords
    except Exception as e:
        return "Error", []









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


    return df


def process_file(filepath):
    df = pd.read_csv(filepath)
    return df


def return_data(data):
    return data