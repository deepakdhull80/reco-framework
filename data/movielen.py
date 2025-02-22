import os
import json
import pandas as pd

from utils import (
    train_test_split,
    dat_to_df,
    get_year,
    GloVeWrapper
)

do_id_based = True
base_path = "/Users/deepakdhull/data/recsys/ml-25m"
print(os.listdir(base_path))

rating_cols = "UserID::MovieID::Rating::Timestamp".split("::")
user_cols = "UserID::Gender::Age::Occupation::ZipCode".split("::")
movie_cols = "MovieID::Title::Genres".split("::")

#### column types #####

rating_types = {
    'UserID': 'int32',
    'MovieID': 'int32',
    'Rating': 'int32',
    'Timestamp': 'int32'
}

user_types = {
    'UserID': 'int32',
    'Gender': 'string',
    'Age': 'int32',
    'Occupation': 'int32',
    'ZipCode': 'string'
}

movie_types = {
    'MovieID': 'int32',
    'Title': 'string',
    'Genres': 'string'
}

if os.path.exists(f'{base_path}/ratings.dat'):
    sort_by_key = "Timestamp"
    rating_df = dat_to_df(f"{base_path}/ratings.dat", rating_cols, rating_types)
    user_df = dat_to_df(f"{base_path}/users.dat", user_cols, user_types)
    movie_df = dat_to_df(f"{base_path}/movies.dat", movie_cols, movie_types, format='rb')
else:
    sort_by_key = "timestamp"
    rating_df = pd.read_csv(f"{base_path}/ratings.csv")
    user_df = None
    movie_df = pd.read_csv(f"{base_path}/movies.csv")

print(rating_df.head(1))
if user_df is not None:
    print(user_df.head(1))
print(movie_df.head(1))


if do_id_based:
    print("Preparing Data ID based system")
    print(rating_df.info())
    print("Total number of events:", rating_df.shape[0])

    train_df, test_df = train_test_split(rating_df, sort_by_key=sort_by_key)
    train_df.to_parquet(f"{base_path}/train.pq", index=False)
    print("Files presents :", f"{base_path}/train.pq")
    test_df.to_parquet(f"{base_path}/val.pq", index=False)
    print("Files presents :", f"{base_path}/val.pq")
    exit()
######### MOVIE DF #############

movie_df['releaseYear'] = movie_df['Title'].map(get_year)
movie_year_count = movie_df['releaseYear'].value_counts()
movie_year_count = sorted([(i,y)for i, y in movie_year_count.items()], key=lambda x: int(x[0]))
x = [i[0] for i in movie_year_count]
y = [i[1] for i in movie_year_count]

def func_genre_split(x):
    x = x.split("|")
    return [i.split("'")[0] for i in x]

movie_df['Genres_list'] = movie_df['Genres'].map(func_genre_split)

genres = set()
for m in movie_df['Genres_list'].tolist():
    genres = genres.union(m)
print("number of genres: ", len(genres))
print(list(genres))

def h(s):
    s = s.split("(")[0].strip()
    return s

movie_df['Title'] = movie_df['Title'].map(h)

# embedding <<<
embedding_index = GloVeWrapper(base_path=base_path)
movie_df['embedding'] = movie_df['Title'].map(lambda t: embedding_index.get_embedding(t))
# <<<

for genre in genres:
    movie_df[genre] = movie_df['Genres_list'].map(lambda li: int(genre in li))

#######################################
if user_df is not None:
    print(user_df['Gender'].value_counts())
    gender_c = {v: i for i, v in enumerate(['M', 'F'])}

    user_df['GenderValue'] = user_df['Gender'].map(lambda x: gender_c[x])

    age2i = { v: i for i,v in enumerate(set(user_df['Age'].tolist()))}
    i2age = { v: i for i,v in age2i.items()}

    user_df['AgeValue'] = user_df['Age'].map(lambda x: age2i[x])

    occupation2i = { v: i for i,v in enumerate(set(user_df['Occupation'].tolist()))}
    i2occupation = { v:i for i,v in occupation2i.items()}

    user_df['OccupationValue'] = user_df['Occupation'].map(lambda x: occupation2i[x])
    stats = {
        "age": {
            'v2i': age2i,
            'i2v': i2age
        },
        "occupation": {
            'v2i': occupation2i,
            'i2v': i2occupation
        },
        "gender": {
            'v2i': gender_c
        }
    }

    json.dump(stats, open(f"{base_path}/stats.json", 'w'))

if user_df is not None:
    rating_df = pd.merge(rating_df, user_df, on="UserID", how='inner')
join_df = pd.merge(rating_df, movie_df, on="MovieID", how='inner')
print(join_df.info())
print("Total number of events:", join_df.shape[0])

train_df, test_df = train_test_split(rating_df, sort_by_key=sort_by_key)
train_df.to_parquet(f"{base_path}/train.pq", index=False)
print("Files presents :", f"{base_path}/train.pq")
test_df.to_parquet(f"{base_path}/val.pq", index=False)
print("Files presents :", f"{base_path}/val.pq")