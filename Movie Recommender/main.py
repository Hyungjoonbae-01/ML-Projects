import pandas as pd
import numpy as np

r_cols = ["user_id", "movie_id", "rating"]
ratings = pd.read_csv(
    "data/u.data",
    sep="\t",
    names=r_cols,
    usecols=range(3),
    encoding="ISO-8859-1",
)

m_cols = ["movie_id", "title"]

movies = pd.read_csv(
    "data/u.item",
    sep="|",
    names=m_cols,
    usecols=range(2),
    encoding="ISO-8859-1",
)

ratings = pd.merge(movies, ratings)
userRatings = ratings.pivot_table(
    index=["user_id"],
    columns=["title"],
    values="rating",
)
user_rating_counts = ratings.groupby("user_id")["rating"].count()
upper_limit = user_rating_counts.quantile(0.99)

normal_users = user_rating_counts[user_rating_counts < upper_limit].index
filtered_ratings = userRatings[userRatings.index.isin(normal_users)]

corrMatrix = filtered_ratings.corr(method="pearson", min_periods=10)

myRatings = filtered_ratings.loc[1].dropna()
myRatings = myRatings - myRatings.mean()


def get_recommandations(myRatings, n=10):
    import pandas as pd

    simCandidates = pd.Series(dtype=float)
    for i in range(0, len(myRatings.index)):
        sims = corrMatrix[myRatings.index[i]].dropna()
        sims = sims.map(lambda x: x * myRatings.iloc[i])
        sims.sort_values(inplace=True, ascending=False)
        simCandidates = pd.concat([simCandidates, sims])

    simCandidates = simCandidates.groupby(simCandidates.index).sum()
    simCandidates.sort_values(inplace=True, ascending=False)
    simCandidates = simCandidates.drop(myRatings.index, errors="ignore")
    return simCandidates.head(n).index.tolist()


print("\n".join(get_recommandations(myRatings, 10)))
