import pandas as pd

df = pd.read_parquet("data/raw/musicbrainz/tracks.parquet")

print("\nAlbums and track counts:")
print(df.groupby("album").size().sort_index())

print("\nTotal tracks:", len(df))
print("\nColumns:")
print(df.columns.tolist())