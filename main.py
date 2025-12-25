import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
import os
import time
import random

st.set_page_config(page_title="Apollo: God of Music", layout="wide")

# These are the weights for each feature in the distance calculation, I tested many combinations and these worked best overall.
FEATURES = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness', 'speechiness']
WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.15, 0.05, 0.05]

# Filtering for moods
MOOD_OPTIONS = {
    "Any": None,
    "Happy / Feel Good": {"valence": (0.65, 1.0), "energy": (0.4, 1.0)},
    "Sad / Melancholic": {"valence": (0.0, 0.35), "energy": (0.0, 0.35)},
    "High Energy / Pump Up": {"valence": (0.3, 1.0), "energy": (0.8, 1.0), "tempo": (140, 300)},
    "Calm / Chill": {"valence": (0.4, 1.0), "energy": (0.0, 0.3), "tempo": (0, 100)},
}


# helper funcs
def init_session_state():
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'generated_playlist' not in st.session_state:
        st.session_state.generated_playlist = []
    if 'total_duration' not in st.session_state:
        st.session_state.total_duration = 0
    if 'selected_tracks_data' not in st.session_state:
        st.session_state.selected_tracks_data = []
    if 'used_filters' not in st.session_state:
        st.session_state.used_filters = {}
    if 'current_search_pool' not in st.session_state:
        st.session_state.current_search_pool = pd.DataFrame()


# makes the addition of selected songs to seeds
def add_selection_to_seeds():
    selected_labels = st.session_state.search_sel

    if not selected_labels:
        return

    pool = st.session_state.current_search_pool
    if pool.empty:
        return

    to_add = pool[pool['label'].isin(selected_labels)]

    added_count = 0
    for _, row in to_add.iterrows():
        d = row.to_dict()
        if not any(s['track_id'] == d['track_id'] for s in st.session_state.selected_tracks_data):
            st.session_state.selected_tracks_data.append(d)
            added_count += 1

    st.session_state.search_sel = []
    if added_count > 0:
        st.toast(f"Added {added_count} songs to seeds!", icon="✅")


# Model training and data loading
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'dataset.csv')

    if not os.path.exists(file_path):
        st.error(f"Dataset not found at: {file_path}")
        return pd.DataFrame(), None

    try:
        df = pd.read_csv(file_path, index_col=0)
    except:
        df = pd.read_csv(file_path)

    text_cols = ['artists', 'album_name', 'track_name']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    genre_col = 'track_genre' if 'track_genre' in df.columns else 'playlist_genre'
    if genre_col in df.columns:
        df[genre_col] = df[genre_col].astype(str).str.title()

    for col in FEATURES:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=FEATURES)

    if 'track_name' in df.columns and 'artists' in df.columns:
        df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')

    df = df.reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(df[FEATURES])

    return df, scaler


# UI components
def render_sidebar(df):
    filters = {}
    with st.sidebar:
        st.header("Preferences")
        filters['target_mins'] = st.slider("Duration (Min)", 10, 300, 60)

        st.markdown("---")
        st.subheader("What is your vibe?")
        st.caption("Your vibe, feel free no one judges here.")

        filters['selected_mood'] = st.selectbox("Vibe", list(MOOD_OPTIONS.keys()))

        genre_col = 'track_genre' if 'track_genre' in df.columns else 'playlist_genre'
        if genre_col in df.columns:
            all_genres = sorted(df[genre_col].unique().tolist())
            filters['selected_genres'] = st.multiselect("Genre", all_genres)
        else:
            filters['selected_genres'] = []

        st.caption("Select your artist or album. If you pick one, I will calculate the vibe based on their songs.")

        if 'popularity' in df.columns:
            artist_ranking = df.groupby('artists')['popularity'].max().sort_values(ascending=False).index.tolist()
        else:
            artist_ranking = sorted(df['artists'].unique().tolist())

        filters['artist_choice'] = st.selectbox("Artist", [None] + artist_ranking)

        album_pool = df[df['artists'] == filters['artist_choice']] if filters['artist_choice'] else df
        if 'popularity' in df.columns:
            album_ranking = album_pool.groupby('album_name')['popularity'].max().sort_values(ascending=False).index.tolist()
        else:
            album_ranking = sorted(album_pool['album_name'].unique().tolist())

        filters['album_choice'] = st.selectbox("Album", [None] + album_ranking)

        st.markdown("---")
        st.subheader(f"Your selections, your destiny ({len(st.session_state.selected_tracks_data)})")
        if len(st.session_state.selected_tracks_data) > 0:
            for idx, t in enumerate(st.session_state.selected_tracks_data):
                st.text(f"{idx + 1}. {t['track_name']}")
            if st.button("Clear Seeds"):
                st.session_state.selected_tracks_data = []
                st.rerun()

    return filters


def render_results():
    if st.button("← Start your new journey", type="secondary"):
        st.session_state.show_results = False
        st.session_state.generated_playlist = []
        st.rerun()

    saved_filters = st.session_state.used_filters
    total_min = st.session_state.total_duration / 60
    count = len(st.session_state.generated_playlist)

    info_parts = []

    # Mood and genre info
    if saved_filters.get('selected_mood') and saved_filters['selected_mood'] != "Any":
        info_parts.append(f"Mood: {saved_filters['selected_mood']}")
    if saved_filters.get('selected_genres'):
        info_parts.append(f"Genres: {', '.join(saved_filters['selected_genres'])}")

    # Reference logic
    if len(st.session_state.selected_tracks_data) > 0:
        seed_names = [t['track_name'] for t in st.session_state.selected_tracks_data]
        if len(seed_names) > 2:
            seed_text = f"{seed_names[0]}, {seed_names[1]} +{len(seed_names) - 2} more"
        else:
            seed_text = ", ".join(seed_names)
        info_parts.append(f"Ref Songs: {seed_text}")
    elif saved_filters.get('artist_choice'):
        info_parts.append(f"Ref Artist: {saved_filters['artist_choice']}")

    filter_text = " • ".join(info_parts) if info_parts else "No specific filters"

    st.markdown(f"###  Here is your personalized, legendary playlist with ({count} songs)")
    st.caption(f"**Total Duration:** {total_min:.1f} min | **Your choices:** {filter_text}")
    st.markdown("---")

    for i, song in enumerate(st.session_state.generated_playlist):
        track_id = str(song['track_id']).strip().replace("spotify:track:", "")

        st.markdown(f"#### {i + 1}. {song['track_name']} — {song['artists']}")

        genre_info = song.get('track_genre', song.get('playlist_genre', '-'))
        pop_info = int(song.get('popularity', 0))

        details_html = f"""
        <div style="
            font-size: 12px; 
            color: #ffffff; 
            background-color: transparent;
            padding: 0px;
            margin-bottom: 8px;
            font-family: monospace;
            white-space: normal; 
            line-height: 1.5;
            opacity: 0.9;
        ">
            <b>Genre:</b> {genre_info} | 
            <b>Pop:</b> {pop_info} | 
            <b>BPM:</b> {int(song.get('tempo', 0))} | 
            <b>Val:</b> {song.get('valence', 0):.2f} | 
            <b>Eng:</b> {song.get('energy', 0):.2f} | 
            <b>Dance:</b> {song.get('danceability', 0):.2f} | 
            <b>Acous:</b> {song.get('acousticness', 0):.2f} | 
            <b>Instr:</b> {song.get('instrumentalness', 0):.2f}
        </div>
        """
        st.markdown(details_html, unsafe_allow_html=True)

        components.html(
            f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="100%" height="80" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
            height=85
        )


# CORE LOGİC FOR PLAYLIST GENERATION

# Weight and KNN based playlist generation
def generate_smart_playlist(pool_df, reference_df, scaler, target_mins):
    pool_df = pool_df.reset_index(drop=True)
    if len(pool_df) < 5:
        return pool_df.to_dict('records'), 0

    # model prep
    x_pool = pool_df[FEATURES]
    x_pool_scaled = scaler.transform(x_pool)
    x_pool_weighted = x_pool_scaled * WEIGHTS

    # calling more neighbors model
    model = NearestNeighbors(n_neighbors=min(100, len(pool_df)), metric='euclidean')
    model.fit(x_pool_weighted)

    selected_seeds = pd.DataFrame(st.session_state.selected_tracks_data)
    target_seconds = target_mins * 60
    playlist = []
    current_duration = 0

    # scenerio 1: Multiple seeds
    if len(selected_seeds) >= 2:
        seed_tracks = selected_seeds.to_dict('records')
        seeds_dur = sum(s['duration_ms'] / 1000 for s in seed_tracks)
        rem_time = target_seconds - seeds_dur

        if rem_time <= 0: return seed_tracks, seeds_dur

        avg_dur = 220
        total_fill = int(rem_time / avg_dur)
        gaps = len(seed_tracks) - 1
        per_gap = max(1, int(total_fill / gaps))

        for i in range(gaps):
            start = seed_tracks[i]
            end = seed_tracks[i + 1]
            if i == 0:
                playlist.append(start)
                current_duration += start['duration_ms'] / 1000

            v1 = scaler.transform([list(start[k] for k in FEATURES)]) * WEIGHTS
            v2 = scaler.transform([list(end[k] for k in FEATURES)]) * WEIGHTS

            for step in range(1, per_gap + 1):
                alpha = step / (per_gap + 1)
                target_vector = v1 * (1 - alpha) + v2 * alpha
                _, indices = model.kneighbors(target_vector, n_neighbors=15)

                candidates = list(indices[0])
                random.shuffle(candidates)

                for idx in candidates:
                    cand = pool_df.iloc[idx].to_dict()
                    if not any(s['track_id'] == cand['track_id'] for s in playlist):
                        if cand['track_id'] == end['track_id']: continue
                        playlist.append(cand)
                        current_duration += cand['duration_ms'] / 1000
                        break

            if not any(s['track_id'] == end['track_id'] for s in playlist):
                playlist.append(end)
                current_duration += end['duration_ms'] / 1000

        return playlist, current_duration

    # Scenario 2 & 3: Single seed or no seed
    target_vector = None
    start_song = None

    if len(selected_seeds) == 1:
        start_song = selected_seeds.iloc[0].to_dict()
        target_vector = scaler.transform([selected_seeds.iloc[0][FEATURES].values]) * WEIGHTS
        playlist.append(start_song)
        current_duration += start_song['duration_ms'] / 1000

    elif not reference_df.empty:
        mean_features = reference_df[FEATURES].mean(axis=0).values.reshape(1, -1)
        target_vector = scaler.transform(mean_features) * WEIGHTS

        dist, ind = model.kneighbors(target_vector, n_neighbors=3)
        start_idx = random.choice(ind[0])
        start_song = pool_df.iloc[start_idx].to_dict()

        playlist.append(start_song)
        current_duration += start_song['duration_ms'] / 1000

    else:
        random_idx = random.randint(0, len(pool_df) - 1)
        start_song = pool_df.iloc[random_idx].to_dict()
        target_vector = scaler.transform([pool_df.iloc[random_idx][FEATURES].values]) * WEIGHTS

        playlist.append(start_song)
        current_duration += start_song['duration_ms'] / 1000

    neighbor_count = min(100, len(pool_df))
    distances, indices = model.kneighbors(target_vector, n_neighbors=neighbor_count)

    all_candidates = list(indices[0])

    if all_candidates[0] == pool_df[pool_df['track_id'] == start_song['track_id']].index[0]:
        all_candidates = all_candidates[1:]

    tier_1 = all_candidates[:3]

    tier_2 = all_candidates[3:23]
    random.shuffle(tier_2)

    tier_3 = all_candidates[23:]
    random.shuffle(tier_3)

    final_order = tier_1 + tier_2 + tier_3

    for idx in final_order:
        if current_duration >= target_seconds: break

        song = pool_df.iloc[idx].to_dict()

        if any(s['track_id'] == song['track_id'] for s in playlist): continue

        playlist.append(song)
        current_duration += song['duration_ms'] / 1000

    return playlist, current_duration


# MAIN EXECUTION
def main():
    init_session_state()
    df, scaler = load_data()

    st.title("God of Music Apollo")
    if df.empty: st.stop()

    if not st.session_state.show_results:
        filters = render_sidebar(df)

        # 1. POOL CREATION
        knn_pool = df.copy()
        if filters['selected_mood'] != "Any":
            v_min, v_max = MOOD_OPTIONS[filters['selected_mood']]["valence"]
            e_min, e_max = MOOD_OPTIONS[filters['selected_mood']]["energy"]
            knn_pool = knn_pool[
                (knn_pool['valence'] >= v_min) & (knn_pool['valence'] <= v_max) &
                (knn_pool['energy'] >= e_min) & (knn_pool['energy'] <= e_max)
            ]
        genre_col = 'track_genre' if 'track_genre' in df.columns else 'playlist_genre'
        if filters['selected_genres']:
            knn_pool = knn_pool[knn_pool[genre_col].isin(filters['selected_genres'])]

        # 2. REFERENCE POOL & STATS
        search_pool = knn_pool.copy()
        if filters['artist_choice']:
            search_pool = search_pool[search_pool['artists'] == filters['artist_choice']]
        if filters['album_choice']:
            search_pool = search_pool[search_pool['album_name'] == filters['album_choice']]

        st.info(f"🧬 Playlist Pool: {len(knn_pool)} songs available.")

        # artis statistics
        if filters['artist_choice'] and len(search_pool) > 0:
            avg_stats = search_pool[FEATURES].mean()
            pop_avg = int(search_pool['popularity'].mean()) if 'popularity' in search_pool.columns else 0

            stats_html = f"""
            <div style="
                font-size: 11px; 
                color: #ffffff; 
                background-color: transparent; 
                padding: 0px; 
                margin-top: 5px; 
                margin-bottom: 15px;
                font-family: monospace;
            ">
                <b>📊 AVERAGE VIBE for {filters['artist_choice'].upper()}:</b><br>
                Pop: {pop_avg} | 
                BPM: {int(avg_stats['tempo'])} | 
                Val: {avg_stats['valence']:.2f} | 
                Eng: {avg_stats['energy']:.2f} | 
                Dance: {avg_stats['danceability']:.2f} | 
                Acous: {avg_stats['acousticness']:.2f}
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)

        if filters['artist_choice']:
            st.caption(f"🎤 Reference Pool: {len(search_pool)} songs used for calculation.")

        if len(search_pool) > 0:
            if 'popularity' in search_pool.columns:
                search_pool = search_pool.sort_values(by='popularity', ascending=False)

            search_pool['label'] = search_pool['track_name'] + " - " + search_pool['artists']
            options = search_pool['label'].head(2000).tolist()

            st.session_state.current_search_pool = search_pool
            st.multiselect(
                "Optional: Pick specific songs to bridge/match and enjoy the show!",
                options,
                key="search_sel",
                on_change=add_selection_to_seeds
            )

        else:
            if filters['artist_choice']:
                st.warning(f"⚠️ {filters['artist_choice']} has no songs matching the selected Mood/Genre.")

        st.markdown("---")

        if st.button("Generate Playlist", type="primary"):
            if len(knn_pool) > 0:
                with st.spinner("Apollo is analyzing the vibes..."):
                    st.session_state.used_filters = filters
                    pl, dur = generate_smart_playlist(knn_pool, search_pool, scaler, filters['target_mins'])
                    st.session_state.generated_playlist = pl
                    st.session_state.total_duration = dur
                    st.session_state.show_results = True
                    st.rerun()
            else:
                st.error("Cannot generate playlist: No songs match your criteria.")

    else:
        render_results()


if __name__ == "__main__":
    main()
