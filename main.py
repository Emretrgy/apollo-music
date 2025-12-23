import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="The Apollo", layout="wide")


if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'generated_playlist' not in st.session_state:
    st.session_state.generated_playlist = []
if 'total_duration' not in st.session_state:
    st.session_state.total_duration = 0
if 'selected_tracks_data' not in st.session_state:
    st.session_state.selected_tracks_data = []

@st.cache_data
def load_and_train():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'dataset.csv')
    
    if not os.path.exists(file_path):
        st.error("Dataset has not been found!")
        return pd.DataFrame(), None, None, None 

    df = pd.read_csv(file_path)
    
    text_cols = ['artists', 'album_name', 'track_name']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    genre_col_temp = 'track_genre' if 'track_genre' in df.columns else 'playlist_genre'
    if genre_col_temp in df.columns:
        df[genre_col_temp] = df[genre_col_temp].astype(str).str.title()
    
    features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness']
    
    if 'track_name' in df.columns and 'artists' in df.columns:
        df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')

    df = df.dropna(subset=features)
    df = df.reset_index(drop=True)
    
    x = df[features]
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    #I chose valence as 25%, energy 20%, danceability 15%, tempo 15%, acousticness 15%, instrumentalness 10%, because this way is most accurate for mood and flow
    weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10] 
    x_weighted = x_scaled * weights
    
    model = NearestNeighbors(n_neighbors=50, metric='euclidean')
    model.fit(x_weighted)
    
    return df, model, scaler, weights


df, model, scaler, weights = load_and_train()

st.title("God of Music Apollo")

if df.empty:
    st.stop()


if not st.session_state.show_results:
    st.markdown("Choose your preferences, add songs to your path, and let Apollo craft the journey.")

    
    with st.sidebar:
        st.header("Choose your preferences")
        target_mins = st.slider("Total Duration (Minutes)", 10, 300, 60, key="main_slider")
        
        st.markdown("---")
        st.subheader("Mood Filters")
        mood_options = {
            "Any": None,
            "I am Happy (Or let me make you happy)": {"valence": (0.7, 1.0), "energy": (0.6, 1.0)},
            "I am Sad and I want to keep it": {"valence": (0.0, 0.4), "energy": (0.0, 0.4)},
            "I am so Energetic! Make me nod my head": {"valence": (0.4, 1.0), "energy": (0.8, 1.0)},
            "Just keep Calm": {"valence": (0.5, 1.0), "energy": (0.0, 0.4)}
        }
        selected_mood = st.selectbox("Pick a vibe to filter songs", list(mood_options.keys()), key="mood_box")

        st.markdown("---")
        genre_col = 'track_genre' if 'track_genre' in df.columns else 'playlist_genre'
        if genre_col in df.columns:
            selected_genres = st.multiselect("Filter by Genre", sorted(df[genre_col].unique().tolist()), key="genre_box")
        else:
            selected_genres = []

        st.markdown("---")
        
        if 'popularity' in df.columns:
            artist_ranking = df.groupby('artists')['popularity'].max().sort_values(ascending=False).index.tolist()
        else:
            artist_ranking = sorted(df['artists'].unique().tolist())
            
        artist_choice = st.selectbox("Filter by Artist", [None] + artist_ranking, key="artist_box")
        
        album_pool = df[df['artists'] == artist_choice] if artist_choice else df
        if 'popularity' in df.columns:
            album_ranking = album_pool.groupby('album_name')['popularity'].max().sort_values(ascending=False).index.tolist()
        else:
            album_ranking = sorted(album_pool['album_name'].unique().tolist())
        album_choice = st.selectbox("Filter by Album", [None] + album_ranking, key="album_box")

        #its for showing selected songs
        st.markdown("---")
        st.subheader(f"Selected Songs ({len(st.session_state.selected_tracks_data)})")
        
        if len(st.session_state.selected_tracks_data) > 0:
            for idx, t in enumerate(st.session_state.selected_tracks_data):
                st.text(f"{idx+1}. {t['track_name']}\n   {t['artists']}")
            
            if st.button("Clear Selection", type="secondary"):
                st.session_state.selected_tracks_data = []
                st.rerun()
        else:
            st.info("No songs added yet.")


    
    search_df = df.copy()
    if selected_genres:
        search_df = search_df[search_df[genre_col].isin(selected_genres)]
    if artist_choice:
        search_df = search_df[search_df['artists'] == artist_choice]
    if album_choice:
        search_df = search_df[search_df['album_name'] == album_choice]
    if selected_mood != "Any":
        v_min, v_max = mood_options[selected_mood]["valence"]
        e_min, e_max = mood_options[selected_mood]["energy"]
        search_df = search_df[(search_df['valence'] >= v_min) & (search_df['valence'] <= v_max) &
                              (search_df['energy'] >= e_min) & (search_df['energy'] <= e_max)]

    st.subheader("Build your path: Search & Add Songs")

    if not search_df.empty:
        if 'popularity' in df.columns:
            search_df = search_df.sort_values(by='popularity', ascending=False)
            search_df['display_label'] = search_df['track_name'] + " - " + search_df['artists'] + " (Pop: " + search_df['popularity'].astype(str) + ")"
        else:
            search_df['display_label'] = search_df['track_name'] + " - " + search_df['artists']
            
        
        songs_to_add_labels = st.multiselect("Search Results (Select to Add):", options=search_df['display_label'].tolist(), key="search_selector")
        
        
        if st.button("➕ Add Selected Songs to List"):
            if songs_to_add_labels:
                songs_to_add = search_df[search_df['display_label'].isin(songs_to_add_labels)]
                for _, row in songs_to_add.iterrows():
                    song_dict = row.to_dict()
                    
                    if not any(s['track_id'] == song_dict['track_id'] for s in st.session_state.selected_tracks_data):
                        st.session_state.selected_tracks_data.append(song_dict)
                st.success(f"Added {len(songs_to_add)} songs!")
                st.rerun()
    else:
        st.info("No matches found with current filters.")

    
    st.markdown("---")
    #in button function, if 1 song is selected, use it as seed, if more than 1, use bridging
    if st.button("Let the music flow !!!", type="primary", key="gen_btn"):
        
        
        if len(st.session_state.selected_tracks_data) > 0:
            selected_seeds = pd.DataFrame(st.session_state.selected_tracks_data)
        else:
            selected_seeds = pd.DataFrame()

        target_seconds = target_mins * 60
        playlist = []
        current_duration = 0
        feature_cols = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness']
        
        
        if len(selected_seeds) < 2:
            if not selected_seeds.empty:
                seed_features = selected_seeds[feature_cols].values
                target_vector = seed_features.mean(axis=0).reshape(1, -1)
                start_song = selected_seeds.iloc[0].to_dict()
                playlist.append(start_song)
                current_duration += start_song['duration_ms'] / 1000
            elif not search_df.empty:
                seed_features = search_df.head(10)[feature_cols].values 
                target_vector = seed_features.mean(axis=0).reshape(1, -1)
            else:
                random_row = df.sample(1)
                target_vector = random_row[feature_cols].values.reshape(1, -1)

            target_scaled = scaler.transform(target_vector)
            target_weighted = target_scaled * weights 
            
            estimated_songs_needed = int((target_seconds - current_duration) / 200) + 10
            distances, indices = model.kneighbors(target_weighted, n_neighbors=max(20, estimated_songs_needed + 50))
            
            for idx in indices[0]:
                if current_duration >= target_seconds: break
                song = df.iloc[idx].to_dict()
                if any(s['track_id'] == song['track_id'] for s in playlist): continue
                playlist.append(song)
                current_duration += song['duration_ms'] / 1000

        
        else:
            with st.spinner('Apollo is building a bridge between your songs...'):
                seed_tracks = selected_seeds.to_dict('records')
                num_gaps = len(seed_tracks) - 1
                seeds_duration = sum(s['duration_ms'] / 1000 for s in seed_tracks)
                remaining_time = target_seconds - seeds_duration
                
                if remaining_time <= 0:
                    playlist = seed_tracks
                    current_duration = seeds_duration
                else:
                    avg_song_duration = 210
                    total_filler_songs = int(remaining_time / avg_song_duration)
                    songs_per_gap = max(1, int(total_filler_songs / num_gaps))
                    
                    for i in range(num_gaps):
                        start_song = seed_tracks[i]
                        end_song = seed_tracks[i+1]
                        
                        if i == 0:
                            playlist.append(start_song)
                            current_duration += start_song['duration_ms'] / 1000
                        
                        v1 = df[df['track_id'] == start_song['track_id']][feature_cols].values
                        v2 = df[df['track_id'] == end_song['track_id']][feature_cols].values
                        
                        v1_scaled = scaler.transform(v1) * weights
                        v2_scaled = scaler.transform(v2) * weights
                        
                        for step in range(1, songs_per_gap + 1):
                            alpha = step / (songs_per_gap + 1)
                            target_vector = v1_scaled * (1 - alpha) + v2_scaled * alpha
                            distances, indices = model.kneighbors(target_vector.reshape(1, -1), n_neighbors=15)
                            
                            for idx in indices[0]:
                                candidate = df.iloc[idx].to_dict()
                                if not any(s['track_id'] == candidate['track_id'] for s in playlist):
                                    if candidate['track_id'] == end_song['track_id']: continue
                                    playlist.append(candidate)
                                    current_duration += candidate['duration_ms'] / 1000
                                    break
                        
                        if not any(s['track_id'] == end_song['track_id'] for s in playlist):
                            playlist.append(end_song)
                            current_duration += end_song['duration_ms'] / 1000

        st.session_state.generated_playlist = playlist
        st.session_state.total_duration = current_duration
        st.session_state.show_results = True
        st.rerun() 


else:
    if st.button("← Create New Playlist", type="secondary"):
        st.session_state.show_results = False
        st.session_state.generated_playlist = []
        
        st.rerun()

    st.success(f"✅ Generated {len(st.session_state.generated_playlist)} tracks | Total Duration: {st.session_state.total_duration/60:.2f} mins")
    st.subheader(f"Here is your personalized playlist")
    
    for i, song in enumerate(st.session_state.generated_playlist):
        track_id = str(song['track_id']).strip()
        if "spotify:track:" in track_id:
            track_id = track_id.replace("spotify:track:", "")
        
        st.markdown(f"**{i+1}. {song['track_name']}** — {song['artists']}")
        
        features_text = (
            f"<b>BPM:</b> {int(song['tempo'])} | "
            f"<b>Dance:</b> {song['danceability']:.2f} | "
            f"<b>Energy:</b> {song['energy']:.2f} | "
            f"<b>Valence:</b> {song['valence']:.2f} | "
            f"<b>Acoustic:</b> {song['acousticness']:.2f} | "
            f"<b>Instru:</b> {song['instrumentalness']:.2f} | "
            f"<b>Speech:</b> {song['speechiness']:.2f} | "
            f"<b>Loud:</b> {song['loudness']:.1f} dB"
        )
        
        st.markdown(
            f"<div style='font-size: 10px; color: #666; margin-bottom: 5px; font-family: monospace;'>{features_text}</div>", 
            unsafe_allow_html=True
        )

        embed_url = f"https://open.spotify.com/embed/track/{track_id}"
        
        components.html(
            f'<iframe src="{embed_url}" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=85
        )