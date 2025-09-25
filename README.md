# Dockerize the Streamlit Spotify App

This folder contains ready-to-use files to build and run your Streamlit app (`spotify_app.py`) in Docker.

## Files
- `Dockerfile` – container build instructions
- `requirements.txt` – Python deps (minimal)
- `.dockerignore` – keeps your image lean
- `docker-compose.yaml` – one-command local run

## Build

```bash
docker build -t spotify-app:latest .
```

## Run
If your artifacts are in `./artifacts/spotify_v3` relative to the project root:

```bash
docker run --rm -p 8501:8501 \
  -v $(pwd)/artifacts/spotify_v3:/app/artifacts/spotify_v3:ro \
  -e SPOTIFY_ARTIFACT_DIR=/app/artifacts/spotify_v3 \
  spotify-app:latest
```

Then open http://localhost:8501

## Using docker-compose

```bash
docker compose up --build
```

## Notes
- The container exposes port **8501** (Streamlit's default).
- The app expects images and static assets from `./assets` (if present) to be copied into the image automatically via `COPY . .` in the Dockerfile. Keep `assets/` next to `spotify_app.py` when building.
- The app reads model artifacts via `SPOTIFY_ARTIFACT_DIR` (default `/app/artifacts/spotify_v3`). If you keep artifacts outside the image, **bind mount** them as shown above.
- If you later add dependencies, put them in `requirements.txt` and rebuild.
