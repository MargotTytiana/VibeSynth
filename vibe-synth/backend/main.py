from fastapi import FastAPI

app = FastAPI(title='Vibe-Synth API')

@app.get('/')
def read_root():
    return {'status': 'Latent Space Engine is running'}
