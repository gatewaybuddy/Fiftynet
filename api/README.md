# Fiftynet REST API

A production-ready REST API for serving Fiftynet models using FastAPI.

## Features

- **Text Generation**: Generate text with flexible sampling strategies
- **Model Management**: Load and switch models dynamically
- **Health Monitoring**: Health check and model status endpoints
- **CORS Support**: Cross-origin requests enabled
- **Request Validation**: Automatic request/response validation with Pydantic
- **OpenAPI Documentation**: Interactive API docs at `/docs`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r api/requirements.txt
```

### 2. Start the Server

```bash
# Set environment variables (optional)
export MODEL_PATH=trained
export TOKENIZER_PATH=tokenizer.json

# Start server
cd api
python server.py
```

Or using uvicorn directly:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 3. Test the API

The server will be available at `http://localhost:8000`

**Interactive documentation**: Visit `http://localhost:8000/docs`

## API Endpoints

### GET /
Root endpoint with API information

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": 123.45
}
```

### GET /model-info
Get information about the loaded model

**Response:**
```json
{
  "model_name": "FFTNet",
  "vocab_size": 5000,
  "dim": 64,
  "num_blocks": 2,
  "device": "cuda",
  "ready": true
}
```

### POST /generate
Generate text from a prompt

**Request:**
```json
{
  "prompt": "The future of AI is",
  "max_new_tokens": 50,
  "temperature": 0.9,
  "top_k": 0,
  "top_p": 0.95
}
```

**Response:**
```json
{
  "generated_text": "The future of AI is bright and full of possibilities...",
  "prompt": "The future of AI is",
  "tokens_generated": 50,
  "inference_time": 0.234
}
```

**Parameters:**
- `prompt` (required): Input text prompt (1-10000 characters)
- `max_new_tokens` (optional): Number of tokens to generate (default: 50, max: 500)
- `temperature` (optional): Sampling temperature 0-2 (default: 1.0)
  - 0 = greedy (deterministic)
  - <1 = more focused
  - >1 = more random
- `top_k` (optional): Top-k sampling (default: 0 = disabled)
- `top_p` (optional): Nucleus sampling threshold 0-1 (default: 1.0 = disabled)

### POST /load-model
Load or reload a model

**Request:**
```json
{
  "model_path": "trained",
  "tokenizer_path": "tokenizer.json"
}
```

## Usage Examples

### Python Client

```python
import requests

# Generate text
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Once upon a time",
        "max_new_tokens": 50,
        "temperature": 0.9,
        "top_p": 0.95
    }
)

result = response.json()
print(result["generated_text"])
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_new_tokens": 30,
    "temperature": 0.9,
    "top_p": 0.95
  }'
```

### Example Client Script

Run the included example client:

```bash
python api/client_example.py
```

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to model weights (default: "trained")
- `TOKENIZER_PATH`: Path to tokenizer file (default: "tokenizer.json")

### Server Configuration

Modify `server.py` or pass arguments to uvicorn:

```bash
uvicorn api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --reload
```

## Production Deployment

### Using Docker

```bash
# Build image
docker build -t fiftynet-api -f api/Dockerfile .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/tokenizer.json:/app/tokenizer.json \
  fiftynet-api
```

### Using Gunicorn + Uvicorn Workers

```bash
gunicorn api.server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Behind a Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.fiftynet.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Performance Tuning

### GPU Inference

The API automatically uses GPU if available. For optimal GPU performance:

```python
# In server.py, adjust batch processing for your GPU
# or use torch.compile for faster inference (PyTorch 2.0+)
state.model = torch.compile(state.model)
```

### Multiple Workers

For CPU inference, use multiple workers:

```bash
uvicorn api.server:app --workers 4
```

**Note:** Each worker loads the model into memory. Adjust based on available RAM.

### Caching

For production, consider adding caching:
- Redis for response caching
- In-memory cache for common prompts

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

## Security Considerations

For production deployment:

1. **CORS**: Configure `allow_origins` in `server.py` appropriately
2. **Rate Limiting**: Add rate limiting middleware
3. **Authentication**: Add API key authentication if needed
4. **HTTPS**: Always use HTTPS in production
5. **Input Validation**: The API validates all inputs, but consider additional sanitization

## Monitoring

### Logging

The server logs all requests. Configure log level:

```bash
uvicorn api.server:app --log-level info
```

### Metrics

For production monitoring, integrate with:
- Prometheus for metrics
- Grafana for visualization
- Sentry for error tracking

## Troubleshooting

### Model not loading

```
503 Service Unavailable: Model not loaded
```

**Solution**: Check that MODEL_PATH and TOKENIZER_PATH are correct, or use `/load-model` endpoint.

### Out of memory

**Solution**: Reduce `max_new_tokens`, use smaller model, or upgrade hardware.

### Slow inference

**Solution**: Use GPU, enable torch.compile, or reduce sequence length.

## Development

### Run in development mode

```bash
uvicorn api.server:app --reload --log-level debug
```

### Run tests

```bash
pytest tests/test_api.py
```

## License

Same as Fiftynet project (MIT)
