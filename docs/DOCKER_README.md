## 🐳 Docker Deployment

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

### Quick Start
```bash
# Clone the repository
git clone https://github.com/tu-usuario/olist-project.git
cd olist-project

# Start the API
docker-compose up

# Access the API
# Swagger UI: http://localhost:8000/docs
# API: http://localhost:8000
```

### Detailed Steps

**1. Build the image:**
```bash
docker-compose build
```

**2. Start the container:**
```bash
docker-compose up
```

**3. Run in background (detached mode):**
```bash
docker-compose up -d
```

**4. View logs:**
```bash
docker-compose logs -f
```

**5. Stop the container:**
```bash
docker-compose down
```

### Testing the API

Once running, test with curl:
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict/shipping \
  -H "Content-Type: application/json" \
  -d '{"latitude": -23.55, "longitude": -46.63}'
```

Or open Swagger UI in your browser:
```
http://localhost:8000/docs
```

### Troubleshooting

**Port already in use:**
```bash
# Stop any process using port 8000
# On Linux/Mac:
lsof -ti:8000 | xargs kill -9

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Container won't start:**
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```
