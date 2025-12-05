#!/bin/bash
# Complete startup script for the unified DataRoom application
# All services (chunking + retriever) are now integrated on port 8002

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Unified DataRoom Application${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
APP_DIR="/home/himanshu-gcp/DataRoom-ai-sheetal"
AUTH_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1XzI2YWExYjEwMzAifQ.AbXATfRDB9eN1TZnlKiq2PzyeIIeA7oG_HOUZt-Cz_U"

# Step 1: Start Infrastructure Services (MongoDB, OpenSearch)
echo -e "\n${YELLOW}[1/3] Starting Infrastructure Services...${NC}"
cd "$APP_DIR"

# Check if containers are already running
if docker ps | grep -q "DataRoom-api\|mongoDataRoom\|opensearchDataRoom"; then
    echo "Some containers already running. Restarting..."
    docker-compose down
fi

# Start MongoDB and OpenSearch
docker-compose up -d mongo opensearch

echo "Waiting for infrastructure services to be ready..."
sleep 10

# Wait for OpenSearch to be ready
echo "Waiting for OpenSearch to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OpenSearch is ready on http://localhost:9200${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ OpenSearch failed to start${NC}"
        exit 1
    fi
    sleep 2
done

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to be ready..."
for i in {1..30}; do
    if docker exec mongoDataRoom mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ MongoDB is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ MongoDB failed to start${NC}"
        exit 1
    fi
    sleep 2
done

# Step 2: Start Unified API Service (includes both chunking and retriever)
echo -e "\n${YELLOW}[2/3] Starting Unified API Service...${NC}"

# Start the API container
docker-compose up -d api

# Wait for API to be ready
echo "Waiting for unified API to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8002/docs > /dev/null 2>&1 || curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Unified API is ready on http://localhost:8002${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${YELLOW}⚠ API may still be starting. Check logs: docker logs -f DataRoom-api${NC}"
    fi
    sleep 2
done

# Step 3: Verify all endpoints
echo -e "\n${YELLOW}[3/3] Verifying endpoints...${NC}"

# Check chunking endpoints
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health endpoint: http://localhost:8002/health${NC}"
else
    echo -e "${YELLOW}⚠ Health endpoint not ready yet${NC}"
fi

# Check retriever endpoints (with timeout to avoid hanging)
if timeout 3 curl -s --max-time 2 http://localhost:8002/chunks/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Retriever health endpoint: http://localhost:8002/chunks/health${NC}"
else
    echo -e "${YELLOW}⚠ Retriever health endpoint not available (this is OK if endpoint doesn't exist)${NC}"
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All Services Started Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Unified Service (All on port 8002):"
echo "  • Main API:            http://localhost:8002"
echo "  • API Docs:            http://localhost:8002/docs"
echo "  • Health Check:        http://localhost:8002/health"
echo "  • Demo Page:           http://localhost:8002/demo"
echo ""
echo "Chunking Endpoints:"
echo "  • Projects:            http://localhost:8002/projects"
echo "  • Files:               http://localhost:8002/projects/{project_id}/files"
echo "  • Parse:               http://localhost:8002/projects/{project_id}/files/{file_id}/parse_latest"
echo ""
echo "Retriever Endpoints (All under /chunks):"
echo "  • Health:              http://localhost:8002/chunks/health"
echo "  • Query:               http://localhost:8002/chunks/query"
echo "  • Agent Query:         http://localhost:8002/chunks/query/agent"
echo "  • Sync All:            http://localhost:8002/chunks/sync-all/{project_id}"
echo "  • Ingest:              http://localhost:8002/chunks/ingest"
echo ""
echo "User Query Endpoint:"
echo "  • Query Response:      http://localhost:8002/USER/query-response"
echo ""
echo "Infrastructure:"
echo "  • OpenSearch:          http://localhost:9200"
echo "  • MongoDB:             localhost:27017"
echo ""
echo "Logs:"
echo "  • API Container:       docker logs -f DataRoom-api"
echo "  • MongoDB:             docker logs -f mongoDataRoom"
echo "  • OpenSearch:          docker logs -f opensearchDataRoom"
echo ""
echo "To stop all services:"
echo "  docker-compose down"
echo ""

