#!/bin/bash

# AlphaMind Securities Research Assistant - Docker deployment script


set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="alphamind"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Helper: printing
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_info "Dependency check complete"
}

# Create the required directories
create_directories() {
    print_info "Creating required directories..."

    mkdir -p data/chroma
    mkdir -p logs
    mkdir -p config/nginx/ssl
    mkdir -p config/grafana/provisioning
    mkdir -p config/grafana/dashboards
    mkdir -p config/alerts

    print_info "Directories created"
}

# Check environment configuration
check_env_file() {
    print_info "Checking environment configuration..."

    if [ ! -f "$ENV_FILE" ]; then
        print_warn ".env not found, creating it from .env.example..."

        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_info ".env created. Please edit it before starting."
            print_warn "In particular, set ANTHROPIC_API_KEY"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_info "Environment file already exists"
    fi
}

# Build images
build_images() {
    print_info "Building Docker images..."

    docker-compose build --no-cache

    print_info "Images built"
}

# Start services
start_services() {
    print_info "Starting services..."

    docker-compose up -d

    print_info "Services started"
}

# Stop services
stop_services() {
    print_info "Stopping services..."

    docker-compose down

    print_info "Services stopped"
}

# Restart services
restart_services() {
    print_info "Restarting services..."

    docker-compose restart

    print_info "Services restarted"
}

# Show service status
status_services() {
    print_info "Service status:"

    docker-compose ps
}

# Show logs
view_logs() {
    local service=$1

    if [ -z "$service" ]; then
        print_info "Tailing logs for all services..."
        docker-compose logs -f
    else
        print_info "Tailing logs for $service..."
        docker-compose logs -f "$service"
    fi
}

# Health check
health_check() {
    print_info "Running health checks..."

    # Wait for services to come up
    sleep 10

    # Application
    if curl -sf http://localhost:8000/health > /dev/null; then
        print_info "✓ application healthy"
    else
        print_error "✗ application unhealthy"
    fi

    # Redis
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        print_info "✓ Redis healthy"
    else
        print_error "✗ Redis unhealthy"
    fi

    # ChromaDB
    if curl -sf http://localhost:8001/api/v1/heartbeat > /dev/null; then
        print_info "✓ ChromaDB healthy"
    else
        print_error "✗ ChromaDB unhealthy"
    fi

    # Prometheus
    if curl -sf http://localhost:9090/-/healthy > /dev/null; then
        print_info "✓ Prometheus healthy"
    else
        print_error "✗ Prometheus unhealthy"
    fi
}

# Clean up resources
cleanup() {
    print_warn "Removing all resources, including volumes..."

    read -p "Confirm cleanup? This deletes all data (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v
        print_info "Cleanup complete"
    else
        print_info "Cleanup cancelled"
    fi
}

# Back up data
backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"

    print_info "Backing up data to $backup_dir..."

    mkdir -p "$backup_dir"

    # Redis data
    docker-compose exec -T redis redis-cli SAVE
    docker cp alphamind-redis:/data/dump.rdb "$backup_dir/"

    # ChromaDB data
    docker cp alphamind-chromadb:/chroma/chroma "$backup_dir/"

    # Configuration
    cp .env "$backup_dir/"
    cp -r config "$backup_dir/"

    print_info "Backup complete: $backup_dir"
}

# Restore data
restore_data() {
    local backup_dir=$1

    if [ -z "$backup_dir" ]; then
        print_error "Please specify a backup directory"
        exit 1
    fi

    if [ ! -d "$backup_dir" ]; then
        print_error "Backup directory not found: $backup_dir"
        exit 1
    fi

    print_warn "Restoring data from $backup_dir..."
    read -p "Confirm restore? This overwrites existing data (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Stop services
        docker-compose stop

        # Redis data
        docker cp "$backup_dir/dump.rdb" alphamind-redis:/data/

        # ChromaDB data
        docker cp "$backup_dir/chroma" alphamind-chromadb:/chroma/

        # Configuration
        cp "$backup_dir/.env" .env
        rm -rf config
        cp -r "$backup_dir/config" config

        # Start services
        docker-compose start

        print_info "Restore complete"
    else
        print_info "Restore cancelled"
    fi
}

# Show help
show_help() {
    cat << EOF
AlphaMind Securities Research Assistant - Docker deployment script

Usage: ./docker-deploy.sh [command]

Commands:
    install     first-time setup (check dependencies, create directories, build images)
    start       start all services
    stop        stop all services
    restart     restart all services
    status      show service status
    logs        tail service logs (optionally for one service)
    health      run health checks
    build       rebuild images
    cleanup     remove all resources, including data
    backup      back up data
    restore     restore data (requires a backup directory)
    help        show this help

Examples:
    ./docker-deploy.sh install
    ./docker-deploy.sh start
    ./docker-deploy.sh logs alphamind
    ./docker-deploy.sh backup
    ./docker-deploy.sh restore backups/20231201_120000

Environment:
    configure settings in the .env file

EOF
}

# Main
main() {
    case "${1:-help}" in
        install)
            check_dependencies
            check_env_file
            create_directories
            build_images
            print_info "Install complete. Run './docker-deploy.sh start' to start the services."
            ;;
        start)
            check_env_file
            start_services
            health_check
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            status_services
            ;;
        logs)
            view_logs "$2"
            ;;
        health)
            health_check
            ;;
        build)
            build_images
            ;;
        cleanup)
            cleanup
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run
main "$@"