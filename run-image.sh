#!/bin/bash

# AlphaMind image run script
# Offers several run configurations

set -e

# Configuration
IMAGE_NAME="alphamind"
CONTAINER_NAME="alphamind-app"
VERSION=${VERSION:-latest}
REGISTRY=""  # set to registry.example.com/ if the image lives in a private registry

# Default port mappings
API_PORT=8000
PROMETHEUS_PORT=9090

# Default volume mappings
DATA_DIR="./data"
LOGS_DIR="./logs"
CONFIG_DIR="./config"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
AlphaMind Docker image runner

Usage: ./run-image.sh [command] [options]

Commands:
    run             run the container (default mode)
    run-dev         run in development mode
    run-test        run the test container
    stop            stop the container
    restart         restart the container
    logs            tail container logs
    shell           open a shell in the container
    status          show container status
    clean           remove the container and its data
    help            show this help

Options:
    --detach        run in the background
    --ports         custom port mappings
    --env-file      environment file to use
    --volume        custom volume mappings
    --name          custom container name
    --network       custom network

Examples:
    ./run-image.sh run
    ./run-image.sh run-dev --detach
    ./run-image.sh run --env-file .env.prod
    ./run-image.sh logs
    ./run-image.sh shell

Production:
    ./run-image.sh run \\
        --env-file .env.prod \\
        --detach \\
        --restart unless-stopped

Development:
    ./run-image.sh run-dev \\
        --volume ./src:/app/src \\
        --detach

EOF
}

# Create the required directories
ensure_directories() {
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$CONFIG_DIR"
}

# Run the container
run_container() {
    local mode=$1
    shift || true

    local detach=false
    local env_file=".env"
    local custom_ports=""
    local custom_volumes=""
    local container_name="$CONTAINER_NAME"
    local restart_policy="no"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --detach|-d)
                detach=true
                shift
                ;;
            --env-file)
                env_file="$2"
                shift 2
                ;;
            --ports|-p)
                custom_ports="-p $2"
                shift 2
                ;;
            --volume|-v)
                custom_volumes="$custom_volumes -v $2"
                shift 2
                ;;
            --name)
                container_name="$2"
                shift 2
                ;;
            --restart)
                restart_policy="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    ensure_directories

    # Check the environment file
    if [ ! -f "$env_file" ]; then
        print_warn "Environment file $env_file not found, falling back to defaults"
        env_file=""
    else
        env_file="--env-file $env_file"
    fi

    # Base configuration
    local image_tag="${REGISTRY}${IMAGE_NAME}:${VERSION}"
    local default_ports="-p ${API_PORT}:8000 -p ${PROMETHEUS_PORT}:9090"
    local default_volumes="-v ${DATA_DIR}:/app/data -v ${LOGS_DIR}:/app/logs -v ${CONFIG_DIR}:/app/config"

    # Adjust configuration per mode
    case $mode in
        dev)
            print_info "Running the container in development mode"
            default_ports="$default_ports -p 5678:5678"  # debugger port
            restart_policy="no"
            ;;
        test)
            print_info "Running the test container"
            restart_policy="no"
            ;;
        prod)
            print_info "Running the container in production mode"
            restart_policy="unless-stopped"
            ;;
        *)
            print_info "Running the container in standard mode"
            ;;
    esac

    # Build the run command
    local run_cmd="docker run"

    if [ "$detach" = true ]; then
        run_cmd="$run_cmd -d"
    fi

    run_cmd="$run_cmd --name $container_name"
    run_cmd="$run_cmd --restart $restart_policy"
    run_cmd="$run_cmd $default_ports $custom_ports"
    run_cmd="$run_cmd $default_volumes $custom_volumes"
    run_cmd="$run_cmd $env_file"
    run_cmd="$run_cmd $image_tag"

    print_info "Starting container: $container_name"
    print_info "Image: $image_tag"

    # Remove an existing container with the same name
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        print_warn "Container $container_name already exists; stopping and removing it"
        docker stop "$container_name" 2>/dev/null || true
        docker rm "$container_name" 2>/dev/null || true
    fi

    # Run the container
    eval $run_cmd

    if [ $? -eq 0 ]; then
        print_info "✓ container started"
        print_info "API: http://localhost:${API_PORT}"
        print_info "Prometheus: http://localhost:${PROMETHEUS_PORT}"

        if [ "$detach" = true ]; then
            print_info "Container is running in the background"
            print_info "Logs: ./run-image.sh logs"
        fi
    else
        print_error "✗ container failed to start"
        exit 1
    fi
}

# Stop the container
stop_container() {
    local name=${1:-$CONTAINER_NAME}

    print_info "Stopping container: $name"

    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        docker stop "$name"
        print_info "✓ container stopped"
    else
        print_warn "Container $name is not running"
    fi
}

# Restart the container
restart_container() {
    local name=${1:-$CONTAINER_NAME}

    print_info "Restarting container: $name"

    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        docker restart "$name"
        print_info "✓ container restarted"
    else
        print_error "Container $name does not exist"
        exit 1
    fi
}

# Tail logs
view_logs() {
    local name=${1:-$CONTAINER_NAME}
    local follow=${2:-true}

    if [ "$follow" = "true" ]; then
        docker logs -f "$name"
    else
        docker logs "$name"
    fi
}

# Open a shell in the container
enter_shell() {
    local name=${1:-$CONTAINER_NAME}

    print_info "Opening a shell in: $name"

    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        docker exec -it "$name" /bin/bash
    else
        print_error "Container $name is not running"
        exit 1
    fi
}

# Show status
show_status() {
    local name=${1:-$CONTAINER_NAME}

    print_info "Container status: $name"

    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        docker ps -a --filter "name=$name" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        print_warn "Container $name does not exist"
    fi
}

# Clean up
clean_container() {
    local name=${1:-$CONTAINER_NAME}

    print_warn "Removing the container and its volumes"

    read -p "Confirm cleanup? This deletes the container and its data (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker stop "$name" 2>/dev/null || true
        docker rm "$name" 2>/dev/null || true
        print_info "✓ container removed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Main
main() {
    local command=${1:-run}
    shift || true

    case $command in
        run)
            run_container "" "$@"
            ;;
        run-dev)
            run_container "dev" "$@"
            ;;
        run-test)
            run_container "test" "$@"
            ;;
        run-prod)
            run_container "prod" "$@"
            ;;
        stop)
            stop_container "$1"
            ;;
        restart)
            restart_container "$1"
            ;;
        logs)
            if [ "$1" = "--no-follow" ]; then
                view_logs "$2" "false"
            else
                view_logs "$1" "true"
            fi
            ;;
        shell)
            enter_shell "$1"
            ;;
        status)
            show_status "$1"
            ;;
        clean)
            clean_container "$1"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run
main "$@"