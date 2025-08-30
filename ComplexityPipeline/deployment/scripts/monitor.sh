#!/bin/bash

# VFX Pipeline Monitoring Script
# Usage: ./monitor.sh [staging|production]

set -e

ENVIRONMENT=${1:-staging}
NAMESPACE="vfx-pipeline-${ENVIRONMENT}"
HELM_RELEASE="vfx-pipeline-${ENVIRONMENT}"

echo "📊 VFX Pipeline Monitoring Dashboard - ${ENVIRONMENT^^} Environment"
echo "🎯 Namespace: ${NAMESPACE}"
echo "📦 Release: ${HELM_RELEASE}"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Function to display section header
show_section() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════"
}

# Function to check if namespace exists
check_namespace() {
    if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
        echo "❌ Namespace ${NAMESPACE} does not exist"
        exit 1
    fi
}

# Function to display deployment status
show_deployment_status() {
    show_section "🚀 DEPLOYMENT STATUS"
    
    echo "📋 Deployment Information:"
    kubectl get deployment ${HELM_RELEASE} --namespace ${NAMESPACE} -o wide 2>/dev/null || echo "❌ Deployment not found"
    
    echo ""
    echo "📊 Replica Set Status:"
    kubectl get replicaset --namespace ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} 2>/dev/null || echo "❌ No replica sets found"
    
    echo ""
    echo "🏃 Pod Status:"
    kubectl get pods --namespace ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} -o wide 2>/dev/null || echo "❌ No pods found"
}

# Function to display service status
show_service_status() {
    show_section "🌐 SERVICE STATUS"
    
    echo "🔗 Services:"
    kubectl get services --namespace ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} 2>/dev/null || echo "❌ No services found"
    
    echo ""
    echo "🔌 Endpoints:"
    kubectl get endpoints --namespace ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} 2>/dev/null || echo "❌ No endpoints found"
}

# Function to display resource usage
show_resource_usage() {
    show_section "💾 RESOURCE USAGE"
    
    echo "📊 CPU and Memory Usage:"
    kubectl top pods --namespace ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} 2>/dev/null || echo "❌ Metrics not available (metrics-server required)"
    
    echo ""
    echo "📈 Node Resource Usage:"
    kubectl top nodes 2>/dev/null || echo "❌ Node metrics not available"
}

# Function to display persistent volumes
show_storage_status() {
    show_section "💽 STORAGE STATUS"
    
    echo "📦 Persistent Volume Claims:"
    kubectl get pvc --namespace ${NAMESPACE} 2>/dev/null || echo "❌ No PVCs found"
    
    echo ""
    echo "🗄️ Persistent Volumes:"
    kubectl get pv | grep ${NAMESPACE} 2>/dev/null || echo "❌ No PVs found for this namespace"
}

# Function to display recent events
show_recent_events() {
    show_section "📰 RECENT EVENTS"
    
    echo "⚡ Recent Events (last 10):"
    kubectl get events --namespace ${NAMESPACE} --sort-by='.lastTimestamp' | tail -10 2>/dev/null || echo "❌ No events found"
}

# Function to display logs
show_logs() {
    show_section "📝 APPLICATION LOGS"
    
    echo "📋 Recent Application Logs (last 50 lines):"
    kubectl logs deployment/${HELM_RELEASE} --namespace ${NAMESPACE} --tail=50 2>/dev/null || echo "❌ No logs available"
}

# Function to display health status
show_health_status() {
    show_section "🔍 HEALTH STATUS"
    
    echo "🏥 Health Check:"
    
    # Try to port-forward and check health
    kubectl port-forward service/${HELM_RELEASE} 8080:8000 --namespace ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    sleep 3
    
    if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Application is healthy"
        
        echo ""
        echo "📊 Health Details:"
        curl -s http://localhost:8080/health | jq . 2>/dev/null || curl -s http://localhost:8080/health
    else
        echo "❌ Application health check failed"
    fi
    
    kill $PORT_FORWARD_PID 2>/dev/null || true
    sleep 1
}

# Function to display HPA status
show_autoscaling_status() {
    show_section "📈 AUTOSCALING STATUS"
    
    echo "🔄 Horizontal Pod Autoscaler:"
    kubectl get hpa --namespace ${NAMESPACE} 2>/dev/null || echo "❌ No HPA found"
    
    echo ""
    echo "📊 HPA Details:"
    kubectl describe hpa ${HELM_RELEASE} --namespace ${NAMESPACE} 2>/dev/null || echo "❌ HPA details not available"
}

# Function to display monitoring endpoints
show_monitoring_endpoints() {
    show_section "📊 MONITORING ENDPOINTS"
    
    echo "🔗 Available Monitoring Endpoints:"
    echo "  • Health Check: http://localhost:8080/health"
    echo "  • Metrics: http://localhost:8080/metrics"
    echo "  • API Documentation: http://localhost:8080/docs"
    echo ""
    echo "📝 To access these endpoints:"
    echo "  kubectl port-forward service/${HELM_RELEASE} 8080:8000 --namespace ${NAMESPACE}"
    echo ""
    
    # Check if Prometheus is available
    if kubectl get service --namespace ${NAMESPACE} | grep prometheus &> /dev/null; then
        echo "📈 Prometheus Dashboard:"
        echo "  kubectl port-forward service/prometheus-server 9090:80 --namespace ${NAMESPACE}"
    fi
    
    # Check if Grafana is available
    if kubectl get service --namespace ${NAMESPACE} | grep grafana &> /dev/null; then
        echo "📊 Grafana Dashboard:"
        echo "  kubectl port-forward service/grafana 3000:80 --namespace ${NAMESPACE}"
    fi
}

# Main monitoring function
main() {
    check_namespace
    
    if [ "$2" = "--watch" ]; then
        echo "👀 Watching mode enabled. Press Ctrl+C to exit."
        while true; do
            clear
            show_deployment_status
            show_resource_usage
            show_health_status
            echo ""
            echo "🔄 Refreshing in 10 seconds... (Press Ctrl+C to exit)"
            sleep 10
        done
    else
        show_deployment_status
        show_service_status
        show_resource_usage
        show_storage_status
        show_autoscaling_status
        show_recent_events
        show_health_status
        show_monitoring_endpoints
        show_logs
        
        echo ""
        echo "🎯 Quick Commands:"
        echo "  • Watch mode: ./monitor.sh ${ENVIRONMENT} --watch"
        echo "  • View logs: kubectl logs -f deployment/${HELM_RELEASE} --namespace ${NAMESPACE}"
        echo "  • Scale up: kubectl scale deployment/${HELM_RELEASE} --replicas=5 --namespace ${NAMESPACE}"
        echo "  • Port forward: kubectl port-forward service/${HELM_RELEASE} 8080:8000 --namespace ${NAMESPACE}"
        echo "  • Debug pod: kubectl exec -it deployment/${HELM_RELEASE} --namespace ${NAMESPACE} -- /bin/bash"
    fi
}

# Run main function
main "$@"
