#!/bin/bash

# VFX Pipeline Deployment Script
# Usage: ./deploy.sh [staging|production] [version]

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
NAMESPACE="vfx-pipeline-${ENVIRONMENT}"
HELM_RELEASE="vfx-pipeline-${ENVIRONMENT}"

echo "🚀 Deploying VFX Pipeline to ${ENVIRONMENT} environment..."
echo "📦 Version: ${VERSION}"
echo "🎯 Namespace: ${NAMESPACE}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    echo "❌ helm is not installed or not in PATH"
    exit 1
fi

# Create namespace if it doesn't exist
echo "📋 Creating namespace ${NAMESPACE}..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Add required Helm repositories
echo "📚 Adding Helm repositories..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Set environment-specific values
VALUES_FILE="deployment/helm/values-${ENVIRONMENT}.yaml"
if [ ! -f "$VALUES_FILE" ]; then
    echo "⚠️  Environment-specific values file not found: $VALUES_FILE"
    echo "📝 Using default values.yaml"
    VALUES_FILE="deployment/helm/values.yaml"
fi

# Deploy dependencies first
echo "🔧 Installing dependencies..."
helm upgrade --install ${HELM_RELEASE}-mongodb bitnami/mongodb \
    --namespace ${NAMESPACE} \
    --set auth.enabled=false \
    --set persistence.enabled=true \
    --set persistence.size=20Gi \
    --wait

helm upgrade --install ${HELM_RELEASE}-redis bitnami/redis \
    --namespace ${NAMESPACE} \
    --set auth.enabled=false \
    --set master.persistence.enabled=true \
    --set master.persistence.size=5Gi \
    --wait

# Deploy the main application
echo "🎯 Deploying VFX Pipeline application..."
helm upgrade --install ${HELM_RELEASE} deployment/helm \
    --namespace ${NAMESPACE} \
    --values ${VALUES_FILE} \
    --set image.tag=${VERSION} \
    --set config.mongodb.uri="mongodb://${HELM_RELEASE}-mongodb:27017" \
    --set config.redis.host="${HELM_RELEASE}-redis-master" \
    --wait \
    --timeout=10m

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available deployment/${HELM_RELEASE} \
    --namespace ${NAMESPACE} \
    --timeout=300s

# Run health check
echo "🔍 Running health check..."
kubectl port-forward service/${HELM_RELEASE} 8080:8000 --namespace ${NAMESPACE} &
PORT_FORWARD_PID=$!
sleep 5

if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    kill $PORT_FORWARD_PID 2>/dev/null || true
    exit 1
fi

kill $PORT_FORWARD_PID 2>/dev/null || true

# Display deployment information
echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Deployment Information:"
echo "  Environment: ${ENVIRONMENT}"
echo "  Version: ${VERSION}"
echo "  Namespace: ${NAMESPACE}"
echo "  Release: ${HELM_RELEASE}"
echo ""
echo "🔗 Access Information:"
kubectl get service ${HELM_RELEASE} --namespace ${NAMESPACE} -o wide
echo ""
echo "📋 Pod Status:"
kubectl get pods --namespace ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE}
echo ""
echo "📝 To access the application:"
echo "  kubectl port-forward service/${HELM_RELEASE} 8080:8000 --namespace ${NAMESPACE}"
echo "  Then open: http://localhost:8080"
echo ""
echo "📊 To view logs:"
echo "  kubectl logs -f deployment/${HELM_RELEASE} --namespace ${NAMESPACE}"
echo ""
echo "🔧 To scale the deployment:"
echo "  kubectl scale deployment/${HELM_RELEASE} --replicas=3 --namespace ${NAMESPACE}"
