#!/bin/bash

# VFX Pipeline Rollback Script
# Usage: ./rollback.sh [staging|production] [revision]

set -e

ENVIRONMENT=${1:-staging}
REVISION=${2}
NAMESPACE="vfx-pipeline-${ENVIRONMENT}"
HELM_RELEASE="vfx-pipeline-${ENVIRONMENT}"

echo "🔄 Rolling back VFX Pipeline in ${ENVIRONMENT} environment..."
echo "🎯 Namespace: ${NAMESPACE}"
echo "📦 Release: ${HELM_RELEASE}"

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

# Show current release history
echo "📋 Current release history:"
helm history ${HELM_RELEASE} --namespace ${NAMESPACE}

# Determine rollback target
if [ -z "$REVISION" ]; then
    echo ""
    echo "🔍 No revision specified, rolling back to previous version..."
    ROLLBACK_CMD="helm rollback ${HELM_RELEASE} --namespace ${NAMESPACE}"
else
    echo ""
    echo "🔍 Rolling back to revision ${REVISION}..."
    ROLLBACK_CMD="helm rollback ${HELM_RELEASE} ${REVISION} --namespace ${NAMESPACE}"
fi

# Confirm rollback
echo ""
read -p "⚠️  Are you sure you want to rollback? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Rollback cancelled"
    exit 1
fi

# Perform rollback
echo "🔄 Performing rollback..."
${ROLLBACK_CMD} --wait --timeout=10m

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete..."
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
    echo "🚨 Rollback may have failed. Check the logs:"
    echo "  kubectl logs -f deployment/${HELM_RELEASE} --namespace ${NAMESPACE}"
    exit 1
fi

kill $PORT_FORWARD_PID 2>/dev/null || true

# Display rollback information
echo ""
echo "🎉 Rollback completed successfully!"
echo ""
echo "📊 Current Status:"
helm status ${HELM_RELEASE} --namespace ${NAMESPACE}
echo ""
echo "📋 Pod Status:"
kubectl get pods --namespace ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE}
echo ""
echo "📝 Updated release history:"
helm history ${HELM_RELEASE} --namespace ${NAMESPACE}
