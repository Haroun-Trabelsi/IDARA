# VFX Pipeline Deployment Guide

This guide provides comprehensive instructions for deploying the VFX Shot Complexity Prediction Pipeline to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration](#configuration)
4. [Deployment Methods](#deployment-methods)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Troubleshooting](#troubleshooting)
7. [Security Considerations](#security-considerations)

## Prerequisites

### System Requirements

- **Kubernetes Cluster**: v1.24+
- **Helm**: v3.8+
- **kubectl**: v1.24+
- **Docker**: v20.10+ (for building images)
- **GPU Support**: NVIDIA GPU with CUDA 11.8+ (optional but recommended)

### Hardware Requirements

#### Minimum (Staging)
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **GPU**: 1x NVIDIA GPU (optional)

#### Recommended (Production)
- **CPU**: 16 cores
- **Memory**: 32GB RAM
- **Storage**: 200GB SSD
- **GPU**: 2x NVIDIA GPU
- **Network**: 1Gbps

### Dependencies

```bash
# Install required tools
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## Environment Setup

### 1. Kubernetes Cluster Setup

#### Option A: Local Development (minikube)

```bash
# Start minikube with GPU support
minikube start --driver=docker --cpus=4 --memory=8192 --disk-size=50g
minikube addons enable nvidia-gpu-device-plugin
```

#### Option B: Cloud Provider (AWS EKS)

```bash
# Create EKS cluster
eksctl create cluster \
  --name vfx-pipeline \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5
```

### 2. GPU Support Setup

```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -l accelerator=nvidia-tesla-k80
```

### 3. Storage Setup

```bash
# Create storage class for fast SSD
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
reclaimPolicy: Retain
allowVolumeExpansion: true
EOF
```

## Configuration

### 1. Environment Variables

Create environment-specific configuration files:

```bash
# Create staging configuration
cp deployment/helm/values-staging.yaml.example deployment/helm/values-staging.yaml

# Create production configuration
cp deployment/helm/values-production.yaml.example deployment/helm/values-production.yaml
```

### 2. Secrets Management

```bash
# Create namespace
kubectl create namespace vfx-pipeline-production

# Create secrets
kubectl create secret generic vfx-pipeline-secrets \
  --from-literal=jwt-secret-key="your-super-secret-jwt-key" \
  --from-literal=api-key-hash-salt="your-salt-here" \
  --from-literal=mongodb-password="secure-mongodb-password" \
  --from-literal=redis-password="secure-redis-password" \
  --namespace vfx-pipeline-production
```

### 3. TLS Certificates

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create cluster issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@company.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Deployment Methods

### Method 1: Automated Deployment Script

```bash
# Make scripts executable
chmod +x deployment/scripts/*.sh

# Deploy to staging
./deployment/scripts/deploy.sh staging v1.0.0

# Deploy to production
./deployment/scripts/deploy.sh production v1.0.0
```

### Method 2: Manual Helm Deployment

```bash
# Add required repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Deploy dependencies
helm install mongodb bitnami/mongodb \
  --namespace vfx-pipeline-production \
  --set auth.enabled=true \
  --set auth.rootPassword="secure-password" \
  --set persistence.size=100Gi

# Deploy application
helm install vfx-pipeline deployment/helm \
  --namespace vfx-pipeline-production \
  --values deployment/helm/values-production.yaml \
  --set image.tag=v1.0.0
```

### Method 3: CI/CD Pipeline

The application includes automated CI/CD pipeline that:

1. **Security Scanning**: Runs safety, pip-audit, and bandit
2. **Testing**: Executes unit tests with coverage >80%
3. **Performance Testing**: Validates response times
4. **Build & Push**: Creates and pushes Docker images
5. **Deployment**: Automatically deploys to staging/production

#### Trigger Deployment

```bash
# Deploy to staging (push to main branch)
git push origin main

# Deploy to production (create release tag)
git tag v1.0.0
git push origin v1.0.0
```

## Monitoring and Maintenance

### 1. Health Monitoring

```bash
# Use monitoring script
./deployment/scripts/monitor.sh production

# Watch mode for real-time monitoring
./deployment/scripts/monitor.sh production --watch
```

### 2. Log Management

```bash
# View application logs
kubectl logs -f deployment/vfx-pipeline-production --namespace vfx-pipeline-production

# View all pod logs
kubectl logs -f -l app.kubernetes.io/name=vfx-pipeline --namespace vfx-pipeline-production

# Export logs for analysis
kubectl logs deployment/vfx-pipeline-production --namespace vfx-pipeline-production --since=1h > app-logs.txt
```

### 3. Metrics and Alerts

```bash
# Access Prometheus
kubectl port-forward service/prometheus-server 9090:80 --namespace vfx-pipeline-production

# Access Grafana
kubectl port-forward service/grafana 3000:80 --namespace vfx-pipeline-production
```

### 4. Backup and Recovery

```bash
# Backup MongoDB
kubectl exec -it mongodb-primary-0 --namespace vfx-pipeline-production -- mongodump --out /backup

# Backup persistent volumes
kubectl get pvc --namespace vfx-pipeline-production -o yaml > pvc-backup.yaml
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures

```bash
# Check pod status
kubectl get pods --namespace vfx-pipeline-production

# Describe problematic pod
kubectl describe pod <pod-name> --namespace vfx-pipeline-production

# Check logs
kubectl logs <pod-name> --namespace vfx-pipeline-production --previous
```

#### 2. Resource Constraints

```bash
# Check resource usage
kubectl top pods --namespace vfx-pipeline-production

# Check node capacity
kubectl describe nodes

# Scale resources
kubectl patch deployment vfx-pipeline-production \
  --namespace vfx-pipeline-production \
  --patch '{"spec":{"template":{"spec":{"containers":[{"name":"vfx-pipeline","resources":{"limits":{"memory":"4Gi","cpu":"2000m"}}}]}}}}'
```

#### 3. Database Connection Issues

```bash
# Test MongoDB connection
kubectl exec -it deployment/vfx-pipeline-production --namespace vfx-pipeline-production -- python -c "
import pymongo
client = pymongo.MongoClient('mongodb://mongodb:27017')
print(client.admin.command('ismaster'))
"

# Check MongoDB logs
kubectl logs mongodb-primary-0 --namespace vfx-pipeline-production
```

#### 4. GPU Issues

```bash
# Check GPU availability
kubectl get nodes -o yaml | grep nvidia.com/gpu

# Verify GPU plugin
kubectl get daemonset nvidia-device-plugin-daemonset --namespace kube-system

# Test GPU in pod
kubectl exec -it deployment/vfx-pipeline-production --namespace vfx-pipeline-production -- nvidia-smi
```

### Performance Optimization

#### 1. Scaling

```bash
# Horizontal scaling
kubectl scale deployment vfx-pipeline-production --replicas=5 --namespace vfx-pipeline-production

# Vertical scaling
kubectl patch deployment vfx-pipeline-production \
  --namespace vfx-pipeline-production \
  --patch '{"spec":{"template":{"spec":{"containers":[{"name":"vfx-pipeline","resources":{"requests":{"memory":"4Gi","cpu":"2000m"}}}]}}}}'
```

#### 2. Caching Optimization

```bash
# Check Redis status
kubectl exec -it redis-master-0 --namespace vfx-pipeline-production -- redis-cli info

# Monitor cache hit rates
kubectl exec -it deployment/vfx-pipeline-production --namespace vfx-pipeline-production -- curl localhost:8000/metrics | grep cache
```

### Rollback Procedures

```bash
# Rollback to previous version
./deployment/scripts/rollback.sh production

# Rollback to specific revision
./deployment/scripts/rollback.sh production 3

# Manual rollback
helm rollback vfx-pipeline-production --namespace vfx-pipeline-production
```

## Security Considerations

### 1. Network Security

```bash
# Create network policies
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vfx-pipeline-network-policy
  namespace: vfx-pipeline-production
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: vfx-pipeline
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: mongodb
    ports:
    - protocol: TCP
      port: 27017
EOF
```

### 2. RBAC Configuration

```bash
# Create service account with minimal permissions
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vfx-pipeline-sa
  namespace: vfx-pipeline-production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: vfx-pipeline-role
  namespace: vfx-pipeline-production
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: vfx-pipeline-binding
  namespace: vfx-pipeline-production
subjects:
- kind: ServiceAccount
  name: vfx-pipeline-sa
  namespace: vfx-pipeline-production
roleRef:
  kind: Role
  name: vfx-pipeline-role
  apiGroup: rbac.authorization.k8s.io
EOF
```

### 3. Security Scanning

```bash
# Scan container images
trivy image vfx-pipeline:latest

# Scan Kubernetes manifests
kube-score score deployment/helm/templates/*.yaml
```

## Best Practices

1. **Resource Management**: Always set resource requests and limits
2. **Health Checks**: Configure proper liveness and readiness probes
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Backup**: Regular backups of data and configurations
5. **Security**: Regular security updates and vulnerability scanning
6. **Testing**: Thorough testing in staging before production deployment
7. **Documentation**: Keep deployment documentation up to date

## Support

For deployment issues or questions:

- **Documentation**: Check this guide and technical documentation
- **Logs**: Review application and system logs
- **Monitoring**: Use monitoring dashboards for insights
- **Support**: Contact the VFX team for assistance

---

*Last updated: 2024-07-17*
