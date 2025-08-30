# VFX Pipeline Operational Runbook

This runbook provides step-by-step procedures for common operational tasks and incident response for the VFX Shot Complexity Prediction Pipeline.

## Table of Contents

1. [Emergency Procedures](#emergency-procedures)
2. [Routine Maintenance](#routine-maintenance)
3. [Troubleshooting Guide](#troubleshooting-guide)
4. [Performance Tuning](#performance-tuning)
5. [Backup and Recovery](#backup-and-recovery)
6. [Security Incidents](#security-incidents)
7. [Scaling Operations](#scaling-operations)

## Emergency Procedures

### 1. Service Outage Response

#### Immediate Actions (0-5 minutes)

```bash
# 1. Check service status
kubectl get pods -n vfx-pipeline-production -l app.kubernetes.io/name=vfx-pipeline

# 2. Check recent events
kubectl get events -n vfx-pipeline-production --sort-by='.lastTimestamp' | tail -20

# 3. Check application logs
kubectl logs -f deployment/vfx-pipeline-production -n vfx-pipeline-production --tail=100

# 4. Check resource usage
kubectl top pods -n vfx-pipeline-production
kubectl top nodes
```

#### Assessment Phase (5-15 minutes)

```bash
# 1. Run monitoring script
./deployment/scripts/monitor.sh production

# 2. Check dependencies
kubectl get pods -n vfx-pipeline-production -l app.kubernetes.io/name=mongodb
kubectl get pods -n vfx-pipeline-production -l app.kubernetes.io/name=redis

# 3. Check ingress and networking
kubectl get ingress -n vfx-pipeline-production
kubectl get services -n vfx-pipeline-production

# 4. Test health endpoints
kubectl port-forward service/vfx-pipeline-production 8080:8000 -n vfx-pipeline-production &
curl -f http://localhost:8080/health || echo "Health check failed"
```

#### Recovery Actions

**Option 1: Pod Restart**
```bash
# Restart deployment
kubectl rollout restart deployment/vfx-pipeline-production -n vfx-pipeline-production

# Wait for rollout
kubectl rollout status deployment/vfx-pipeline-production -n vfx-pipeline-production --timeout=300s
```

**Option 2: Scale Down/Up**
```bash
# Scale to 0 replicas
kubectl scale deployment vfx-pipeline-production --replicas=0 -n vfx-pipeline-production

# Wait for pods to terminate
kubectl wait --for=delete pod -l app.kubernetes.io/name=vfx-pipeline -n vfx-pipeline-production --timeout=60s

# Scale back up
kubectl scale deployment vfx-pipeline-production --replicas=3 -n vfx-pipeline-production
```

**Option 3: Rollback**
```bash
# Check rollout history
kubectl rollout history deployment/vfx-pipeline-production -n vfx-pipeline-production

# Rollback to previous version
kubectl rollout undo deployment/vfx-pipeline-production -n vfx-pipeline-production

# Or rollback to specific revision
kubectl rollout undo deployment/vfx-pipeline-production --to-revision=2 -n vfx-pipeline-production
```

### 2. Database Emergency

#### MongoDB Issues

```bash
# Check MongoDB status
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongosh --eval "db.adminCommand('ping')"

# Check MongoDB logs
kubectl logs mongodb-primary-0 -n vfx-pipeline-production --tail=100

# Check disk space
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- df -h

# Emergency restart MongoDB
kubectl delete pod mongodb-primary-0 -n vfx-pipeline-production
```

#### Redis Issues

```bash
# Check Redis status
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli ping

# Check Redis memory usage
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli info memory

# Flush Redis cache if needed (CAUTION: Data loss)
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli flushall

# Restart Redis
kubectl delete pod redis-master-0 -n vfx-pipeline-production
```

### 3. Resource Exhaustion

#### High CPU Usage

```bash
# Check CPU usage
kubectl top pods -n vfx-pipeline-production --sort-by=cpu

# Scale up replicas
kubectl scale deployment vfx-pipeline-production --replicas=5 -n vfx-pipeline-production

# Increase CPU limits
kubectl patch deployment vfx-pipeline-production -n vfx-pipeline-production -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "vfx-pipeline",
          "resources": {
            "limits": {"cpu": "4000m"},
            "requests": {"cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n vfx-pipeline-production --sort-by=memory

# Increase memory limits
kubectl patch deployment vfx-pipeline-production -n vfx-pipeline-production -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "vfx-pipeline",
          "resources": {
            "limits": {"memory": "8Gi"},
            "requests": {"memory": "4Gi"}
          }
        }]
      }
    }
  }
}'

# Restart pods to apply new limits
kubectl rollout restart deployment/vfx-pipeline-production -n vfx-pipeline-production
```

#### Disk Space Issues

```bash
# Check disk usage
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- df -h

# Clean up temporary files
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- find /tmp -type f -mtime +1 -delete

# Expand PVC if needed
kubectl patch pvc vfx-pipeline-storage -n vfx-pipeline-production -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

## Routine Maintenance

### 1. Daily Health Checks

```bash
#!/bin/bash
# daily-health-check.sh

echo "=== Daily Health Check $(date) ==="

# Check pod status
echo "1. Pod Status:"
kubectl get pods -n vfx-pipeline-production -l app.kubernetes.io/name=vfx-pipeline

# Check resource usage
echo "2. Resource Usage:"
kubectl top pods -n vfx-pipeline-production --sort-by=cpu | head -10

# Check disk usage
echo "3. Disk Usage:"
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- df -h | grep -E "(Filesystem|/data|/models)"

# Check application health
echo "4. Application Health:"
kubectl port-forward service/vfx-pipeline-production 8080:8000 -n vfx-pipeline-production &
sleep 3
curl -s http://localhost:8080/health | jq .
kill %1

# Check database connections
echo "5. Database Health:"
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongosh --eval "db.adminCommand('ping')" --quiet
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli ping

echo "=== Health Check Complete ==="
```

### 2. Weekly Maintenance

```bash
#!/bin/bash
# weekly-maintenance.sh

echo "=== Weekly Maintenance $(date) ==="

# Update Helm repositories
helm repo update

# Check for security updates
kubectl get pods -n vfx-pipeline-production -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u | while read image; do
    echo "Checking $image for vulnerabilities..."
    trivy image --severity HIGH,CRITICAL $image
done

# Clean up completed jobs
kubectl delete jobs --field-selector status.successful=1 -n vfx-pipeline-production

# Backup database
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongodump --out /backup/$(date +%Y%m%d)

# Check log rotation
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- find /var/log -name "*.log" -size +100M

echo "=== Weekly Maintenance Complete ==="
```

### 3. Monthly Tasks

```bash
#!/bin/bash
# monthly-tasks.sh

echo "=== Monthly Tasks $(date) ==="

# Performance analysis
echo "1. Performance Analysis:"
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- python -c "
import psutil
import time
print(f'CPU Usage: {psutil.cpu_percent(interval=1)}%')
print(f'Memory Usage: {psutil.virtual_memory().percent}%')
print(f'Disk Usage: {psutil.disk_usage(\"/\").percent}%')
"

# Update dependencies
echo "2. Dependency Updates:"
# Check for outdated packages
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- pip list --outdated

# Certificate renewal check
echo "3. Certificate Check:"
kubectl get certificates -n vfx-pipeline-production

# Capacity planning
echo "4. Capacity Planning:"
kubectl describe nodes | grep -A 5 "Allocated resources"

echo "=== Monthly Tasks Complete ==="
```

## Troubleshooting Guide

### 1. Common Issues

#### Issue: Pods Stuck in Pending State

**Symptoms:**
- Pods show "Pending" status
- Events show scheduling failures

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n vfx-pipeline-production
kubectl get events -n vfx-pipeline-production --field-selector involvedObject.name=<pod-name>
```

**Solutions:**
```bash
# Check node resources
kubectl describe nodes

# Check if PVC is bound
kubectl get pvc -n vfx-pipeline-production

# Check node selectors and affinity rules
kubectl get pod <pod-name> -n vfx-pipeline-production -o yaml | grep -A 10 nodeSelector
```

#### Issue: High Response Times

**Symptoms:**
- API responses taking >5 seconds
- Users reporting slow performance

**Diagnosis:**
```bash
# Check application metrics
kubectl port-forward service/vfx-pipeline-production 8080:8000 -n vfx-pipeline-production &
curl -s http://localhost:8080/metrics | grep response_time

# Check resource usage
kubectl top pods -n vfx-pipeline-production

# Check database performance
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongosh --eval "db.runCommand({serverStatus: 1}).opcounters"
```

**Solutions:**
```bash
# Scale up replicas
kubectl scale deployment vfx-pipeline-production --replicas=5 -n vfx-pipeline-production

# Increase resource limits
kubectl patch deployment vfx-pipeline-production -n vfx-pipeline-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"vfx-pipeline","resources":{"limits":{"cpu":"2000m","memory":"4Gi"}}}]}}}}'

# Clear Redis cache
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli flushall
```

#### Issue: Memory Leaks

**Symptoms:**
- Memory usage continuously increasing
- Pods being killed due to OOMKilled

**Diagnosis:**
```bash
# Monitor memory usage over time
kubectl top pods -n vfx-pipeline-production --sort-by=memory

# Check for memory leaks in application
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- python -c "
import psutil
import gc
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'GC stats: {gc.get_stats()}')
"
```

**Solutions:**
```bash
# Restart pods to clear memory
kubectl rollout restart deployment/vfx-pipeline-production -n vfx-pipeline-production

# Implement memory limits
kubectl patch deployment vfx-pipeline-production -n vfx-pipeline-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"vfx-pipeline","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# Enable memory profiling
kubectl set env deployment/vfx-pipeline-production PYTHONMALLOC=debug -n vfx-pipeline-production
```

### 2. Database Issues

#### MongoDB Connection Issues

```bash
# Check MongoDB status
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongosh --eval "db.adminCommand('ping')"

# Check connection pool
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- python -c "
import pymongo
client = pymongo.MongoClient('mongodb://mongodb:27017')
print(client.admin.command('serverStatus')['connections'])
"

# Restart MongoDB if needed
kubectl delete pod mongodb-primary-0 -n vfx-pipeline-production
```

#### Redis Connection Issues

```bash
# Check Redis connectivity
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli ping

# Check Redis info
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli info

# Clear Redis if corrupted
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli flushall
```

## Performance Tuning

### 1. Application Performance

#### CPU Optimization

```bash
# Enable CPU profiling
kubectl set env deployment/vfx-pipeline-production ENABLE_PROFILING=true -n vfx-pipeline-production

# Adjust worker processes
kubectl set env deployment/vfx-pipeline-production WORKERS=4 -n vfx-pipeline-production

# Enable CPU affinity
kubectl patch deployment vfx-pipeline-production -n vfx-pipeline-production -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "vfx-pipeline",
          "resources": {
            "requests": {"cpu": "1000m"},
            "limits": {"cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

#### Memory Optimization

```bash
# Tune garbage collection
kubectl set env deployment/vfx-pipeline-production PYTHONOPTIMIZE=1 -n vfx-pipeline-production

# Adjust batch sizes
kubectl set env deployment/vfx-pipeline-production BATCH_SIZE=16 -n vfx-pipeline-production

# Enable memory mapping
kubectl set env deployment/vfx-pipeline-production USE_MEMORY_MAPPING=true -n vfx-pipeline-production
```

#### GPU Optimization

```bash
# Check GPU utilization
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- nvidia-smi

# Enable mixed precision
kubectl set env deployment/vfx-pipeline-production MIXED_PRECISION=true -n vfx-pipeline-production

# Adjust GPU memory fraction
kubectl set env deployment/vfx-pipeline-production GPU_MEMORY_FRACTION=0.8 -n vfx-pipeline-production
```

### 2. Database Performance

#### MongoDB Tuning

```bash
# Check slow queries
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongosh --eval "db.setProfilingLevel(2, {slowms: 100})"

# Create indexes
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongosh --eval "
db.predictions.createIndex({video_path: 1});
db.predictions.createIndex({timestamp: -1});
db.predictions.createIndex({complexity_score: 1});
"

# Optimize connection pool
kubectl set env deployment/vfx-pipeline-production MONGODB_MAX_POOL_SIZE=50 -n vfx-pipeline-production
```

#### Redis Tuning

```bash
# Configure Redis memory policy
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli config set maxmemory-policy allkeys-lru

# Set memory limit
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli config set maxmemory 2gb

# Enable persistence
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli config set save "900 1 300 10 60 10000"
```

## Backup and Recovery

### 1. Database Backup

#### MongoDB Backup

```bash
#!/bin/bash
# mongodb-backup.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/mongodb_$BACKUP_DATE"

echo "Starting MongoDB backup: $BACKUP_DATE"

# Create backup
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongodump --out $BACKUP_DIR

# Compress backup
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR

# Upload to cloud storage (example with AWS S3)
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- aws s3 cp $BACKUP_DIR.tar.gz s3://vfx-pipeline-backups/mongodb/

echo "MongoDB backup completed: $BACKUP_DIR.tar.gz"
```

#### Redis Backup

```bash
#!/bin/bash
# redis-backup.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)

echo "Starting Redis backup: $BACKUP_DATE"

# Create RDB snapshot
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli bgsave

# Wait for background save to complete
kubectl exec -it redis-master-0 -n vfx-pipeline-production -- redis-cli lastsave

# Copy RDB file
kubectl cp vfx-pipeline-production/redis-master-0:/data/dump.rdb ./redis_backup_$BACKUP_DATE.rdb

echo "Redis backup completed: redis_backup_$BACKUP_DATE.rdb"
```

### 2. Application Recovery

#### Restore from Backup

```bash
#!/bin/bash
# restore-from-backup.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Restoring from backup: $BACKUP_FILE"

# Scale down application
kubectl scale deployment vfx-pipeline-production --replicas=0 -n vfx-pipeline-production

# Restore MongoDB
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongorestore --drop $BACKUP_FILE

# Scale up application
kubectl scale deployment vfx-pipeline-production --replicas=3 -n vfx-pipeline-production

echo "Restore completed"
```

#### Disaster Recovery

```bash
#!/bin/bash
# disaster-recovery.sh

echo "=== Disaster Recovery Procedure ==="

# 1. Assess damage
echo "1. Assessing system status..."
kubectl get pods -n vfx-pipeline-production
kubectl get pvc -n vfx-pipeline-production

# 2. Restore from latest backup
echo "2. Restoring from latest backup..."
LATEST_BACKUP=$(aws s3 ls s3://vfx-pipeline-backups/mongodb/ | sort | tail -n 1 | awk '{print $4}')
aws s3 cp s3://vfx-pipeline-backups/mongodb/$LATEST_BACKUP ./

# 3. Redeploy application
echo "3. Redeploying application..."
helm upgrade vfx-pipeline-production deployment/helm \
  --namespace vfx-pipeline-production \
  --values deployment/helm/values-production.yaml

# 4. Verify recovery
echo "4. Verifying recovery..."
kubectl rollout status deployment/vfx-pipeline-production -n vfx-pipeline-production
./deployment/scripts/monitor.sh production

echo "=== Disaster Recovery Complete ==="
```

## Security Incidents

### 1. Unauthorized Access

#### Immediate Response

```bash
# 1. Check for suspicious activity
kubectl logs deployment/vfx-pipeline-production -n vfx-pipeline-production | grep -i "unauthorized\|failed\|error"

# 2. Review authentication logs
kubectl exec -it deployment/vfx-pipeline-production -n vfx-pipeline-production -- grep "authentication" /var/log/app.log

# 3. Check network policies
kubectl get networkpolicies -n vfx-pipeline-production

# 4. Rotate secrets immediately
kubectl delete secret vfx-pipeline-secrets -n vfx-pipeline-production
kubectl create secret generic vfx-pipeline-secrets \
  --from-literal=jwt-secret-key="new-secret-key" \
  --from-literal=api-key-hash-salt="new-salt" \
  --namespace vfx-pipeline-production

# 5. Restart application to use new secrets
kubectl rollout restart deployment/vfx-pipeline-production -n vfx-pipeline-production
```

### 2. Data Breach Response

```bash
#!/bin/bash
# data-breach-response.sh

echo "=== Data Breach Response ==="

# 1. Isolate affected systems
echo "1. Isolating systems..."
kubectl patch deployment vfx-pipeline-production -n vfx-pipeline-production -p '{"spec":{"replicas":0}}'

# 2. Preserve evidence
echo "2. Preserving evidence..."
kubectl logs deployment/vfx-pipeline-production -n vfx-pipeline-production > breach-logs-$(date +%Y%m%d).txt

# 3. Assess impact
echo "3. Assessing impact..."
kubectl exec -it mongodb-primary-0 -n vfx-pipeline-production -- mongosh --eval "db.predictions.count()"

# 4. Notify stakeholders
echo "4. Notification procedures initiated..."

# 5. Implement additional security measures
echo "5. Implementing additional security..."
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: vfx-pipeline-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF

echo "=== Data Breach Response Complete ==="
```

## Scaling Operations

### 1. Horizontal Scaling

#### Auto-scaling Configuration

```bash
# Configure HPA
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vfx-pipeline-hpa
  namespace: vfx-pipeline-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vfx-pipeline-production
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
```

#### Manual Scaling

```bash
# Scale up for high load
kubectl scale deployment vfx-pipeline-production --replicas=10 -n vfx-pipeline-production

# Scale down during low usage
kubectl scale deployment vfx-pipeline-production --replicas=2 -n vfx-pipeline-production

# Check scaling status
kubectl get hpa -n vfx-pipeline-production -w
```

### 2. Vertical Scaling

```bash
# Increase resource limits
kubectl patch deployment vfx-pipeline-production -n vfx-pipeline-production -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "vfx-pipeline",
          "resources": {
            "requests": {"cpu": "2000m", "memory": "4Gi"},
            "limits": {"cpu": "4000m", "memory": "8Gi"}
          }
        }]
      }
    }
  }
}'

# Apply changes
kubectl rollout restart deployment/vfx-pipeline-production -n vfx-pipeline-production
```

### 3. Database Scaling

#### MongoDB Scaling

```bash
# Scale MongoDB replica set
helm upgrade mongodb bitnami/mongodb \
  --namespace vfx-pipeline-production \
  --set replicaSet.enabled=true \
  --set replicaSet.replicas.secondary=2

# Add MongoDB sharding (for very large datasets)
helm upgrade mongodb bitnami/mongodb-sharded \
  --namespace vfx-pipeline-production \
  --set shards=3 \
  --set mongos.replicaCount=2
```

#### Redis Scaling

```bash
# Scale Redis with clustering
helm upgrade redis bitnami/redis-cluster \
  --namespace vfx-pipeline-production \
  --set cluster.nodes=6 \
  --set cluster.replicas=1
```

## Contact Information

### Escalation Matrix

| Severity | Contact | Response Time |
|----------|---------|---------------|
| P1 (Critical) | On-call Engineer | 15 minutes |
| P2 (High) | Team Lead | 1 hour |
| P3 (Medium) | Development Team | 4 hours |
| P4 (Low) | Product Owner | 24 hours |

### Emergency Contacts

- **On-call Engineer**: +1-555-0123
- **Team Lead**: +1-555-0124
- **DevOps Lead**: +1-555-0125
- **Security Team**: +1-555-0126

### Communication Channels

- **Slack**: #vfx-pipeline-alerts
- **Email**: vfx-pipeline-team@company.com
- **Incident Management**: PagerDuty

---

*Last updated: 2024-07-17*
*Next review: 2024-08-17*
