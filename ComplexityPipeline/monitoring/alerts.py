# monitoring/alerts.py

import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import time

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AlertManager:
    """Comprehensive alerting system for the VFX pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('alerts', {})
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_lock = threading.Lock()
        
        # Alert thresholds
        self.thresholds = self.config.get('thresholds', {
            'error_rate_percent': 5.0,
            'processing_time_seconds': 300.0,
            'queue_size': 50,
            'memory_usage_percent': 85.0,
            'cpu_usage_percent': 90.0,
            'disk_usage_percent': 90.0
        })
        
        # Notification settings
        self.email_config = self.config.get('email', {})
        self.slack_config = self.config.get('slack', {})
        self.webhook_config = self.config.get('webhook', {})
        
        # Alert suppression (prevent spam)
        self.suppression_window = self.config.get('suppression_window_minutes', 15)
        self.last_alerts: Dict[str, datetime] = {}
        
        # Start background monitoring
        self._start_alert_monitoring()
    
    def _start_alert_monitoring(self):
        """Start background thread for alert monitoring."""
        def monitor_alerts():
            while True:
                try:
                    self._check_system_alerts()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Alert monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_alerts, daemon=True)
        monitor_thread.start()
    
    def _check_system_alerts(self):
        """Check for system-level alerts."""
        import psutil
        
        # CPU usage alert
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.thresholds['cpu_usage_percent']:
            self.create_alert(
                'high_cpu_usage',
                'High CPU Usage',
                f'CPU usage is {cpu_percent:.1f}% (threshold: {self.thresholds["cpu_usage_percent"]}%)',
                AlertSeverity.HIGH,
                'system_monitor',
                {'cpu_percent': cpu_percent}
            )
        
        # Memory usage alert
        memory = psutil.virtual_memory()
        if memory.percent > self.thresholds['memory_usage_percent']:
            self.create_alert(
                'high_memory_usage',
                'High Memory Usage',
                f'Memory usage is {memory.percent:.1f}% (threshold: {self.thresholds["memory_usage_percent"]}%)',
                AlertSeverity.HIGH,
                'system_monitor',
                {'memory_percent': memory.percent}
            )
        
        # Disk usage alert
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > self.thresholds['disk_usage_percent']:
            self.create_alert(
                'high_disk_usage',
                'High Disk Usage',
                f'Disk usage is {disk_percent:.1f}% (threshold: {self.thresholds["disk_usage_percent"]}%)',
                AlertSeverity.MEDIUM,
                'system_monitor',
                {'disk_percent': disk_percent}
            )
    
    def create_alert(self, alert_id: str, title: str, description: str, 
                    severity: AlertSeverity, source: str, metadata: Dict[str, Any] = None):
        """Create a new alert."""
        with self.alert_lock:
            # Check if alert should be suppressed
            if self._should_suppress_alert(alert_id):
                return
            
            # Create alert
            alert = Alert(
                id=alert_id,
                title=title,
                description=description,
                severity=severity,
                timestamp=datetime.now(),
                source=source,
                metadata=metadata or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.last_alerts[alert_id] = alert.timestamp
            
            # Send notifications
            self._send_notifications(alert)
            
            logger.warning(f"Alert created: {alert_id} - {title}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        with self.alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id}")
    
    def _should_suppress_alert(self, alert_id: str) -> bool:
        """Check if alert should be suppressed due to recent similar alert."""
        if alert_id not in self.last_alerts:
            return False
        
        last_alert_time = self.last_alerts[alert_id]
        time_diff = datetime.now() - last_alert_time
        
        return time_diff < timedelta(minutes=self.suppression_window)
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications via configured channels."""
        # Email notification
        if self.email_config.get('enabled', False):
            self._send_email_notification(alert)
        
        # Slack notification
        if self.slack_config.get('enabled', False):
            self._send_slack_notification(alert)
        
        # Webhook notification
        if self.webhook_config.get('enabled', False):
            self._send_webhook_notification(alert)
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        try:
            smtp_server = self.email_config['smtp_server']
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config['username']
            password = self.email_config['password']
            recipients = self.email_config['recipients']
            
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[VFX Pipeline Alert] {alert.title}"
            
            body = f"""
Alert Details:
- ID: {alert.id}
- Severity: {alert.severity.value.upper()}
- Source: {alert.source}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Description: {alert.description}

Metadata:
{json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.sendmail(username, recipients, msg.as_string())
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification."""
        try:
            webhook_url = self.slack_config['webhook_url']
            
            color_map = {
                AlertSeverity.LOW: 'good',
                AlertSeverity.MEDIUM: 'warning',
                AlertSeverity.HIGH: 'danger',
                AlertSeverity.CRITICAL: 'danger'
            }
            
            payload = {
                'text': f'VFX Pipeline Alert: {alert.title}',
                'attachments': [{
                    'color': color_map.get(alert.severity, 'warning'),
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True},
                        {'title': 'Description', 'value': alert.description, 'short': False}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        try:
            webhook_url = self.webhook_config['url']
            
            payload = {
                'alert_id': alert.id,
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.value,
                'timestamp': alert.timestamp.isoformat(),
                'source': alert.source,
                'metadata': alert.metadata
            }
            
            headers = self.webhook_config.get('headers', {})
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.alert_lock:
            return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        with self.alert_lock:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp > last_24h
            ]
            
            severity_counts = {severity.value: 0 for severity in AlertSeverity}
            for alert in recent_alerts:
                severity_counts[alert.severity.value] += 1
            
            return {
                'active_alerts': len(self.active_alerts),
                'alerts_last_24h': len(recent_alerts),
                'severity_breakdown': severity_counts,
                'most_recent_alert': max(
                    (alert.timestamp for alert in recent_alerts),
                    default=None
                )
            }
