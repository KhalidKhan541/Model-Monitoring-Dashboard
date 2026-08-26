import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.slack_webhook = config.get('slack_webhook')
        self.email_config = config.get('email', {})
    
    def send_drift_alert(self, model_name: str, drift_type: str, severity: str,
                        drifted_features: List[str], drift_scores: Dict[str, Any]):
        message = {
            'text': f"🚨 Drift Alert: {drift_type} detected for model {model_name}",
            'blocks': [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 Model Drift Alert - {severity.upper()}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Model:*\n{model_name}"},
                        {"type": "mrkdwn", "text": f"*Drift Type:*\n{drift_type}"},
                        {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"}
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Drifted Features:*\n{', '.join(drifted_features[:10]) if drifted_features else 'N/A'}"
                    }
                }
            ]
        }
        
        if self.slack_webhook:
            self._send_slack(message)
        
        if self.email_config.get('enabled'):
            self._send_email(
                subject=f"🚨 Drift Alert: {drift_type} - {model_name}",
                body=self._format_email_body(model_name, drift_type, severity, drifted_features, drift_scores)
            )
        
        logger.warning(f"Drift alert sent for {model_name}: {drift_type}")
    
    def send_latency_alert(self, model_name: str, endpoint: str, 
                          metric_name: str, value: float, threshold: float):
        message = {
            'text': f"⚠️ Latency Alert: {metric_name} = {value:.2f}ms for {model_name}",
            'blocks': [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "⚠️ Latency Alert"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Model:*\n{model_name}"},
                        {"type": "mrkdwn", "text": f"*Endpoint:*\n{endpoint}"},
                        {"type": "mrkdwn", "text": f"*Metric:*\n{metric_name}"},
                        {"type": "mrkdwn", "text": f"*Value:*\n{value:.2f}ms"},
                        {"type": "mrkdwn", "text": f"*Threshold:*\n{threshold:.2f}ms"}
                    ]
                }
            ]
        }
        
        if self.slack_webhook:
            self._send_slack(message)
        
        logger.warning(f"Latency alert: {model_name} - {metric_name} = {value:.2f}ms")
    
    def send_model_health_alert(self, model_name: str, status: str, 
                               issues: List[str]):
        severity = 'critical' if status == 'critical' else 'warning'
        
        message = {
            'text': f"⚠️ Model Health Alert: {model_name} is {status}",
            'blocks': [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"⚠️ Model Health Alert - {severity.upper()}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Model:*\n{model_name}"},
                        {"type": "mrkdwn", "text": f"*Status:*\n{status}"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"}
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Issues:*\n" + "\n".join(f"• {issue}" for issue in issues[:5])
                    }
                }
            ]
        }
        
        if self.slack_webhook:
            self._send_slack(message)
        
        logger.warning(f"Health alert for {model_name}: {status}")
    
    def send_daily_report(self, report: Dict[str, Any]):
        model_name = report.get('model_name', 'Unknown')
        
        message = {
            'text': f"📊 Daily Monitoring Report: {model_name}",
            'blocks': [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📊 Daily Report - {model_name}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Total Predictions:*\n{report.get('total_predictions', 0)}"},
                        {"type": "mrkdwn", "text": f"*Avg Latency:*\n{report.get('avg_latency_ms', 0):.2f}ms"},
                        {"type": "mrkdwn", "text": f"*Drift Alerts:*\n{report.get('drift_alerts', 0)}"},
                        {"type": "mrkdwn", "text": f"*Status:*\n{report.get('status', 'unknown')}"}
                    ]
                }
            ]
        }
        
        if self.slack_webhook:
            self._send_slack(message)
        
        if self.email_config.get('enabled'):
            self._send_email(
                subject=f"📊 Daily Report: {model_name}",
                body=json.dumps(report, indent=2)
            )
    
    def _format_email_body(self, model_name: str, drift_type: str, severity: str,
                          drifted_features: List[str], drift_scores: Dict[str, Any]) -> str:
        body = f"""
Model Drift Alert
================

Model: {model_name}
Drift Type: {drift_type}
Severity: {severity}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

Drifted Features:
{chr(10).join(f'  - {f}' for f in drifted_features[:20])}

Drift Scores:
{json.dumps(drift_scores, indent=2)}

Please investigate and take appropriate action.
"""
        return body
    
    def _send_slack(self, message: Dict[str, Any]):
        if not self.slack_webhook:
            logger.warning("Slack webhook not configured")
            return
        
        try:
            response = requests.post(self.slack_webhook, json=message)
            response.raise_for_status()
            logger.info("Slack message sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
    
    def _send_email(self, subject: str, body: str, to: str = None):
        if not self.email_config.get('enabled'):
            return
        
        smtp_host = self.email_config.get('smtp_host')
        smtp_port = self.email_config.get('smtp_port', 587)
        smtp_user = self.email_config.get('smtp_user')
        smtp_password = self.email_config.get('smtp_password')
        from_email = self.email_config.get('from_email')
        to_email = to or self.email_config.get('to_email')
        
        if not all([smtp_host, smtp_user, smtp_password, from_email, to_email]):
            logger.warning("Email configuration incomplete")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")