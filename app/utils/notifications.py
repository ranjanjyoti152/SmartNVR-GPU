import smtplib
import os
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def load_config():
    """Load email configuration from settings file"""
    config_file = os.path.join('config', 'settings.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get('email', {})
        except Exception as e:
            logger.error(f"Error loading email config: {str(e)}")
    return {}

def send_detection_email(camera, detection):
    """Send email notification for a detection event"""
    # Load email config
    email_config = load_config()
    
    # Check if email notifications are enabled
    if not email_config.get('enabled', False):
        logger.info("Email notifications are disabled")
        return False
    
    # Check required parameters
    smtp_server = email_config.get('smtp_server')
    smtp_port = email_config.get('smtp_port', 587)
    smtp_username = email_config.get('smtp_username')
    smtp_password = email_config.get('smtp_password')
    from_email = email_config.get('from_email')
    recipients = email_config.get('recipients', [])
    
    if not all([smtp_server, smtp_username, smtp_password, from_email, recipients]):
        logger.warning("Incomplete email configuration")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"SmartNVR Alert: {detection.class_name} detected on {camera.name}"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>SmartNVR Detection Alert</h2>
            <p><strong>Camera:</strong> {camera.name}</p>
            <p><strong>Object:</strong> {detection.class_name}</p>
            <p><strong>Confidence:</strong> {detection.confidence:.2%}</p>
            <p><strong>Time:</strong> {detection.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Add image if available
        image_path = detection.image_path
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', '<detection_image>')
                msg.attach(img)
            body += '<p><img src="cid:detection_image" width="640" /></p>'
        
        # Add video link if available
        if detection.video_path:
            video_url = f"/playback?video={os.path.basename(detection.video_path)}&camera={camera.id}"
            body += f'<p><a href="{video_url}">View Recorded Video</a></p>'
        
        body += """
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server and send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Detection alert email sent for {detection.class_name} on {camera.name}")
        
        # Update notification status
        detection.notified = True
        
        return True
        
    except Exception as e:
        logger.error(f"Error sending email notification: {str(e)}")
        return False

def send_test_email(smtp_server, smtp_port, smtp_username, smtp_password, recipients):
    """
    Send a test email to verify SMTP configuration
    
    Args:
        smtp_server (str): SMTP server address
        smtp_port (int): SMTP server port
        smtp_username (str): SMTP username
        smtp_password (str): SMTP password
        recipients (list): List of email addresses to send to
        
    Returns:
        dict: Result with success status and error message if applicable
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = "SmartNVR - Test Email"
        
        # Email body with HTML
        body = f"""
        <html>
        <body>
            <h2>SmartNVR Email Test</h2>
            <p>This is a test email from your SmartNVR system.</p>
            <p>If you're receiving this message, your email notifications are configured correctly.</p>
            <p><strong>Configuration:</strong></p>
            <ul>
                <li>SMTP Server: {smtp_server}</li>
                <li>SMTP Port: {smtp_port}</li>
                <li>Username: {smtp_username}</li>
            </ul>
            <p>This email was sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Thank you for using SmartNVR!</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server and send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Test email sent to {', '.join(recipients)}")
        
        return {
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error sending test email: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }