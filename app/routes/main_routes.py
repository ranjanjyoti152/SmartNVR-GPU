"""
Main routes for SmartNVR application
"""
from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from app.models.camera import Camera
from app.models.ai_model import AIModel

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Landing page / redirect to dashboard if logged in"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Dashboard route with live camera feeds"""
    cameras = Camera.query.filter_by(is_active=True).all()
    return render_template('dashboard.html', title='Dashboard', cameras=cameras)

@main_bp.route('/playback')
@login_required
def playback():
    """Video playback route"""
    cameras = Camera.query.filter_by(is_active=True).all()
    return render_template('playback.html', title='Playback', cameras=cameras)

@main_bp.route('/monitor')
@login_required
def monitor():
    """System monitoring route"""
    from app.utils.system_monitor import get_system_resources
    resources = get_system_resources()
    return render_template('monitor.html', title='System Monitor', resources=resources)

@main_bp.route('/profile')
@login_required
def profile():
    """User profile route"""
    return render_template('profile.html', title='My Profile')

@main_bp.route('/camera-management')
@login_required
def camera_management():
    """Camera management route"""
    cameras = Camera.query.all()
    models = AIModel.query.all()
    return render_template('camera_management.html', title='Camera Management', cameras=cameras, models=models)

@main_bp.route('/add-camera', methods=['POST'])
@login_required
def add_camera():
    """Add new camera"""
    from flask import request, flash
    
    name = request.form.get('name')
    rtsp_url = request.form.get('rtsp_url')
    username = request.form.get('username')
    password = request.form.get('password')
    model_id = request.form.get('model_id')
    confidence = request.form.get('confidence', 0.45)
    
    # Validate inputs
    if not name or not rtsp_url:
        flash('Camera name and RTSP URL are required', 'danger')
        return redirect(url_for('main.camera_management'))
    
    # Create new camera
    camera = Camera(
        name=name,
        rtsp_url=rtsp_url,
        username=username,
        password=password,
        model_id=model_id,
        confidence_threshold=confidence,
        is_active=True
    )
    
    # Save to database
    from app import db
    db.session.add(camera)
    db.session.commit()
    
    # Start camera processor after adding the camera
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        if manager.start_camera(camera):
            flash(f'Camera {name} added and started successfully', 'success')
        else:
            flash(f'Camera {name} added but could not be started. Check camera URL and credentials.', 'warning')
    except Exception as e:
        flash(f'Camera {name} added but an error occurred when starting: {str(e)}', 'warning')
    
    return redirect(url_for('main.camera_management'))

@main_bp.route('/edit-camera/<int:camera_id>', methods=['POST'])
@login_required
def edit_camera(camera_id):
    """Edit existing camera"""
    from flask import request, flash
    
    # Find camera
    camera = Camera.query.get_or_404(camera_id)
    
    # Store old active state to check if we need to start/stop the camera
    old_is_active = camera.is_active
    
    # Update camera details
    camera.name = request.form.get('name', camera.name)
    camera.rtsp_url = request.form.get('rtsp_url', camera.rtsp_url)
    camera.username = request.form.get('username')
    camera.password = request.form.get('password')
    camera.location = request.form.get('location')
    camera.is_active = 'enabled' in request.form
    camera.recording_enabled = 'recording_enabled' in request.form
    camera.detection_enabled = 'detection_enabled' in request.form
    
    if request.form.get('model_id'):
        camera.model_id = request.form.get('model_id')
        
    if request.form.get('confidence'):
        camera.confidence_threshold = float(request.form.get('confidence'))
    
    # Save to database
    from app import db
    db.session.commit()
    
    # Restart camera processor if needed
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        
        # If camera was active and still is, or settings changed, restart it
        if camera.is_active:
            # Stop camera if it's running
            manager.stop_camera(camera.id)
            
            # Start camera with new settings
            if manager.start_camera(camera):
                flash(f'Camera {camera.name} updated and restarted successfully', 'success')
            else:
                flash(f'Camera {camera.name} updated but could not be started. Check camera URL and credentials.', 'warning')
        elif old_is_active and not camera.is_active:
            # Camera was disabled, make sure it's stopped
            manager.stop_camera(camera.id)
            flash(f'Camera {camera.name} updated and stopped successfully', 'success')
        else:
            flash(f'Camera {camera.name} updated successfully', 'success')
    except Exception as e:
        flash(f'Camera {camera.name} updated but an error occurred when restarting: {str(e)}', 'warning')
    
    return redirect(url_for('main.camera_management'))

@main_bp.route('/delete-camera/<int:camera_id>', methods=['POST'])
@login_required
def delete_camera(camera_id):
    """Delete camera"""
    from flask import flash
    
    # Find camera
    camera = Camera.query.get_or_404(camera_id)
    camera_name = camera.name
    
    # Stop camera processor first if it's running
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        manager.stop_camera(camera_id)
    except Exception as e:
        flash(f'Warning: Error stopping camera processor: {str(e)}', 'warning')
    
    # Delete from database
    from app import db
    db.session.delete(camera)
    db.session.commit()
    
    flash(f'Camera {camera_name} deleted successfully', 'success')
    return redirect(url_for('main.camera_management'))

@main_bp.route('/settings')
@login_required
def settings():
    """System settings route"""
    # Load current settings
    import os
    import json
    settings_file = os.path.join('config', 'settings.json')
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            # Ensure all required sections exist in settings
            if 'recording' not in settings:
                settings['recording'] = {
                    'retention_days': 30,
                    'storage_path': 'storage/recordings',
                    'clip_length': 60,
                    'format': 'mp4'
                }
                
            if 'notifications' not in settings:
                settings['notifications'] = {
                    'email_enabled': False,
                    'smtp_server': '',
                    'smtp_port': 587,
                    'smtp_username': '',
                    'smtp_password': '',
                    'from_email': '',
                    'email_to': ''
                }
                
            if 'system' not in settings:
                settings['system'] = {
                    'log_level': 'info'
                }
                
            if 'detection' not in settings:
                settings['detection'] = {
                    'default_confidence': 0.45,
                    'default_model': 'yolov5s'
                }
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # Default settings if JSON is invalid
            settings = create_default_settings()
    else:
        # Default settings if file doesn't exist
        settings = create_default_settings()
        
    # Get available AI models
    from app.models.ai_model import AIModel
    ai_models = AIModel.query.all()
    
    return render_template('settings.html', title='Settings', settings=settings, ai_models=ai_models)

def create_default_settings():
    """Create default settings dictionary"""
    return {
        'recording': {
            'retention_days': 30,
            'storage_path': 'storage/recordings',
            'clip_length': 60,
            'format': 'mp4'
        },
        'notifications': {
            'email_enabled': False,
            'smtp_server': '',
            'smtp_port': 587,
            'smtp_username': '',
            'smtp_password': '',
            'from_email': '',
            'email_to': ''
        },
        'system': {
            'log_level': 'info'
        },
        'detection': {
            'default_confidence': 0.45,
            'default_model': 'yolov5s'
        }
    }

@main_bp.route('/save-settings', methods=['POST'])
@login_required
def save_settings():
    """Save system settings"""
    from flask import request, flash
    import os
    import json
    
    # Get form data
    settings = {
        'recording': {
            'retention_days': int(request.form.get('retention_days', 30)),
            'storage_path': request.form.get('storage_path', 'storage/recordings'),
            'clip_length': int(request.form.get('clip_length', 60)),
            'format': request.form.get('format', 'mp4')
        },
        'notifications': {
            'email_enabled': 'email_enabled' in request.form,
            'smtp_server': request.form.get('smtp_server', ''),
            'smtp_port': int(request.form.get('smtp_port', 587)),
            'smtp_username': request.form.get('smtp_username', ''),
            'from_email': request.form.get('from_email', ''),
            'email_to': request.form.get('email_to', ''),
            'push_enabled': 'push_enabled' in request.form
        },
        'system': {
            'log_level': request.form.get('log_level', 'info')
        },
        'detection': {
            'default_confidence': float(request.form.get('default_confidence', 0.45)),
            'default_model': request.form.get('default_model', 'yolov5s')
        }
    }
    
    # Only update password if provided
    if request.form.get('smtp_password'):
        settings['notifications']['smtp_password'] = request.form.get('smtp_password')
    else:
        # Preserve existing password if it exists
        settings_file = os.path.join('config', 'settings.json')
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    old_settings = json.load(f)
                    if 'notifications' in old_settings and 'smtp_password' in old_settings['notifications']:
                        settings['notifications']['smtp_password'] = old_settings['notifications']['smtp_password']
            except:
                pass
    
    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)
    
    # Save settings to JSON file
    with open(os.path.join('config', 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)
    
    flash('Settings saved successfully', 'success')
    return redirect(url_for('main.settings'))