from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
import os
import json
import torch
import shutil
from werkzeug.utils import secure_filename

from app import db
from app.models import User, Camera
from app.models.ai_model import AIModel
from app.utils.decorators import admin_required

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/')
@login_required
@admin_required
def admin_index():
    """Admin dashboard"""
    return render_template('admin/dashboard.html', title='Admin Dashboard')

@admin_bp.route('/users')
@login_required
@admin_required
def user_management():
    """User management page"""
    users = User.query.all()
    return render_template('admin/users.html', title='User Management', users=users)

@admin_bp.route('/users/create', methods=['GET', 'POST'])
@login_required
@admin_required
def create_user():
    """Create a new user"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        is_admin = True if request.form.get('is_admin') else False
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('admin.create_user'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('admin.create_user'))
        
        # Create new user
        user = User(username=username, email=email, is_admin=is_admin)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('User created successfully.', 'success')
        return redirect(url_for('admin.user_management'))
    
    return render_template('admin/create_user.html', title='Create User')

@admin_bp.route('/users/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit a user"""
    user = User.query.get_or_404(user_id)
    
    # Prevent admins from editing themselves (to avoid removing their own admin rights)
    if user.id == current_user.id:
        flash('You cannot edit your own account from here. Use profile page instead.', 'warning')
        return redirect(url_for('admin.user_management'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        is_admin = True if request.form.get('is_admin') else False
        password = request.form.get('password')
        
        # Update user
        if email != user.email:
            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'danger')
                return redirect(url_for('admin.edit_user', user_id=user_id))
            user.email = email
        
        user.is_admin = is_admin
        
        if password:
            user.set_password(password)
        
        db.session.commit()
        flash('User updated successfully.', 'success')
        return redirect(url_for('admin.user_management'))
    
    return render_template('admin/edit_user.html', title='Edit User', user=user)

@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user"""
    user = User.query.get_or_404(user_id)
    
    # Prevent admins from deleting themselves
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    db.session.delete(user)
    db.session.commit()
    
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin.user_management'))

# API endpoints
@admin_bp.route('/api/users', methods=['GET'])
@login_required
@admin_required
def api_get_users():
    """API endpoint to get all users"""
    users = User.query.all()
    return jsonify([{
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'is_admin': user.is_admin,
        'last_login': user.last_login.isoformat() if user.last_login else None,
        'created_at': user.created_at.isoformat()
    } for user in users])

@admin_bp.route('/models')
@login_required
@admin_required
def manage_models():
    """AI model management page"""
    models = AIModel.query.all()
    return render_template('admin/models.html', title='AI Model Management', models=models)

@admin_bp.route('/models/add', methods=['POST'])
@login_required
@admin_required
def add_model():
    """Add a new AI model reference"""
    file_path = request.form.get('file_path')
    name = request.form.get('name')
    description = request.form.get('description', '')
    is_custom = True if request.form.get('is_custom') else False
    is_default = True if request.form.get('is_default') else False
    
    # Validate inputs
    if not all([file_path, name]):
        flash('Name and file path are required.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Check if file exists
    if not os.path.exists(file_path):
        flash(f'Model file not found at {file_path}. Please ensure the file exists.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Check if model name already exists
    if AIModel.query.filter_by(name=name).first():
        flash('A model with this name already exists.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # If this is set to default, clear default flag on all other models
    if is_default:
        AIModel.query.filter_by(is_default=True).update({'is_default': False})
    
    # Create new model
    model = AIModel(
        name=name,
        file_path=file_path,
        description=description,
        is_custom=is_custom,
        is_default=is_default
    )
    
    db.session.add(model)
    db.session.commit()
    
    flash('Model added successfully.', 'success')
    return redirect(url_for('admin.manage_models'))

@admin_bp.route('/models/upload', methods=['POST'])
@login_required
@admin_required
def upload_model():
    """Upload a new custom YOLOv5 model"""
    try:
        # Check if the post request has the file part
        if 'model_file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file part in the request'
            }), 400
            
        model_file = request.files['model_file']
        
        if model_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        name = request.form.get('name')
        description = request.form.get('description', '')
        is_default = True if request.form.get('is_default') else False
        
        # Validate inputs
        if not name:
            return jsonify({
                'success': False,
                'message': 'Model name is required'
            }), 400
        
        # Check if model name already exists
        if AIModel.query.filter_by(name=name).first():
            return jsonify({
                'success': False,
                'message': 'A model with this name already exists'
            }), 400
        
        # Print debugging information
        print(f"Received file: {model_file.filename}, size: {model_file.content_length}")
        
        # Sanitize filename and save model file
        filename = secure_filename(model_file.filename)
        if not filename.endswith('.pt'):
            filename = f"{filename}.pt"
        
        models_dir = os.path.join('storage', 'models')
        os.makedirs(models_dir, exist_ok=True)
        file_path = os.path.join(models_dir, filename)
        
        # If file exists, add timestamp to filename
        if os.path.exists(file_path):
            import time
            timestamp = int(time.time())
            name_parts = filename.rsplit('.', 1)
            filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            file_path = os.path.join(models_dir, filename)
        
        print(f"Saving file to: {file_path}")
        model_file.save(file_path)
        print(f"File saved successfully")
        
        # Verify it's a valid YOLOv5 model
        try:
            print(f"Attempting to verify model file with PyTorch")
            # Load model in a safe way - this will validate the file is a PyTorch model
            model_data = torch.load(file_path, map_location=torch.device('cpu'))
            print(f"Model verified successfully")
        except Exception as e:
            # Remove the invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            print(f"Model verification failed: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Invalid PyTorch model file: {str(e)}'
            }), 400
        
        # If this is set to default, clear default flag on all other models
        if is_default:
            AIModel.query.filter_by(is_default=True).update({'is_default': False})
        
        # Create model record
        model = AIModel(
            name=name,
            file_path=file_path,
            description=description,
            is_custom=True,
            is_default=is_default
        )
        
        db.session.add(model)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Model uploaded successfully',
            'model': {
                'id': model.id,
                'name': model.name,
                'file_path': model.file_path,
                'is_default': model.is_default
            }
        })
    except Exception as e:
        print(f"Unexpected error during model upload: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@admin_bp.route('/models/download-pretrained', methods=['POST'])
@login_required
@admin_required
def download_pretrained_model():
    """Download a pre-trained YOLOv5 model from the official repository"""
    data = request.json
    model_name = data.get('model')
    
    if not model_name or model_name not in ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
        return jsonify({
            'success': False,
            'message': 'Invalid model name'
        }), 400
    
    try:
        # Download model using torch hub
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Load model from PyTorch hub
        model = torch.hub.load('ultralytics/yolov5', model_name)
        
        # Save model
        model_path = os.path.join(models_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Add model to database if not already present
        existing_model = AIModel.query.filter_by(name=model_name.upper()).first()
        if not existing_model:
            model_record = AIModel(
                name=model_name.upper(),
                file_path=model_path,
                description=f"Pre-trained {model_name.upper()} model",
                is_custom=False,
                is_default=(model_name == 'yolov5s' and AIModel.query.filter_by(is_default=True).count() == 0)
            )
            
            db.session.add(model_record)
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Model {model_name} downloaded successfully',
            'model_path': model_path
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error downloading model: {str(e)}'
        }), 500

@admin_bp.route('/models/<int:model_id>/update', methods=['POST'])
@login_required
@admin_required
def update_model(model_id):
    """Update an AI model's metadata"""
    model = AIModel.query.get_or_404(model_id)
    
    name = request.form.get('name')
    description = request.form.get('description', '')
    
    # Validate inputs
    if not name:
        flash('Model name is required.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Check if name already exists for a different model
    existing = AIModel.query.filter_by(name=name).first()
    if existing and existing.id != model_id:
        flash('A model with this name already exists.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Update model
    model.name = name
    model.description = description
    
    db.session.commit()
    
    flash('Model updated successfully.', 'success')
    return redirect(url_for('admin.manage_models'))

@admin_bp.route('/models/<int:model_id>/set-default', methods=['POST'])
@login_required
@admin_required
def set_default_model(model_id):
    """Set an AI model as the default"""
    model = AIModel.query.get_or_404(model_id)
    
    # Clear default flag on all models
    AIModel.query.filter_by(is_default=True).update({'is_default': False})
    
    # Set this model as default
    model.is_default = True
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'{model.name} set as default model'
    })

@admin_bp.route('/models/<int:model_id>/delete', methods=['DELETE'])
@login_required
@admin_required
def delete_model(model_id):
    """Delete an AI model"""
    model = AIModel.query.get_or_404(model_id)
    
    # Prevent deleting default model
    if model.is_default:
        return jsonify({
            'success': False,
            'message': 'Cannot delete the default model. Set another model as default first.'
        }), 400
    
    # Delete file if it's a custom model
    if model.is_custom and os.path.exists(model.file_path):
        try:
            os.remove(model.file_path)
        except Exception as e:
            # Log error but continue with deletion
            print(f"Error deleting model file: {str(e)}")
    
    # Delete from database
    db.session.delete(model)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Model deleted successfully'
    })