"""
TensorBoard utility functions for SPTnano transformer monitoring
"""

import os
import subprocess
import webbrowser
import time
import threading
from IPython.display import display, HTML

# Import config from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

def launch_tensorboard(log_dir=None, port=6006, auto_open=True):
    """
    Launch TensorBoard server and optionally open in browser
    
    Parameters:
    -----------
    log_dir : str, optional
        Directory containing TensorBoard logs. If None, uses default config directory
    port : int
        Port for TensorBoard server (default 6006)
    auto_open : bool
        Whether to automatically open browser window
        
    Returns:
    --------
    process : subprocess.Popen
        TensorBoard server process
    url : str
        URL where TensorBoard is running
    """
    
    if log_dir is None:
        log_dir = config.TENSORBOARD_LOGS
    
    if not os.path.exists(log_dir):
        print(f"❌ TensorBoard log directory not found: {log_dir}")
        return None, None
    
    # Check if TensorBoard is already running on this port
    try:
        import requests
        url = f"http://localhost:{port}"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print(f"⚠ TensorBoard already running at {url}")
            if auto_open:
                webbrowser.open(url)
            return None, url
    except:
        pass  # Port is free, continue
    
    # Launch TensorBoard
    cmd = ['tensorboard', '--logdir', log_dir, '--port', str(port), '--host', 'localhost']
    
    try:
        print(f"🚀 Launching TensorBoard...")
        print(f"   Log directory: {log_dir}")
        print(f"   Port: {port}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        url = f"http://localhost:{port}"
        
        # Check if server started successfully
        try:
            import requests
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ TensorBoard running at: {url}")
                
                if auto_open:
                    print("🌐 Opening browser...")
                    webbrowser.open(url)
                
                return process, url
            else:
                print(f"❌ TensorBoard server not responding")
                process.terminate()
                return None, None
        except Exception as e:
            print(f"❌ Could not connect to TensorBoard: {e}")
            process.terminate()
            return None, None
            
    except FileNotFoundError:
        print("❌ TensorBoard not found. Install with: mamba install tensorboard")
        return None, None
    except Exception as e:
        print(f"❌ Failed to launch TensorBoard: {e}")
        return None, None

def stop_tensorboard(process):
    """
    Stop TensorBoard server process
    
    Parameters:
    -----------
    process : subprocess.Popen
        TensorBoard process to stop
    """
    if process:
        process.terminate()
        process.wait()
        print("🛑 TensorBoard server stopped")

def tensorboard_widget(log_dir=None, port=6006, height=600):
    """
    Create an inline TensorBoard widget for Jupyter notebooks
    
    Parameters:
    -----------
    log_dir : str, optional
        Directory containing TensorBoard logs
    port : int
        Port for TensorBoard server
    height : int
        Height of iframe in pixels
        
    Returns:
    --------
    IPython.display.HTML widget
    """
    
    if log_dir is None:
        log_dir = config.TENSORBOARD_LOGS
    
    # Launch TensorBoard if not running
    process, url = launch_tensorboard(log_dir, port, auto_open=False)
    
    if url:
        iframe_html = f'''
        <div style="border: 2px solid #1f77b4; border-radius: 5px; padding: 10px; margin: 10px 0;">
            <h3 style="margin-top: 0; color: #1f77b4;">📊 TensorBoard Monitor</h3>
            <p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>
            <iframe src="{url}" width="100%" height="{height}px" frameborder="0"></iframe>
        </div>
        '''
        return HTML(iframe_html)
    else:
        return HTML('<div style="color: red;">❌ Failed to launch TensorBoard</div>')

def get_tensorboard_status():
    """
    Check TensorBoard server status and available log directories
    
    Returns:
    --------
    dict with status information
    """
    
    status = {
        'tensorboard_available': False,
        'server_running': False,
        'log_dirs': [],
        'latest_run': None
    }
    
    # Check if TensorBoard is installed
    try:
        subprocess.run(['tensorboard', '--version'], 
                      capture_output=True, check=True)
        status['tensorboard_available'] = True
    except:
        return status
    
    # Check if server is running
    try:
        import requests
        response = requests.get('http://localhost:6006', timeout=2)
        if response.status_code == 200:
            status['server_running'] = True
    except:
        pass
    
    # List available log directories
    if os.path.exists(config.TENSORBOARD_LOGS):
        for item in os.listdir(config.TENSORBOARD_LOGS):
            item_path = os.path.join(config.TENSORBOARD_LOGS, item)
            if os.path.isdir(item_path):
                status['log_dirs'].append(item)
        
        # Find latest run
        if status['log_dirs']:
            status['log_dirs'].sort(reverse=True)  # Most recent first
            status['latest_run'] = status['log_dirs'][0]
    
    return status

def print_tensorboard_instructions():
    """
    Print instructions for using TensorBoard
    """
    print("=" * 60)
    print("📊 TENSORBOARD MONITORING INSTRUCTIONS")
    print("=" * 60)
    print()
    print("🚀 AUTOMATIC LAUNCH (Recommended):")
    print("   from SPTnano.tensorboard_utils import launch_tensorboard")
    print("   process, url = launch_tensorboard()")
    print()
    print("🌐 MANUAL LAUNCH (Terminal):")
    print(f"   tensorboard --logdir {config.TENSORBOARD_LOGS} --port 6006")
    print("   Then open: http://localhost:6006")
    print()
    print("📱 JUPYTER WIDGET (Inline):")
    print("   from SPTnano.tensorboard_utils import tensorboard_widget")
    print("   tensorboard_widget()")
    print()
    print("📊 WHAT YOU'LL SEE:")
    print("   • Loss curves for each scale (30f, 60f, 120f, 240f)")
    print("   • Learning rate schedules")
    print("   • Real-time training progress")
    print("   • Compare different training runs")
    print()
    print("🛑 TO STOP:")
    print("   • Close browser tab")
    print("   • Ctrl+C in terminal")
    print("   • Use stop_tensorboard(process) if launched programmatically")
    print("=" * 60)

# Convenience function for notebook use
def start_monitoring():
    """
    Quick start function for notebook users
    """
    status = get_tensorboard_status()
    
    if not status['tensorboard_available']:
        print("❌ TensorBoard not installed. Run: mamba install tensorboard")
        return None
    
    if status['server_running']:
        print("✅ TensorBoard already running at http://localhost:6006")
        return None
    
    if not status['log_dirs']:
        print("⚠ No TensorBoard logs found yet. Train a model first.")
        return None
    
    print(f"📊 Found {len(status['log_dirs'])} log directories")
    print(f"🎯 Latest run: {status['latest_run']}")
    
    return launch_tensorboard() 