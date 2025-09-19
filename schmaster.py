#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master script for batch processing with memory leak prevention
Processes files in separate processes to avoid memory accumulation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
import psutil
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(log_dir):
    """Setup logging to both file and console"""
    log_file = Path(log_dir) / 'batch_processing.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Clear existing handlers and add new ones
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_video_files(input_dir, extensions=None):
    """Get all video files from input directory"""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    video_files = []
    for ext in extensions:
        video_files.extend(input_path.glob(f'**/*{ext}'))
        video_files.extend(input_path.glob(f'**/*{ext.upper()}'))
    
    return sorted(video_files)

def monitor_process_memory(process, max_memory_gb=8):
    """Monitor process memory usage and kill if it exceeds limit"""
    try:
        if process.poll() is None:  # Process is still running
            proc = psutil.Process(process.pid)
            memory_gb = proc.memory_info().rss / (1024**3)
            if memory_gb > max_memory_gb:
                logger.warning(f"Process memory exceeded {max_memory_gb}GB ({memory_gb:.2f}GB), terminating...")
                process.terminate()
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return False

def process_single_file(video_file, script_path, base_args, output_dir, timeout_seconds=3600, max_memory_gb=8, show_output=True, realtime_output=False):
    """Process a single video file in a separate process"""
    process_logger = logging.getLogger(f"worker-{os.getpid()}")
    process_logger.info(f"🎬 Starting: {video_file.name}")
    
    # Build command arguments
    cmd = [sys.executable, str(script_path)]
    cmd.extend(base_args)
    cmd.extend(['--source', str(video_file)])
    
    # Set output name based on video file name
    output_name = video_file.stem
    cmd.extend(['--project', str(output_dir)])
    cmd.extend(['--name', output_name])
    
    start_time = time.time()
    process = None
    output_lines = []
    
    try:
        # Start process
        process_logger.info(f"Command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        if realtime_output and show_output:
            # Real-time output streaming
            import threading
            import queue
            
            def read_output(pipe, q, stream_name):
                try:
                    for line in iter(pipe.readline, ''):
                        q.put((stream_name, line))
                    pipe.close()
                except:
                    pass
            
            q = queue.Queue()
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, q, 'stdout'))
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, q, 'stderr'))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitor with real-time output
            while process.poll() is None:
                elapsed = time.time() - start_time
                
                # Check timeout
                if elapsed > timeout_seconds:
                    process_logger.error(f"⏱️  Timeout ({timeout_seconds}s) for {video_file.name}")
                    process.terminate()
                    process.wait(timeout=10)
                    return video_file, False, f"Timeout after {timeout_seconds}s"
                
                # Check memory
                if monitor_process_memory(process, max_memory_gb):
                    process.terminate()
                    process.wait(timeout=10)
                    return video_file, False, "Memory limit exceeded"
                
                # Get output from queue
                try:
                    while True:
                        stream_name, line = q.get_nowait()
                        line = line.strip()
                        if line:
                            output_lines.append(line)
                            if stream_name == 'stdout':
                                process_logger.info(f"[{video_file.stem}] {line}")
                            else:
                                process_logger.warning(f"[{video_file.stem}] ERR: {line}")
                except queue.Empty:
                    time.sleep(0.5)
            
            # Get remaining output
            try:
                while True:
                    stream_name, line = q.get_nowait()
                    line = line.strip()
                    if line:
                        output_lines.append(line)
                        if stream_name == 'stdout':
                            process_logger.info(f"[{video_file.stem}] {line}")
                        else:
                            process_logger.warning(f"[{video_file.stem}] ERR: {line}")
            except queue.Empty:
                pass
                
        else:
            # Batch output (current simple approach)
            while process.poll() is None:
                elapsed = time.time() - start_time
                
                if elapsed > timeout_seconds:
                    process_logger.error(f"⏱️  Timeout ({timeout_seconds}s) for {video_file.name}")
                    process.terminate()
                    process.wait(timeout=10)
                    return video_file, False, f"Timeout after {timeout_seconds}s"
                
                if monitor_process_memory(process, max_memory_gb):
                    process.terminate()
                    process.wait(timeout=10)
                    return video_file, False, "Memory limit exceeded"
                
                time.sleep(1)
        
        # Get final output if not using real-time
        if not realtime_output:
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            
            # Show batch output
            if show_output and stdout:
                for line in stdout.strip().split('\n'):
                    if line.strip():
                        process_logger.info(f"[{video_file.stem}] {line.strip()}")
            
            if stderr:
                for line in stderr.strip().split('\n'):
                    if line.strip():
                        process_logger.warning(f"[{video_file.stem}] ERR: {line.strip()}")
        
        return_code = process.returncode
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            process_logger.info(f"✅ Completed {video_file.name} in {elapsed_time:.1f}s")
            time.sleep(0.5)
            return video_file, True, f"Success in {elapsed_time:.1f}s"
        else:
            process_logger.error(f"❌ Failed {video_file.name} (code {return_code}) in {elapsed_time:.1f}s")
            return video_file, False, f"Process failed with code {return_code}"
            
    except Exception as e:
        process_logger.error(f"💥 Exception processing {video_file.name}: {e}")
        if process:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
            except:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except:
                    pass
        return video_file, False, str(e)
    
    finally:
        if process:
            try:
                if process.poll() is None:
                    process_logger.warning(f"🧹 Force cleaning process for {video_file.name}")
                    process.kill()
                    process.wait(timeout=5)
            except:
                pass

def cleanup_and_copy_videos(project_base_dir, skip_existing=True):
    """
    Clean up video files in project subdirectories and copy them to project base directory.
    
    Args:
        project_base_dir (str or Path): Base project directory to scan subdirectories and copy files to
        skip_existing (bool): If True, skip copying files that already exist in destination
    """
    project_base_path = Path(project_base_dir)
    
    # Ensure project base directory exists
    if not project_base_path.exists():
        print(f"Error: Project directory '{project_base_path}' does not exist")
        return
    
    print(f"\n{'='*50}")
    print("VIDEO CLEANUP AND COPY OPERATION")
    print(f"{'='*50}")
    print(f"Scanning project directory: {project_base_path}")
    print(f"Will copy files to: {project_base_path}")
    print(f"Skip existing files: {skip_existing}")
    
    # Find all .mp4 files in project subdirectories (not in the base directory itself)
    mp4_files = []
    for subdir in project_base_path.iterdir():
        if subdir.is_dir():
            subdir_mp4s = list(subdir.glob('**/*.mp4'))
            mp4_files.extend(subdir_mp4s)
            if subdir_mp4s:
                print(f"Found {len(subdir_mp4s)} .mp4 files in {subdir.name}/")
    
    if not mp4_files:
        print("No .mp4 files found in project subdirectories")
        return
    
    print(f"Total found: {len(mp4_files)} .mp4 files")
    
    files_to_delete = []
    files_to_keep = []
    
    # Separate files based on '_trace' suffix
    for mp4_file in mp4_files:
        filename = mp4_file.stem  # filename without extension
        if '_trace' in filename:
            files_to_keep.append(mp4_file)
            print(f"✓ Keeping: {mp4_file.name}")
        else:
            files_to_delete.append(mp4_file)
            print(f"✗ Marked for deletion: {mp4_file.name}")
    
    # Delete files without '_trace'
    deleted_count = 0
    print(f"\nDeleting {len(files_to_delete)} files...")
    for file_to_delete in files_to_delete:
        try:
            file_to_delete.unlink()
            print(f"Deleted: {file_to_delete}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_to_delete}: {e}")
    
    print(f"Successfully deleted {deleted_count} files")
    
    # Copy remaining .mp4 files to project base directory
    # Re-scan for remaining .mp4 files after deletion
    remaining_mp4_files = []
    for subdir in project_base_path.iterdir():
        if subdir.is_dir():
            remaining_mp4_files.extend(list(subdir.glob('**/*.mp4')))
    
    copied_count = 0
    skipped_count = 0
    
    print(f"\nCopying {len(remaining_mp4_files)} files to project base directory...")
    for mp4_file in remaining_mp4_files:
        try:
            destination = project_base_path / mp4_file.name
            
            # Check if file already exists and skip if requested
            if destination.exists():
                if skip_existing:
                    print(f"⏭️  Skipping existing file: {mp4_file.name}")
                    skipped_count += 1
                    continue
                else:
                    # Create unique name if not skipping
                    counter = 1
                    stem = mp4_file.stem
                    suffix = mp4_file.suffix
                    while destination.exists():
                        new_name = f"{stem}_{counter}{suffix}"
                        destination = project_base_path / new_name
                        counter += 1
                    print(f"File exists, renamed to: {destination.name}")
            
            shutil.copy2(mp4_file, destination)
            print(f"Copied: {mp4_file.name} -> {destination}")
            copied_count += 1
            
        except Exception as e:
            print(f"Error copying {mp4_file}: {e}")
    
    print(f"\n{'='*50}")
    print(f"OPERATION COMPLETED!")
    print(f"Deleted: {deleted_count} files")
    print(f"Copied: {copied_count} files to project base directory")
    print(f"Skipped existing: {skipped_count} files")
    print(f"{'='*50}")


def post_process_cleanup(output_dir, video_file_name, skip_existing=True):
    """
    Run cleanup for a specific video's output directory
    
    Args:
        output_dir (Path): The base output directory 
        video_file_name (str): Name of the video file (without extension)
        skip_existing (bool): Whether to skip files that already exist in destination
    """
    video_output_dir = Path(output_dir) / video_file_name
    if video_output_dir.exists():
        print(f"Running cleanup for {video_file_name}...")
        cleanup_and_copy_videos(video_output_dir, skip_existing=skip_existing)
    parser = argparse.ArgumentParser(description='Batch process videos with memory leak prevention')
    parser.add_argument('--input_dir', required=True, help='Directory containing input videos')
    parser.add_argument('--output_dir', required=True, help='Directory for output results')
    parser.add_argument('--script_path', default='main.py', help='Path to your main processing script')
    parser.add_argument('--weights', required=True, help='Path to model weights file')
    parser.add_argument('--baseline', action='store_true', help='Use baseline processing')
    parser.add_argument('--norecognition', action='store_true', help='Disable recognition module')
    parser.add_argument('--kalmanfilter', action='store_true', help='Enable kalman filter')
    parser.add_argument('--target_gesture', default='five', help='Target gesture for recognition')
    parser.add_argument('--color-shift', action='store_true', help='Enable color shift preprocessing')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='Confidence threshold')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout per file in seconds')
    parser.add_argument('--max_memory_gb', type=float, default=8.0, help='Maximum memory per process (GB)')
    parser.add_argument('--extensions', nargs='+', default=['.mp4', '.avi'], help='Video file extensions to process')
    parser.add_argument('--resume', action='store_true', help='Resume from last processed file')
    parser.add_argument('--show-output', action='store_true', default=True, help='Show real-time output from main.py processes')
    parser.add_argument('--quiet', action='store_true', help='Suppress real-time output from main.py processes')
    parser.add_argument('--realtime', action='store_true', help='Show main.py output in real-time (may be less stable)')
    parser.add_argument('--max_workers', type=int, default=2, help='Maximum number of concurrent processes')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    script_path = Path(args.script_path)
    
    # Validate inputs
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Get video files
    video_files = get_video_files(input_dir, args.extensions)
    logger.info(f"Found {len(video_files)} video files to process")
    logger.info(f"Using {args.max_workers} concurrent workers")
    
    if not video_files:
        logger.warning("No video files found!")
        return
    
    # Build base arguments for the processing script
    base_args = ['--weights', str(args.weights)]
    base_args.extend(['--conf-thres', str(args.conf_thres)])
    
    if args.baseline:
        base_args.append('--baseline')
    if args.norecognition:
        base_args.append('--norecognition')
    if args.kalmanfilter:
        base_args.append('--kalmanfilter')
    if args.color_shift:
        base_args.append('--color-shift')
    
    base_args.extend(['--target_gesture', args.target_gesture])
    
    # Resume functionality
    processed_files_log = output_dir / 'processed_files.txt'
    processed_files = set()
    if args.resume and processed_files_log.exists():
        with open(processed_files_log, 'r') as f:
            processed_files = set(line.strip() for line in f)
        logger.info(f"Resuming: {len(processed_files)} files already processed")
    
    # Filter out already processed files
    if args.resume:
        video_files = [f for f in video_files if str(f) not in processed_files]
        logger.info(f"Remaining files to process: {len(video_files)}")
    
    # Process files with concurrent workers
    successful = 0
    failed = 0
    start_time = time.time()
    completed_count = len(processed_files) if args.resume else 0
    total_files = len(get_video_files(input_dir, args.extensions))
    
    logger.info(f"Starting batch processing with {args.max_workers} workers...")
    
    # Use ProcessPoolExecutor for concurrent processing
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_video = {}
        show_output = args.show_output and not args.quiet
        
        for video_file in video_files:
            future = executor.submit(
                process_single_file, 
                video_file, 
                script_path, 
                base_args, 
                output_dir, 
                args.timeout,
                args.max_memory_gb,
                show_output,
                args.realtime  # Add realtime parameter
            )
            future_to_video[future] = video_file
            logger.info(f"📝 Queued: {video_file.name}")
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_video):
            try:
                video_file, success, message = future.result(timeout=args.timeout + 60)
                completed_count += 1
                
                if success:
                    successful += 1
                    # Log successful processing
                    with open(processed_files_log, 'a') as f:
                        f.write(f"{video_file}\n")
                    logger.info(f"✅ SUCCESS: {video_file.name} - {message}")
                else:
                    failed += 1
                    logger.error(f"❌ FAILED: {video_file.name} - {message}")
                
                # Progress update
                elapsed = time.time() - start_time
                remaining = total_files - completed_count
                active_workers = len([f for f in future_to_video if not f.done()])
                
                if completed_count > 0:
                    avg_time = elapsed / (completed_count - (len(processed_files) if args.resume else 0))
                    eta = remaining * avg_time if remaining > 0 else 0
                    
                    logger.info(f"📊 Progress: {completed_count}/{total_files} | "
                               f"✅ Success: {successful} | ❌ Failed: {failed} | "
                               f"⏱️  ETA: {eta/60:.1f}min | "
                               f"🔄 Active: {active_workers}")
                
                # Small delay between job completions for system stability
                time.sleep(0.1)
                
            except Exception as e:
                video_file = future_to_video.get(future, "unknown")
                failed += 1
                completed_count += 1
                logger.error(f"💥 EXCEPTION: {video_file} - {str(e)}")
    
    logger.info("🏁 All workers completed, finalizing...")
    time.sleep(2)  # Allow final cleanup
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH PROCESSING COMPLETED!")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Processed this run: {len(video_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    if len(video_files) > 0:
        logger.info(f"Average time per file: {total_time/len(video_files):.1f} seconds")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()