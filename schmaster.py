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

def process_single_file(video_file, script_path, base_args, output_dir, timeout_seconds=3600, max_memory_gb=8):
    """Process a single video file in a separate process"""
    process_logger = logging.getLogger(f"worker-{os.getpid()}")
    process_logger.info(f"Processing: {video_file}")
    
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
    
    try:
        # Start process
        process_logger.info(f"Starting command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor process with timeout and memory checking
        while process.poll() is None:
            elapsed = time.time() - start_time
            
            # Check timeout
            if elapsed > timeout_seconds:
                process_logger.error(f"Process timeout ({timeout_seconds}s) reached, terminating...")
                process.terminate()
                process.wait(timeout=10)
                return video_file, False, f"Timeout after {timeout_seconds}s"
            
            # Check memory usage
            if monitor_process_memory(process, max_memory_gb):
                process.wait(timeout=10)
                return video_file, False, "Memory limit exceeded"
            
            time.sleep(5)  # Check every 5 seconds
        
        # Get process output
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            process_logger.info(f"Successfully processed {video_file} in {elapsed_time:.1f}s")
            return video_file, True, f"Success in {elapsed_time:.1f}s"
        else:
            process_logger.error(f"Process failed with return code {return_code}")
            if stderr:
                process_logger.error(f"STDERR: {stderr}")
            return video_file, False, f"Process failed with code {return_code}"
            
    except Exception as e:
        process_logger.error(f"Error processing {video_file}: {e}")
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        return video_file, False, str(e)

def main():
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
        for video_file in video_files:
            future = executor.submit(
                process_single_file, 
                video_file, 
                script_path, 
                base_args, 
                output_dir, 
                args.timeout,
                args.max_memory_gb
            )
            future_to_video[future] = video_file
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_video):
            video_file, success, message = future.result()
            completed_count += 1
            
            if success:
                successful += 1
                # Log successful processing
                with open(processed_files_log, 'a') as f:
                    f.write(f"{video_file}\n")
                logger.info(f"✓ Completed {video_file.name}: {message}")
            else:
                failed += 1
                logger.error(f"✗ Failed {video_file.name}: {message}")
            
            # Progress update
            elapsed = time.time() - start_time
            remaining = total_files - completed_count
            if completed_count > 0:
                avg_time = elapsed / (completed_count - (len(processed_files) if args.resume else 0))
                eta = remaining * avg_time
                
                logger.info(f"Progress: {completed_count}/{total_files} | "
                           f"Success: {successful} | Failed: {failed} | "
                           f"ETA: {eta/60:.1f}min | "
                           f"Active workers: {len([f for f in future_to_video if not f.done()])}")
    
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