import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from vision_pipeline import RealTimeVisionPipeline, PipelineConfig
from scripts.calibrate_camera import calibrate_from_live_camera, calibrate_from_images
from scripts.aruco_demo import generate_markers
from scripts.pose_estimation_demo import main as pose_demo_main
from scripts.visualization_demo import main as viz_demo_main
from scripts.distortion_correction_demo import main as distortion_main

def create_default_config() -> PipelineConfig:
    return PipelineConfig(camera_id=0, camera_width=1280, camera_height=720, camera_fps=30, aruco_dictionary='DICT_4X4_50', marker_size=50.0, pose_algorithm='SOLVEPNP_EPNP', use_ransac=True, temporal_filtering=True, filter_alpha=0.3, show_axes=True, show_cube=True, show_trajectory=True, show_info=True, save_frames=False, output_directory='output', save_trajectory=True)

def run_realtime_pipeline(config: PipelineConfig) -> None:
    print('=' * 60)
    print('6-DoF POSE ESTIMATION PIPELINE')
    print('=' * 60)
    print(f'Camera ID: {config.camera_id}')
    print(f'Resolution: {config.camera_width}x{config.camera_height}')
    print(f'Target FPS: {config.camera_fps}')
    print(f'ArUco Dictionary: {config.aruco_dictionary}')
    print(f'Marker Size: {config.marker_size}mm')
    print(f'Pose Algorithm: {config.pose_algorithm}')
    print(f"RANSAC: {('Enabled' if config.use_ransac else 'Disabled')}")
    print(f"Temporal Filtering: {('Enabled' if config.temporal_filtering else 'Disabled')}")
    print('=' * 60)
    pipeline = RealTimeVisionPipeline(config)
    try:
        pipeline.run_realtime()
    except KeyboardInterrupt:
        print('\nPipeline interrupted by user')
    except Exception as e:
        print(f'\nPipeline error: {e}')
    finally:
        pipeline.stop()

def run_benchmark(config: PipelineConfig, duration: float) -> None:
    print('=' * 60)
    print('PIPELINE BENCHMARK')
    print('=' * 60)
    print(f'Duration: {duration} seconds')
    print(f'Resolution: {config.camera_width}x{config.camera_height}')
    print('=' * 60)
    pipeline = RealTimeVisionPipeline(config)
    try:
        results = pipeline.run_benchmark(duration)
        import json
        results_file = Path(config.output_directory) / f'benchmark_{int(duration)}s.json'
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'\nBenchmark results saved to: {results_file}')
    except Exception as e:
        print(f'Benchmark error: {e}')

def run_calibration(mode: str, **kwargs) -> None:
    print('=' * 60)
    print('CAMERA CALIBRATION')
    print('=' * 60)
    if mode == 'live':
        camera_id = kwargs.get('camera', 0)
        pattern_width = kwargs.get('pattern_width', 9)
        pattern_height = kwargs.get('pattern_height', 6)
        square_size = kwargs.get('square_size', 25.0)
        calibrator = calibrate_from_live_camera(camera_id, (pattern_width, pattern_height), square_size)
    elif mode == 'images':
        image_path = kwargs.get('path', '')
        pattern_width = kwargs.get('pattern_width', 9)
        pattern_height = kwargs.get('pattern_height', 6)
        square_size = kwargs.get('square_size', 25.0)
        if not image_path:
            print('Error: --path required for image calibration')
            return
        calibrator = calibrate_from_images(image_path, (pattern_width, pattern_height), square_size)
    else:
        print(f'Unknown calibration mode: {mode}')
        return
    if calibrator:
        print('\nCalibration completed successfully!')
        calibrator.print_camera_parameters()

def run_marker_generation(marker_ids: list, output_dir: str, dictionary: str) -> None:
    print('=' * 60)
    print('ARUCO MARKER GENERATION')
    print('=' * 60)
    print(f'Dictionary: {dictionary}')
    print(f'Marker IDs: {marker_ids}')
    print(f'Output directory: {output_dir}')
    print('=' * 60)
    from aruco_tracking.aruco_detector import ArUcoDetector
    detector = ArUcoDetector(dictionary_name=dictionary)
    generate_markers(detector, marker_ids, output_dir)

def main():
    parser = argparse.ArgumentParser(description='6-DoF Pose Estimation System', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  # Calibrate camera live\n  python main.py --calibrate --mode live --camera 0\n  \n  # Run real-time pose estimation\n  python main.py --run --calibration camera_calibration.pkl\n  \n  # Run benchmark\n  python main.py --benchmark --calibration camera_calibration.pkl --duration 30\n  \n  # Generate markers\n  python main.py --generate-markers --ids 0 1 2 3 --output markers/\n  \n  # Run pose estimation demo\n  python main.py --demo-pose --calibration camera_calibration.pkl --synthetic\n  \n  # Run visualization demo\n  python main.py --demo-viz --calibration camera_calibration.pkl --camera 0\n        ')
    parser.add_argument('--run', action='store_true', help='Run real-time pose estimation pipeline')
    parser.add_argument('--benchmark', action='store_true', help='Run pipeline benchmark')
    parser.add_argument('--calibrate', action='store_true', help='Run camera calibration')
    parser.add_argument('--generate-markers', action='store_true', help='Generate ArUco markers')
    parser.add_argument('--demo-pose', action='store_true', help='Run pose estimation demo')
    parser.add_argument('--demo-viz', action='store_true', help='Run visualization demo')
    parser.add_argument('--demo-distortion', action='store_true', help='Run distortion correction demo')
    parser.add_argument('--calibration', type=str, help='Camera calibration file path')
    parser.add_argument('--config', type=str, help='Configuration file path (JSON)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--dictionary', type=str, default='DICT_4X4_50', choices=['DICT_4X4_50', 'DICT_4X4_100', 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_6X6_50', 'DICT_6X6_100', 'DICT_7X7_50', 'DICT_7X7_100'], help='ArUco dictionary (default: DICT_4X4_50)')
    parser.add_argument('--marker-size', type=float, default=50.0, help='Physical marker size in mm (default: 50.0)')
    parser.add_argument('--mode', choices=['live', 'images'], help='Calibration mode')
    parser.add_argument('--path', type=str, help='Path to calibration images')
    parser.add_argument('--pattern-width', type=int, default=9, help='Checkerboard pattern width (default: 9)')
    parser.add_argument('--pattern-height', type=int, default=6, help='Checkerboard pattern height (default: 6)')
    parser.add_argument('--square-size', type=float, default=25.0, help='Checkerboard square size in mm (default: 25.0)')
    parser.add_argument('--ids', nargs='+', type=int, help='Marker IDs to generate')
    parser.add_argument('--output', type=str, default='aruco_markers', help='Output directory for markers (default: aruco_markers)')
    parser.add_argument('--duration', type=float, default=30.0, help='Benchmark duration in seconds (default: 30.0)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--save-frames', action='store_true', help='Save processed frames')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for demos')
    parser.add_argument('--image', type=str, help='Input image for demos')
    args = parser.parse_args()
    if args.calibrate:
        run_calibration(mode=args.mode or 'live', camera=args.camera, path=args.path, pattern_width=args.pattern_width, pattern_height=args.pattern_height, square_size=args.square_size)
    elif args.generate_markers:
        if not args.ids:
            print('Error: --ids required for marker generation')
            return
        run_marker_generation(args.ids, args.output, args.dictionary)
    elif args.run:
        config = create_default_config()
        config.camera_id = args.camera
        config.camera_width = args.width
        config.camera_height = args.height
        config.camera_fps = args.fps
        config.aruco_dictionary = args.dictionary
        config.marker_size = args.marker_size
        config.output_directory = args.output_dir
        config.save_frames = args.save_frames
        if args.calibration:
            config.calibration_file = args.calibration
        elif Path('camera_calibration.pkl').exists():
            config.calibration_file = 'camera_calibration.pkl'
            print(f'Using existing calibration: camera_calibration.pkl')
        else:
            print('Warning: No calibration file specified. Using default parameters.')
            print('For best results, calibrate your camera first:')
            print('  python main.py --calibrate --mode live')
        run_realtime_pipeline(config)
    elif args.benchmark:
        config = create_default_config()
        config.camera_id = args.camera
        config.camera_width = args.width
        config.camera_height = args.height
        config.camera_fps = args.fps
        config.aruco_dictionary = args.dictionary
        config.marker_size = args.marker_size
        config.output_directory = args.output_dir
        if args.calibration:
            config.calibration_file = args.calibration
        elif Path('camera_calibration.pkl').exists():
            config.calibration_file = 'camera_calibration.pkl'
        else:
            print('Warning: No calibration file specified. Using default parameters.')
        run_benchmark(config, args.duration)
    elif args.demo_pose:
        if not args.calibration:
            print('Error: --calibration required for pose demo')
            return
        demo_args = ['--calibration', args.calibration, '--synthetic' if args.synthetic else '', '--image', args.image if args.image else '']
        demo_args = [arg for arg in demo_args if arg]
        original_argv = sys.argv
        sys.argv = ['pose_estimation_demo.py'] + demo_args
        try:
            pose_demo_main()
        finally:
            sys.argv = original_argv
    elif args.demo_viz:
        if not args.calibration:
            print('Error: --calibration required for visualization demo')
            return
        demo_args = ['--calibration', args.calibration, '--image', args.image if args.image else '', '--camera', str(args.camera) if not args.image else '', '--trajectory' if not args.image and (not args.camera) else '']
        demo_args = [arg for arg in demo_args if arg]
        original_argv = sys.argv
        sys.argv = ['visualization_demo.py'] + demo_args
        try:
            viz_demo_main()
        finally:
            sys.argv = original_argv
    elif args.demo_distortion:
        if not args.calibration:
            print('Error: --calibration required for distortion demo')
            return
        demo_args = ['--calibration', args.calibration, '--image', args.image if args.image else '', '--camera', str(args.camera) if not args.image else '']
        demo_args = [arg for arg in demo_args if arg]
        original_argv = sys.argv
        sys.argv = ['distortion_correction_demo.py'] + demo_args
        try:
            distortion_main()
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()
        print('\n' + '=' * 60)
        print('QUICK START GUIDE')
        print('=' * 60)
        print('1. Calibrate your camera:')
        print('   python main.py --calibrate --mode live')
        print()
        print('2. Run real-time pose estimation:')
        print('   python main.py --run --calibration camera_calibration.pkl')
        print()
        print('3. Generate ArUco markers:')
        print('   python main.py --generate-markers --ids 0 1 2 3')
        print()
        print('4. Run demos:')
        print('   python main.py --demo-pose --calibration camera_calibration.pkl --synthetic')
        print('   python main.py --demo-viz --calibration camera_calibration.pkl --camera 0')
        print()
        print('5. Run benchmark:')
        print('   python main.py --benchmark --calibration camera_calibration.pkl --duration 30')
        print('=' * 60)
if __name__ == '__main__':
    main()