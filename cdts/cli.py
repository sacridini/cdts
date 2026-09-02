import argparse
import sys

from .raster import run_landtrendr_image, run_ccdc_image

def run_landtrendr(args):
    try:
        run_landtrendr_image(
            input_path=args.input,
            output_dir=args.output_dir,
            start_year=args.start_year,
            max_segments=args.max_segments,
            chunk_size=args.chunk_size,
            n_jobs=args.jobs,
            save_vertices=args.save_vertices,
            event_type=args.event_type,
            sort_by=args.sort_by,
            min_mag=args.min_mag,
            min_dur=args.min_dur,
            pre_val_thresh=args.pre_val_thresh,
            prefix=args.prefix,
            output_scale_factor=args.output_scale
        )
    except Exception as e:
        print(f"Error running LandTrendr: {e}")
        sys.exit(1)

def run_ccdc_cli(args):
    try:
        # User needs to provide a list of dates. We'll read it from a text/csv file.
        # Alternatively, if not provided, we can simulate dates for testing (but throw warning)
        import os
        import numpy as np
        
        dates = []
        if args.dates_file and os.path.exists(args.dates_file):
            with open(args.dates_file, 'r') as f:
                dates = [int(line.strip()) for line in f if line.strip().isdigit()]
        else:
            print("WARNING: No --dates-file provided. Assuming 1 observation every 16 days (Landsat).")
            # We don't know the number of dates until we open the file, so we'll 
            # let process_file_ccdc handle it or we can hack it.
            # Actually, we should force it or read the TIF first.
            import rasterio
            with rasterio.open(args.input) as src:
                num_dates = src.count // args.num_bands
            dates = list(np.arange(1, 1 + num_dates * 16, 16))
            
        run_ccdc_image(
            input_path=args.input,
            output_dir=args.output_dir,
            dates=dates,
            num_bands=args.num_bands,
            qa_band_idx=args.qa_band,
            max_segments=args.max_segments,
            chunk_size=args.chunk_size,
            n_jobs=args.jobs,
            prefix=args.prefix,
            conseq_anom=6 if args.cold else 3
        )
    except Exception as e:
        print(f"Error running CCDC: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="cdts: Change Detection Python Library")
    subparsers = parser.add_subparsers(dest="command", help="Available algorithms")
    
    # LandTrendr Subparser
    lt_parser = subparsers.add_parser("landtrendr", help="Run LandTrendr algorithm")
    lt_parser.add_argument("input", help="Path to input multi-band GeoTIFF")
    lt_parser.add_argument("output_dir", help="Directory to save the outputs")
    lt_parser.add_argument("--start-year", type=int, default=2000, help="Year of the first band (default: 2000)")
    lt_parser.add_argument("--max-segments", type=int, default=6, help="Maximum number of segments (default: 6)")
    lt_parser.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores to use (-1 for all, default: -1)")
    lt_parser.add_argument("--save-vertices", action="store_true", help="Save the raw vertices stack")
    lt_parser.add_argument("--chunk-size", type=int, default=512, help="Size of the image chunks to process at once (default: 512)")
    
    # Event Extraction options
    lt_parser.add_argument("--event-type", choices=["loss", "gain"], default="loss", help="Event type to map (default: loss)")
    lt_parser.add_argument("--sort-by", choices=["greatest", "newest", "fastest", "longest"], default="greatest", help="How to select the event (default: greatest)")
    lt_parser.add_argument("--min-mag", type=float, default=0.0, help="Minimum magnitude filter")
    lt_parser.add_argument("--min-dur", type=int, default=1, help="Minimum duration filter")
    lt_parser.add_argument("--pre-val-thresh", type=float, default=0.0, help="Pre-value threshold filter")
    lt_parser.add_argument("--prefix", default="lt_event", help="Prefix for output metric files")
    lt_parser.add_argument("--output-scale", type=float, default=1.0, help="Scale factor to multiply output values (e.g. 0.0001 to convert back to float NDVI)")
    
    # CCDC Subparser
    ccdc_parser = subparsers.add_parser("ccdc", help="Run CCDC algorithm")
    ccdc_parser.add_argument("input", help="Path to input stacked multi-band GeoTIFF")
    ccdc_parser.add_argument("output_dir", help="Directory to save the outputs")
    ccdc_parser.add_argument("--num-bands", type=int, default=6, help="Number of bands per date in the stack (default: 6)")
    ccdc_parser.add_argument("--qa-band", type=int, default=-1, help="Index of QA band for cloud masking within the block (0-based, default: -1 for none)")
    ccdc_parser.add_argument("--dates-file", help="Path to text file containing Julian dates (one per line)")
    ccdc_parser.add_argument("--max-segments", type=int, default=6, help="Maximum number of segments (default: 6)")
    ccdc_parser.add_argument("--chunk-size", type=int, default=512, help="Size of the image chunks to process at once (default: 512)")
    ccdc_parser.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores to use (-1 for all, default: -1)")
    ccdc_parser.add_argument("--cold", action="store_true", help="Use COLD algorithm logic (6 consecutive anomalies instead of 3)")
    ccdc_parser.add_argument("--prefix", default="ccdc", help="Prefix for output files (default: ccdc)")
    
    args = parser.parse_args()
    
    if args.command == "landtrendr":
        run_landtrendr(args)
    elif args.command == "ccdc":
        run_ccdc_cli(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
