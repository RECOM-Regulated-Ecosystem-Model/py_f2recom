"""
Command-line interface for py_f2recom.
"""
import argparse
import sys

def main():
    """Main entry point for the py_f2recom CLI."""
    parser = argparse.ArgumentParser(
        description="FESOM2-REcoM2 model analysis tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add subcommands here
    plot_parser = subparsers.add_parser('plot', help='Plot model output')
    plot_parser.add_argument('input', help='Input NetCDF file')
    plot_parser.add_argument('-v', '--variable', help='Variable to plot')
    plot_parser.add_argument('-o', '--output', help='Output file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'plot':
        print(f"Plotting {args.variable} from {args.input}")
        # TODO: Implement plotting logic
        print("Plotting not yet implemented")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
