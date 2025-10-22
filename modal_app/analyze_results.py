#!/usr/bin/env python3
"""
Script to analyze existing results and create dataset-level summaries.
Usage: python analyze_results.py path/to/results.csv
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def analyze_results_by_dataset(results_file):
    """
    Analyze results file and create summaries by dataset.
    
    Args:
        results_file: Path to the CSV file containing full results
    """
    try:
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} rows from {results_file}")
        print(f"Columns: {list(df.columns)}")
        
        if 'dataset' not in df.columns:
            print("Warning: 'dataset' column not found. Available columns:", df.columns.tolist())
            return
            
        print(f"\nDatasets found: {df['dataset'].unique()}")
        print(f"Number of unique datasets: {df['dataset'].nunique()}")
        
        # Group by dataset and calculate aggregated metrics
        dataset_summary = df.groupby('dataset').agg({
            'num_samples': 'sum',
            'total_time': 'sum',
            'total_runtime': 'first',  # Should be the same for all
            'job_id': 'first',
            'wer': lambda x: (x * df.loc[x.index, 'num_samples']).sum() / df.loc[x.index, 'num_samples'].sum(),  # Weighted average WER
            'total_audio_length': 'sum'
        }).reset_index()
        
        # Recalculate RTFx for each dataset
        dataset_summary['rtfx'] = dataset_summary['total_audio_length'] / dataset_summary['total_time']
        
        print("\n" + "="*80)
        print("RESULTS BY DATASET")
        print("="*80)
        
        for _, row in dataset_summary.iterrows():
            print(f"\nDataset: {row['dataset']}")
            print(f"  Samples: {row['num_samples']}")
            print(f"  Total Audio Length: {row['total_audio_length']:.2f}s ({row['total_audio_length']/60:.1f}min)")
            print(f"  Total Processing Time: {row['total_time']:.2f}s")
            print(f"  WER: {row['wer']:.2f}%")
            print(f"  RTFx: {row['rtfx']:.2f}")
        
        # Save to file
        output_file = Path(results_file).parent / f"results_by_dataset_{Path(results_file).stem}.csv"
        dataset_summary.to_csv(output_file, index=False)
        print(f"\nDataset summary saved to: {output_file}")
        
        return dataset_summary
        
    except Exception as e:
        print(f"Error processing {results_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze ASR evaluation results by dataset')
    parser.add_argument('results_file', help='Path to the CSV file containing results')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: File {args.results_file} not found")
        sys.exit(1)
    
    analyze_results_by_dataset(args.results_file)

if __name__ == "__main__":
    main()