"""
Ferro Alloy Plant Optimization System
Main entry point for the application
"""
import os
import argparse
from datetime import datetime

from modules.data_loader import DataLoader
from modules.baseline_assessment import BaselineAssessment
from modules.energy_analysis import EnergyAnalysis
from modules.material_analysis import MaterialAnalysis
from modules.furnace_optimization import FurnaceOptimization
from modules.electrode_optimization import ElectrodeOptimization
from modules.process_integration import ProcessIntegration
from modules.what_if_engine import WhatIfEngine
from modules.visualization import DashboardGenerator


def main():
    parser = argparse.ArgumentParser(description='Ferro Alloy Plant Optimization System')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory with input CSV files')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Configuration file')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'baseline', 'energy', 'material', 'furnace', 'electrode', 'process', 'what_if'],
                        help='Analysis mode to run')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting Ferro Alloy Plant Optimization - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Loading data from {args.data_dir}")
    
    # Initialize data loader
    data_loader = DataLoader(data_dir=args.data_dir, config_path=args.config)
    datasets = data_loader.load_all_datasets()
    
    # Run specified analysis mode
    if args.mode in ['full', 'baseline']:
        baseline = BaselineAssessment(datasets)
        baseline_results = baseline.run_assessment()
        baseline.generate_reports(args.output_dir)
        
    if args.mode in ['full', 'energy']:
        energy = EnergyAnalysis(datasets)
        energy_results = energy.analyze()
        energy.generate_reports(args.output_dir)
        
    if args.mode in ['full', 'material']:
        material = MaterialAnalysis(datasets)
        material_results = material.analyze()
        material.generate_reports(args.output_dir)
        
    if args.mode in ['full', 'furnace']:
        furnace = FurnaceOptimization(datasets)
        furnace_results = furnace.optimize()
        furnace.generate_reports(args.output_dir)
        
    if args.mode in ['full', 'electrode']:
        electrode = ElectrodeOptimization(datasets)
        electrode_results = electrode.optimize()
        electrode.generate_reports(args.output_dir)
        
    if args.mode in ['full', 'process']:
        process = ProcessIntegration(datasets)
        process_results = process.optimize()
        process.generate_reports(args.output_dir)
        
    if args.mode in ['full', 'what_if']:
        what_if = WhatIfEngine(datasets)
        scenarios = what_if.generate_scenarios()
        what_if.run_scenarios(scenarios)
        what_if.generate_reports(args.output_dir)
        
    # Generate comprehensive dashboard if running full analysis
    if args.mode == 'full':
        dashboard = DashboardGenerator(
            baseline_results=baseline_results,
            energy_results=energy_results,
            material_results=material_results,
            furnace_results=furnace_results,
            electrode_results=electrode_results,
            process_results=process_results,
            what_if_scenarios=scenarios
        )
        dashboard.generate_dashboard(args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
