import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import argparse
import sys

def analyze_weather_data(csv_file, output_format='text', min_days=None):
    """
    Analyze weather data from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        output_format (str): Output format ('text' or 'json')
        min_days (int): Minimum number of days required for analysis
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Initialize results dictionary for JSON output
        results = {
            'temporal_analysis': {},
            'spatial_analysis': {},
            'data_quality': {},
            'weather_patterns': {},
            'statistical_summary': {},
            'ml_suitability': {},
            'distribution_analysis': {}
        }
        
        # 1. Temporal Analysis
        date_range = pd.to_datetime(df['date'])
        
        # Calculate unique days per location
        location_days = df.groupby(['latitude', 'longitude'])['date'].nunique().describe()
        
        # Calculate unique days per city
        city_days = df.groupby('city')['date'].nunique().describe()
        
        # Get overall date range
        temporal_data = {
            'date_range': {
                'start': date_range.min().strftime('%Y-%m-%d'),
                'end': date_range.max().strftime('%Y-%m-%d')
            },
            'days_per_location': {
                'min': int(location_days['min']),
                'max': int(location_days['max']),
                'mean': round(location_days['mean'], 2),
                'median': int(location_days['50%'])
            },
            'days_per_city': {
                'min': int(city_days['min']),
                'max': int(city_days['max']),
                'mean': round(city_days['mean'], 2),
                'median': int(city_days['50%'])
            }
        }
        
        # For minimum days check, use the minimum days per location
        if min_days and temporal_data['days_per_location']['min'] < min_days:
            print(f"Error: Some locations have fewer than {min_days} days of data")
            print(f"Minimum days per location: {temporal_data['days_per_location']['min']}")
            return None
        results['temporal_analysis'] = temporal_data
        
        # 2. Spatial Coverage Analysis
        cities = df['city'].unique()
        city_locations = {}
        
        for city in cities:
            locations = df[df['city'] == city].groupby(['latitude', 'longitude']).size()
            if len(locations) > 1:
                city_locations[city] = len(locations)
                
        spatial_data = {
            'total_cities': len(cities),
            'cities_with_multiple_points': city_locations
        }
        results['spatial_analysis'] = spatial_data
        
        # 3. Data Quality Analysis
        missing_values = df.isnull().sum().to_dict()
        temp_inconsistencies = df[
            (df['temp_min'] > df['temp']) | 
            (df['temp_max'] < df['temp']) |
            (df['temp_min'] > df['temp_max'])
        ]
        
        quality_data = {
            'missing_values': missing_values,
            'temperature_inconsistencies': len(temp_inconsistencies)
        }
        results['data_quality'] = quality_data
        
        # 4. Weather Patterns
        weather_patterns = df['description'].value_counts().to_dict()
        results['weather_patterns'] = weather_patterns
        
        # 5. Statistical Summary
        numeric_columns = ['temp', 'pressure', 'humidity', 'wind_speed']
        stats_summary = df[numeric_columns].describe().round(2).to_dict()
        results['statistical_summary'] = stats_summary
        
        # 6. ML Suitability Assessment
        ml_warnings = []
        min_location_days = temporal_data['days_per_location']['min']
        
        if min_location_days < 365:
            ml_warnings.append("Some locations have less than 1 year of data - limited seasonal patterns")
        if min_location_days < 30:
            ml_warnings.append("Some locations have less than 30 days of data - not suitable for monthly patterns")
        if min_location_days < 7:
            ml_warnings.append("Some locations have less than 7 days of data - only suitable for very short-term predictions")
            
        results['ml_suitability'] = {
            'warnings': ml_warnings,
            'recommended_for_ml': min_location_days >= 30,
            'min_days_per_location': min_location_days
        }
        
        # 7. Distribution Analysis
        # Calculate quartiles and IQR for temperature
        temp_stats = {
            'Q1': df['temp'].quantile(0.25),
            'Q3': df['temp'].quantile(0.75),
            'median': df['temp'].quantile(0.5)
        }
        temp_stats['IQR'] = temp_stats['Q3'] - temp_stats['Q1']
        
        # Calculate outlier bounds
        temp_stats['lower_bound'] = temp_stats['Q1'] - 1.5 * temp_stats['IQR']
        temp_stats['upper_bound'] = temp_stats['Q3'] + 1.5 * temp_stats['IQR']
        
        # Identify outliers
        temp_outliers = df[
            (df['temp'] < temp_stats['lower_bound']) | 
            (df['temp'] > temp_stats['upper_bound'])
        ].copy()
        
        # Add outlier classification
        temp_outliers['outlier_type'] = np.where(
            temp_outliers['temp'] < temp_stats['lower_bound'],
            'low',
            'high'
        )
        
        # Calculate deviation from bounds for outliers
        temp_outliers['deviation'] = np.where(
            temp_outliers['outlier_type'] == 'low',
            temp_stats['lower_bound'] - temp_outliers['temp'],
            temp_outliers['temp'] - temp_stats['upper_bound']
        )
        
        # Prepare outlier details
        outlier_details = []
        for _, row in temp_outliers.iterrows():
            outlier_details.append({
                'city': row['city'],
                'date': row['date'],
                'temperature': row['temp'],
                'type': row['outlier_type'],
                'deviation': round(row['deviation'], 2)
            })
        
        distribution_data = {
            'temperature_stats': {
                'Q1': round(temp_stats['Q1'], 2),
                'median': round(temp_stats['median'], 2),
                'Q3': round(temp_stats['Q3'], 2),
                'IQR': round(temp_stats['IQR'], 2),
                'lower_bound': round(temp_stats['lower_bound'], 2),
                'upper_bound': round(temp_stats['upper_bound'], 2)
            },
            'outliers': {
                'count': len(temp_outliers),
                'details': outlier_details
            }
        }
        results['distribution_analysis'] = distribution_data
        
        if output_format == 'json':
            return results
        else:
            # Print text format report
            print("\n=== Weather Data Analysis Report ===")
            print(f"Total records analyzed: {len(df)}")
            
            print(f"\n1. Temporal Coverage:")
            print(f"   Date range: {temporal_data['date_range']['start']} to {temporal_data['date_range']['end']}")
            print("\n   Days per location:")
            print(f"   - Minimum: {temporal_data['days_per_location']['min']}")
            print(f"   - Maximum: {temporal_data['days_per_location']['max']}")
            print(f"   - Mean: {temporal_data['days_per_location']['mean']}")
            print(f"   - Median: {temporal_data['days_per_location']['median']}")
            print("\n   Days per city:")
            print(f"   - Minimum: {temporal_data['days_per_city']['min']}")
            print(f"   - Maximum: {temporal_data['days_per_city']['max']}")
            print(f"   - Mean: {temporal_data['days_per_city']['mean']}")
            print(f"   - Median: {temporal_data['days_per_city']['median']}")
            
            print("\n2. Spatial Coverage Analysis:")
            print(f"   - Total number of cities: {len(cities)}")
            for city, locations in city_locations.items():
                print(f"   - {city}: {locations} locations")
            
            print("\n3. Data Quality Analysis:")
            if any(missing_values.values()):
                print("   Missing values detected:")
                for col, count in missing_values.items():
                    if count > 0:
                        print(f"   - {col}: {count}")
            else:
                print("   No missing values found")
            
            print("\n4. Weather Patterns Analysis:")
            for weather, count in weather_patterns.items():
                print(f"   - {weather}: {count}")
            
            print("\n5. Statistical Summary:")
            for col in numeric_columns:
                print(f"\n   {col}:")
                for stat, value in stats_summary[col].items():
                    print(f"   - {stat}: {value}")
            
            print("\n6. Machine Learning Suitability Assessment:")
            for warning in ml_warnings:
                print(f"   WARNING: {warning}")
            
            print("\n7. Data Distribution Analysis:")
            print("\n   Temperature Distribution Statistics:")
            print(f"   - First Quartile (Q1): {distribution_data['temperature_stats']['Q1']}°K")
            print(f"   - Median: {distribution_data['temperature_stats']['median']}°K")
            print(f"   - Third Quartile (Q3): {distribution_data['temperature_stats']['Q3']}°K")
            print(f"   - Interquartile Range (IQR): {distribution_data['temperature_stats']['IQR']}°K")
            print(f"   - Outlier Bounds: [{distribution_data['temperature_stats']['lower_bound']}°K, "
                  f"{distribution_data['temperature_stats']['upper_bound']}°K]")
            
            if distribution_data['outliers']['count'] > 0:
                print(f"\n   Found {distribution_data['outliers']['count']} temperature outliers:")
                for outlier in distribution_data['outliers']['details']:
                    print(f"\n   Outlier in {outlier['city']} on {outlier['date']}:")
                    print(f"   - Temperature: {outlier['temperature']}°K")
                    print(f"   - Type: {outlier['type'].upper()} outlier")
                    print(f"   - Deviates from {'lower' if outlier['type'] == 'low' else 'upper'} "
                          f"bound by {outlier['deviation']}°K")
            else:
                print("\n   No temperature outliers detected.")
            
            return df
            
    except Exception as e:
        print(f"Error processing the CSV file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze weather data from a CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--output', choices=['text', 'json'], default='text',
                      help='Output format (text or json)')
    parser.add_argument('--min-days', type=int,
                      help='Minimum number of days required in the dataset')
    
    args = parser.parse_args()
    
    result = analyze_weather_data(
        args.csv_file,
        output_format=args.output,
        min_days=args.min_days
    )
    
    if args.output == 'json' and result is not None:
        import json
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()