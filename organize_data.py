#!/usr/bin/env python3
"""
Organize raw futures data files by commodity type.
Creates organized_data/ structure and data_inventory.csv
"""
import os
import shutil
import csv
from pathlib import Path
from collections import defaultdict

# Commodity mappings - symbol to category and name
COMMODITY_MAP = {
    # Metals
    'HG': ('copper', 'Copper'),
    'GC': ('gold', 'Gold'),
    'SI': ('silver', 'Silver'),
    'PL': ('platinum', 'Platinum'),
    'PA': ('palladium', 'Palladium'),
    'ALI': ('aluminum', 'Aluminum'),
    
    # Energy
    'CL': ('crude_oil', 'Crude Oil WTI'),
    'BZ': ('brent_crude', 'Brent Crude'),
    'NG': ('natural_gas', 'Natural Gas'),
    'HO': ('heating_oil', 'Heating Oil'),
    'RB': ('gasoline', 'RBOB Gasoline'),
    
    # Grains
    'ZC': ('corn', 'Corn'),
    'ZW': ('wheat', 'Wheat'),
    'ZS': ('soybeans', 'Soybeans'),
    'ZL': ('soybean_oil', 'Soybean Oil'),
    'ZM': ('soybean_meal', 'Soybean Meal'),
    'ZO': ('oats', 'Oats'),
    'ZR': ('rice', 'Rice'),
    
    # Softs
    'SB': ('sugar', 'Sugar'),
    'KC': ('coffee', 'Coffee'),
    'CC': ('cocoa', 'Cocoa'),
    'CT': ('cotton', 'Cotton'),
    'OJ': ('orange_juice', 'Orange Juice'),
    
    # Meats
    'LE': ('live_cattle', 'Live Cattle'),
    'HE': ('lean_hogs', 'Lean Hogs'),
    'GF': ('feeder_cattle', 'Feeder Cattle'),
    
    # Currencies
    'AD': ('currencies', 'Australian Dollar'),
    'A6': ('currencies', 'Australian Dollar (micro)'),
    'DX': ('currencies', 'US Dollar Index'),
    'CNH': ('currencies', 'Offshore RMB'),
    'JY': ('currencies', 'Japanese Yen'),
    'PJY': ('currencies', 'Japanese Yen (mini)'),
    'NOK': ('currencies', 'Norwegian Krone'),
    'SEK': ('currencies', 'Swedish Krona'),
    'KRW': ('currencies', 'Korean Won'),
    'PRK': ('currencies', 'Polish Zloty'),
    'TWN': ('currencies', 'Taiwan Dollar'),
    
    # Equity Indices
    'ES': ('equity_indices', 'E-mini S&P 500'),
    'MES': ('equity_indices', 'Micro E-mini S&P 500'),
    'NQ': ('equity_indices', 'E-mini NASDAQ-100'),
    'MNQ': ('equity_indices', 'Micro E-mini NASDAQ-100'),
    'YM': ('equity_indices', 'E-mini Dow'),
    'RTY': ('equity_indices', 'E-mini Russell 2000'),
    'NK': ('equity_indices', 'Nikkei 225 (USD)'),
    'NKD': ('equity_indices', 'Nikkei 225 (Yen)'),
    'NIY': ('equity_indices', 'Nikkei 225 Yen'),
    
    # Fixed Income
    'ZN': ('fixed_income', '10-Year T-Note'),
    'ZB': ('fixed_income', '30-Year T-Bond'),
    'ZF': ('fixed_income', '5-Year T-Note'),
    'ZT': ('fixed_income', '2-Year T-Note'),
    'ZQ': ('fixed_income', '30-Day Fed Funds'),
    'UB': ('fixed_income', 'Ultra T-Bond'),
    'TN': ('fixed_income', 'Ultra 10-Year'),
    
    # Crypto
    'BTC': ('crypto', 'Bitcoin'),
    'MBT': ('crypto', 'Micro Bitcoin'),
    'ETH': ('crypto', 'Ethereum'),
    'MET': ('crypto', 'Micro Ether'),
    
    # Volatility
    'VX': ('volatility', 'VIX Futures'),
    'VXM': ('volatility', 'Mini VIX'),
}

def get_commodity_info(filename):
    """Extract commodity symbol from filename and return category info."""
    # Pattern: SYMBOL_MONTHYEAR_1min.txt
    parts = filename.split('_')
    if len(parts) >= 2:
        symbol = parts[0]
        if symbol in COMMODITY_MAP:
            folder, name = COMMODITY_MAP[symbol]
            return symbol, folder, name
    return None, 'other', 'Unknown'

def main():
    script_dir = Path(__file__).parent
    organized_dir = script_dir / 'organized_data'
    
    # Create organized_data structure
    print("Creating organized_data structure...")
    organized_dir.mkdir(exist_ok=True)
    
    # Scan all .txt files
    txt_files = list(script_dir.glob('*.txt'))
    print(f"Found {len(txt_files)} .txt files")
    
    # Group files by category
    files_by_category = defaultdict(list)
    inventory = []
    
    for txt_file in txt_files:
        symbol, folder, name = get_commodity_info(txt_file.name)
        files_by_category[folder].append((txt_file, symbol, name))
        
        # Add to inventory
        size_mb = txt_file.stat().st_size / (1024 * 1024)
        inventory.append({
            'filename': txt_file.name,
            'symbol': symbol or 'UNKNOWN',
            'commodity': name,
            'category': folder,
            'size_mb': f'{size_mb:.2f}'
        })
    
    # Create folders and move files
    print("\nOrganizing files by commodity...")
    moved_count = 0
    
    for folder, files_list in sorted(files_by_category.items()):
        folder_path = organized_dir / folder
        folder_path.mkdir(exist_ok=True)
        
        print(f"\n{folder}/: {len(files_list)} files")
        
        for src_file, symbol, name in files_list:
            dest_file = folder_path / src_file.name
            if not dest_file.exists():
                shutil.move(str(src_file), str(dest_file))
                moved_count += 1
        
        print(f"  Moved {len(files_list)} files to {folder}/")
    
    # Create inventory CSV
    print("\nCreating data inventory...")
    inventory_file = script_dir / 'data_inventory.csv'
    
    with open(inventory_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'symbol', 'commodity', 'category', 'size_mb'])
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda x: (x['category'], x['symbol'], x['filename'])))
    
    print(f"\n✓ Organized {moved_count} files into {len(files_by_category)} categories")
    print(f"✓ Created inventory: {inventory_file}")
    print(f"\nStructure:")
    print(f"  organized_data/")
    for folder in sorted(files_by_category.keys()):
        count = len(files_by_category[folder])
        print(f"    {folder}/  ({count} files)")

if __name__ == '__main__':
    main()

