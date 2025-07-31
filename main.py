import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit { color: #00ff00; font-weight: bold; }
    .loss { color: #ff0000; font-weight: bold; }
    .neutral { color: #888888; }
    .last-update {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Crypto Trading Dashboard")

# Sidebar for configuration
st.sidebar.header("Configuration")
csv_url = st.sidebar.text_input(
    "Google Sheet CSV URL",
    value="https://docs.google.com/spreadsheets/d/e/2PACX-1vQgiqkaWzOnXJBIeNEzvUXaGPS0f3gHytC7A1wlohkFScEhVbururPv9amRuAop5ooqY_BJU23XKlL_/pub?output=csv",
    help="Paste the public CSV URL from your Google Sheet"
)

# Auto-refresh controls
col1, col2 = st.sidebar.columns(2)
with col1:
    auto_refresh = st.checkbox("Auto Refresh", value=True)
with col2:
    refresh_interval = st.selectbox("Interval (sec)", [5, 10, 15, 30, 60], index=1)

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False, help="Enable detailed debugging output")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()

@st.cache_data(ttl=5)
def load_sheet_data(url):
    """Load data from Google Sheet CSV URL"""
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=2)
def get_binance_price(symbol):
    """Fetch live price from Binance Futures API"""
    try:
        clean_symbol = symbol.replace('/', '').replace('-', '').upper()
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={clean_symbol}USDT"
        
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        
        data = response.json()
        price = float(data['price'])
        
        if debug_mode:
            st.write(f"💰 {symbol} Live Price: {price}")
        
        return price
    except Exception as e:
        if debug_mode:
            st.write(f"❌ Error fetching price for {symbol}: {str(e)}")
        return None

def parse_date_flexible(date_str):
    """Parse date with multiple format support - assumes recent dates"""
    if pd.isna(date_str) or date_str == '':
        return None
        
    # Clean the date string
    date_str = str(date_str).strip()
    
    # Try different date formats
    date_formats = [
        "%d/%m/%y",   # 01/07/25
        "%d/%m/%Y",   # 01/07/2025
        "%d-%m-%y",   # 01-07-25
        "%d-%m-%Y",   # 01-07-2025
        "%Y-%m-%d",   # 2025-07-01
        "%m/%d/%y",   # 07/01/25
        "%m/%d/%Y",   # 07/01/2025
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            
            # Smart year handling for 2-digit years
            if parsed_date.year < 100:
                current_year = datetime.now().year
                current_2digit = current_year % 100
                
                # If parsed year is greater than current 2-digit year, assume previous century
                # Otherwise, use current century
                if parsed_date.year > current_2digit:
                    parsed_date = parsed_date.replace(year=parsed_date.year + 1900)
                else:
                    parsed_date = parsed_date.replace(year=parsed_date.year + 2000)
            
            # Additional validation: don't allow future dates beyond today
            if parsed_date.date() > datetime.now().date():
                if debug_mode:
                    st.write(f"⚠️ Date {parsed_date.strftime('%Y-%m-%d')} is in the future, adjusting to previous year")
                parsed_date = parsed_date.replace(year=parsed_date.year - 1)
            
            return parsed_date
        except ValueError:
            continue
    
    if debug_mode:
        st.write(f"❌ Could not parse date: {date_str}")
    return None

@st.cache_data(ttl=60)
def fetch_1d_ohlc_to_today(symbol, start_date):
    """Fetch daily OHLC data from start_date to TODAY with better error handling"""
    try:
        clean_symbol = symbol.replace("/", "").replace("-", "").upper() + "USDT"
        url = "https://fapi.binance.com/fapi/v1/klines"
        
        # Parse start date
        parsed_start_date = parse_date_flexible(start_date)
        
        if parsed_start_date is None:
            if debug_mode:
                st.write(f"❌ Could not parse start date: {start_date}")
            return []
        
        # Get current date/time but adjust to UTC
        end_date = datetime.now()
        
        start_ts = int(parsed_start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        params = {
            "symbol": clean_symbol,
            "interval": "1d",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        
        if debug_mode:
            st.write(f"📡 API Request: {url}")
            st.write(f"📡 Parameters: {params}")
            st.write(f"📡 Fetching {clean_symbol} from {parsed_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Make the API request
        response = requests.get(url, params=params, timeout=15)
        
        if debug_mode:
            st.write(f"📡 Response Status: {response.status_code}")
        
        response.raise_for_status()
        candles = response.json()
        
        # Validate the response
        if not isinstance(candles, list):
            if debug_mode:
                st.write(f"❌ Invalid response format: {type(candles)}")
            return []
        
        if debug_mode:
            st.write(f"✅ Retrieved {len(candles)} candles for {clean_symbol}")
            if candles:
                first_candle_date = datetime.fromtimestamp(candles[0][0] / 1000).strftime('%Y-%m-%d')
                last_candle_date = datetime.fromtimestamp(candles[-1][0] / 1000).strftime('%Y-%m-%d')
                st.write(f"📅 Actual date range: {first_candle_date} to {last_candle_date}")
                
                # Show sample data structure
                st.write(f"📊 Sample candle data:")
                st.write(f"First candle: {candles[0]}")
        
        return candles
        
    except requests.exceptions.RequestException as e:
        if debug_mode:
            st.write(f"❌ Network error fetching OHLC for {symbol}: {str(e)}")
        return []
    except Exception as e:
        if debug_mode:
            st.write(f"❌ General error fetching OHLC for {symbol}: {str(e)}")
        return []

def check_entries_hit_sequentially(candles, entries, symbol=""):
    """
    Check entries hit sequentially - if a lower entry is hit, all higher entries are also hit
    """
    if not candles or not entries:
        return [False] * len(entries), []
    
    entries_hit = [False] * len(entries)
    hit_dates = [None] * len(entries)
    
    # Sort candles by timestamp
    sorted_candles = sorted(candles, key=lambda x: x[0])
    
    if debug_mode and symbol.upper() == "CRV":
        st.write(f"🔍 Sequential Entry Check for {symbol}:")
        st.write(f"Entries: {entries}")
    
    for i, candle in enumerate(sorted_candles):
        try:
            low_price = float(candle[3])
            candle_date = datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d')
            
            # Check each entry level
            for entry_idx, entry_price in enumerate(entries):
                entry_price = float(entry_price)
                
                # If this entry hasn't been hit yet and the low price hit it
                if not entries_hit[entry_idx] and low_price <= entry_price:
                    entries_hit[entry_idx] = True
                    hit_dates[entry_idx] = candle_date
                    
                    if debug_mode and symbol.upper() == "CRV":
                        st.write(f"✅ Entry {entry_idx + 1} ({entry_price}) HIT on {candle_date} (Low: {low_price:.4f})")
                    
                    # IMPORTANT: If a lower entry is hit, all higher entries must also be hit
                    # This handles the case where price gaps down below multiple entries at once
                    for higher_entry_idx in range(entry_idx):
                        if not entries_hit[higher_entry_idx]:
                            entries_hit[higher_entry_idx] = True
                            hit_dates[higher_entry_idx] = candle_date
                            
                            if debug_mode and symbol.upper() == "CRV":
                                st.write(f"✅ Entry {higher_entry_idx + 1} ({entries[higher_entry_idx]}) also HIT (price gapped down)")
            
        except (ValueError, TypeError, IndexError) as e:
            if debug_mode:
                st.write(f"Error processing candle: {e}")
            continue
    
    if debug_mode and symbol.upper() == "CRV":
        st.write(f"Final Results: {entries_hit}")
        st.write(f"Hit Dates: {hit_dates}")
    
    return entries_hit, hit_dates

def safe_float(value):
    """Safely convert value to float, return None if not possible"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return None
        return float(value)
    except:
        return None

def calculate_metrics(row, live_price, symbol):
    """Calculate all trading metrics for a row using historical candle data from start date to today"""
    try:
        # Extract entry prices in order
        entries = []
        entry_columns = ['Entry 1', 'Entry 2', 'Entry 3']
        
        # Try alternative column names if primary ones don't exist
        if not any(col in row.index for col in entry_columns):
            entry_columns = ['1st entry', '2nd entry', '3rd entry']
        
        for col in entry_columns:
            if col in row.index:
                entry = safe_float(row[col])
                if entry is not None and entry > 0:
                    entries.append(entry)
                else:
                    entries.append(None)
            else:
                entries.append(None)
        
        # Remove trailing None values but keep the order for first entries
        while entries and entries[-1] is None:
            entries.pop()
        
        # Get start date for historical analysis
        start_date = None
        date_columns = ['Date of given', 'Date', 'Start Date', 'Given Date']
        for col in date_columns:
            if col in row.index and pd.notna(row[col]):
                start_date = row[col]
                break
        
        if not entries or live_price is None:
            return {
                'entry_hit': False,
                'entries_hit_status': "–",
                'avg_entry': None,
                'pl': None,
                'entry_down_pct': None,
                'roi_pct': None
            }
        
        # Filter out None entries for processing
        valid_entries = [e for e in entries if e is not None]
        
        if not valid_entries:
            return {
                'entry_hit': False,
                'entries_hit_status': "–",
                'avg_entry': None,
                'pl': None,
                'entry_down_pct': None,
                'roi_pct': None
            }
        
        # 🔥 DEBUG CODE FOR CRV 🔥
        if symbol.upper() == "CRV" and debug_mode:
            st.write(f"\n🔍 DETAILED CRV DEBUG:")
            st.write(f"Raw entries from sheet: {[row.get('Entry 1'), row.get('Entry 2'), row.get('Entry 3')]}")
            st.write(f"Valid entries: {valid_entries}")
            st.write(f"Entry types: {[type(e) for e in valid_entries]}")
            st.write(f"Start date: {start_date}")
            st.write(f"Live price: {live_price}")
        
        # Get historical data from start date to TODAY and check entries hit sequentially
        entries_hit_flags = [False] * len(valid_entries)
        entries_hit_status = "–"
        hit_dates = []
        
        if start_date:
            candles = fetch_1d_ohlc_to_today(symbol, start_date)
            if candles:
                # 🔥 MORE DEBUG CODE FOR CRV 🔥
                if symbol.upper() == "CRV" and debug_mode:
                    st.write(f"Number of candles retrieved: {len(candles)}")
                    st.write(f"Sample candle lows from July 2025:")
                    for i, candle in enumerate(candles):
                        low = float(candle[3])
                        high = float(candle[2])
                        close = float(candle[4])
                        date = datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d')
                        st.write(f"  {date}: Low = {low:.4f}, High = {high:.4f}, Close = {close:.4f}")
                        
                        # Manual check for each entry
                        for j, entry in enumerate(valid_entries):
                            entry_float = float(entry)
                            if low <= entry_float:
                                st.write(f"    ✅ SHOULD HIT Entry {j+1} ({entry_float}) on {date} (Low: {low:.4f})")
                
                entries_hit_flags, hit_dates = check_entries_hit_sequentially(candles, valid_entries, symbol)
                
                # Additional debug for CRV
                if symbol.upper() == "CRV" and debug_mode:
                    st.write(f"After sequential check:")
                    st.write(f"  Entries hit flags: {entries_hit_flags}")
                    st.write(f"  Hit dates: {hit_dates}")
                
                # Create status string showing which entries were hit with dates
                hit_count = sum(entries_hit_flags)
                if hit_count == 0:
                    entries_hit_status = "No entries hit"
                else:
                    hit_entries = []
                    for i, (hit, date) in enumerate(zip(entries_hit_flags, hit_dates)):
                        if hit and date:
                            hit_entries.append(f"Entry {i+1} ({date})")
                    entries_hit_status = " → ".join(hit_entries)
            else:
                entries_hit_status = "No candle data"
        else:
            entries_hit_status = "No start date provided"
        
        # Calculate average of hit entries, or all entries if none hit yet
        hit_entries = [entry for entry, hit in zip(valid_entries, entries_hit_flags) if hit]
        
        if hit_entries:
            avg_entry = sum(hit_entries) / len(hit_entries)
        else:
            # If no entries hit yet, use all valid entries for potential calculations
            avg_entry = sum(valid_entries) / len(valid_entries)
        
        # Get quantity
        quantity = safe_float(row.get('Quantity', 1)) or 1
        
        # Calculate P/L based on average entry
        pl = (live_price - avg_entry) * quantity if avg_entry else None
        
        # Calculate percentage down from average entry
        entry_down_pct = ((live_price - avg_entry) / avg_entry) * 100 if avg_entry else None
        
        # Calculate ROI percentage
        roi_pct = (pl / (avg_entry * quantity)) * 100 if avg_entry and pl is not None else None
        
        return {
            'entry_hit': any(entries_hit_flags),
            'entries_hit_status': entries_hit_status,
            'avg_entry': avg_entry,
            'pl': pl,
            'entry_down_pct': entry_down_pct,
            'roi_pct': roi_pct
        }
    
    except Exception as e:
        if debug_mode:
            st.write(f"❌ Error in calculate_metrics for {symbol}: {str(e)}")
        return {
            'entry_hit': False,
            'entries_hit_status': f"Error: {str(e)[:50]}",
            'avg_entry': None,
            'pl': None,
            'entry_down_pct': None,
            'roi_pct': None
        }

def format_currency(value):
    """Format currency values"""
    if value is None or pd.isna(value):
        return "–"
    return f"${value:,.2f}"

def format_percentage(value):
    """Format percentage values with colors"""
    if value is None or pd.isna(value):
        return "–"
    
    color_class = "profit" if value > 0 else "loss" if value < 0 else "neutral"
    sign = "+" if value > 0 else ""
    return f'<span class="{color_class}">{sign}{value:.2f}%</span>'

def format_pl(value):
    """Format P/L values with colors"""
    if value is None or pd.isna(value):
        return "–"
    
    color_class = "profit" if value > 0 else "loss" if value < 0 else "neutral"
    sign = "+" if value > 0 else ""
    return f'<span class="{color_class}">{sign}{format_currency(value)}</span>'

def test_crv_manually():
    """Manual test for CRV data with actual sheet entries"""
    if debug_mode:
        st.write("🧪 **MANUAL CRV TEST WITH ACTUAL SHEET ENTRIES**")
        
        # Test API call directly
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": "CRVUSDT",
            "interval": "1d",
            "startTime": int(datetime(2025, 7, 1).timestamp() * 1000),
            "endTime": int(datetime.now().timestamp() * 1000),
            "limit": 50
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            st.write(f"✅ API Response received: {len(data)} candles")
            
            # Use actual Google Sheet entries for CRV
            entries = [0.516, 0.46, 0.41]  # Actual sheet values
            st.write(f"Testing with actual sheet entries: {entries}")
            
            for i, candle in enumerate(data):
                date = datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d')
                low = float(candle[3])
                
                st.write(f"{date}: Low = {low:.4f}")
                
                for j, entry in enumerate(entries):
                    if low <= entry:
                        st.write(f"  🎯 Would hit Entry {j+1} ({entry})")
                        
        except Exception as e:
            st.write(f"❌ Test failed: {e}")

# Main application logic
if csv_url:
    # Call test function if debug mode is on
    if debug_mode:
        test_crv_manually()
        st.write("---")  # Add separator
    
    # Create placeholder for the dashboard
    dashboard_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Auto-refresh loop
    while True:
        with dashboard_placeholder.container():
            # Load data from Google Sheet
            df, error = load_sheet_data(csv_url)
            
            if error:
                st.error(f"Error loading data: {error}")
                st.stop()
            
            if df is None or df.empty:
                st.warning("No data found in the sheet")
                st.stop()
            
            # Debug: Show loaded data
            if debug_mode:
                st.write("🔍 **Loaded Data from Google Sheet:**")
                st.dataframe(df)
            
            # Identify symbol column (try common names)
            symbol_col = None
            for col in ['Symbol', 'PAIR NAME', 'Pair', 'symbol', 'pair']:
                if col in df.columns:
                    symbol_col = col
                    break
            
            if symbol_col is None:
                st.error("Could not find Symbol/Pair column in the data")
                st.stop()
            
            # Process each row
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (_, row) in enumerate(df.iterrows()):
                symbol = row[symbol_col]
                
                # Update progress
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {symbol}... ({idx + 1}/{len(df)})")
                
                # Get live price
                live_price = get_binance_price(symbol)
                
                # Calculate metrics with historical data
                metrics = calculate_metrics(row, live_price, symbol)
                
                # Prepare result row
                result_row = {
                    'Symbol': symbol,
                    'Live Price': format_currency(live_price),
                    'Entry Status': metrics['entries_hit_status'],
                    'Entry Hit': '✅' if metrics['entry_hit'] else '❌',
                    'Avg Entry': format_currency(metrics['avg_entry']),
                    'P/L': format_pl(metrics['pl']),
                    'Entry % Down': format_percentage(metrics['entry_down_pct']),
                    'ROI %': format_percentage(metrics['roi_pct']),
                }
                
                # Add original columns
                for col in df.columns:
                    if col not in result_row and col != symbol_col:
                        result_row[col] = row[col]
                
                results.append(result_row)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display summary metrics
            total_rows = len(results_df)
            entries_hit = sum(1 for r in results if r['Entry Hit'] == '✅')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Pairs", total_rows)
            
            with col2:
                st.metric("Entries Hit", f"{entries_hit}/{total_rows}")
            
            with col3:
                hit_rate = (entries_hit / total_rows * 100) if total_rows > 0 else 0
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
            
            with col4:
                st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
            
            # Display main table
            st.subheader("Trading Dashboard")
            
            # Display the table with HTML formatting
            st.markdown(
                results_df.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
            
            # Summary statistics
            st.subheader("Summary")
            
            # Calculate total P/L if possible
            try:
                total_pl = 0
                valid_pl_count = 0
                
                for idx, row in df.iterrows():
                    symbol = row[symbol_col]
                    live_price = get_binance_price(symbol)
                    metrics = calculate_metrics(row, live_price, symbol)
                    
                    if metrics['pl'] is not None:
                        total_pl += metrics['pl']
                        valid_pl_count += 1
                
                if valid_pl_count > 0:
                    st.metric("Total P/L", format_currency(total_pl))
                
            except:
                pass
        
        # Status and timestamp
        with status_placeholder.container():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f'<p class="last-update">Last updated: {current_time}</p>', 
                       unsafe_allow_html=True)
        
        # Auto-refresh logic
        if not auto_refresh:
            break
        
        # Wait for next refresh
        time.sleep(refresh_interval)
        
        # Clear cache for fresh data
        st.cache_data.clear()

else:
    st.info("👆 Please enter your Google Sheet CSV URL in the sidebar to get started")
    
    st.markdown("""
    ### How to set up your Google Sheet:
    
    1. **Create a Google Sheet** with columns like:
       - `Symbol` or `PAIR NAME` (e.g., BTC, ETH, CRV)
       - `Entry 1`, `Entry 2`, `Entry 3` (entry prices)
       - `Date of given` (start date - use format like 01/07/25 for July 1, 2025)
       - `Quantity` (position size)
    
    2. **Date Format Examples**:
       - `01/07/25` (July 1, 2025)
       - `15/07/25` (July 15, 2025)
       - `01/06/25` (June 1, 2025)
    
    3. **Publish as CSV**:
       - Go to File → Share → Publish to web
       - Choose "Comma-separated values (.csv)" 
       - Copy the URL and paste it above
    
    ### Key Features:
    
    ✅ **Historical Analysis**: Fetches daily candles from your given date **to TODAY**
    
    📊 **Sequential Entry Tracking**: 
    - Entry 1 must be hit before Entry 2 is considered
    - Entry 2 must be hit before Entry 3 is considered
    
    📅 **Smart Date Handling**: 
    - Automatically interprets 2-digit years correctly
    - Prevents future date errors
    
    🎯 **Entry Status Display**: 
    - Shows which entries were hit with dates
    - Example: "Entry 1 (2025-07-15) → Entry 2 (2025-07-20)"
    
    💰 **Accurate P/L**: Based on actual hit entries and their average
    
    🔄 **Real-time Updates**: Auto-refresh with live price data
    
    🐛 **Debug Mode**: Enable in sidebar for detailed troubleshooting
    
    ### Debug Features:
    - **Raw Data Display**: Shows exactly what's loaded from Google Sheets
    - **CRV-Specific Debugging**: Detailed analysis for CRV entry detection
    - **Candle Data Inspection**: Shows daily lows/highs for verification
    - **Manual Entry Checks**: Displays when entries should be hit
    
    ### Perfect for Your Use Case:
    - Put dates from **recent months** (like 01/07/25 for July 1, 2025)
    - System checks **daily candles from that date to today**
    - Shows exactly **when and which entries were hit**
    - **Enable Debug Mode** to troubleshoot any detection issues
    """)
