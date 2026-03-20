import pandas as pd
import json
from groq import Groq
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Currency conversion: USD to INR (change this value as needed)
USD_TO_INR = 83.0

def convert_to_rupees(value: float) -> float:
    """Convert USD value to INR"""
    return value * USD_TO_INR

def detect_currency(df: pd.DataFrame) -> bool:
    """Detect if data might be in dollars based on column names"""
    columns = [col.lower() for col in df.columns]
    column_text = ' '.join(columns)
    
    # Check if data is already in Rupees (don't convert)
    rupee_keywords = ['rupees', 'rs ', 'rs.', 'inr', '₹']
    if any(keyword in column_text for keyword in rupee_keywords):
        return False  # Already in rupees, don't convert
    
    # Check if data is in Dollars (convert to rupees)
    dollar_keywords = ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'income', 'dollar', 'usd']
    return any(keyword in column_text for keyword in dollar_keywords)


def suggest_currency(df: pd.DataFrame) -> dict:
    """Analyze Excel data and suggest the currency type with explanation"""
    columns = [col.lower() for col in df.columns]
    column_text = ' '.join(columns)
    
    # Check for explicit currency indicators in column names
    if any(kw in column_text for kw in ['rupees', 'rs', 'rs.', 'inr', '₹']):
        return {
            "detected_currency": "INR",
            "currency_symbol": "₹",
            "reason": "Column names contain 'rupees', 'rs', or 'inr'",
            "should_convert": False
        }
    
    if any(kw in column_text for kw in ['dollar', 'usd', '$']):
        return {
            "detected_currency": "USD",
            "currency_symbol": "$",
            "reason": "Column names contain 'dollar', 'usd', or '$'",
            "should_convert": True
        }
    
    # Analyze numeric columns for currency indicators
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if numeric_cols:
        sample_values = []
        for col in numeric_cols[:3]:
            sample = df[col].dropna().head(100)
            sample_values.extend(sample.tolist())
        
        if sample_values:
            avg_value = sum(sample_values) / len(sample_values)
            max_value = max(sample_values)
            
            # If values are very large (like in millions/billions), likely dollars
            if max_value > 1000000:  # > 10 lakhs, likely dollars
                return {
                    "detected_currency": "USD (likely)",
                    "currency_symbol": "$",
                    "reason": f"Large values detected (avg: {avg_value:,.0f}, max: {max_value:,.0f}) - likely in USD",
                    "should_convert": True
                }
            else:
                return {
                    "detected_currency": "INR (likely)",
                    "currency_symbol": "₹",
                    "reason": f"Smaller values detected (avg: {avg_value:,.0f}, max: {max_value:,.0f}) - likely in INR",
                    "should_convert": False
                }
    
    # Default: no conversion
    return {
        "detected_currency": "Unknown",
        "currency_symbol": "₹",
        "reason": "No currency indicators found - defaulting to INR",
        "should_convert": False
    }


def get_column_info(df: pd.DataFrame) -> dict:
    """Intelligently detect column types"""
    columns = df.columns.tolist()
    column_types = {}
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() or 'month' in col.lower():
                column_types[col] = 'date'
            else:
                column_types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'date'
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    column_types[col] = 'date'
                except:
                    column_types[col] = 'categorical'
            else:
                column_types[col] = 'categorical'
        else:
            column_types[col] = 'categorical'
    
    return {
        "columns": columns,
        "column_types": column_types
    }


def generate_enhanced_insight(query: str, df: pd.DataFrame, chart_specs: list, convert_to_inr: bool = False) -> str:
    """Generate rich narrative insight based on actual computed data"""
    
    # Currency settings
    currency_symbol = "₹" if convert_to_inr else "$"
    usd_to_inr = USD_TO_INR if convert_to_inr else 1.0
    
    # Get computed statistics for each chart
    stats_data = []
    for spec in chart_specs[:2]:  # Focus on first 2 charts
        x_col = spec.get('x')
        y_col = spec.get('y')
        agg = spec.get('aggregation', 'sum')
        
        if x_col and x_col in df.columns:
            if y_col and y_col in df.columns:
                if agg == 'sum':
                    grouped = df.groupby(x_col)[y_col].sum()
                elif agg == 'count':
                    grouped = df.groupby(x_col).size()
                elif agg == 'average':
                    grouped = df.groupby(x_col)[y_col].mean()
                elif agg == 'max':
                    grouped = df.groupby(x_col)[y_col].max()
                elif agg == 'min':
                    grouped = df.groupby(x_col)[y_col].min()
                else:
                    grouped = df.groupby(x_col).size()
                
                total = grouped.sum() if hasattr(grouped, 'sum') else len(grouped)
                mean_val = grouped.mean() if hasattr(grouped, 'mean') else 0
                max_val = grouped.max() if hasattr(grouped, 'max') else 0
                min_val = grouped.min() if hasattr(grouped, 'min') else 0
                top_item = grouped.idxmax() if len(grouped) > 0 else "N/A"
                
                # Apply currency conversion
                total = total * usd_to_inr
                mean_val = mean_val * usd_to_inr
                max_val = max_val * usd_to_inr
                min_val = min_val * usd_to_inr
                
                stats_data.append({
                    "x_column": x_col,
                    "y_column": y_col,
                    "aggregation": agg,
                    "total": float(total) if total else 0,
                    "mean": float(mean_val) if mean_val else 0,
                    "max": float(max_val) if max_val else 0,
                    "min": float(min_val) if min_val else 0,
                    "top_item": str(top_item),
                    "item_count": len(grouped),
                    "currency": currency_symbol
                })
    
    # Create enhanced prompt for insight generation
    insight_prompt = f"""You are a business intelligence analyst. Generate a compelling, narrative insight 
about the user's query based on the computed statistics.

USER QUERY: {query}

COMPUTED STATISTICS:
{json.dumps(stats_data, indent=2, default=str)}

IMPORTANT: Use the currency symbol from the data: {currency_symbol}

Write a 2-3 sentence insight that:
- Starts with a key finding or summary
- Includes specific numbers with currency symbol {currency_symbol} (totals, percentages, comparisons)
- Mentions the top performing category/item
- Provides business context when possible
- Is conversational but professional

Example: "Sales show strong performance with total revenue of {currency_symbol}2.5M across {len(df)} transactions. The Electronics category leads with {currency_symbol}800K, representing 32% of all sales. Overall, there's a clear upward trend indicating positive business growth."

Respond with ONLY the insight text, no JSON or formatting."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": insight_prompt
                }
            ],
            temperature=0,  # Deterministic output
            max_tokens=300
        )
        
        insight = response.choices[0].message.content.strip()
        return insight
    except Exception as e:
        print(f"Insight generation error: {e}")
        # Fallback to basic insight
        return generate_basic_insight(stats_data)


def generate_basic_insight(stats_data: list, convert_to_inr: bool = False) -> str:
    """Generate basic insight when AI fails"""
    currency_symbol = "₹" if convert_to_inr else "$"
    
    if not stats_data:
        return "Analysis of your data shows interesting patterns and trends."
    
    stats = stats_data[0]
    total = stats.get('total', 0)
    top = stats.get('top_item', 'N/A')
    y_col = stats.get('y_column', 'values')
    
    return f"Analysis shows a total of {currency_symbol}{total:,.2f} in {y_col}. The top performer is {top} with the highest value in this category."


def analyze_with_ai(query: str, df: pd.DataFrame) -> dict:
    """Use Groq AI to analyze the query and data, return chart specifications"""
    
    # Get column info
    column_info = get_column_info(df)
    columns = column_info["columns"]
    column_types = column_info["column_types"]
    
    # Get sample data
    sample_data = df.head(10).to_dict(orient='records')
    
    # Get row count for context
    row_count = len(df)
    
    # Build the prompt
    prompt = f"""You are a data visualization expert. Given a user's query and the data structure, 
generate appropriate chart specifications.

USER QUERY: {query}

DATA COLUMNS: {columns}
COLUMN TYPES: {column_types}
TOTAL ROWS: {row_count}

SAMPLE DATA (first 10 rows):
{json.dumps(sample_data, indent=2, default=str)}

Respond with a JSON object containing:
1. "insight": A brief natural language insight about the data
2. "charts": An array of chart specifications, each with:
   - "id": unique identifier (e.g., "chart1", "chart2")
   - "title": descriptive title
   - "type": chart type ("line", "bar", "pie", "scatter", "area", "donut")
   - "x": the column name for x-axis (category/timeline)
   - "y": the column name for y-axis (values)
   - "aggregation": how to aggregate ("sum", "count", "average", "min", "max", "none")
   
Generate 2-4 charts that would best answer the user's query. For each chart:
- Use appropriate aggregation (sum for totals, count for frequency, average for means)
- Choose chart type based on data: line for trends, bar for comparisons, pie for proportions
- Use numeric columns for y-axis, categorical or date columns for x-axis
- If query asks about specific metrics (revenue, sales, profit), prioritize those columns
- If query asks about comparison, use bar charts
- If query asks about trends over time, use line charts
- If query asks about distribution or proportions, use pie/donut charts

Return ONLY valid JSON, no other text."""

    # Generate deterministic seed from query
    query_hash = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,  # Set to 0 for consistent results with same query
            max_tokens=2000,
            seed=query_hash  # Use deterministic seed
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = result[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
            
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        # Fallback to basic analysis
        return fallback_analysis(query, df, column_info)


def fallback_analysis(query: str, df: pd.DataFrame, column_info: dict) -> dict:
    """Fallback analysis when AI fails"""
    columns = column_info["columns"]
    column_types = column_info["column_types"]
    
    charts = []
    
    # Find numeric and categorical columns
    numeric_cols = [c for c, t in column_types.items() if t == 'numeric']
    categorical_cols = [c for c, t in column_types.items() if t == 'categorical']
    date_cols = [c for c, t in column_types.items() if t == 'date']
    
    # Use first available columns
    x_col = date_cols[0] if date_cols else (categorical_cols[0] if categorical_cols else columns[0])
    y_col = numeric_cols[0] if numeric_cols else (columns[1] if len(columns) > 1 else columns[0])
    
    if x_col and y_col:
        try:
            grouped = df.groupby(x_col)[y_col].sum().reset_index()
            charts.append({
                "id": "chart1",
                "title": f"{y_col} by {x_col}",
                "type": "bar",
                "x": x_col,
                "y": y_col,
                "aggregation": "sum"
            })
        except:
            pass
    
    # Add a second chart if possible
    if len(numeric_cols) >= 2:
        charts.append({
            "id": "chart2",
            "title": f"{numeric_cols[1]} by {x_col}" if x_col else f"{numeric_cols[1]} Distribution",
            "type": "line",
            "x": x_col if x_col else categorical_cols[0] if categorical_cols else columns[0],
            "y": numeric_cols[1],
            "aggregation": "sum" if x_col else "none"
        })
    elif len(categorical_cols) >= 2:
        charts.append({
            "id": "chart2",
            "title": f"{categorical_cols[1]} Breakdown",
            "type": "pie",
            "x": categorical_cols[0],
            "y": y_col,
            "aggregation": "count"
        })
    
    return {
        "insight": "Analysis of your data shows the following patterns.",
        "charts": charts
    }


def generate_chart_data(df: pd.DataFrame, chart_spec: dict, convert_to_inr: bool = False) -> dict:
    """Generate chart data from dataframe based on chart specification"""
    
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    aggregation = chart_spec.get("aggregation", "sum")
    chart_type = chart_spec.get("type", "bar")
    
    if not x_col or x_col not in df.columns:
        return {"x": [], "y": [], "raw_data": []}
    
    # Apply aggregation
    if aggregation == "sum":
        grouped = df.groupby(x_col)[y_col].sum() if y_col and y_col in df.columns else df.groupby(x_col).size()
    elif aggregation == "count":
        grouped = df.groupby(x_col).size()
    elif aggregation == "average":
        grouped = df.groupby(x_col)[y_col].mean() if y_col and y_col in df.columns else df.groupby(x_col).size()
    elif aggregation == "min":
        grouped = df.groupby(x_col)[y_col].min() if y_col and y_col in df.columns else df.groupby(x_col).size()
    elif aggregation == "max":
        grouped = df.groupby(x_col)[y_col].max() if y_col and y_col in df.columns else df.groupby(x_col).size()
    else:
        grouped = df.set_index(x_col)[y_col] if y_col and y_col in df.columns else df[x_col]
    
    # Convert to appropriate format
    x_values = [str(x) for x in grouped.index]
    y_values = []
    for v in grouped.values:
        try:
            val = float(v)
            if convert_to_inr:
                val = convert_to_rupees(val)
            y_values.append(val)
        except:
            y_values.append(v)
    
    # Build raw data for verification
    raw_data = []
    for x_val, y_val in zip(grouped.index, grouped.values):
        try:
            val = float(y_val)
            if convert_to_inr:
                val = convert_to_rupees(val)
        except:
            val = y_val
        raw_data.append({
            "label": str(x_val),
            "value": val
        })
    
    return {
        "x": x_values,
        "y": y_values,
        "raw_data": raw_data,
        "aggregation": aggregation,
        "x_column": x_col,
        "y_column": y_col if y_col else "count"
    }


def analyze_query(query: str, df: pd.DataFrame) -> dict:
    """Main function to analyze query and generate charts"""
    
    # Detect if currency conversion is needed
    should_convert = detect_currency(df)
    currency_symbol = "₹" if should_convert else ""
    
    # Get AI analysis
    ai_result = analyze_with_ai(query, df)
    
    # Generate chart data
    charts = []
    chart_specs = ai_result.get("charts", [])
    
    for chart_spec in chart_specs:
        chart_data = generate_chart_data(df, chart_spec, convert_to_inr=should_convert)
        
        if chart_data["x"] and chart_data["y"]:
            charts.append({
                "id": chart_spec.get("id", f"chart{len(charts)+1}"),
                "title": chart_spec.get("title", "Chart"),
                "type": chart_spec.get("type", "bar"),
                "x": chart_data["x"],
                "y": chart_data["y"],
                "raw_data": chart_data.get("raw_data", []),
                "aggregation": chart_data.get("aggregation", "sum"),
                "x_column": chart_data.get("x_column", chart_spec.get("x")),
                "y_column": chart_data.get("y_column", chart_spec.get("y"))
            })
    
    # Generate enhanced insight using actual computed data
    try:
        enhanced_insight = generate_enhanced_insight(query, df, chart_specs, should_convert)
    except Exception as e:
        print(f"Enhanced insight error: {e}")
        enhanced_insight = ai_result.get("insight", "Analysis complete.")
    
    return {
        "insight": enhanced_insight,
        "charts": charts
    }
