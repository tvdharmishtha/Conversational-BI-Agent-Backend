import pandas as pd
import json
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def get_column_info(df: pd.DataFrame) -> dict:
    """Intelligently detect column types"""
    columns = df.columns.tolist()
    column_types = {}
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it looks like a date column with numeric year/month
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() or 'month' in col.lower():
                column_types[col] = 'date'
            else:
                column_types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'date'
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            # Check if it looks like a date
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


def analyze_with_ai(query: str, df: pd.DataFrame) -> dict:
    """Use Groq AI to analyze the query and data, return chart specifications"""
    
    # Get column info
    column_info = get_column_info(df)
    columns = column_info["columns"]
    column_types = column_info["column_types"]
    
    # Get sample data
    sample_data = df.head(10).to_dict(orient='records')
    
    # Build the prompt
    prompt = f"""You are a data visualization expert. Given a user's query and the data structure, 
generate appropriate chart specifications.

USER QUERY: {query}

DATA COLUMNS: {columns}
COLUMN TYPES: {column_types}

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

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        # Find JSON in response (in case there's extra text)
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


def generate_chart_data(df: pd.DataFrame, chart_spec: dict) -> dict:
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
        # No aggregation - use raw values
        grouped = df.set_index(x_col)[y_col] if y_col and y_col in df.columns else df[x_col]
    
    # Convert to appropriate format
    x_values = [str(x) for x in grouped.index]
    y_values = []
    for v in grouped.values:
        try:
            y_values.append(float(v))
        except:
            y_values.append(v)
    
    # Build raw data for verification
    raw_data = []
    for x_val, y_val in zip(grouped.index, grouped.values):
        try:
            val = float(y_val)
        except:
            val = y_val
        raw_data.append({
            "label": str(x_val),
            "value": val
        })
    
    # Show all data - no limit
    # Backend sends all data, frontend can handle display
    
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
    
    # Get AI analysis
    ai_result = analyze_with_ai(query, df)
    
    # Generate chart data
    charts = []
    for chart_spec in ai_result.get("charts", []):
        chart_data = generate_chart_data(df, chart_spec)
        
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
    
    return {
        "insight": ai_result.get("insight", "Analysis complete."),
        "charts": charts
    }
