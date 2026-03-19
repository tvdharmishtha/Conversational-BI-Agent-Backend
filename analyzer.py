import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=__import__("os").environ.get("GROQ_API_KEY"))


def get_intent(query):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Classify into one:
- revenue_trend
- product_comparison
- stock_analysis

Query: {query}
Only return category.
"""
                }
            ]
        )

        return response.choices[0].message.content.strip().lower()

    except:
        return "revenue_trend"  # fallback


def analyze_query(query, df):
    intent = get_intent(query)

    # fallback keyword safety
    if "stock" in query.lower():
        intent = "stock_analysis"
    elif "product" in query.lower():
        intent = "product_comparison"
    elif "revenue" in query.lower():
        intent = "revenue_trend"
    
    charts = []

    if "revenue" in intent:
        month_col = 'Month'
        revenue_col = 'Revenue'
        
        if month_col in df.columns and revenue_col in df.columns:
            grouped = df.groupby(month_col)[revenue_col].sum()
        else:
            # Use first two columns as fallback
            if len(df.columns) >= 2:
                grouped = df.groupby(df.columns[0])[df.columns[1]].sum()
            else:
                return {"insight": "Insufficient data columns", "charts": []}

        charts.append({
            "id": "chart1",
            "title": "Revenue Trend",
            "type": "line",
            "x": [str(x) for x in grouped.index],
            "y": grouped.values.tolist()
        })

        insight = "Revenue is showing a trend over time."

    elif "comparison" in intent:
        product_col = 'Product'
        revenue_col = 'Revenue'
        
        if product_col in df.columns and revenue_col in df.columns:
            grouped = df.groupby(product_col)[revenue_col].sum()
        else:
            # Use first two columns as fallback
            if len(df.columns) >= 2:
                grouped = df.groupby(df.columns[0])[df.columns[1]].sum()
            else:
                return {"insight": "Insufficient data columns", "charts": []}

        charts.append({
            "id": "chart1",
            "title": "Product Comparison",
            "type": "bar",
            "x": grouped.index.tolist(),
            "y": grouped.values.tolist()
        })

        insight = "Some products are outperforming others."

    elif "stock" in intent:
        product_col = 'Product'
        stock_col = 'Stock'
        
        if product_col in df.columns and stock_col in df.columns:
            grouped = df.groupby(product_col)[stock_col].sum()
        else:
            # Use first two columns as fallback
            if len(df.columns) >= 2:
                grouped = df.groupby(df.columns[0])[df.columns[1]].sum()
            else:
                return {"insight": "Insufficient data columns", "charts": []}

        charts.append({
            "id": "chart1",
            "title": "Stock Analysis",
            "type": "bar",
            "x": grouped.index.tolist(),
            "y": grouped.values.tolist()
        })

        insight = "Stock levels vary across products."

    else:
        return {"insight": "Could not understand query", "charts": []}

    return {
        "insight": insight,
        "charts": charts
    }
