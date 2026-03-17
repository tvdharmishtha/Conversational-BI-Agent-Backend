import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_data():
    return pd.read_excel("data/business_data.xlsx")

def get_intent(query):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"""
                Classify this query into one of:
                - revenue_trend
                - product_comparison
                - stock_analysis

                Query: {query}
                Only return the category name.
                """
            }
        ]
    )

    return response.choices[0].message.content.strip().lower()


def analyze_query(query, df):
    intent = get_intent(query)

    if "revenue" in intent:
        grouped = df.groupby("Month")["Revenue"].sum()
        return {
            "type": "line",
            "x": [str(x) for x in grouped.index],   
            "y": [int(y) for y in grouped.values],
            "title": "Revenue Trend",
            "insight": "Revenue trend shows growth over time."
        }

    elif "comparison" in intent:
        grouped = df.groupby("Product")["Revenue"].sum()
        return {
            "type": "bar",
            "x": [str(x) for x in grouped.index],   
            "y": [int(y) for y in grouped.values],
            "title": "Product Comparison",
            "insight": "Boat Headphones outperform Plane Toy."
        }

    elif "stock" in intent:
        grouped = df.groupby("Product")["Stock"].sum()
        return {
            "type": "bar",
            "x": [str(x) for x in grouped.index],   
            "y": [int(y) for y in grouped.values],
            "title": "Stock Analysis",
            "insight": "Plane Toy stock is lower — needs restock."
        }

    return {"error": "AI could not understand"}