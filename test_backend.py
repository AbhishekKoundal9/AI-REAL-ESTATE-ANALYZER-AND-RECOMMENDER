import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.agent import InvestmentAgent
from backend.database import get_history_records

def test_pipeline():
    print("Testing Investment Agent & Pipeline...")
    agent = InvestmentAgent()
    
    result = agent.run_investment_agent(
        location="Whitefield",
        sqft=1500.0,
        bhk=3,
        bath=3,
        balcony=2,
        budget=95.0
    )
    
    print("================ AGENT EXECUTION RESULT ================")
    print(f"Predicted Price: {result['predicted_price_lakhs']} Lakhs ({result['predicted_price_inr'].encode('utf-8')})")
    print(f"Price / Sqft: {result['price_per_sqft']}/sq.ft")
    print(f"5-Year ROI: {result['roi_percent']}%")
    print(f"Rental Yield: {result['rental_yield_percent']}%")
    print(f"Investment Score: {result['investment_score']} / 100")
    print(f"Risk Classification: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Reasons: {result['reasons']}")
    print(f"PDF Generated: {result['pdf_filename']}")
    print(f"PDF Exists: {os.path.exists(os.path.join('pdf_reports', result['pdf_filename']))}")
    print("========================================================\n")
    
    history = get_history_records()
    print(f"History Records Count in SQLite DB: {len(history)}")

if __name__ == "__main__":
    test_pipeline()
