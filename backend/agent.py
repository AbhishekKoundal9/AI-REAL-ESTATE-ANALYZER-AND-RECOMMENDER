import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import datetime
import joblib
import pandas as pd
import numpy as np

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdf_reports")
os.makedirs(PDF_DIR, exist_ok=True)

class InvestmentAgent:
    def __init__(self):
        self.model_path = os.path.join(SAVED_MODELS_DIR, "best_model.joblib")
        self.dataset_path = os.path.join(SAVED_MODELS_DIR, "cleaned_dataset.csv")
        self.locations_path = os.path.join(SAVED_MODELS_DIR, "locations.json")
        self.metrics_path = os.path.join(SAVED_MODELS_DIR, "metrics.json")
        
        self.model = None
        self.dataset = None
        self.locations = []
        self.metrics = {}
        
        self.load_resources()

    def load_resources(self):
        """Loads trained ML model, preprocessed Kaggle dataset, and metadata."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.dataset_path):
            self.dataset = pd.read_csv(self.dataset_path)
        if os.path.exists(self.locations_path):
            with open(self.locations_path, 'r') as f:
                self.locations = json.load(f)
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                self.metrics = json.load(f)

    # 1. Read & Validate Input
    def read_input(self, location: str, sqft: float, bhk: int, bath: int, balcony: int):
        loc = location.strip() if location else "other"
        # Match location format or default to 'other'
        if loc not in self.locations:
            loc = "other"
        return {
            "location": loc,
            "total_sqft": float(max(100.0, sqft)),
            "bhk": int(max(1, bhk)),
            "bath": int(max(1, bath)),
            "balcony": int(max(0, balcony))
        }

    # 2. Search Kaggle Dataset Statistics
    def search_dataset(self, location: str, sqft: float):
        if self.dataset is None or self.dataset.empty:
            return {"avg_price_per_sqft": 6500.0, "location_count": 0}
        
        loc_df = self.dataset[self.dataset['location'] == location]
        if loc_df.empty:
            loc_df = self.dataset
            
        avg_pps = loc_df['price_per_sqft'].mean() if 'price_per_sqft' in loc_df.columns else 6000.0
        min_price = loc_df['price'].min()
        max_price = loc_df['price'].max()
        avg_price = loc_df['price'].mean()
        
        return {
            "avg_price_per_sqft": round(float(avg_pps), 2),
            "location_sample_count": len(loc_df),
            "market_min_price": round(float(min_price), 2),
            "market_max_price": round(float(max_price), 2),
            "market_avg_price": round(float(avg_price), 2)
        }

    # 3. Predict Property Price using ML Model
    def predict_price(self, input_data: dict) -> float:
        if self.model is None:
            self.load_resources()
            
        if self.model is None:
            # Fallback estimation formula if model not loaded
            return round((input_data["total_sqft"] * 6500.0) / 100000.0, 2)
            
        df_input = pd.DataFrame([{
            "location": input_data["location"],
            "total_sqft": input_data["total_sqft"],
            "bhk": input_data["bhk"],
            "bath": input_data["bath"],
            "balcony": input_data["balcony"]
        }])
        
        pred_lakhs = float(self.model.predict(df_input)[0])
        return max(5.0, round(pred_lakhs, 2))  # Minimum clamp 5 Lakhs

    # 4. Find Similar Properties (Top 5 Matching Recommendations)
    def find_similar_properties(self, location: str, bhk: int, budget_lakhs: float = None, top_n: int = 5):
        if self.dataset is None or self.dataset.empty:
            return []
            
        df = self.dataset.copy()
        
        # Filter location preference if matches exist
        loc_matches = df[df['location'] == location]
        if len(loc_matches) >= top_n:
            df = loc_matches
            
        # Calculate similarity match score (based on BHK, location match, price proximity to budget)
        target_budget = budget_lakhs if budget_lakhs and budget_lakhs > 0 else df['price'].mean()
        
        scores = []
        for idx, row in df.iterrows():
            loc_score = 100 if row['location'] == location else 60
            bhk_diff = abs(row['bhk'] - bhk)
            bhk_score = max(0, 100 - (bhk_diff * 20))
            
            price_diff_pct = abs(row['price'] - target_budget) / target_budget
            price_score = max(0, 100 - (price_diff_pct * 50))
            
            total_match_score = round((loc_score * 0.4) + (bhk_score * 0.3) + (price_score * 0.3), 1)
            
            scores.append({
                "id": int(idx),
                "location": str(row['location']),
                "total_sqft": float(row['total_sqft']),
                "price_lakhs": float(row['price']),
                "bhk": int(row['bhk']),
                "bath": int(row.get('bath', 2)),
                "balcony": int(row.get('balcony', 1)),
                "match_score": min(99.9, max(50.0, total_match_score))
            })
            
        # Sort by match score descending
        sorted_props = sorted(scores, key=lambda x: x['match_score'], reverse=True)
        return sorted_props[:top_n]

    # 5. Calculate ROI (5-Year Estimated Capital Appreciation)
    def calculate_roi(self, predicted_price: float, avg_pps: float):
        # Base annual appreciation rate: 6% - 11% depending on location demand
        annual_growth_rate = 0.085 if avg_pps > 7000 else 0.072
        future_val_5yr = predicted_price * ((1 + annual_growth_rate) ** 5)
        total_roi_pct = ((future_val_5yr - predicted_price) / predicted_price) * 100.0
        return round(total_roi_pct, 2)

    # 6. Calculate Rental Yield
    def calculate_rental_yield(self, predicted_price_lakhs: float, total_sqft: float):
        # Estimated monthly rent: ₹18 - ₹35 per sq.ft in Bengaluru
        est_monthly_rent_per_sqft = 24.0
        annual_rent_inr = total_sqft * est_monthly_rent_per_sqft * 12.0
        property_val_inr = predicted_price_lakhs * 100000.0
        
        rental_yield_pct = (annual_rent_inr / property_val_inr) * 100.0
        return round(rental_yield_pct, 2)

    # 7. Calculate Investment Score (0 to 100)
    def calculate_investment_score(self, roi: float, rental_yield: float, price_per_sqft: float, avg_pps: float):
        # Normalized ROI score (0-40 points)
        roi_points = min(40.0, (roi / 55.0) * 40.0)
        
        # Normalized Yield score (0-30 points)
        yield_points = min(30.0, (rental_yield / 5.0) * 30.0)
        
        # Value ratio score (is price per sqft below or above market avg?) (0-30 points)
        price_diff_pct = (price_per_sqft - avg_pps) / avg_pps
        value_points = max(5.0, min(30.0, 20.0 - (price_diff_pct * 25.0)))
        
        total_score = round(roi_points + yield_points + value_points, 1)
        return min(98.5, max(35.0, total_score))

    # 8. Classify Risk (Low, Medium, High)
    def classify_risk(self, pps: float, avg_pps: float, investment_score: float):
        dev = (pps - avg_pps) / avg_pps
        if dev > 0.25 or investment_score < 55:
            return "High Risk"
        elif dev > 0.08 or investment_score < 75:
            return "Medium Risk"
        else:
            return "Low Risk"

    # 9. Future Price Estimation (1-Year & 3-Years)
    def estimate_future_prices(self, current_price: float, risk_level: str):
        growth_rates = {
            "Low Risk": (0.085, 0.28),    # 8.5% 1-yr, ~28% 3-yr
            "Medium Risk": (0.07, 0.22),  # 7.0% 1-yr, ~22% 3-yr
            "High Risk": (0.045, 0.14)    # 4.5% 1-yr, ~14% 3-yr
        }
        r1, r3 = growth_rates.get(risk_level, (0.07, 0.22))
        p_1yr = round(current_price * (1 + r1), 2)
        p_3yr = round(current_price * (1 + r3), 2)
        return {"price_1yr": p_1yr, "price_3yr": p_3yr}

    # 10. Generate AI Insights & Recommendation (BUY / HOLD / AVOID)
    def generate_recommendation_and_insights(self, predicted_price: float, sqft: float, pps: float, avg_pps: float, roi: float, yield_pct: float, score: float, risk: str):
        insights = []
        
        # Price position insight
        if pps < avg_pps:
            pct_below = round(((avg_pps - pps) / avg_pps) * 100, 1)
            insights.append(f"Property price is {pct_below}% below the local market average of ₹{avg_pps:,.0f}/sq.ft.")
        else:
            pct_above = round(((pps - avg_pps) / avg_pps) * 100, 1)
            insights.append(f"Property is priced at a premium ({pct_above}% above average area rate of ₹{avg_pps:,.0f}/sq.ft).")

        # Yield insight
        if yield_pct >= 3.5:
            insights.append(f"Strong rental yield of {yield_pct}% makes it an attractive income-generating asset.")
        else:
            insights.append(f"Moderate rental yield of {yield_pct}% is typical for residential capital growth areas.")

        # ROI & Growth insight
        if roi >= 40:
            insights.append(f"Projected 5-year capital appreciation (ROI: {roi}%) indicates strong long-term value creation.")
        else:
            insights.append(f"Steady projected capital growth (ROI: {roi}%) provides defensive capital preservation.")

        # Final recommendation logic
        reasons = []
        if score >= 75 and risk in ["Low Risk", "Medium Risk"]:
            recommendation = "BUY"
            reasons.append("Solid investment score with strong growth prospects.")
            reasons.append(f"Low to manageable risk profile ({risk}).")
            reasons.append("Favorable purchase valuation relative to market metrics.")
        elif score >= 60:
            recommendation = "HOLD"
            reasons.append("Moderate return metrics with stable market demand.")
            reasons.append(f"Acceptable risk level ({risk}).")
            reasons.append("Consider negotiating price before making a firm purchase.")
        else:
            recommendation = "AVOID"
            reasons.append("Underwhelming ROI and rental yield metrics.")
            reasons.append(f"Higher risk classification ({risk}).")
            reasons.append("Property is overvalued compared to local benchmark rates.")

        return {
            "insights": insights,
            "recommendation": recommendation,
            "reasons": reasons
        }

    # 11. Complete Agent Execution Workflow
    def run_investment_agent(self, location: str, sqft: float, bhk: int, bath: int, balcony: int, budget: float = None):
        # Step 1: Read input
        input_data = self.read_input(location, sqft, bhk, bath, balcony)
        
        # Step 2: Search dataset stats
        stats = self.search_dataset(input_data["location"], input_data["total_sqft"])
        
        # Step 3: Predict price
        predicted_price = self.predict_price(input_data)
        
        # Calculate price per sqft
        pps = (predicted_price * 100000.0) / input_data["total_sqft"]
        avg_pps = stats["avg_price_per_sqft"]
        
        # Step 4: Find top 5 similar properties
        recommendations = self.find_similar_properties(
            input_data["location"], input_data["bhk"], budget if budget else predicted_price
        )
        
        # Step 5: Calculate ROI
        roi = self.calculate_roi(predicted_price, avg_pps)
        
        # Step 6: Calculate Rental Yield
        rental_yield = self.calculate_rental_yield(predicted_price, input_data["total_sqft"])
        
        # Step 7: Calculate Investment Score
        investment_score = self.calculate_investment_score(roi, rental_yield, pps, avg_pps)
        
        # Step 8: Classify Risk
        risk_level = self.classify_risk(pps, avg_pps, investment_score)
        
        # Step 9: Future price estimates
        future_prices = self.estimate_future_prices(predicted_price, risk_level)
        
        # Step 10: AI Insights & Recommendation
        rec_info = self.generate_recommendation_and_insights(
            predicted_price, input_data["total_sqft"], pps, avg_pps, roi, rental_yield, investment_score, risk_level
        )
        
        # Step 11: Generate PDF Report
        report_id = f"report_{uuid.uuid4().hex[:8]}.pdf"
        pdf_path = self.generate_pdf_report(
            report_id=report_id,
            input_data=input_data,
            predicted_price=predicted_price,
            pps=pps,
            avg_pps=avg_pps,
            roi=roi,
            rental_yield=rental_yield,
            investment_score=investment_score,
            risk_level=risk_level,
            future_prices=future_prices,
            recommendation=rec_info["recommendation"],
            reasons=rec_info["reasons"],
            insights=rec_info["insights"]
        )
        
        return {
            "input_data": input_data,
            "market_stats": stats,
            "predicted_price_lakhs": predicted_price,
            "predicted_price_inr": f"₹{predicted_price * 100000:,.0f}",
            "price_per_sqft": round(pps, 2),
            "avg_price_per_sqft": avg_pps,
            "roi_percent": roi,
            "rental_yield_percent": rental_yield,
            "investment_score": investment_score,
            "risk_level": risk_level,
            "future_prices": future_prices,
            "recommendation": rec_info["recommendation"],
            "reasons": rec_info["reasons"],
            "insights": rec_info["insights"],
            "recommendations_list": recommendations,
            "pdf_filename": report_id
        }

    # 12. PDF Report Generator (ReportLab)
    def generate_pdf_report(self, report_id, input_data, predicted_price, pps, avg_pps, roi, rental_yield, investment_score, risk_level, future_prices, recommendation, reasons, insights):
        file_path = os.path.join(PDF_DIR, report_id)
        doc = SimpleDocTemplate(file_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#8B5CF6'), alignment=1
        )
        subtitle_style = ParagraphStyle(
            'SubTitleStyle', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#64748B'), alignment=1
        )
        heading_style = ParagraphStyle(
            'HeadStyle', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#06B6D4')
        )
        body_style = ParagraphStyle(
            'BodyStyle', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.HexColor('#1E293B')
        )
        
        story = []
        
        # Header
        story.append(Paragraph("AI REAL ESTATE ANALYZER", title_style))
        story.append(Paragraph(f"Automated Agentic Investment Evaluation Report • Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
        story.append(Spacer(1, 15))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#8B5CF6'), spaceAfter=15))
        
        # Section 1: Property Specifications & Valuation
        story.append(Paragraph("1. Property Summary & Machine Learning Valuation", heading_style))
        story.append(Spacer(1, 8))
        
        prop_data = [
            [Paragraph("<b>Location:</b>", body_style), Paragraph(str(input_data['location']), body_style),
             Paragraph("<b>Predicted Price:</b>", body_style), Paragraph(f"<b>₹{predicted_price:,.2f} Lakhs</b>", body_style)],
            [Paragraph("<b>Total Area:</b>", body_style), Paragraph(f"{input_data['total_sqft']} sq.ft.", body_style),
             Paragraph("<b>Price / sq.ft:</b>", body_style), Paragraph(f"₹{pps:,.0f}", body_style)],
            [Paragraph("<b>BHK / Bath / Balcony:</b>", body_style), Paragraph(f"{input_data['bhk']} BHK | {input_data['bath']} Bath | {input_data['balcony']} Balc", body_style),
             Paragraph("<b>Area Avg Rate:</b>", body_style), Paragraph(f"₹{avg_pps:,.0f}", body_style)]
        ]
        
        t1 = Table(prop_data, colWidths=[130, 140, 130, 140])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t1)
        story.append(Spacer(1, 15))
        
        # Section 2: Financial Metrics & Risk Classification
        story.append(Paragraph("2. Financial Investment Metrics & Risk Profile", heading_style))
        story.append(Spacer(1, 8))
        
        rec_color = colors.HexColor('#10B981') if recommendation == 'BUY' else (colors.HexColor('#F59E0B') if recommendation == 'HOLD' else colors.HexColor('#EF4444'))
        
        metrics_data = [
            [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Assessment</b>", body_style)],
            [Paragraph("Estimated 5-Yr ROI", body_style), Paragraph(f"<b>{roi}%</b>", body_style), Paragraph("Capital Appreciation Potential", body_style)],
            [Paragraph("Gross Rental Yield", body_style), Paragraph(f"<b>{rental_yield}%</b>", body_style), Paragraph("Annual Rental Income Return", body_style)],
            [Paragraph("Investment Score", body_style), Paragraph(f"<b>{investment_score} / 100</b>", body_style), Paragraph("Weighted Quality Score", body_style)],
            [Paragraph("Risk Classification", body_style), Paragraph(f"<b>{risk_level}</b>", body_style), Paragraph("Volatility & Pricing Variance Risk", body_style)],
            [Paragraph("1-Yr Future Valuation", body_style), Paragraph(f"₹{future_prices['price_1yr']:,.2f} Lakhs", body_style), Paragraph("Projected 12-Month Value", body_style)],
            [Paragraph("3-Yr Future Valuation", body_style), Paragraph(f"₹{future_prices['price_3yr']:,.2f} Lakhs", body_style), Paragraph("Projected 36-Month Value", body_style)],
        ]
        
        t2 = Table(metrics_data, colWidths=[170, 150, 220])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EDF2F7')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t2)
        story.append(Spacer(1, 15))
        
        # Section 3: Recommendation & AI Decision Rationale
        story.append(Paragraph(f"3. Agentic Recommendation: <font color='{rec_color.hexval()}'>{recommendation}</font>", heading_style))
        story.append(Spacer(1, 8))
        
        rec_text = f"<b>Final Recommendation:</b> {recommendation}<br/><br/><b>Key Rationale:</b><br/>"
        for r in reasons:
            rec_text += f"• {r}<br/>"
            
        rec_text += "<br/><b>AI Market Insights:</b><br/>"
        for ins in insights:
            rec_text += f"• {ins}<br/>"
            
        story.append(Paragraph(rec_text, body_style))
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94A3B8'), spaceAfter=10))
        story.append(Paragraph("AI Real Estate Analyzer • Machine Learning & Agentic AI Platform • Powered by Scikit-Learn & FastAPI", subtitle_style))
        
        doc.build(story)
        return file_path
