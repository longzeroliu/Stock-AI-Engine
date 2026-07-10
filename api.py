from fastapi import FastAPI, HTTPException
import uvicorn
from stock_ai_engine_v2 import APILayer, IndicatorEngine, PatternEngine, ChipEngine, AIScoreEngine

app = FastAPI(title="Stock AI Engine API", version="2.0")

@app.get("/api/v1/analyze/{stock_id}")
async def analyze_stock(stock_id: str):
    try:
        # 1. 獲取 K 線數據
        raw_df = APILayer.get_kline_data(stock_id, days=180)
        
        # 2. 計算技術指標
        ta_df = IndicatorEngine.add_all_indicators(raw_df)
        
        # 3. 進行型態辨識
        current_patterns = PatternEngine.analyze_patterns(ta_df)
        
        # 4. 獲取籌碼數據 (三大法人、融資券、大戶)
        chip_data = ChipEngine.analyze_chips(stock_id)
        
        # 5. AI 綜合評分
        ai_report = AIScoreEngine.calculate_score(stock_id, ta_df, current_patterns, chip_data)
        
        return ai_report
        
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {str(e)}")

if __name__ == "__main__":
    print("🚀 正在啟動 Stock AI Engine API 伺服器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
