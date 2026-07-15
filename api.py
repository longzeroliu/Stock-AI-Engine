from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from stock_ai_engine_v2 import APILayer, IndicatorEngine, PatternEngine, MarketChipEngine, AIScoreEngine

app = FastAPI(title="Stock AI Engine API")

# 加入 CORS 確保 Dify 呼叫暢通無阻
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running", "message": "Stock AI Engine V2 is live!"}

@app.get("/api/v1/analyze/{stock_id}")
def analyze(stock_id: str):
    try:
        print(f"📥 正在獲取 {stock_id} 的 K 線數據...")
        df = APILayer.get_kline_data(stock_id)
        
        print("🧮 正在計算技術指標與支撐壓力...")
        df = IndicatorEngine.add_all_indicators(df)
        
        print("🔍 正在進行莎拉型態辨識...")
        patterns = PatternEngine.analyze_patterns(df)
        
        print("🕵️ 正在分析大戶與法人籌碼...")
        chip_data = MarketChipEngine.analyze_chips(stock_id)
        
        print("⚖️ 正在計算 AI 綜合評分 (包含外資期貨)...")
        # 呼叫評分引擎，產出最終報告
        report = AIScoreEngine.calculate_score(stock_id, df, patterns, chip_data, {})
        
        return report
    except Exception as e:
        print(f"❌ 發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
