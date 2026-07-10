from fastapi import FastAPI, HTTPException
import uvicorn

# 匯入我們剛剛寫好的引擎模組
from stock_ai_engine_v2 import APILayer, IndicatorEngine, PatternEngine, AIScoreEngine

# 建立 FastAPI 應用程式
app = FastAPI(title="Stock AI Engine API", version="2.0")

@app.get("/api/v1/analyze/{stock_id}")
async def analyze_stock(stock_id: str):
    """
    接收股票代號，回傳 AI 綜合分析 JSON 報告
    """
    try:
        # 1. 獲取原始數據
        raw_df = APILayer.get_finmind_data(stock_id, days=180)
        raw_df.attrs["stock_id"] = stock_id
        
        # 2. 計算技術指標
        ta_df = IndicatorEngine.add_all_indicators(raw_df)
        
        # 3. 進行型態辨識
        current_patterns = PatternEngine.analyze_patterns(ta_df)
        
        # 4. AI 綜合評分
        ai_report = AIScoreEngine.calculate_score(ta_df, current_patterns)
        
        # FastAPI 會自動將 Python 字典轉換為標準的 JSON 格式回傳
        return ai_report
        
    except ValueError as ve:
        # 處理找不到股票代號等預期內的錯誤
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        # 處理其他未知的系統錯誤
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {str(e)}")

# 啟動伺服器
if __name__ == "__main__":
    print("🚀 正在啟動 Stock AI Engine API 伺服器...")
    # 運行在 port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
