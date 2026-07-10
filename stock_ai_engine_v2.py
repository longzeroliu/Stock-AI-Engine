import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta


# ==========================================
# 1. API Layer (數據獲取層)
# ==========================================
class APILayer:
    @staticmethod
    def get_finmind_data(stock_id: str, days: int = 180) -> pd.DataFrame:
        """
        從 FinMind API 獲取台股歷史 K 線數據
        :param stock_id: 股票代號 (例如 '2330')
        :param days: 往前抓取的天數 (預設 180 天，確保長天期均線算得出來)
        """
        print(f"📥 正在從 FinMind 獲取 {stock_id} 的歷史數據...")

        # 計算開始與結束日期
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date
        }

        response = requests.get(url, params=parameter)
        data = response.json()

        if data["msg"] != "success" or not data["data"]:
            raise ValueError(f"獲取數據失敗，請檢查股票代號 {stock_id} 是否正確，或 API 是否超載。")

        # 將 JSON 轉為 Pandas DataFrame
        df = pd.DataFrame(data["data"])

        # 重新命名欄位，以符合 pandas-ta 的標準格式 (Open, High, Low, Close, Volume)
        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume"
        })

        # 設定日期為 Index，並轉換數值型態
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric)

        print(f"✅ 成功獲取 {len(df)} 筆 K 線數據！")
        return df


# ==========================================
# 2. Indicator Engine (技術指標引擎)
# ==========================================
class IndicatorEngine:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        計算所有技術指標並合併到 DataFrame 中
        """
        print("🧮 正在計算技術指標 (MA, MACD, RSI, KD, BOLL, ATR)...")

        # 為了避免修改原始數據，我們複製一份
        df_ta = df.copy()

        # 1. 均線 (MA)
        df_ta['MA5'] = df_ta.ta.sma(length=5)
        df_ta['MA20'] = df_ta.ta.sma(length=20)
        df_ta['MA60'] = df_ta.ta.sma(length=60)

        # 2. 指數移動平均線 (EMA)
        df_ta['EMA12'] = df_ta.ta.ema(length=12)
        df_ta['EMA26'] = df_ta.ta.ema(length=26)

        # 3. MACD (會產生 MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9 三個欄位)
        macd = df_ta.ta.macd(fast=12, slow=26, signal=9)
        df_ta = pd.concat([df_ta, macd], axis=1)

        # 4. RSI (相對強弱指標)
        df_ta['RSI14'] = df_ta.ta.rsi(length=14)

        # 5. KD (隨機指標，會產生 STOCHk_14_3_3, STOCHd_14_3_3)
        stoch = df_ta.ta.stoch(k=14, d=3, smooth_k=3)
        df_ta = pd.concat([df_ta, stoch], axis=1)

        # 6. BOLL (布林通道，會產生 BBL_5_2.0, BBM_5_2.0, BBU_5_2.0 等)
        boll = df_ta.ta.bbands(length=20, std=2)
        df_ta = pd.concat([df_ta, boll], axis=1)

        # 7. ATR (真實波動幅度)
        df_ta['ATR14'] = df_ta.ta.atr(length=14)

        # 8. 成交量均線 (Volume MA)
        df_ta['Vol_MA5'] = df_ta['Volume'].rolling(window=5).mean()

        # 移除因為計算長天期指標而產生的 NaN (空值) 列
        df_ta.dropna(inplace=True)

        print("✅ 技術指標計算完成！")
        return df_ta

import numpy as np

# ==========================================
# 3. Pattern Engine (型態辨識引擎)
# ==========================================
class PatternEngine:
    @staticmethod
    def analyze_patterns(df: pd.DataFrame) -> dict:
        """
        分析最新一個交易日的技術型態
        回傳一個字典，包含各項型態是否成立 (True/False)
        """
        print("🔍 正在進行 K 線與波段型態辨識...")
        
        # 確保資料夠長 (至少需要 30 天來判斷 W 底)
        if len(df) < 30:
            return {"error": "數據不足，無法分析型態"}

        # 取得最後幾天的數據 (用於短線 K 線判斷)
        # iloc[-1] 是最新一天, iloc[-2] 是昨天, iloc[-3] 是前天
        today = df.iloc[-1]
        yest = df.iloc[-2]
        day3 = df.iloc[-3]
        
        patterns = {}

        # -----------------------------------
        # 1. 均線型態判斷
        # -----------------------------------
        # 均線多頭排列: MA5 > MA20 > MA60，且 MA5 正在上升
        patterns["均線多頭排列"] = bool(
            today['MA5'] > today['MA20'] > today['MA60'] and 
            today['MA5'] > yest['MA5']
        )
        
        # 均線空頭排列: MA5 < MA20 < MA60
        patterns["均線空頭排列"] = bool(
            today['MA5'] < today['MA20'] < today['MA60']
        )

        # -----------------------------------
        # 2. 短線 K 線組合判斷
        # -----------------------------------
        # 判斷單根 K 線是否為紅K (收盤 > 開盤) 或 黑K (收盤 < 開盤)
        is_red_today = today['Close'] > today['Open']
        is_red_yest = yest['Close'] > yest['Open']
        is_red_day3 = day3['Close'] > day3['Open']
        
        is_black_today = today['Close'] < today['Open']
        is_black_yest = yest['Close'] < yest['Open']
        is_black_day3 = day3['Close'] < day3['Open']

        # 紅三兵: 連續三天紅K，且收盤價一天比一天高
        patterns["紅三兵"] = bool(
            is_red_today and is_red_yest and is_red_day3 and
            today['Close'] > yest['Close'] > day3['Close']
        )

        # 黑三兵: 連續三天黑K，且收盤價一天比一天低
        patterns["黑三兵"] = bool(
            is_black_today and is_black_yest and is_black_day3 and
            today['Close'] < yest['Close'] < day3['Close']
        )

        # 晨星 (Morning Star): 跌勢中，長黑K -> 十字星/小實體 -> 長紅K
        # 簡化邏輯: 前天黑K，昨天實體很小(震幅小)，今天紅K且收盤價超過前天黑K的一半
        body_yest = abs(yest['Close'] - yest['Open'])
        patterns["晨星"] = bool(
            is_black_day3 and is_red_today and
            body_yest < (yest['Close'] * 0.01) and # 昨天實體小於股價1%
            today['Close'] > (day3['Open'] + day3['Close']) / 2 # 今天收復前天一半
        )

        # -----------------------------------
        # 3. 波段型態判斷 (W底 / M頭)
        # -----------------------------------
        # 取近 30 天的收盤價來尋找波段
        recent_30 = df['Close'].tail(30).values
        
        # 尋找局部低點 (比前後兩天都低)
        local_minima = []
        for i in range(1, 29):
            if recent_30[i] < recent_30[i-1] and recent_30[i] < recent_30[i+1]:
                local_minima.append((i, recent_30[i]))
                
        # W底邏輯: 至少有兩個局部低點，且最後兩個低點的價格相近 (誤差 3% 內)，且目前價格已經反彈
        is_w_bottom = False
        if len(local_minima) >= 2:
            # 取最後兩個低點
            idx1, val1 = local_minima[-2]
            idx2, val2 = local_minima[-1]
            
            # 兩個低點距離至少 3 天，且價格誤差在 3% 以內
            if (idx2 - idx1) >= 3 and abs(val1 - val2) / val1 < 0.03:
                # 找出兩個低點之間的最高點 (頸線)
                neckline = max(recent_30[idx1:idx2])
                # 如果今天收盤價突破頸線，或者正在從第二個低點強勢反彈
                if today['Close'] > neckline or (today['Close'] > val2 * 1.02):
                    is_w_bottom = True
                    
        patterns["W底"] = is_w_bottom

        print("✅ 型態辨識完成！")
        return patterns

# ==========================================
# 4. AI Score Engine (綜合評分引擎)
# ==========================================
class AIScoreEngine:
    @staticmethod
    def calculate_score(df: pd.DataFrame, patterns: dict) -> dict:
        """
        根據技術指標與型態，計算 0~100 的綜合 AI 評分
        """
        print("⚖️ 正在計算 AI 綜合評分...")
        today = df.iloc[-1]
        
        # -----------------------------------
        # 1. 技術面評分 (滿分 40 分)
        # -----------------------------------
        tech_score = 0
        tech_details = []
        
        # 條件 A: 均線趨勢 (10分)
        if today['MA5'] > today['MA20']:
            tech_score += 10
            tech_details.append("短均線大於長均線，趨勢偏多 (+10分)")
        else:
            tech_details.append("短均線小於長均線，趨勢偏空 (+0分)")
            
        # 條件 B: MACD 動能 (10分)
        # MACDh_12_26_9 是 MACD 的柱狀圖 (OSC)
        if today['MACDh_12_26_9'] > 0:
            tech_score += 10
            tech_details.append("MACD 柱狀圖翻紅，多方動能強 (+10分)")
        else:
            tech_details.append("MACD 柱狀圖為綠，空方動能強 (+0分)")
            
        # 條件 C: RSI 強弱 (10分)
        if 50 <= today['RSI14'] <= 80:
            tech_score += 10
            tech_details.append("RSI 處於 50~80 強勢區間 (+10分)")
        elif today['RSI14'] < 30:
            tech_score += 5
            tech_details.append("RSI 低於 30，超賣醞釀反彈 (+5分)")
        else:
            tech_details.append("RSI 處於弱勢或超買區間 (+0分)")
            
        # 條件 D: KD 指標 (10分)
        if today['STOCHk_14_3_3'] > today['STOCHd_14_3_3']:
            tech_score += 10
            tech_details.append("KD 指標 K值大於D值，呈現多頭 (+10分)")
        else:
            tech_details.append("KD 指標死亡交叉或偏空 (+0分)")

        # -----------------------------------
        # 2. 型態面評分 (滿分 20 分)
        # -----------------------------------
        # 預設給予中性 10 分，出現好型態加分，壞型態扣分
        pattern_score = 10 
        pattern_details = []
        
        if patterns.get("W底"):
            pattern_score += 10
            pattern_details.append("出現 W底 底部反轉型態 (+10分)")
        if patterns.get("紅三兵"):
            pattern_score += 5
            pattern_details.append("出現 紅三兵 強勢攻擊型態 (+5分)")
        if patterns.get("晨星"):
            pattern_score += 5
            pattern_details.append("出現 晨星 止跌回升型態 (+5分)")
        if patterns.get("均線多頭排列"):
            pattern_score += 5
            pattern_details.append("均線呈現完美多頭排列 (+5分)")
            
        if patterns.get("黑三兵"):
            pattern_score -= 5
            pattern_details.append("出現 黑三兵 弱勢下跌型態 (-5分)")
        if patterns.get("均線空頭排列"):
            pattern_score -= 5
            pattern_details.append("均線呈現空頭排列 (-5分)")
            
        # 確保分數落在 0~20 之間
        pattern_score = max(0, min(20, pattern_score))
        if not pattern_details:
            pattern_details.append("目前無明顯特殊 K 線型態")

        # -----------------------------------
        # 3. 籌碼面 (滿分 20 分) - 尚未實作，給予中性分
        # -----------------------------------
        chip_score = 10
        chip_details = ["籌碼面數據尚未串接 (預設中性 10分)"]

        # -----------------------------------
        # 4. 期貨面 (滿分 20 分) - 尚未實作，給予中性分
        # -----------------------------------
        futures_score = 10
        futures_details = ["期貨大盤數據尚未串接 (預設中性 10分)"]

        # -----------------------------------
        # 總分與 AI 評語結算
        # -----------------------------------
        total_score = tech_score + pattern_score + chip_score + futures_score
        
        if total_score >= 80:
            action = "強烈看多 (Strong Buy)"
        elif total_score >= 60:
            action = "偏多看待 (Buy)"
        elif total_score >= 40:
            action = "中性震盪 (Neutral)"
        else:
            action = "偏空看待 (Sell)"

        print(f"✅ 評分完成！總分: {total_score} 分 ({action})")
        
        return {
            "stock_id": df.attrs.get("stock_id", "Unknown"),
            "date": today.name.strftime('%Y-%m-%d'),
            "total_score": total_score,
            "action": action,
            "breakdown": {
                "Technical (40%)": {"score": tech_score, "details": tech_details},
                "Pattern (20%)": {"score": pattern_score, "details": pattern_details},
                "Chip (20%)": {"score": chip_score, "details": chip_details},
                "Futures (20%)": {"score": futures_score, "details": futures_details}
            }
        }

# ==========================================
# 測試執行區塊 (最終整合版)
# ==========================================
if __name__ == "__main__":
    import json
    stock_id = "2330" # 測試台積電
    
    try:
        # 1. 獲取原始數據
        raw_df = APILayer.get_finmind_data(stock_id, days=180)
        raw_df.attrs["stock_id"] = stock_id # 偷偷把股票代號塞進去傳遞
        
        # 2. 計算技術指標
        ta_df = IndicatorEngine.add_all_indicators(raw_df)
        
        # 3. 進行型態辨識
        current_patterns = PatternEngine.analyze_patterns(ta_df)
        
        # 4. AI 綜合評分
        ai_report = AIScoreEngine.calculate_score(ta_df, current_patterns)
        
        # 5. 印出漂亮的 JSON 報告 (這就是未來要餵給 Dify 的格式)
        print("\n" + "="*50)
        print(f" 📊 {stock_id} AI 股票分析報告")
        print("="*50)
        # 使用 json.dumps 讓輸出格式化，方便閱讀
        print(json.dumps(ai_report, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

