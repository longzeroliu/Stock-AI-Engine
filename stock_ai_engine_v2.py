import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. API Layer (數據獲取層)
# ==========================================
class APILayer:
    @staticmethod
    def fetch_finmind(dataset: str, data_id: str, days: int) -> pd.DataFrame:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": dataset, "data_id": data_id, "start_date": start_date, "end_date": end_date}
        try:
            response = requests.get(url, params=params, timeout=10 )
            data = response.json()
            if data.get("msg") == "success" and data.get("data"):
                return pd.DataFrame(data["data"])
        except Exception as e:
            print(f"⚠️ 獲取 {dataset} 失敗: {e}")
        return pd.DataFrame()

    @staticmethod
    def get_kline_data(stock_id: str, days: int = 180) -> pd.DataFrame:
        df = APILayer.fetch_finmind("TaiwanStockPrice", stock_id, days)
        if df.empty:
            raise ValueError(f"獲取 K 線失敗，請檢查股票代號 {stock_id}。")
        df = df.rename(columns={"date": "Date", "open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric)
        return df

# ==========================================
# 2. Indicator Engine (技術指標引擎)
# ==========================================
class IndicatorEngine:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df_ta = df.copy()
        df_ta['MA5'] = df_ta.ta.sma(length=5)
        df_ta['MA20'] = df_ta.ta.sma(length=20)
        df_ta['MA60'] = df_ta.ta.sma(length=60)
        macd = df_ta.ta.macd(fast=12, slow=26, signal=9)
        df_ta = pd.concat([df_ta, macd], axis=1)
        df_ta['RSI14'] = df_ta.ta.rsi(length=14)
        stoch = df_ta.ta.stoch(k=14, d=3, smooth_k=3)
        df_ta = pd.concat([df_ta, stoch], axis=1)
        df_ta.dropna(inplace=True)
        return df_ta

# ==========================================
# 3. Pattern Engine (型態辨識引擎)
# ==========================================
class PatternEngine:
    @staticmethod
    def analyze_patterns(df: pd.DataFrame) -> dict:
        if len(df) < 30: return {}
        today, yest, day3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        patterns = {}

        patterns["均線多頭排列"] = bool(today['MA5'] > today['MA20'] > today['MA60'] and today['MA5'] > yest['MA5'])
        patterns["均線空頭排列"] = bool(today['MA5'] < today['MA20'] < today['MA60'])

        is_red_today = today['Close'] > today['Open']
        is_red_yest = yest['Close'] > yest['Open']
        is_red_day3 = day3['Close'] > day3['Open']
        is_black_today = today['Close'] < today['Open']
        is_black_yest = yest['Close'] < yest['Open']
        is_black_day3 = day3['Close'] < day3['Open']

        patterns["紅三兵"] = bool(is_red_today and is_red_yest and is_red_day3 and today['Close'] > yest['Close'] > day3['Close'])
        patterns["黑三兵"] = bool(is_black_today and is_black_yest and is_black_day3 and today['Close'] < yest['Close'] < day3['Close'])

        body_yest = abs(yest['Close'] - yest['Open'])
        patterns["晨星"] = bool(is_black_day3 and is_red_today and body_yest < (yest['Close'] * 0.01) and today['Close'] > (day3['Open'] + day3['Close']) / 2)

        recent_30 = df['Close'].tail(30).values
        local_minima = [(i, recent_30[i]) for i in range(1, 29) if recent_30[i] < recent_30[i-1] and recent_30[i] < recent_30[i+1]]
        is_w_bottom = False
        if len(local_minima) >= 2:
            idx1, val1 = local_minima[-2]
            idx2, val2 = local_minima[-1]
            if (idx2 - idx1) >= 3 and abs(val1 - val2) / val1 < 0.03:
                neckline = max(recent_30[idx1:idx2])
                if today['Close'] > neckline or (today['Close'] > val2 * 1.02):
                    is_w_bottom = True
        patterns["W底"] = is_w_bottom
        return patterns

# ==========================================
# 4. Chip Engine (籌碼引擎)
# ==========================================
class ChipEngine:
    @staticmethod
    def analyze_chips(stock_id: str) -> dict:
        chip_score = 10
        chip_details = []
        lending_warning = False
        try:
            df_inst = APILayer.fetch_finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, 10)
            if not df_inst.empty:
                df_inst['diff'] = df_inst['buy'] - df_inst['sell']
                recent_date = df_inst['date'].max()
                today_inst = df_inst[df_inst['date'] == recent_date]
                foreign = today_inst[today_inst['name'].str.contains('外資', na=False)]['diff'].sum()
                trust = today_inst[today_inst['name'].str.contains('投信', na=False)]['diff'].sum()
                if foreign > 0 and trust > 0: chip_score += 5; chip_details.append("外資與投信同步買超 (+5分)")
                elif foreign < 0 and trust < 0: chip_score -= 5; chip_details.append("外資與投信同步賣超 (-5分)")
                elif foreign > 0: chip_score += 3; chip_details.append("外資買超 (+3分)")
                elif foreign < 0: chip_score -= 3; chip_details.append("外資賣超 (-3分)")
                    
            df_margin = APILayer.fetch_finmind("TaiwanStockMarginPurchaseShortSale", stock_id, 10)
            if not df_margin.empty:
                df_margin = df_margin.sort_values('date')
                if len(df_margin) >= 2:
                    margin_today = df_margin.iloc[-1]['MarginPurchaseTodayBalance']
                    margin_yest = df_margin.iloc[-2]['MarginPurchaseTodayBalance']
                    if margin_today > margin_yest: chip_score -= 2; chip_details.append("融資餘額增加 (-2分)")
                    else: chip_score += 2; chip_details.append("融資餘額減少 (+2分)")
                    
                    short_today = df_margin.iloc[-1]['ShortSaleTodayBalance']
                    short_yest = df_margin.iloc[-2]['ShortSaleTodayBalance']
                    if short_today > short_yest * 1.05:
                        lending_warning = True
                        chip_score -= 3; chip_details.append("⚠️ 融券/借券餘額顯著增加 (-3分)")
        except Exception as e:
            chip_details.append(f"籌碼數據獲取異常: {str(e)}")
            
        chip_score = max(0, min(20, chip_score))
        if not chip_details: chip_details.append("籌碼面數據無明顯變化 (中性)")
        return {"score": chip_score, "details": chip_details, "lending_warning": lending_warning}

# ==========================================
# 5. AI Score Engine (綜合評分引擎)
# ==========================================
class AIScoreEngine:
    @staticmethod
    def calculate_score(stock_id: str, df: pd.DataFrame, patterns: dict, chip_data: dict) -> dict:
        today = df.iloc[-1]
        tech_score = 0
        tech_details = []
        
        if today['MA5'] > today['MA20']: tech_score += 10; tech_details.append("短均線大於長均線，趨勢偏多 (+10分)")
        else: tech_details.append("短均線小於長均線，趨勢偏空 (+0分)")
        if today.get('MACDh_12_26_9', 0) > 0: tech_score += 10; tech_details.append("MACD 柱狀圖翻紅，多方動能強 (+10分)")
        else: tech_details.append("MACD 柱狀圖為綠，空方動能強 (+0分)")
        if 50 <= today['RSI14'] <= 80: tech_score += 10; tech_details.append("RSI 處於 50~80 強勢區間 (+10分)")
        elif today['RSI14'] < 30: tech_score += 5; tech_details.append("RSI 低於 30，超賣醞釀反彈 (+5分)")
        else: tech_details.append("RSI 處於弱勢或超買區間 (+0分)")
        if today.get('STOCHk_14_3_3', 0) > today.get('STOCHd_14_3_3', 0): tech_score += 10; tech_details.append("KD 指標 K值大於D值，呈現多頭 (+10分)")
        else: tech_details.append("KD 指標死亡交叉或偏空 (+0分)")

        pattern_score = 10 
        pattern_details = []
        if patterns.get("W底"): pattern_score += 10; pattern_details.append("出現 W底 底部反轉型態 (+10分)")
        if patterns.get("紅三兵"): pattern_score += 5; pattern_details.append("出現 紅三兵 強勢攻擊型態 (+5分)")
        if patterns.get("晨星"): pattern_score += 5; pattern_details.append("出現 晨星 止跌回升型態 (+5分)")
        if patterns.get("均線多頭排列"): pattern_score += 5; pattern_details.append("均線呈現完美多頭排列 (+5分)")
        if patterns.get("黑三兵"): pattern_score -= 5; pattern_details.append("出現 黑三兵 弱勢下跌型態 (-5分)")
        if patterns.get("均線空頭排列"): pattern_score -= 5; pattern_details.append("均線呈現空頭排列 (-5分)")
        pattern_score = max(0, min(20, pattern_score))
        if not pattern_details: pattern_details.append("目前無明顯特殊 K 線型態")

        chip_score = chip_data["score"]
        chip_details = chip_data["details"]
        futures_score = 10
        futures_details = ["期貨大盤數據尚未串接 (預設中性 10分)"]

        total_score = tech_score + pattern_score + chip_score + futures_score
        
        if total_score >= 80: action = "強烈看多 (Strong Buy)"
        elif total_score >= 60: action = "偏多看待 (Buy)"
        elif total_score >= 40: action = "中性震盪 (Neutral)"
        else: action = "偏空看待 (Sell)"

        return {
            "stock_id": stock_id,
            "date": today.name.strftime('%Y-%m-%d'),
            "total_score": total_score,
            "action": action,
            "breakdown": {
                "Technical (40%)": {"score": tech_score, "details": tech_details},
                "Pattern (20%)": {"score": pattern_score, "details": pattern_details},
                "Chip (20%)": {"score": chip_score, "details": chip_details, "lending_warning": chip_data["lending_warning"]},
                "Futures (20%)": {"score": futures_score, "details": futures_details}
            }
        }
