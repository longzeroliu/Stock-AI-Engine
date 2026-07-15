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
    def get_kline_data(stock_id: str, days: int = 400) -> pd.DataFrame:
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
# 2. Indicator Engine (技術指標與支撐壓力)
# ==========================================
class IndicatorEngine:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df_ta = df.copy()
        df_ta['MA5'] = df_ta.ta.sma(length=5)
        df_ta['MA10'] = df_ta.ta.sma(length=10)
        df_ta['MA20'] = df_ta.ta.sma(length=20)
        df_ta['MA60'] = df_ta.ta.sma(length=60)
        df_ta['MA120'] = df_ta.ta.sma(length=120)
        df_ta['MA240'] = df_ta.ta.sma(length=240)
        
        macd = df_ta.ta.macd(fast=12, slow=26, signal=9)
        df_ta = pd.concat([df_ta, macd], axis=1)
        df_ta['RSI14'] = df_ta.ta.rsi(length=14)
        stoch = df_ta.ta.stoch(k=14, d=3, smooth_k=3)
        df_ta = pd.concat([df_ta, stoch], axis=1)
        
        bbands = df_ta.ta.bbands(length=20, std=2)
        df_ta = pd.concat([df_ta, bbands], axis=1)
        
        df_ta['Support'] = df_ta['Low'].rolling(20).min()
        df_ta['Resistance'] = df_ta['High'].rolling(20).max()
        
        df_ta.dropna(inplace=True)
        return df_ta

# ==========================================
# 3. Pattern Engine (莎拉新新型態學)
# ==========================================
class PatternEngine:
    @staticmethod
    def analyze_patterns(df: pd.DataFrame) -> dict:
        if len(df) < 5: return {}
        today, yest, day3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        patterns = {}

        is_red_today = today['Close'] > today['Open']
        is_red_yest = yest['Close'] > yest['Open']
        is_red_day3 = day3['Close'] > day3['Open']
        is_black_today = today['Close'] < today['Open']
        is_black_yest = yest['Close'] < yest['Open']
        is_black_day3 = day3['Close'] < day3['Open']

        patterns["莎拉_正多頭開花"] = bool(today['MA5'] > today['MA10'] > today['MA20'] > today['MA60'] > today['MA120'] > today['MA240'])
        ma60_up = today['MA60'] > yest['MA60']
        break_long = (today['Close'] > today['MA120'] and yest['Close'] <= yest['MA120']) or (today['Close'] > today['MA240'] and yest['Close'] <= yest['MA240'])
        patterns["莎拉_金包銀"] = bool(ma60_up and today['Close'] > today['MA60'] and break_long)

        bias_60 = (today['Close'] - today['MA60']) / today['MA60']
        patterns["莎拉_正乖離過大(牽狗繩)"] = bool(bias_60 > 0.15)
        patterns["莎拉_負乖離過大(牽狗繩)"] = bool(bias_60 < -0.15)

        patterns["莎拉_小碎步"] = bool(is_red_today and is_red_yest and is_red_day3 and today['Close'] > yest['Close'] > day3['Close'])
        patterns["莎拉_下樓梯"] = bool(is_black_today and is_black_yest and is_black_day3 and today['Close'] < yest['Close'] < day3['Close'])

        if 'BBU_20_2.0' in today and 'BBL_20_2.0' in today:
            patterns["莎拉_穿心箭"] = bool(day3['High'] > day3['BBU_20_2.0'] and today['Close'] < today['BBU_20_2.0'] and yest['Close'] < yest['BBU_20_2.0'])
            patterns["莎拉_有底撐"] = bool(day3['Low'] < day3['BBL_20_2.0'] and today['Close'] > today['BBL_20_2.0'] and today['Low'] >= day3['Low'])

        return patterns

# ==========================================
# 4. Market & Chip Engine (大盤與籌碼引擎)
# ==========================================
class MarketChipEngine:
    @staticmethod
    def analyze_market() -> dict:
        try:
            df_taiex = APILayer.fetch_finmind("TaiwanStockPrice", "TAIEX", 60)
            if not df_taiex.empty:
                df_taiex['close'] = pd.to_numeric(df_taiex['close'])
                ma5 = df_taiex['close'].rolling(5).mean().iloc[-1]
                ma20 = df_taiex['close'].rolling(20).mean().iloc[-1]
                if ma5 > ma20: return {"score": 10, "details": ["大盤 (TAIEX) 站上月線，整體市場氣氛偏多 (+10分)"]}
                else: return {"score": 0, "details": ["大盤 (TAIEX) 跌破月線，需留意系統性風險 (+0分)"]}
        except: pass
        return {"score": 5, "details": ["大盤數據獲取異常 (預設中性)"]}

    @staticmethod
    def analyze_chips(stock_id: str) -> dict:
        chip_score = 10
        chip_details = []
        lending_warning = False
        
        try:
            df_inst = APILayer.fetch_finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, 20)
            if not df_inst.empty:
                df_inst['diff'] = df_inst['buy'] - df_inst['sell']
                recent_date = df_inst['date'].max()
                today_inst = df_inst[df_inst['date'] == recent_date]
                foreign = today_inst[today_inst['name'].str.contains('外資', na=False)]['diff'].sum()
                trust = today_inst[today_inst['name'].str.contains('投信', na=False)]['diff'].sum()
                f_sheets = int(foreign / 1000)
                t_sheets = int(trust / 1000)
                if foreign > 0 and trust > 0: chip_score += 5; chip_details.append(f"外資與投信同步買超 (外資 {f_sheets}張, 投信 {t_sheets}張) (+5分)")
                elif foreign < 0 and trust < 0: chip_score -= 5; chip_details.append(f"外資與投信同步賣超 (外資 {f_sheets}張, 投信 {t_sheets}張) (-5分)")
                else: chip_details.append(f"法人動向分歧 (外資 {f_sheets}張, 投信 {t_sheets}張) (+0分)")
            else: chip_details.append("三大法人：近期無資料")
        except Exception as e: chip_details.append(f"三大法人：獲取失敗 ({str(e)})")

        try:
            df_shares = APILayer.fetch_finmind("TaiwanStockHoldingSharesPer", stock_id, 60)
            if not df_shares.empty:
                df_shares['HoldingSharesLevel'] = df_shares['HoldingSharesLevel'].astype(str)
                df_big = df_shares[df_shares['HoldingSharesLevel'] == '15'].sort_values('date')
                if len(df_big) >= 2:
                    latest_pct = df_big.iloc[-1]['percent']
                    prev_pct = df_big.iloc[-2]['percent']
                    if latest_pct > prev_pct: chip_score += 5; chip_details.append(f"千張大戶持股增加：{prev_pct}% ➡️ {latest_pct}% (+5分)")
                    elif latest_pct < prev_pct: chip_score -= 2; chip_details.append(f"千張大戶持股減少：{prev_pct}% ➡️ {latest_pct}% (-2分)")
                    else: chip_details.append(f"千張大戶持股維持：{latest_pct}% (+0分)")
                else: chip_details.append("千張大戶：資料筆數不足")
            else: chip_details.append("千張大戶：近期無資料")
        except Exception as e: chip_details.append(f"千張大戶：獲取失敗 ({str(e)})")
                    
        try:
            df_margin = APILayer.fetch_finmind("TaiwanStockMarginPurchaseShortSale", stock_id, 20)
            if not df_margin.empty:
                df_margin = df_margin.sort_values('date')
                if len(df_margin) >= 2:
                    margin_today = df_margin.iloc[-1]['MarginPurchaseTodayBalance']
                    margin_yest = df_margin.iloc[-2]['MarginPurchaseTodayBalance']
                    if margin_today > margin_yest: chip_score -= 2; chip_details.append("融資餘額(散戶)增加 (-2分)")
                    else: chip_score += 2; chip_details.append("融資餘額(散戶)減少 (+2分)")
                    
                    short_today = df_margin.iloc[-1]['ShortSaleTodayBalance']
                    short_yest = df_margin.iloc[-2]['ShortSaleTodayBalance']
                    if short_today > short_yest * 1.05:
                        lending_warning = True
                        chip_score -= 5; chip_details.append("⚠️ 融券/借券賣出餘額顯著增加，有避險賣壓 (-5分)")
            else: chip_details.append("融資券：近期無資料")
        except Exception as e: chip_details.append(f"融資券：獲取失敗 ({str(e)})")
            
        chip_score = max(0, min(20, chip_score))
        if not chip_details: chip_details.append("籌碼面數據無明顯變化 (中性)")
        return {"score": chip_score, "details": chip_details, "lending_warning": lending_warning}

# ==========================================
# 5. Futures Engine (期貨引擎)
# ==========================================
class FuturesEngine:
    @staticmethod
    def analyze_futures() -> dict:
        futures_score = 10
        futures_details = []
        try:
            df_fut = APILayer.fetch_finmind("TaiwanFuturesInstitutionalInvestors", "TX", 10)
            if not df_fut.empty:
                df_foreign = df_fut[df_fut['name'] == '外資及陸資'].sort_values('date')
                if len(df_foreign) >= 1:
                    latest_oi = df_foreign.iloc[-1]['open_interest_netlot']
                    if latest_oi > 10000: futures_score += 10; futures_details.append(f"📈 外資台指期淨多單高達 {latest_oi} 口，大盤極度偏多 (+10分)")
                    elif latest_oi > 0: futures_score += 5; futures_details.append(f"📈 外資台指期淨多單 {latest_oi} 口，大盤偏多 (+5分)")
                    elif latest_oi < -10000: futures_score -= 10; futures_details.append(f"📉 外資台指期淨空單高達 {latest_oi} 口，大盤極度偏空 (-10分)")
                    elif latest_oi < 0: futures_score -= 5; futures_details.append(f"📉 外資台指期淨空單 {latest_oi} 口，大盤偏空 (-5分)")
                    else: futures_details.append(f"外資台指期未平倉量為 {latest_oi} 口 (中性)")
            else: futures_details.append("外資期貨未平倉：近期無資料")
        except Exception as e: futures_details.append(f"期貨數據獲取失敗 ({str(e)})")
            
        futures_score = max(0, min(20, futures_score))
        return {"score": futures_score, "details": futures_details}

# ==========================================
# 6. AI Score Engine (綜合評分引擎)
# ==========================================
class AIScoreEngine:
    @staticmethod
    def calculate_score(stock_id: str, df: pd.DataFrame, patterns: dict, chip_data: dict, market_data: dict = None) -> dict:
        today = df.iloc[-1]
        tech_score = 0
        tech_details = []
        
        tech_details.append(f"🎯 關鍵價位：近期支撐 {today['Support']:.2f} / 近期壓力 {today['Resistance']:.2f}")
        
        if today['MA5'] > today['MA20']: tech_score += 10; tech_details.append("短均線大於長均線，趨勢偏多 (+10分)")
        if today.get('MACDh_12_26_9', 0) > 0: tech_score += 10; tech_details.append("MACD 柱狀圖翻紅，多方動能強 (+10分)")
        if 50 <= today['RSI14'] <= 80: tech_score += 10; tech_details.append("RSI 處於 50~80 強勢區間 (+10分)")
        elif today['RSI14'] < 30: tech_score += 5; tech_details.append("RSI 低於 30，超賣醞釀反彈 (+5分)")

        pattern_score = 10 
        pattern_details = []
        if patterns.get("莎拉_正多頭開花"): pattern_score += 10; pattern_details.append("🌸 [莎拉型態] 正多頭開花：均線完美發散 (+10分)")
        if patterns.get("莎拉_金包銀"): pattern_score += 10; pattern_details.append("🥇 [莎拉型態] 金包銀：突破長均線壓力 (+10分)")
        if patterns.get("莎拉_小碎步"): pattern_score += 5; pattern_details.append("🚶 [莎拉型態] 小碎步：多方量縮推升 (+5分)")
        if patterns.get("莎拉_有底撐"): pattern_score += 5; pattern_details.append("🛡️ [莎拉型態] 有底撐：低檔反轉 (+5分)")
        if patterns.get("莎拉_下樓梯"): pattern_score -= 5; pattern_details.append("📉 [莎拉型態] 下樓梯：高檔走勢轉弱 (-5分)")
        if patterns.get("莎拉_穿心箭"): pattern_score -= 10; pattern_details.append("🏹 [莎拉型態] 穿心箭：短線逃頂訊號 (-10分)")
        if patterns.get("莎拉_正乖離過大(牽狗繩)"): pattern_score -= 5; pattern_details.append("🐕 [莎拉型態] 牽狗繩：正乖離過大，留意回檔 (-5分)")
        if patterns.get("莎拉_負乖離過大(牽狗繩)"): pattern_score += 5; pattern_details.append("🐕 [莎拉型態] 牽狗繩：負乖離過大，醞釀反彈 (+5分)")

        pattern_score = max(0, min(20, pattern_score))
        if not pattern_details: pattern_details.append("目前無觸發特殊莎拉型態")

        futures_data = FuturesEngine.analyze_futures()

        total_score = tech_score + pattern_score + chip_data["score"] + futures_data["score"]
        
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
                "Chip (20%)": {"score": chip_data["score"], "details": chip_data["details"], "lending_warning": chip_data["lending_warning"]},
                "Futures (20%)": {"score": futures_data["score"], "details": futures_data["details"]}
            }
        }
