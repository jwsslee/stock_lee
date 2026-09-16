"""이종완의 주식투자 현황 @ 한국투자증권
실행: python -m streamlit run app_jwlee.py
국내주식 조회 전용. 자동 주문 없음. 자세한 안내: manual.pdf
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ================= 사용자 설정: 아래 네 값을 직접 입력 =================
APP_KEY = os.environ.get("X_APP_KEY")

APP_SECRET = os.environ.get("X_APP_SECRET")

CANO = "50205776"

ACNT_PRDT_CD = "01"

IS_PAPER = True            # True: 모의투자 / False: 실전(조회만 수행)
CASH_REFERENCE_CODE = "005930"  # 주문가능현금 조회 기준 종목
# ====================================================================
TITLE = "이종완의 주식투자 현황 @ 한국투자증권"
KST = ZoneInfo("Asia/Seoul")
BASE_URL = ("https://openapivts.koreainvestment.com:29443" if IS_PAPER
            else "https://openapi.koreainvestment.com:9443")
TRAIN_YEARS = 3
FEATURES = ["r1", "r5", "r20", "ma5_gap", "ma20_gap", "ma60_gap",
            "vol20", "rsi14", "range", "vol_ratio", "macd_gap"]


def now():
    return datetime.now(KST)


def number(value):
    """빈 값/누락은 0 대신 NaN으로 표시."""
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return float("nan")


def won(value):
    n = number(value)
    return "조회 불가" if not np.isfinite(n) else f"{n:,.0f} 원"


class KISError(RuntimeError):
    pass


class KISClient:
    def __init__(self):
        self.http = requests.Session()
        self.last_call = 0.0
        self.token = ""
        self.expires = 0.0
        # 토큰도 비밀정보: 개인 홈 아래 로컬 파일, 공개 폴더에는 저장하지 않음.
        identity = hashlib.sha256((BASE_URL + APP_KEY + APP_SECRET).encode()).hexdigest()[:24]
        self.token_path = Path.home() / ".jwlee_kis" / f"token_{identity}.json"

    def access_token(self):
        if self.token and time.time() < self.expires - 120:
            return self.token
        try:
            saved = json.loads(self.token_path.read_text(encoding="utf-8"))
            if time.time() < saved["expires"] - 120:
                self.token, self.expires = saved["token"], saved["expires"]
                return self.token
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            r = self.http.post(BASE_URL + "/oauth2/tokenP", json={
                "grant_type": "client_credentials", "appkey": APP_KEY,
                "appsecret": APP_SECRET}, timeout=20)
            body = r.json()
            if not r.ok or not body.get("access_token"):
                raise KISError("토큰 발급 실패: APP KEY/SECRET, 실전·모의 설정 및 발급 제한을 확인하세요.")
        except (requests.RequestException, ValueError) as e:
            raise KISError("토큰 서버 연결/응답 오류. 잠시 후 재시도하세요.") from e
        self.token = body["access_token"]
        self.expires = time.time() + int(body.get("expires_in", 86400))
        try:
            self.token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"token": self.token, "expires": self.expires}, f)
        except OSError:
            pass  # 파일 저장 불가 시 현재 세션 메모리에서만 재사용
        return self.token

    def get(self, path, tr_id, params, continuation=""):
        for attempt in range(4):
            token = self.access_token()
            time.sleep(max(0, 0.65 - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            try:
                r = self.http.get(BASE_URL + path, headers={
                    "authorization": f"Bearer {token}", "appkey": APP_KEY,
                    "appsecret": APP_SECRET, "tr_id": tr_id, "custtype": "P",
                    "tr_cont": continuation, "content-type": "application/json; charset=utf-8"},
                    params=params, timeout=20)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                b = r.json()
                if b.get("msg_cd") == "EGW00201":
                    time.sleep(2 ** attempt)
                    continue
                if not r.ok or str(b.get("rt_cd")) != "0":
                    # 키/계좌/응답 전체를 화면이나 로그에 노출하지 않음
                    raise KISError(f"API 조회 실패 ({b.get('msg_cd', r.status_code)}). 계좌·권한·운영시간을 확인하세요.")
                return b, r.headers.get("tr_cont", "")
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise KISError("네트워크 또는 JSON 응답 오류: 4회 시도 후 중단했습니다.")
                time.sleep(2 ** attempt)
        raise KISError("API 호출 제한/서버 오류: 잠시 후 새로고침하세요.")

    def balance(self):
        rows, summary, fk, nk, cont = [], {}, "", "", ""
        seen = set()
        for _ in range(100):
            b, header = self.get("/uapi/domestic-stock/v1/trading/inquire-balance",
                "VTTC8434R" if IS_PAPER else "TTTC8434R", {
                    "CANO": CANO, "ACNT_PRDT_CD": ACNT_PRDT_CD,
                    "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
                    "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00",
                    "CTX_AREA_FK100": fk, "CTX_AREA_NK100": nk}, cont)
            rows.extend(b.get("output1", []))
            s = b.get("output2", [])
            if not summary and s:
                summary = s[0] if isinstance(s, list) else s
            if header not in ("M", "F"):
                return rows, summary
            fk, nk = b.get("ctx_area_fk100", ""), b.get("ctx_area_nk100", "")
            if not (fk or nk) or (fk, nk) in seen:
                raise KISError("잔고 연속조회 키 오류: 부분 잔고를 전체 잔고로 표시하지 않습니다.")
            seen.add((fk, nk))
            cont = "N"
        raise KISError("잔고 연속조회 한도 초과: 전체 잔고 조회를 중단했습니다.")

    def cash(self):
        b, _ = self.get("/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            "VTTC8908R" if IS_PAPER else "TTTC8908R", {
                "CANO": CANO, "ACNT_PRDT_CD": ACNT_PRDT_CD,
                "PDNO": CASH_REFERENCE_CODE, "ORD_UNPR": "0", "ORD_DVSN": "01",
                "CMA_EVLU_AMT_ICLD_YN": "N", "OVRS_ICLD_YN": "N"})
        return b.get("output", {})

    def history(self, code, start, end):
        frames, cursor = [], pd.Timestamp(end)
        start = pd.Timestamp(start)
        # 최대 100개 제한: 90 달력일씩 분할하면 거래일 기준 제한을 넘지 않음.
        while cursor >= start:
            lo = max(start, cursor - pd.Timedelta(days=89))
            b, _ = self.get("/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                "FHKST03010100", {"FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": lo.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": cursor.strftime("%Y%m%d"), "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0"})
            records = b.get("output2", [])
            if records:
                frames.append(pd.DataFrame(records))
            cursor = lo - pd.Timedelta(days=1)
        if not frames:
            raise KISError("일봉 데이터가 없습니다. 상장일·거래종목·API 제공 범위를 확인하세요.")
        d = pd.concat(frames, ignore_index=True).rename(columns={
            "stck_bsop_date": "date", "stck_oprc": "open", "stck_hgpr": "high",
            "stck_lwpr": "low", "stck_clpr": "close", "acml_vol": "volume"})
        required = ["date", "open", "high", "low", "close", "volume"]
        if not set(required).issubset(d.columns):
            raise KISError("일봉 응답 필드가 예상과 다릅니다.")
        d = d[required].copy()
        d["date"] = pd.to_datetime(d["date"], format="%Y%m%d", errors="coerce")
        for c in required[1:]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna().drop_duplicates("date").sort_values("date").set_index("date")
        d = d.loc[(d.index >= start) & (d.index <= pd.Timestamp(end))]
        return d.loc[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]


def holdings_table(rows):
    out = []
    for r in rows:
        qty = number(r.get("hldg_qty"))
        if not np.isfinite(qty):
            raise KISError("보유수량이 누락된 잔고가 있습니다. 다시 조회하세요.")
        if qty <= 0:
            continue
        out.append({"종목코드": str(r.get("pdno", "")), "종목명": r.get("prdt_name", ""),
            "보유수량": qty, "매입단가": number(r.get("pchs_avg_pric")),
            "평가손익": number(r.get("evlu_pfls_amt")), "수익률(%)": number(r.get("evlu_pfls_rt")),
            "현재가": number(r.get("prpr"))})
    return pd.DataFrame(out, columns=["종목코드", "종목명", "보유수량", "매입단가", "평가손익", "수익률(%)", "현재가"])


def features(d):
    c = d.close
    x = pd.DataFrame(index=d.index)
    for n in (1, 5, 20):
        x[f"r{n}"] = c.pct_change(n, fill_method=None)
    for n in (5, 20, 60):
        x[f"ma{n}_gap"] = c / c.rolling(n).mean() - 1
    x["vol20"] = x.r1.rolling(20).std()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    x["rsi14"] = (100 - 100 / (1 + rs)).where(down != 0, 100).where((up + down) != 0, 50) / 100
    x["range"] = (d.high - d.low) / c
    x["vol_ratio"] = d.volume / d.volume.rolling(20).mean().replace(0, np.nan)
    x["macd_gap"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    return x.replace([np.inf, -np.inf], np.nan)


def new_model():
    return RandomForestRegressor(n_estimators=160, max_depth=5, min_samples_leaf=15,
                                 max_features=0.8, random_state=42, n_jobs=-1)


def analyze(d, today, cost=0.003):
    """오늘 봉은 마감 이후에도 제외. 전일 확정봉 -> 다음 관측 거래일 수익률."""
    d = d.loc[d.index < pd.Timestamp(today)].copy()
    fallback = {"의견": "유지", "사유": "확정 일봉/학습 데이터 부족: 판단 유보"}
    if len(d) < 320:
        return fallback
    x = features(d)
    target = d.close.shift(-1) / d.close - 1
    valid = x.notna().all(axis=1) & target.notna()
    X, y = x.loc[valid, FEATURES], target.loc[valid]
    if len(X) < 250 or x.iloc[-1].isna().any():
        return fallback
    if (pd.Timestamp(today) - d.index[-1]).days > 7 or d.volume.iloc[-1] <= 0:
        return {"의견": "유지", "사유": "최근 데이터 지연/거래정지 가능성: 판단 유보"}
    preds, truths, bases, meanpreds = [], [], [], []
    # 시계열 3분할, 각 학습/검증 사이 1행 제거: 익일 레이블 경계 누출 방지.
    for train, test in TimeSeriesSplit(n_splits=3, gap=1).split(X):
        model = new_model().fit(X.iloc[train], y.iloc[train])
        preds.extend(model.predict(X.iloc[test]))
        truths.extend(y.iloc[test])
        bases.extend([float((y.iloc[train] > 0).mean() >= 0.5)] * len(test))
        meanpreds.extend([y.iloc[train].mean()] * len(test))
    pred, actual = np.asarray(preds), np.asarray(truths)
    accuracy = float(((pred > 0) == (actual > 0)).mean())
    baseline = float((np.asarray(bases, dtype=bool) == (actual > 0)).mean())
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    zero_rmse = np.sqrt(np.mean(actual ** 2))
    mean_rmse = np.sqrt(mean_squared_error(actual, meanpreds))
    qualified = accuracy >= max(0.52, baseline + 0.02) and rmse < min(zero_rmse, mean_rmse)
    model = new_model().fit(X, y)
    forecast = float(model.predict(x.iloc[[-1]][FEATURES])[0])
    threshold = max(cost, float(x.vol20.iloc[-1]) * 0.25)
    signal = "유지"
    reason = "예상 변동이 거래비용/변동성 기준 미만"
    if not qualified:
        reason = "시계열 검증이 기준모형 대비 우수하지 않음: 판단 유보"
    elif forecast > threshold and x.ma20_gap.iloc[-1] > 0:
        signal, reason = "매수", "검증 통과 + 예상 상승폭이 기준 초과 + 20일선 위"
    elif forecast < -threshold and x.ma20_gap.iloc[-1] < 0:
        signal, reason = "매도", "검증 통과 + 예상 하락폭이 기준 초과 + 20일선 아래"
    elif abs(forecast) > threshold:
        reason = "예측 방향과 20일 이동평균 추세가 불일치"
    importance = pd.DataFrame({"특성": FEATURES, "중요도": model.feature_importances_}).sort_values("중요도", ascending=False)
    return {"의견": signal, "사유": reason, "기준일": str(d.index[-1].date()),
        "예측수익률(%)": forecast * 100, "방향정확도(%)": accuracy * 100,
        "다수방향 기준(%)": baseline * 100, "MAE(%p)": mae * 100,
        "RMSE(%p)": rmse * 100, "0수익률 RMSE(%p)": zero_rmse * 100,
        "평균수익률 RMSE(%p)": mean_rmse * 100, "판단임계값(%)": threshold * 100,
        "학습표본": len(X), "검증표본": len(actual), "검증통과": qualified,
        "중요도": importance}


def chart(d, name, today):
    cutoff = pd.Timestamp(today) - pd.DateOffset(months=6)
    a = d.copy()
    a["MA20"] = a.close.rolling(20).mean()
    a["MA60"] = a.close.rolling(60).mean()
    a = a.loc[a.index >= cutoff]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.76, 0.24])
    fig.add_trace(go.Candlestick(x=a.index, open=a.open, high=a.high, low=a.low, close=a.close,
        name="일봉: 상승 파랑 / 하락 빨강", increasing_line_color="#1751D0",
        decreasing_line_color="#C1232C"), row=1, col=1)
    for col, color in [("MA20", "#E2661E"), ("MA60", "#0E9E8F")]:
        fig.add_trace(go.Scatter(x=a.index, y=a[col], name=col, line=dict(color=color, width=1.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=a.index, y=a.volume, name="거래량", marker_color="#1751D0", showlegend=False), row=2, col=1)
    fig.update_layout(height=530, title=f"{name} · 최근 6개월 수정주가 일봉 (KRX)",
        template="plotly_white", paper_bgcolor="#FDFDFF", plot_bgcolor="#FDFDFF",
        font=dict(color="#1E2433"), hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.12), margin=dict(l=30, r=20, t=100, b=25))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], gridcolor="#E2E6F0")
    fig.update_yaxes(gridcolor="#E2E6F0")
    fig.update_yaxes(title_text="원", row=1, col=1)
    fig.update_yaxes(title_text="주", rangemode="tozero", row=2, col=1)
    return fig, a


def main():
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption("국내주식 · KRX 일봉 · 계좌 조회 전용 · 자동 주문 없음")
    st.warning("AI 신호는 실험적 참고 의견입니다. 수익을 보장하지 않으며 재무·뉴스·개인별 위험성향은 반영하지 않습니다.")
    if APP_KEY == "YOUR_APP_KEY" or APP_SECRET == "YOUR_APP_SECRET":
        st.info("app_jwlee.py 상단에 APP_KEY, APP_SECRET, CANO, ACNT_PRDT_CD를 입력한 뒤 다시 실행하세요.")
        st.stop()
    if not (CANO.isdigit() and len(CANO) == 8 and ACNT_PRDT_CD.isdigit() and len(ACNT_PRDT_CD) == 2):
        st.error("계좌번호는 앞 8자리 CANO와 뒤 2자리 ACNT_PRDT_CD로 구분하세요.")
        st.stop()
    identity = hashlib.sha256((BASE_URL + APP_KEY + APP_SECRET + CANO + ACNT_PRDT_CD).encode()).hexdigest()
    if st.session_state.get("identity") != identity:
        st.session_state.clear()
        st.session_state.identity = identity
        st.session_state.client = KISClient()
    client = st.session_state.client
    st.sidebar.header("조회 설정")
    st.sidebar.write("모의투자" if IS_PAPER else "실전계좌 · 조회 전용")
    st.sidebar.caption(f"계좌: ****{CANO[-4:]}-{ACNT_PRDT_CD}")
    cost = st.sidebar.number_input("매매비용·안전마진 가정(%)", min_value=0.05, max_value=5.0, value=0.30, step=0.05) / 100
    refresh = st.sidebar.button("계좌·시세 새로고침", type="primary")
    st.sidebar.caption("최초 로딩은 보유종목 수에 따라 수분 소요됩니다. 자동 갱신하지 않습니다.")
    if refresh:
        for key in ["snapshot", "histories", "analyses"]:
            st.session_state.pop(key, None)
    if "snapshot" not in st.session_state:
        try:
            with st.spinner("계좌 잔고 조회 중..."):
                begun = now()
                rows, summary = client.balance()
                table = holdings_table(rows)
                cash, cash_error = {}, ""
                try:
                    cash = client.cash()
                except KISError as e:
                    cash_error = str(e)
                st.session_state.snapshot = (table, summary, cash, cash_error, begun, now())
        except KISError as e:
            st.error(str(e))
            st.stop()
    table, summary, cash, cash_error, begun, ended = st.session_state.snapshot
    st.caption(f"API 조회 일시(KST): {begun:%Y-%m-%d %H:%M:%S} ~ {ended:%H:%M:%S} | 화면 일시: {now():%Y-%m-%d %H:%M:%S}")
    a, b, c = st.columns(3)
    a.metric("예수금", won(summary.get("dnca_tot_amt")))
    b.metric("주문가능현금", won(cash.get("ord_psbl_cash")))
    c.metric("미수 없는 매수가능금액", won(cash.get("nrcvb_buy_amt")))
    st.caption(f"주문가능금액: 기준종목 {CASH_REFERENCE_CODE}, 시장가, CMA·해외 평가금액 제외. 종목·미체결 주문·증거금 조건에 따라 달라질 수 있습니다. 예수금은 출금가능액이 아닙니다.")
    if cash_error:
        st.warning("주문가능현금만 조회 실패: " + cash_error)
    st.subheader("보유종목 잔고")
    st.dataframe(table.style.format({"보유수량": "{:,.0f}", "매입단가": "{:,.2f}", "평가손익": "{:,.0f}",
        "수익률(%)": "{:.2f}", "현재가": "{:,.0f}"}, na_rep="조회 불가"), use_container_width=True, hide_index=True)
    st.caption("금액 단위: 원. 현재가는 잔고 API 조회 시점의 평가가격이며 실시간 스트리밍 체결가가 아닙니다. 평가손익/수익률은 증권사 응답 그대로 표시합니다.")
    st.download_button("잔고 CSV 저장", table.to_csv(index=False).encode("utf-8-sig"), "holdings.csv", "text/csv")
    if table.empty:
        st.info("보유수량이 0보다 큰 국내주식이 없습니다.")
        return
    histories = st.session_state.setdefault("histories", {})
    analyses = st.session_state.setdefault("analyses", {})
    today = now().date()
    start = (pd.Timestamp(today) - pd.DateOffset(years=TRAIN_YEARS)).date()
    st.subheader("종목별 6개월 일봉 및 AI 의견")
    st.info("AI는 오늘의 미확정 봉을 사용하지 않습니다. 오늘 실행하더라도 직전 확정 거래일 종가를 기준으로 다음 관측 거래일 수익률을 예측합니다. 주말·휴일에는 오늘의 거래 신호가 아닙니다.")
    opinions = []
    for _, row in table.iterrows():
        code, name = row["종목코드"], row["종목명"]
        st.markdown(f"### {name} ({code})")
        try:
            hkey = (code, str(today))
            if hkey not in histories:
                with st.spinner(f"{name} · 최대 3년 일봉 수집 중..."):
                    histories[hkey] = (client.history(code, start, today), now())
            d, fetched = histories[hkey]
            if d.empty:
                raise KISError("유효한 일봉이 없습니다.")
            fig, visible = chart(d, name, today)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"일봉 조회: {fetched:%Y-%m-%d %H:%M:%S} KST | 최종 봉: {d.index[-1]:%Y-%m-%d} | 당일 봉이 있으면 차트에는 표시하되 장중에는 변동 가능")
            with st.expander("일봉 원자료 보기"):
                st.dataframe(visible, use_container_width=True)
            key = (hkey, cost)
            if key not in analyses:
                with st.spinner(f"{name} · Random Forest 시계열 학습/검증..."):
                    analyses[key] = analyze(d, today, cost)
            result = analyses[key]
            st.markdown(f"**AI 의견: {result['의견']}** — {result['사유']}")
            details = {k: v for k, v in result.items() if k not in ("중요도", "사유", "의견")}
            if details:
                st.dataframe(pd.DataFrame([details]), hide_index=True, use_container_width=True)
                with st.expander("특성 중요도 · 해석 주의"):
                    st.dataframe(result["중요도"], hide_index=True)
                    st.caption("불순도 기반 특성 중요도이며 인과관계나 수익 기여도는 아닙니다. 예측수익률은 확률이 아닙니다. 검증은 예측 오차 평가이지 비용 반영 전략 수익률 검증이 아닙니다.")
            opinions.append({"종목코드": code, "종목명": name, **{k: v for k, v in result.items() if k != "중요도"}})
        except (KISError, ValueError, KeyError) as e:
            st.warning(f"{name}: {e}")
            opinions.append({"종목코드": code, "종목명": name, "의견": "유지", "사유": "데이터/분석 오류: 판단 유보"})
    st.subheader("AI 의견 요약")
    result_table = pd.DataFrame(opinions)
    st.dataframe(result_table, hide_index=True, use_container_width=True)
    st.download_button("AI 의견 CSV 저장", result_table.to_csv(index=False).encode("utf-8-sig"), "ai_opinions.csv", "text/csv")
    st.caption("표시된 계좌의 명의는 사용자가 입력한 인증정보로 결정됩니다. 제목의 이름은 명의 확인을 의미하지 않습니다.")


if __name__ == "__main__":
    main()
