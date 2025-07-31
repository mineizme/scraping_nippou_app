import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import pandas as pd
import altair as alt
from serpapi import GoogleSearch
import sqlite3  # ← 追加：カラム追加用
from serpapi import GoogleSearch


# --- DBファイル名 ---
DB_FILE = "project_reports.db"

# --- DB破損時の初期化関数を追加 ---
def initialize_database():
    if os.path.exists(DB_FILE):
        try:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("SELECT name FROM sqlite_master LIMIT 1")
        except sqlite3.DatabaseError:
            print("⚠️ DBファイルが壊れているため削除して再作成します。")
            os.remove(DB_FILE)

    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                goal TEXT,
                estimate_weeks INTEGER,
                analysis TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY,
                project_id INTEGER,
                content TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                progress TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS progress_steps (
                id INTEGER PRIMARY KEY,
                project_id INTEGER,
                step_index INTEGER,
                checked BOOLEAN DEFAULT 0
            )
        """)
        conn.commit()

# ← ここで初期化実行
initialize_database()

# --- 必要なら analysis カラムを追加 ---
def ensure_analysis_column():
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(projects)")
        columns = [row[1] for row in cur.fetchall()]
        if "analysis" not in columns:
            cur.execute("ALTER TABLE projects ADD COLUMN analysis TEXT")
            conn.commit()

# --- 環境変数読み込み ---
load_dotenv("mine.env")
SERPAPI_API_KEY = "d849feab8005780b037956c5b80c9fbee2da30597c51302c0e9d930021945ba0"
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- DBファイル名 ---
DB_FILE = "project_reports.db"

# --- DB接続設定 ---
engine = create_engine(f"sqlite:///{DB_FILE}")
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# --- 必要なら analysis カラムを追加 ---
def ensure_analysis_column():
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(projects)")
        columns = [row[1] for row in cur.fetchall()]
        if "analysis" not in columns:
            cur.execute("ALTER TABLE projects ADD COLUMN analysis TEXT")
            conn.commit()

ensure_analysis_column()  # ← 呼び出し

# --- テーブル定義 ---
class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    goal = Column(Text)
    estimate_weeks = Column(Integer)
    analysis = Column(Text)  # ← 追加（DBにも手動追加済）
    reports = relationship("DailyReport", back_populates="project", cascade="all, delete-orphan")
    progress_steps = relationship("ProgressStep", back_populates="project", cascade="all, delete-orphan")

class DailyReport(Base):
    __tablename__ = "daily_reports"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    progress = Column(String, default="未対応")
    project = relationship("Project", back_populates="reports")

class ProgressStep(Base):
    __tablename__ = "progress_steps"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    step_index = Column(Integer)
    checked = Column(Boolean, default=False)
    project = relationship("Project", back_populates="progress_steps")

Base.metadata.create_all(engine)

# --- SerpAPI Google検索関数 ---
def serpapi_google_search(query, num_results=5):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    return [{"title": r.get("title"), "link": r.get("link")} for r in organic_results[:num_results]]

# --- GPTで目標から課題と工程を抽出 ---
def analyze_goal_with_gpt(goal_text):
    prompt = f"""
以下のプロジェクト目標に対して、重要な課題と実行すべき工程を3つずつ抽出してください。
フロントエンド、バックエンドで作業を分担するのでそれぞれの作業内容も抽出してください。
フォーマットはJSON形式で出力してください。

プロジェクト目標: {goal_text}

# 出力例:
{{
  "tasks": ["課題1", "課題2", "課題3"],
  "processes": ["工程1", "工程2", "工程3"]
  -----------------------------------------------
  "フロントエンド": ["工程1", "工程2", "工程3"]
  "バックエンド": ["工程1", "工程2", "工程3"]
}}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# --- UI 開始 ---
st.title("📝 日報課題抽出アプリ（ステップ進捗＋Google検索連携＋GPT目標分析）")

# --- プロジェクト選択 ---
project_list = session.query(Project).order_by(Project.name).all()
project_names = [p.name for p in project_list]

col1, col2 = st.columns([3, 1])
with col1:
    selected_project_name = st.selectbox("プロジェクトを選択", [""] + project_names)
with col2:
    if selected_project_name:
        if st.button("❌ 選択プロジェクトを削除"):
            proj_to_delete = session.query(Project).filter_by(name=selected_project_name).first()
            if proj_to_delete:
                session.delete(proj_to_delete)
                session.commit()
                st.success(f"プロジェクト「{selected_project_name}」を削除しました。")
                st.rerun 
                st.warning("ページをリロードしてください。")

# --- プロジェクト新規作成 ---
with st.expander("➕ プロジェクト新規作成"):
    new_name = st.text_input("プロジェクト名")
    new_goal = st.text_area("目標")
    new_weeks = st.number_input("目安週数", min_value=1, value=4)
    if st.button("プロジェクトを追加"):
        if new_name.strip() == "":
            st.warning("プロジェクト名は必須です。")
        else:
            existing = session.query(Project).filter_by(name=new_name.strip()).first()
            if existing:
                st.warning("同名のプロジェクトが既に存在します。")
            else:
                gpt_analysis = analyze_goal_with_gpt(new_goal.strip()) if new_goal.strip() else ""
                new_project = Project(
                    name=new_name.strip(),
                    goal=new_goal.strip(),
                    estimate_weeks=new_weeks,
                    analysis=gpt_analysis
                )
                session.add(new_project)
                session.commit()
                st.success("プロジェクトを追加しました（目標から課題と工程を抽出済）")
                st.warning("ページをリロードしてください。")

if selected_project_name:
    project = session.query(Project).filter_by(name=selected_project_name).first()

    if project:
        # 🔽 分析結果の表示
        st.subheader("🧠 GPTによる目標分析（課題と工程）")

        # 1. 編集可能なテキストエリア
        edited_analysis = st.text_area("抽出内容", value=project.analysis or "未分析", height=300)

        # 2. 保存ボタン
        if st.button("編集内容を保存"):
            project.analysis = edited_analysis
            session.commit()
            st.success("抽出内容を保存しました。ページを再読み込みしてください。")

        # 本日の作業記録入力欄はボタン外に配置（必ず定義される）
        st.subheader("■ 本日の作業記録")
        report_text = st.text_area(
            "日報入力欄",
            height=200,
            placeholder="例：今日は新しい見積もりシステムの導入準備を行ったが、仕様書が不完全で判断に迷う場面が多かった…"
        )

        if st.button("日報を送信し、課題を抽出"):
            if report_text.strip():
                prompt = f"""
以下は業務日報の一部です。この中から課題・懸念事項・障害・悩みを箇条書きで抽出してください。
業務内容: {report_text}
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                extracted = response.choices[0].message.content.strip()
                new_report = DailyReport(
                    project_id=project.id,
                    content=f"{report_text}\n\n[GPT抽出課題]\n{extracted}"
                )
                session.add(new_report)
                session.commit()
                st.success("日報を登録し、課題を抽出しました。ページを再読み込みしてください。")

        st.subheader("🔍 日報キーワード検索とGoogle検索")
        search_keyword = st.text_input("キーワードを入力してください")

        if search_keyword:
            filtered_reports = session.query(DailyReport).filter(
                DailyReport.project_id == project.id,
                DailyReport.content.ilike(f"%{search_keyword}%")
            ).all()
            st.write(f"🔎 キーワード「{search_keyword}」を含む日報：{len(filtered_reports)}件")
            for r in filtered_reports:
                date_str = (r.created_at + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
                st.markdown(f"- {date_str} : {r.content[:150]}...")

            st.subheader("🌐 Google検索結果")
            try:
                results = serpapi_google_search(search_keyword, num_results=5)
                if results:
                    for res in results:
                        st.markdown(f"[{res['title']}]({res['link']})")
                else:
                    st.write("検索結果がありません。")
            except Exception as e:
                st.error(f"Google検索中にエラーが発生しました: {e}")

        st.subheader("📋 登録された日報一覧")
        reports = session.query(DailyReport).filter_by(project_id=project.id).order_by(DailyReport.created_at.desc()).all()
        for report in reports:
            date_str = (report.created_at + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
            with st.expander(f"🗓️ {date_str} の日報", expanded=False):
                st.text_area("内容", value=report.content, height=200, disabled=False)

        st.subheader("📊 進捗ステップチェック")

        step_names = [
            "要件定義",
            "基本設計",
            "詳細設計",
            "フロントエンド実装",
            "バックエンド実装",
            "テスト",
            "リリース準備",
            "本番リリース"
        ]
        step_weights = [1, 1, 3, 5, 5, 3, 1, 1]

        db_steps = session.query(ProgressStep).filter_by(project_id=project.id).order_by(ProgressStep.step_index).all()
        if not db_steps or len(db_steps) != len(step_names):
            for i in range(len(step_names)):
                ps = ProgressStep(project_id=project.id, step_index=i, checked=False)
                session.add(ps)
            session.commit()
            db_steps = session.query(ProgressStep).filter_by(project_id=project.id).order_by(ProgressStep.step_index).all()

        updated_any = False
        checked_steps = []

        # カラム分割：左にチェックリスト、右にグラフ
        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.markdown("###### ✅ ステップ進捗チェック")
            for ps in db_steps:
                checked = st.checkbox(step_names[ps.step_index], value=ps.checked, key=f"step_{ps.step_index}")
                if checked != ps.checked:
                    ps.checked = checked
                    updated_any = True
                checked_steps.append(checked)

        if updated_any:
            session.commit()
            st.warning("進捗が更新されました。ページを再読み込みしてください。")

        total_weight = sum(step_weights)
        achieved_weight = sum(w for w, c in zip(step_weights, checked_steps) if c)
        progress_percent = round(achieved_weight / total_weight * 100, 1) if total_weight > 0 else 0

        estimate_weeks = project.estimate_weeks or 1
        ideal_points = list(range(estimate_weeks + 1))
        ideal_progress = [min(100, int((w / estimate_weeks) * 100)) for w in ideal_points]

        ideal_df = pd.DataFrame({
            "週数": ideal_points,
            "達成度": ideal_progress,
            "タイプ": ["理想進捗"] * len(ideal_points)
        })

        # --- 修正：現状進捗を初日報日からの週数で表示 ---
        first_report = session.query(DailyReport).filter_by(project_id=project.id)\
                        .order_by(DailyReport.created_at.asc()).first()
        if first_report:
            start_date = first_report.created_at
            today = datetime.utcnow()
            weeks_since_start = (today - start_date).days / 7
            actual_df = pd.DataFrame({
                "週数": [0, round(weeks_since_start, 1)],
                "達成度": [0, progress_percent],
                "タイプ": ["現状"] * 2
            })
        else:
            actual_df = pd.DataFrame(columns=["週数", "達成度", "タイプ"])

        combined_df = pd.concat([ideal_df, actual_df])

        with right_col:
            chart = alt.Chart(combined_df).mark_line(point=True).encode(
                x=alt.X("週数:Q", title="週数（週）"),
                y=alt.Y("達成度:Q", title="達成度（％）", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("タイプ:N",
                                scale=alt.Scale(domain=["理想進捗", "現状"], range=["orange", "black"]),
                                legend=alt.Legend(title="進捗タイプ"))
            ).properties(
                width=400,
                height=500,
                title=f"案件「{selected_project_name}」の理想進捗と現状の比較"
            )

            st.altair_chart(chart, use_container_width=True)
            st.markdown(f"### 現状進捗達成度: **{progress_percent} %**")

    else:
        st.warning("選択されたプロジェクトは存在しません。")

else:
    st.info("まだ日報が登録されていません。")
    if selected_project_name:
        st.markdown(f"### 案件「{selected_project_name}」", unsafe_allow_html=True)

session.close()
