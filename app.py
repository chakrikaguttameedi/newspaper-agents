import streamlit as st
import feedparser
import requests
import time
from bs4 import BeautifulSoup
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📰 Newspaper Research Agents",
    page_icon="📰",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #0f0f0f;
    color: #f0ece2;
}
.main { background-color: #0f0f0f; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 12px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 { font-size: 3rem; color: #f0ece2; margin: 0; letter-spacing: -1px; }
.hero p { color: #a09080; font-size: 1.1rem; margin-top: 0.5rem; }
.accent { color: #e94560; }

.agent-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 4px solid;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.agent-card.sports        { border-left-color: #00b4d8; }
.agent-card.politics      { border-left-color: #e94560; }
.agent-card.entertainment { border-left-color: #f4a261; }
.agent-card.technology    { border-left-color: #06d6a0; }
.agent-card.summary       { border-left-color: #9b5de5; border-left-width: 6px; }

.agent-title { font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem; }
.agent-content { font-size: 0.95rem; line-height: 1.7; color: #c8c0b0; white-space: pre-wrap; }

.badge { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px; }
.badge-done    { background: #06d6a0; color: #000; }
.badge-running { background: #f4a261; color: #000; }
.badge-wait    { background: #2a2a2a; color: #888; }

[data-testid="stSidebar"] { background: #111 !important; border-right: 1px solid #2a2a2a; }

.stButton > button {
    background: linear-gradient(135deg, #e94560, #c1121f) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
}
.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0ece2 !important;
    border-radius: 8px !important;
}
hr { border-color: #2a2a2a !important; }
.stProgress > div > div { background: #e94560 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
class NewsState(TypedDict):
    newspaper_url: str
    raw_articles: str
    sports_analysis: str
    politics_analysis: str
    entertainment_analysis: str
    technology_analysis: str
    summary: str

# ─────────────────────────────────────────────
# SCRAPER
# ─────────────────────────────────────────────
def scrape_newspaper(state: NewsState) -> NewsState:
    url = state["newspaper_url"]
    try:
        feed = feedparser.parse(url)
        articles = []
        if feed.entries:
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                articles.append(f"TITLE: {title}\nSUMMARY: {summary}\n")
            raw_text = "\n---\n".join(articles)
        else:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            headlines = soup.find_all(["h1", "h2", "h3"], limit=20)
            raw_text = "\n".join([h.get_text(strip=True) for h in headlines])
    except Exception as e:
        raw_text = f"Error fetching content: {str(e)}"
    return {**state, "raw_articles": raw_text}

# ─────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────
def make_agent(llm, system_prompt, output_key, delay=10):
    def agent(state: NewsState) -> NewsState:
        time.sleep(delay)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Here are today's newspaper articles:\n\n{state['raw_articles']}")
        ]
        response = llm.invoke(messages)
        return {**state, output_key: response.content}
    return agent

def make_summary_agent(llm, delay=10):
    def agent(state: NewsState) -> NewsState:
        time.sleep(delay)
        messages = [
            SystemMessage(content="""You are a Chief News Editor.
Create a concise, well-structured DAILY NEWS BRIEFING from the analyses provided.
Format it clearly with sections, key highlights, and an overall summary.
Make it professional and easy to read."""),
            HumanMessage(content=f"""
⚽ SPORTS: {state['sports_analysis']}
🏛️ POLITICS: {state['politics_analysis']}
🎬 ENTERTAINMENT: {state['entertainment_analysis']}
💻 TECHNOLOGY: {state['technology_analysis']}
Create a unified daily news briefing.""")
        ]
        response = llm.invoke(messages)
        return {**state, "summary": response.content}
    return agent

# ─────────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────────
def build_graph(llm):
    workflow = StateGraph(NewsState)

    workflow.add_node("scraper", scrape_newspaper)
    workflow.add_node("sports", make_agent(llm,
        """You are a Sports News Analyst.
        Extract and analyze ONLY sports-related news. Provide:
        1. Key sports stories found
        2. Teams/players mentioned
        3. Match results or upcoming events
        4. Your analysis of the sports coverage
        If no sports news found, state that clearly.""",
        "sports_analysis"))
    workflow.add_node("politics", make_agent(llm,
        """You are a Political News Analyst.
        Extract and analyze ONLY politics-related news. Provide:
        1. Key political stories and events
        2. Politicians, parties, or governments mentioned
        3. Policy changes or decisions
        4. Your analysis of the political landscape
        If no political news found, state that clearly.""",
        "politics_analysis"))
    workflow.add_node("entertainment", make_agent(llm,
        """You are an Entertainment News Analyst.
        Extract and analyze ONLY entertainment-related news. Provide:
        1. Celebrity news
        2. Movies, TV shows, or music mentioned
        3. Awards, events, or launches
        4. Your analysis of entertainment coverage
        If no entertainment news found, state that clearly.""",
        "entertainment_analysis"))
    workflow.add_node("technology", make_agent(llm,
        """You are a Technology News Analyst.
        Extract and analyze ONLY technology-related news. Provide:
        1. Key tech stories (AI, gadgets, companies)
        2. Companies or products mentioned
        3. Innovations or breakthroughs
        4. Your analysis of the tech news landscape
        If no tech news found, state that clearly.""",
        "technology_analysis"))
    workflow.add_node("summary", make_summary_agent(llm))

    workflow.set_entry_point("scraper")
    workflow.add_edge("scraper", "sports")
    workflow.add_edge("sports", "politics")
    workflow.add_edge("politics", "entertainment")
    workflow.add_edge("entertainment", "technology")
    workflow.add_edge("technology", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()

# ─────────────────────────────────────────────
# UI — HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📰 Newspaper Research <span class="accent">Agents</span></h1>
    <p>Powered by LangGraph · Gemini 2.5 Flash · Multi-Agent Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Enter your Gemini API key..."
    )

    st.markdown("### 🗞️ Newspaper RSS Feed")
    preset_urls = {
        "BBC News (World)": "https://feeds.bbci.co.uk/news/rss.xml",
        "BBC Sport": "https://feeds.bbci.co.uk/sport/rss.xml",
        "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
        "Custom URL": "custom"
    }

    selected = st.selectbox("Choose a newspaper:", list(preset_urls.keys()))

    if preset_urls[selected] == "custom":
        newspaper_url = st.text_input("Enter RSS URL:", placeholder="https://example.com/rss.xml")
    else:
        newspaper_url = preset_urls[selected]
        st.code(newspaper_url, language=None)

    st.markdown("---")
    st.markdown("### 🤖 Active Agents")
    st.markdown("⚽ **Sports** Agent")
    st.markdown("🏛️ **Politics** Agent")
    st.markdown("🎬 **Entertainment** Agent")
    st.markdown("💻 **Technology** Agent")
    st.markdown("📋 **Summary** Editor")
    st.markdown("---")
    st.info("🧠 Model: **gemini-2.5-flash**")
    run_btn = st.button("🚀 Run Agents", use_container_width=True)

# ── MAIN CONTENT ──
if run_btn:
    if not api_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
        st.stop()
    if not newspaper_url:
        st.error("⚠️ Please select or enter a newspaper URL.")
        st.stop()

    # ✅ CORRECT model string: gemini-2.5-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    app = build_graph(llm)

    st.markdown("## 📡 Live Agent Pipeline")
    progress_bar = st.progress(0)

    agents_info = [
        ("🔍 Scraper",             "Fetching articles from RSS feed..."),
        ("⚽ Sports Agent",         "Analyzing sports news..."),
        ("🏛️ Politics Agent",       "Analyzing political news..."),
        ("🎬 Entertainment Agent",  "Analyzing entertainment news..."),
        ("💻 Technology Agent",     "Analyzing technology news..."),
        ("📋 Summary Editor",       "Creating daily briefing..."),
    ]

    status_placeholders = []
    cols = st.columns(3)
    for i, (name, desc) in enumerate(agents_info):
        with cols[i % 3]:
            ph = st.empty()
            ph.markdown(f"""
            <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;
                        padding:0.8rem 1rem;margin-bottom:0.5rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:600;color:#f0ece2;">{name}</div>
                <div style="font-size:0.78rem;color:#666;margin-top:4px;">{desc}</div>
                <span class="badge badge-wait">WAITING</span>
            </div>
            """, unsafe_allow_html=True)
            status_placeholders.append(ph)

    st.markdown("---")

    def update_status(idx, name, desc, status):
        color = {"RUNNING": "#f4a261", "DONE": "#06d6a0", "WAITING": "#555"}.get(status, "#555")
        badge_class = {"RUNNING": "badge-running", "DONE": "badge-done", "WAITING": "badge-wait"}.get(status, "badge-wait")
        status_placeholders[idx].markdown(f"""
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-left:3px solid {color};
                    border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem;text-align:center;">
            <div style="font-size:1.1rem;font-weight:600;color:#f0ece2;">{name}</div>
            <div style="font-size:0.78rem;color:#999;margin-top:4px;">{desc}</div>
            <span class="badge {badge_class}">{status}</span>
        </div>
        """, unsafe_allow_html=True)

    node_map = {
        "scraper": 0, "sports": 1, "politics": 2,
        "entertainment": 3, "technology": 4, "summary": 5
    }

    initial_state = {
        "newspaper_url": newspaper_url,
        "raw_articles": "",
        "sports_analysis": "",
        "politics_analysis": "",
        "entertainment_analysis": "",
        "technology_analysis": "",
        "summary": ""
    }

    try:
        update_status(0, "🔍 Scraper", "Fetching articles...", "RUNNING")
        progress_bar.progress(5)

        result = {}

        for event in app.stream(initial_state):
            node = list(event.keys())[0]
            result.update(event[node])

            idx = node_map.get(node, 0)
            update_status(idx, agents_info[idx][0], "✅ Complete!", "DONE")
            progress_bar.progress(int((idx + 1) / 6 * 100))

            if idx + 1 < len(agents_info):
                update_status(idx + 1, agents_info[idx + 1][0], agents_info[idx + 1][1], "RUNNING")

        progress_bar.progress(100)
        st.success("✅ All agents completed successfully!")

        # ── RESULTS TABS ──
        st.markdown("## 📊 Analysis Results")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚽ Sports", "🏛️ Politics", "🎬 Entertainment", "💻 Technology", "📋 Full Briefing"
        ])

        with tab1:
            st.markdown(f"""
            <div class="agent-card sports">
                <div class="agent-title" style="color:#00b4d8;">⚽ Sports Analysis</div>
                <div class="agent-content">{result.get('sports_analysis', 'No data')}</div>
            </div>""", unsafe_allow_html=True)

        with tab2:
            st.markdown(f"""
            <div class="agent-card politics">
                <div class="agent-title" style="color:#e94560;">🏛️ Politics Analysis</div>
                <div class="agent-content">{result.get('politics_analysis', 'No data')}</div>
            </div>""", unsafe_allow_html=True)

        with tab3:
            st.markdown(f"""
            <div class="agent-card entertainment">
                <div class="agent-title" style="color:#f4a261;">🎬 Entertainment Analysis</div>
                <div class="agent-content">{result.get('entertainment_analysis', 'No data')}</div>
            </div>""", unsafe_allow_html=True)

        with tab4:
            st.markdown(f"""
            <div class="agent-card technology">
                <div class="agent-title" style="color:#06d6a0;">💻 Technology Analysis</div>
                <div class="agent-content">{result.get('technology_analysis', 'No data')}</div>
            </div>""", unsafe_allow_html=True)

        with tab5:
            st.markdown(f"""
            <div class="agent-card summary">
                <div class="agent-title" style="color:#9b5de5;">📋 Daily News Briefing</div>
                <div class="agent-content">{result.get('summary', 'No data')}</div>
            </div>""", unsafe_allow_html=True)

        with st.expander("🔍 View Raw Scraped Articles"):
            st.text(result.get("raw_articles", "No articles scraped"))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            st.warning("⏳ Quota exceeded! Wait 1-2 minutes and try again.")
        elif "404" in str(e) or "NOT_FOUND" in str(e):
            st.warning("⚠️ Model not found. Please check your API key supports Gemini 2.5 Flash.")
        else:
            st.info("💡 Check your API key and internet connection.")

else:
    # ── LANDING PAGE ──
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">🤖</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#888;margin-bottom:0.5rem;">
            Ready to Analyze
        </div>
        <div style="font-size:1rem;color:#444;">
            Enter your API key → Select a newspaper → Click <strong style="color:#e94560;">Run Agents</strong>
        </div>
    </div>
    <div style="display:flex;gap:1rem;justify-content:center;margin-top:2rem;flex-wrap:wrap;">
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-top:3px solid #00b4d8;border-radius:8px;padding:1.2rem 1.5rem;width:180px;text-align:center;">
            <div style="font-size:2rem;">⚽</div>
            <div style="font-weight:600;margin-top:0.3rem;color:#f0ece2;">Sports</div>
            <div style="font-size:0.8rem;color:#555;margin-top:4px;">Scores, teams, events</div>
        </div>
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-top:3px solid #e94560;border-radius:8px;padding:1.2rem 1.5rem;width:180px;text-align:center;">
            <div style="font-size:2rem;">🏛️</div>
            <div style="font-weight:600;margin-top:0.3rem;color:#f0ece2;">Politics</div>
            <div style="font-size:0.8rem;color:#555;margin-top:4px;">Policies, leaders, events</div>
        </div>
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-top:3px solid #f4a261;border-radius:8px;padding:1.2rem 1.5rem;width:180px;text-align:center;">
            <div style="font-size:2rem;">🎬</div>
            <div style="font-weight:600;margin-top:0.3rem;color:#f0ece2;">Entertainment</div>
            <div style="font-size:0.8rem;color:#555;margin-top:4px;">Celebs, movies, music</div>
        </div>
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-top:3px solid #06d6a0;border-radius:8px;padding:1.2rem 1.5rem;width:180px;text-align:center;">
            <div style="font-size:2rem;">💻</div>
            <div style="font-weight:600;margin-top:0.3rem;color:#f0ece2;">Technology</div>
            <div style="font-size:0.8rem;color:#555;margin-top:4px;">AI, gadgets, startups</div>
        </div>
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-top:3px solid #9b5de5;border-radius:8px;padding:1.2rem 1.5rem;width:180px;text-align:center;">
            <div style="font-size:2rem;">📋</div>
            <div style="font-weight:600;margin-top:0.3rem;color:#f0ece2;">Summary</div>
            <div style="font-size:0.8rem;color:#555;margin-top:4px;">Daily briefing report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
