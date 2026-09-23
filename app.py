"""Streamlit frontend for TinyAgent with interactive chat, real-time trajectory traces, and debugging tools."""

import json
import os
import streamlit as st

from TinyAgent import (
    LLM,
    TinyAgent,
    Memory,
    TrimmingMemory,
    SummarizationMemory,
    RAGMemory,
    Tools,
    NativeTools,
    ReAct,
    toolbox,
)

# Page configuration
st.set_page_config(
    page_title="TinyAgent Workbench",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean developer experience
st.markdown(
    """
    <style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    /* Fixed viewport split: disable outer page bounce, let sub-containers scroll */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    /* Scrollable container styling */
    .scroll-container::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    .scroll-container::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    .scroll-container::-webkit-scrollbar-thumb:hover {
        background: #8b949e;
    }
    .step-card {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid #30363d;
        background-color: #161b22;
    }
    .step-badge {
        font-weight: 600;
        font-size: 0.85rem;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .badge-thought { background-color: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
    .badge-action { background-color: #d2992233; color: #e3b341; border: 1px solid #d29922; }
    .badge-obs { background-color: #23863633; color: #3fb950; border: 1px solid #238636; }
    .badge-answer { background-color: #a371f733; color: #bc8cff; border: 1px solid #a371f7; }
    </style>
    """,
    unsafe_allow_html=True,
)



def initialize_session():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str, "run_index": int}
    if "runs_history" not in st.session_state:
        st.session_state.runs_history = []  # list of serialized trajectory runs
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_config_hash" not in st.session_state:
        st.session_state.agent_config_hash = None


def build_tools(tool_mode: str, selected_toolbox: list[str]) -> Tools:
    """Instantiate and populate the tool registry."""
    if tool_mode == "Native Function Calling":
        tool_reg = NativeTools()
    else:
        tool_reg = Tools()

    tool_mapping = {
        "multiply": toolbox.multiply,
        "add": toolbox.add,
        "subtract": toolbox.subtract,
        "divide": toolbox.divide,
        "power": toolbox.power,
        "execute_command": toolbox.execute_command,
        "final_answer": toolbox.final_answer,
    }

    for name in selected_toolbox:
        if name in tool_mapping:
            tool_reg.add_tool(name=name, func=tool_mapping[name])

    return tool_reg


def build_memory(memory_type: str, trim_k: int, llm: LLM) -> Memory:
    """Instantiate the selected memory store."""
    if memory_type == "Trimming Memory":
        return TrimmingMemory(recent=trim_k)
    elif memory_type == "Summarization Memory":
        return SummarizationMemory(llm=llm)
    elif memory_type == "RAG Memory":
        sample_docs = [
            "TinyAgent is an educational agent framework created in Python.",
            "Supported memory models include Base, Trimming, Summarization, and RAG.",
            "ReAct planner uses THOUGHT, ACTION, and final_answer syntax.",
        ]
        return RAGMemory(embedding_model=None, documents=sample_docs, top_k=2)
    return Memory()


def get_or_create_agent(
    provider: str,
    model: str,
    custom_base_url: str,
    api_key: str,
    tool_mode: str,
    selected_tools: list[str],
    memory_type: str,
    trim_k: int,
    use_react: bool,
    max_steps: int,
) -> TinyAgent:
    """Create or return cached TinyAgent instance."""
    config_key = f"{provider}:{model}:{custom_base_url}:{tool_mode}:{selected_tools}:{memory_type}:{trim_k}:{use_react}:{max_steps}"
    
    if st.session_state.agent is not None and st.session_state.agent_config_hash == config_key:
        return st.session_state.agent

    base_url = custom_base_url.strip() if custom_base_url.strip() else None
    key = api_key.strip() if api_key.strip() else None

    llm = LLM(
        model=model,
        provider=provider if not base_url else None,
        base_url=base_url,
        api_key=key,
    )

    tools = build_tools(tool_mode, selected_tools)
    memory = build_memory(memory_type, trim_k, llm)
    planner = ReAct(max_steps=max_steps) if use_react else None

    agent = TinyAgent(
        llm=llm,
        memory=memory,
        tools=tools,
        planner=planner,
        record_trajectory=True,
    )

    st.session_state.agent = agent
    st.session_state.agent_config_hash = config_key
    return agent


initialize_session()

# Sidebar: Controls & Configuration
with st.sidebar:
    st.title("⚙️ Agent Controls")
    
    with st.expander("🤖 LLM Provider & Model", expanded=True):
        provider = st.selectbox(
            "Provider",
            options=["ollama_cloud", "ollama", "groq", "openai", "openrouter", "deepseek", "gemini", "together"],
            index=0,
        )
        model = st.text_input("Model Name", value="gemma4:31b" if provider == "ollama_cloud" else "llama-3.3-70b-versatile")
        custom_base_url = st.text_input("Custom Base URL (optional)", placeholder="http://localhost:11434/v1")
        api_key = st.text_input("API Key (optional, falls back to env)", type="password")

    with st.expander("🛠️ Tools & Mode", expanded=True):
        tool_mode = st.radio("Tool Execution Mode", ["Prompt-based (JSON / ReAct)", "Native Function Calling"], index=0)
        selected_tools = st.multiselect(
            "Active Toolbox",
            options=["multiply", "add", "subtract", "divide", "power", "execute_command", "final_answer"],
            default=["multiply", "add", "execute_command", "final_answer"],
        )

    with st.expander("🧠 Memory & Planning", expanded=False):
        memory_type = st.selectbox("Memory Architecture", ["Summarization Memory", "Trimming Memory", "Base Memory"], index=0)
        trim_k = st.slider("Trimming Window Size", min_value=2, max_value=20, value=6)
        use_react = st.checkbox("Enable ReAct Planner", value=True)
        max_steps = st.slider("Max ReAct Steps", min_value=1, max_value=20, value=10)

    st.divider()
    if st.button("🧹 Clear Chat & Traces", use_container_width=True):
        st.session_state.messages = []
        st.session_state.runs_history = []
        st.session_state.agent = None
        st.session_state.agent_config_hash = None
        st.rerun()

# Build or refresh the agent
agent = get_or_create_agent(
    provider=provider,
    model=model,
    custom_base_url=custom_base_url,
    api_key=api_key,
    tool_mode=tool_mode,
    selected_tools=selected_tools,
    memory_type=memory_type,
    trim_k=trim_k,
    use_react=use_react,
    max_steps=max_steps,
)

# Header
st.subheader("🤖 TinyAgent Interactive Workbench")
st.caption(f"Connected to **{agent.llm.base_url}** (Model: `{agent.llm.model}`) | Planner: `{type(agent.planner).__name__ if agent.planner else 'None'}` | Tools: `{len(agent.tools.registry)} registered`")

# Main Layout: 2 columns (Chat Interface & Debug/Traces Viewer)
col_chat, col_debug = st.columns([1.1, 0.9], gap="large")

# Render Left Column: Chat Conversation
with col_chat:
    st.markdown("### 💬 Chat")
    chat_container = st.container(height=640)
    with chat_container:

        if not st.session_state.messages:
            st.info("👋 Send a message or query to start. Example:\n- *What is 42 * 19?*\n- *Run bash command 'ls -la'*")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "run_index" in msg and msg["run_index"] is not None:
                    st.caption(f"Traced in Run #{msg['run_index'] + 1}")

    user_query = st.chat_input("Ask TinyAgent anything...")
    if user_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query, "run_index": None})
        
        with col_chat:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)

        # Run agent
        with col_chat:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Agent is reasoning and executing tools..."):
                        try:
                            response_text = agent.run(user_query)
                        except Exception as e:
                            response_text = f"⚠️ Execution Error: {e}"
                    st.markdown(response_text)

        # Save trajectory run
        current_run_idx = None
        if agent.trajectory and agent.trajectory.runs:
            latest_run = agent.trajectory.runs[-1]
            st.session_state.runs_history.append(latest_run)
            current_run_idx = len(st.session_state.runs_history) - 1

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "run_index": current_run_idx,
        })
        st.rerun()

# Render Right Column: Traces, Step Debugger, & System State
with col_debug:
    st.markdown("### 🔍 Trace & Debug Center")
    
    debug_container = st.container(height=640)
    with debug_container:
        tabs = st.tabs(["📍 Execution Trace", "📦 Memory State", "🛠️ Tool Registry", "📋 Terminal Box"])

        with tabs[0]:
            if not st.session_state.runs_history:
                st.info("No execution traces yet. Send a prompt to record steps!")
            else:
                run_options = [f"Run #{i + 1}: {run['query'][:35]}" for i, run in enumerate(st.session_state.runs_history)]
                selected_run_str = st.selectbox("Select Trajectory Run", run_options, index=len(run_options) - 1)
                selected_idx = run_options.index(selected_run_str)
                target_run = st.session_state.runs_history[selected_idx]

                st.markdown(f"**Query:** `{target_run['query']}`")
                st.markdown(f"**Total Steps:** `{len(target_run['steps'])}`")
                
                for s_idx, step in enumerate(target_run["steps"], 1):
                    with st.expander(f"Step {s_idx}: {('Action: ' + str(step.action.get('tool') if isinstance(step.action, dict) else step.action)) if step.action else ('Answer' if step.answer else 'Thought')}", expanded=True):
                        if step.thought:
                            st.markdown('<span class="step-badge badge-thought">💭 Thought</span>', unsafe_allow_html=True)
                            st.info(step.thought)
                        
                        if step.action:
                            st.markdown('<span class="step-badge badge-action">🛠️ Action</span>', unsafe_allow_html=True)
                            st.json(step.action)

                        if step.observation is not None:
                            st.markdown('<span class="step-badge badge-obs">👁️ Observation</span>', unsafe_allow_html=True)
                            st.code(str(step.observation), language="text")

                        if step.answer:
                            st.markdown('<span class="step-badge badge-answer">💬 Final Answer</span>', unsafe_allow_html=True)
                            st.success(step.answer)

        with tabs[1]:
            st.markdown("**Current Conversation Memory Turns:**")
            if agent.memory:
                raw_msgs = agent.memory.get_messages()
                st.metric("Total Messages in Memory", len(raw_msgs))
                for i, m in enumerate(raw_msgs):
                    with st.expander(f"#{i + 1} Role: {m.get('role', 'unknown').upper()}", expanded=(i == 0 or i == len(raw_msgs) - 1)):
                        st.code(m.get("content", ""), language="markdown")
                        if m.get("tool_calls"):
                            st.caption("Native Tool Calls:")
                            st.json(m.get("tool_calls"))
            else:
                st.warning("No memory module attached.")

        with tabs[2]:
            st.markdown("**Registered Tools & Schemas:**")
            if agent.tools:
                st.metric("Total Registered Tools", len(agent.tools.registry))
                for t_name, t_meta in agent.tools.registry.items():
                    with st.expander(f"`{t_name}`", expanded=False):
                        st.write(f"**Description:** {t_meta.get('description', '')}")
                        if t_meta.get("schema"):
                            st.caption("OpenAI Function Schema:")
                            st.json(t_meta.get("schema"))
                
                st.markdown("**Active System Tool Prompt:**")
                st.code(agent.tools.prompt if agent.tools.prompt else "(Empty / Native Mode)", language="markdown")
            else:
                st.warning("No tools registered.")

        with tabs[3]:
            st.markdown("**Unicode Box Trajectory (Terminal Format):**")
            if agent.trajectory and agent.trajectory.runs:
                st.code(agent.trajectory.format_latest_run(width=68), language="text")
            else:
                st.info("No box formatted trajectory recorded yet.")

