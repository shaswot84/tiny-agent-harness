"""Context window visualizer component for TinyAgent Streamlit Workbench."""

import json
import streamlit as st


def analyze_context_window(messages: list[dict], max_context_tokens: int = 8192) -> dict:
    """Analyze messages in the context window and compute token statistics and segment breakdowns."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    blocks = []
    category_tokens = {
        "system": 0,
        "history_user": 0,
        "history_assistant": 0,
        "tool_observation": 0,
        "current_query": 0,
    }

    for idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        tool_calls = msg.get("tool_calls")

        # Estimate tokens (~3.8 chars per token + structural overhead)
        token_count = max(1, int(len(content) / 3.8))
        if tool_calls:
            token_count += max(1, int(len(json.dumps(tool_calls)) / 3.8))

        # Categorize
        if role == "system":
            cat = "system"
            label = "System Prompt & Rules"
            badge_color = "#a371f7"
        elif role == "tool" or (role == "user" and content.startswith("OBSERVATION:")):
            cat = "tool_observation"
            label = "Tool Observation"
            badge_color = "#3fb950"
        elif role == "assistant":
            cat = "history_assistant"
            label = "Assistant Response" if not tool_calls else "Assistant (Tool Call Action)"
            badge_color = "#d29922" if tool_calls else "#58a6ff"
        else:
            if idx == len(messages) - 1 or (idx == len(messages) - 2 and messages[-1].get("role") == "assistant"):
                cat = "current_query"
                label = "Current User Query"
                badge_color = "#f0883e"
            else:
                cat = "history_user"
                label = "User Message History"
                badge_color = "#58a6ff"

        category_tokens[cat] += token_count
        blocks.append({
            "index": idx + 1,
            "role": role,
            "category": cat,
            "label": label,
            "badge_color": badge_color,
            "content": content,
            "tool_calls": tool_calls,
            "tokens": token_count,
            "chars": len(content),
        })

    total_tokens = sum(b["tokens"] for b in blocks)
    pct_used = min(100.0, (total_tokens / max_context_tokens) * 100) if max_context_tokens > 0 else 0.0

    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "max_context_tokens": max_context_tokens,
        "pct_used": pct_used,
        "blocks": blocks,
        "category_tokens": category_tokens,
    }


def render_context_window_visualizer(agent, snapshots: list[dict] | None = None) -> None:
    """Render the interactive context window visualizer at the bottom of the Streamlit app.
    
    Args:
        agent: The active TinyAgent instance.
        snapshots: Optional list of historical context snapshots from session state.
    """
    if snapshots is None:
        snapshots = st.session_state.get("context_snapshots", [])

    st.divider()
    st.markdown("### 🪟 LLM Context Window Visualizer")
    st.caption("Inspect live token distribution, prompt structures, memory layers, and historical context states fed into the LLM.")

    # Controls row
    col_snap, col_limit, col_filter = st.columns([2, 1.5, 2.5])

    with col_snap:
        snapshot_choices = ["Latest (Live Memory)"]
        if snapshots:
            for s in snapshots:
                snapshot_choices.append(f"Turn #{s['turn']} ({s['timestamp']}) - {s['query'][:25]}...")
        chosen_snapshot = st.selectbox("Context Snapshot", snapshot_choices, index=0)

    with col_limit:
        window_options = [8192, 16384, 32768, 64000, 128000, 1000000]
        selected_limit = st.selectbox(
            "Window Budget (Tokens)",
            window_options,
            index=window_options.index(32768) if 32768 in window_options else 2,
            format_func=lambda x: f"{x:,} tokens"
        )

    with col_filter:
        category_filter = st.selectbox(
            "Filter Category",
            ["All Blocks", "System Prompt & Rules", "Tool Observations", "User Messages", "Assistant Responses"],
            index=0
        )

    # Determine messages to inspect
    if chosen_snapshot == "Latest (Live Memory)" or not snapshots:
        active_messages = agent.memory.get_messages() if agent and agent.memory else []
    else:
        # Extract turn number
        turn_id = int(chosen_snapshot.split("#")[1].split(" ")[0])
        match = next((s for s in snapshots if s["turn"] == turn_id), None)
        active_messages = match["messages"] if match else (agent.memory.get_messages() if agent and agent.memory else [])

    ctx_data = analyze_context_window(active_messages, max_context_tokens=selected_limit)

    # Top-level metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Est. Total Tokens", f"{ctx_data['total_tokens']:,}", help="Approximated at ~3.8 chars/token including payload metadata.")
    m2.metric("Window Usage", f"{ctx_data['pct_used']:.2f}%", f"of {selected_limit:,}")
    m3.metric("Context Blocks", f"{len(ctx_data['blocks'])} msgs")
    m4.metric("Total Characters", f"{ctx_data['total_chars']:,}")

    # Color legend mapping
    legend_map = {
        "system": ("#a371f7", "System & Rules"),
        "current_query": ("#f0883e", "Current Query"),
        "history_user": ("#58a6ff", "User History"),
        "history_assistant": ("#388bfd", "Assistant Responses"),
        "tool_observation": ("#3fb950", "Tool Observations"),
    }

    # Proportional Visual Context Bar
    if ctx_data["total_tokens"] > 0:
        bar_segments = []
        for cat_key, cat_tokens in ctx_data["category_tokens"].items():
            if cat_tokens > 0:
                width_pct = (cat_tokens / selected_limit) * 100
                color, label = legend_map.get(cat_key, ("#8b949e", cat_key))
                bar_segments.append(
                    f'<div style="width: {max(width_pct, 1.0):.2f}%; background-color: {color}; height: 16px;" '
                    f'title="{label}: {cat_tokens:,} tokens ({width_pct:.2f}%)"></div>'
                )
        
        # Remaining free context space
        free_pct = max(0.0, 100.0 - ctx_data["pct_used"])
        if free_pct > 0:
            bar_segments.append(
                f'<div style="width: {free_pct:.2f}%; background-color: #21262d; height: 16px;" '
                f'title="Available Window Budget: {free_pct:.2f}%"></div>'
            )

        st.markdown(
            f'<div style="display: flex; width: 100%; border-radius: 8px; overflow: hidden; border: 1px solid #30363d; margin-top: 8px; margin-bottom: 8px;">'
            f'{"".join(bar_segments)}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Legend Chips
        legend_html = []
        for cat_key, (color, label) in legend_map.items():
            t_count = ctx_data["category_tokens"].get(cat_key, 0)
            legend_html.append(
                f'<span style="display: inline-flex; align-items: center; margin-right: 15px; font-size: 0.8rem; color: #8b949e;">'
                f'<span style="display: inline-block; width: 10px; height: 10px; background-color: {color}; border-radius: 50%; margin-right: 5px;"></span>'
                f'<strong style="color: #c9d1d9; margin-right: 4px;">{label}:</strong> {t_count:,} tok'
                f'</span>'
            )
        st.markdown(f'<div style="margin-bottom: 16px;">{"".join(legend_html)}</div>', unsafe_allow_html=True)

    # Filtered Blocks View
    filter_role_map = {
        "System Prompt & Rules": ["system"],
        "Tool Observations": ["tool_observation"],
        "User Messages": ["history_user", "current_query"],
        "Assistant Responses": ["history_assistant"],
    }

    filtered_blocks = ctx_data["blocks"]
    if category_filter in filter_role_map:
        allowed = filter_role_map[category_filter]
        filtered_blocks = [b for b in filtered_blocks if b["category"] in allowed]

    st.markdown(f"**Context Blocks ({len(filtered_blocks)}):** Click any block below to inspect its prompt payload.")

    if not filtered_blocks:
        st.info("No blocks match the selected filter.")
    else:
        for b in filtered_blocks:
            pct_block = (b["tokens"] / max(ctx_data["total_tokens"], 1)) * 100
            snippet = b["content"][:60].replace("\n", " ") + ("..." if len(b["content"]) > 60 else "")
            
            exp_title = f"#{b['index']} [{b['role'].upper()}] - {b['label']} | ~{b['tokens']:,} tokens ({pct_block:.1f}%) — \"{snippet}\""
            
            with st.expander(exp_title, expanded=(b["category"] == "current_query")):
                col_b1, col_b2, col_b3 = st.columns([1.5, 1.5, 3])
                col_b1.markdown(f"**Role:** `{b['role']}`")
                col_b2.markdown(f"**Token Weight:** `{b['tokens']:,}` tokens")
                col_b3.markdown(f"**Character Length:** `{b['chars']:,}` chars")

                # Deconstruct System Prompt if applicable
                if b["role"] == "system" and "# ReAct (Reason and Act)" in b["content"]:
                    sys_subtab1, sys_subtab2, sys_subtab3 = st.tabs(["Full System Prompt", "ReAct Framework Rules", "Tool Definitions"])
                    with sys_subtab1:
                        st.code(b["content"], language="markdown")
                    with sys_subtab2:
                        react_part = b["content"].split("# ReAct (Reason and Act)")[1].split("# Tools")[0] if "# Tools" in b["content"] else b["content"]
                        st.code("# ReAct (Reason and Act)" + react_part, language="markdown")
                    with sys_subtab3:
                        tool_part = b["content"].split("# Tools")[1] if "# Tools" in b["content"] else "(No tools section)"
                        st.code("# Tools" + tool_part, language="markdown")
                else:
                    st.code(b["content"], language="markdown")

                if b["tool_calls"]:
                    st.markdown("**Tool Calls:**")
                    st.json(b["tool_calls"])
