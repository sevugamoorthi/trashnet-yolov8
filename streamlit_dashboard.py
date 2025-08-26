# streamlit_dashboard.py
import streamlit as st
import time
import json
from pathlib import Path
from collections import Counter
import pandas as pd

ipc_path = Path("last_detections.json")

st.set_page_config(layout="wide", page_title="TrashNet Live Dashboard")
st.title("TrashNet - Live Dashboard (optional)")

placeholder = st.empty()
while True:
    if ipc_path.exists():
        data = json.loads(ipc_path.read_text())
        detections = data.get("detections", [])
        ts = data.get("timestamp", 0)
        frame = data.get("frame", 0)
        names = [d["name"] for d in detections]
        counts = Counter(names)
        df = pd.DataFrame(detections)
        with placeholder.container():
            st.subheader(f"Frame {frame}   timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}")
            st.write("Counts:", dict(counts))
            if not df.empty:
                st.dataframe(df[["name","conf","bbox","cx","cy"]])
            else:
                st.info("No detections in last frame.")
    else:
        st.info("Waiting for detections... Run detect_realtime.py first.")
    time.sleep(1.0)
