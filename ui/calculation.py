"""Shared, truthful progress display for iterative fitting."""
from __future__ import annotations

import streamlit as st


class FitProgress:
    def __init__(self, title):
        self.title = title
        self.bar = None

    def __enter__(self):
        st.markdown("""
<style>
@keyframes wtw-coffee { 50% { transform: translateY(-4px); } }
@media (prefers-reduced-motion: no-preference) {
  .wtw-coffee { display:inline-block; animation:wtw-coffee 1.5s ease-in-out infinite; }
}
</style>
<div role="status" style="padding:16px;border:1px solid #9bb0c8;border-radius:8px;background:#eef2ff;color:#051c2c">
<span class="wtw-coffee" aria-hidden="true">&#9749;</span>
<strong> Calculation in progress</strong><br>
The model is fitting. You can take a coffee while the solver works.
Progress counts completed fitting attempts, not elapsed time. Some attempts take longer.
</div>
""", unsafe_allow_html=True)
        self.bar = st.progress(0., text=f"{self.title}: preparing calculation...")
        return self.update

    def update(self, completed, total, message):
        fraction = min(.99, max(0., completed / total)) if total else 0.
        self.bar.progress(fraction, text=f"{self.title}: {message}")

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.bar.progress(1., text=f"{self.title}: calculation finished.")
        else:
            self.bar.empty()
        return False
