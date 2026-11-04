"""User-facing privacy limits and educational-use notice."""
import streamlit as st

DISCLAIMER = (
    "Made by Merouane Hamdani. Licensed under the MIT License. "
    "For educational purposes only. Provided AS IS, without warranty. "
    "Not validated for operational, safety, regulatory or commercial decisions; "
    "independently verify all inputs, assumptions and results."
)

PRIVACY = (
    "Uploads, entered parameters and results are processed in the app server's memory "
    "and retained in your active session so the app can work. They are not placed in "
    "shared application caches, automatically saved to a database or written to files "
    "by the app. Temporary numerical workspaces exist only during calculations. "
    "Save project and export buttons create downloads only when you choose to save them; "
    "download payloads are temporarily held in server memory. "
    "Use Clear session data to remove this session's app and upload-widget references. "
    "This is not a guarantee of immediate secure erasure from memory. "
    "When hosted, uploaded data leaves your device and reaches the hosting server. "
    "Hosting/platform logs, backups, infrastructure retention and browser/download storage "
    "are outside the app's control. Do not upload confidential or personal data without "
    "your organization's approval; use an approved local deployment for sensitive work."
)


def render():
    st.info("Data privacy: session memory only; no shared user-data cache or automatic file/database saving.")
    st.warning(DISCLAIMER)
    with st.expander("Data privacy, storage and downloads"):
        st.markdown(PRIVACY)
