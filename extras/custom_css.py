import base64
import os

def get_base64_image(image_path):
    """Reads a local image and converts it to a base64 string for HTML."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at: {image_path}") 
        return "" 
        
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_custom_css() -> str:
    """Returns all custom CSS as a string"""
    return """
<style>
    /* Main app background - black */
    .stApp {
        background-color: #000000;
    }
    
    .stAppHeader {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }

    

    /* Multiselect input wrapper - pink border */
    div[data-testid="stMultiSelect"] > div {
        border-color: #F25081 !important;
    }

    /* Multiselect inner div - pink border */
    div[data-testid="stMultiSelect"] > div > div {
        border-color: #F25081 !important;
        background-color: #1a1a1a !important;
    }

    /* Multiselect actual input component - pink border */
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] {
        border-color: #F25081 !important;
        background-color: #1a1a1a !important;
    }

    /* Selected tags/pills - pink background, white text */
    span[data-baseweb="tag"] {
        background-color: #F25081 !important;
        color: #ffffff !important;
    }

    /* Dropdown menu container - dark background */
    div[role="listbox"] {
        background-color: #1a1a1a !important;
        border-color: #F25081 !important;
    }

    /* Dropdown menu items - white text on dark */
    div[role="listbox"] li {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* Dropdown menu selected items - pink background */
    div[role="listbox"] li[aria-selected="true"] {
        background-color: #F25081 !important;
        color: #ffffff !important;
    }

    /* Placeholder text - gray */
    div[data-testid="stMultiSelect"] input::placeholder {
        color: #888888 !important;
    }
    .stSpinner > div > div {
        border-top-color: #F25081;
        color:#F25081 !important;
    }

    .stChatInputContainer > div > div > input {
        border-color: #F25081 !important;
        background-color: #1a1a1a;
        color: #ffffff !important;
    }
    .stChatInputContainer > div > div > input:focus {
        border-color: #F25081 !important;
        box-shadow: 0 0 0 0.2rem rgba(242, 80, 129, 0.25) !important;
    }
</style>
"""

def apply_form_button_styles():
    """Returns form-specific button CSS"""
    return """
    <style>
    div[data-testid="stForm"] button {
        background-color: #7c52f4 !important;
        border-color: #7c52f4 !important;
    }
    
    div[data-testid="stForm"] button p {
        color: #ffffff !important;
    }

    div[data-testid="stForm"] button:hover {
        background-color: #a53254 !important;
        border-color: #a53254 !important;
    }

    div[data-testid="stForm"] button:hover p {
        color: #ffffff !important;
    }

    div[data-testid="stForm"] button:focus, 
    div[data-testid="stForm"] button:active {
        background-color: #7c52f4 !important;
        border-color: #7c52f4 !important;
        color: #ffffff !important;
    }
    </style>"""


