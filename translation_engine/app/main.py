import streamlit as st
import torch
from translation_engine.app.HTMLS import SMILE_SPINNER
from translation_engine.app.utils import add_background_image, init_session_state, load_model, load_tokenizer

def main():
    # Initialize session state variables
    init_session_state(st)

    # Set background image
    bk_img = "./translation_engine/src/img/image_2.jpg"
    add_background_image(bk_img, st)
    
    # ==============================[ Main Page ]================================
    st.markdown('''
    <div class="title-container">
        <h1 class="title">🌐 BIA Lingua-Net Translator 🌐</h1>
    </div>''', unsafe_allow_html=True)
    
    # Input section
    st.markdown(
        '<div class="input-section"><p class="input-instructions">Enter text for translation:</p></div>',
        unsafe_allow_html=True,
    )
    in_text = st.text_area("", height=150)
    
    # ==============================[ Sidebar ]================================
    st.sidebar.markdown('''
    <div class="sidebar-title-container">
        <h2 class="sidebar-title">Translation Settings</h2>
    </div>''', unsafe_allow_html=True)

    # Translation direction
    tran_drt = st.sidebar.radio("Choose Translation Direction:", ['English to Darija', 'Darija to English'], horizontal=True)
    
    # Model settings
    st.sidebar.markdown('<div class="section-header">Adjust Model Settings</div>', unsafe_allow_html=True)
    temperature = st.sidebar.slider("Temperature:", min_value=0.1, max_value=1.0, value=1.0, step=0.01)
    top_k = st.sidebar.slider("Top-K Sampling:", min_value=1, max_value=6, value=1, step=1)
    beam_size = st.sidebar.slider("Beam Size:", min_value=1, max_value=10, value=4, step=1)
    len_norm_coeff = st.sidebar.slider("Length Normalization Coefficient:", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    text_drt = st.sidebar.radio("Result Text Direction:", ['rtl', 'ltr'], horizontal=True)
    max_beam_fork = st.sidebar.slider("Max Beam Fork:", min_value=16, max_value=128, value=128, step=1)

    # Load the appropriate model
    model = st.session_state.en_dr_model if tran_drt == "English to Darija" else st.session_state.dr_en_model

    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)

    if st.sidebar.button(f"Translate: {tran_drt}") and in_text.strip() != "":
        spinner_holder = st.empty()
        spinner_holder.markdown(SMILE_SPINNER, unsafe_allow_html=True)
        
        # Translation logic
        best_hypo, all_hypos = model.translate(
            in_text,
            st.session_state.tokenizer,
            temperature=temperature,
            beam_size=beam_size,
            len_norm_coeff=len_norm_coeff,
            is_ltr=(text_drt == 'ltr'),
            max_beam_fork=max_beam_fork
        )
        
        spinner_holder.empty()
        
        if best_hypo:                
            st.markdown(
                f"<div class='input-section'><p class='input-instructions'>Best Translation:</p><p style='direction:{text_drt}; font-weight:bold;'>{best_hypo}</p></div>",
                unsafe_allow_html=True,
            )

            with st.expander("Show All Hypotheses"):
                st.markdown("<div class='input-instructions'>Alternative Translations:</div>", unsafe_allow_html=True)
                for hypo in all_hypos:
                    st.markdown(
                        f"<p style='direction:{text_drt};'>{hypo['hypothesis']} <span style='color: gray;'>[Score: {hypo['score']:.2f}]</span></p>",
                        unsafe_allow_html=True,
                    )
        else:
            error_message = """
            <div class="error-component">
                <p><strong>Oops!</strong> It looks like there was a problem with the translation. Please try again.</p>
                <p class="help-text">You might want to adjust the settings or modify the input text for better results.</p>
                <p class="encouragement">We’re here to help you create meaningful translations, so don’t hesitate to try again!</p>
            </div>
            """
            st.markdown(error_message, unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="input-section">
            <p class="input-instructions">Type your text and click translate</p>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

    
#poetry run streamlit run translation_engine/app/main.py 