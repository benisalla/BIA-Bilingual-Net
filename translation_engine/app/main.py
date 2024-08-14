import streamlit as st
import time
from translation_engine.app.HTMLS import SMILE_SPINNER
from translation_engine.app.utils import (
    add_background_image,
    init_session_state,
)


def main():
    # Initialize session state variables
    init_session_state(st)

    # Set background image
    bk_img = "./translation_engine/src/img/image_3.jpg"
    add_background_image(bk_img, st)

    # ==============================[ Main Page ]================================
    st.markdown(
        """
        <div class="title-container" style="
            background-color: #2C3E50;
            border-radius: 20px;
            text-align: center;
            box-shadow: rgb(251 192 95 / 73%) 0px -1px 20px 0px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
            ">
            <h1 class="title" style="
                color: rgb(255 255 255);
                font-size: 2.2em;
                font-family: "Helvetica Neue", sans-serif;
                font-weight: 700;
                letter-spacing: 2px;
                text-shadow: rgb(132 126 126) 5px -6px 5px;
                ">
                🌐 BIA Lingua-Net Translator 🌐
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    in_text = st.text_area(
        "",
        height=20,
        max_chars=500,
        placeholder="Type your text here...",
        help="Enter the text you want to translate.",
    )

    # ==============================[ Sidebar ]================================
    st.sidebar.markdown(
        """
    <div class="sidebar-title-container">
        <h2 class="sidebar-title">Lingua-Net Settings</h2>
    </div>""",
        unsafe_allow_html=True,
    )

    # Translation direction
    tran_drt = st.sidebar.radio(
        "Choose Translation Direction:",
        ["English to Darija", "Darija to English"],
        horizontal=True,
    )

    # Model settings
    st.sidebar.markdown(
        '<div class="section-header">Adjust Model Settings</div>',
        unsafe_allow_html=True,
    )
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=10.0, value=0.0, step=0.01
    )

    top_k = st.sidebar.slider(
        "Top-K Sampling:", min_value=10, max_value=200, value=50, step=1
    )
    top_p = st.sidebar.slider(
        "Top-P Sampling:", min_value=0.0, max_value=1.0, value=0.95, step=0.01
    )

    beam_size = st.sidebar.slider(
        "Beam Size:", min_value=1, max_value=10, value=4, step=1
    )
    len_norm_coeff = st.sidebar.slider(
        "Length Normalization Coefficient:",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
    )
    # text_drt = st.sidebar.radio(
    #     "Result Text Direction:", ["rtl", "ltr"], index=0, horizontal=False
    # )
    max_beam_fork = st.sidebar.slider(
        "Max Beam Fork:", min_value=16, max_value=128, value=128, step=1
    )

    # Load the appropriate model
    model = (
        st.session_state.en_dr_model
        if tran_drt == "English to Darija"
        else st.session_state.dr_en_model
    )

    st.sidebar.markdown('<hr class="hr-style">', unsafe_allow_html=True)

    if st.button(f"Translate: {tran_drt}") and in_text.strip() != "":
        spinner_holder = st.empty()
        spinner_holder.markdown(SMILE_SPINNER, unsafe_allow_html=True)

        best_hypo, all_hypos = model.translate(
            sx=in_text.strip(),
            tokenizer=st.session_state.tokenizer,
            temperature=temperature,
            beam_size=beam_size,
            len_norm_coeff=len_norm_coeff,
            top_k=top_k,
            top_p=top_p,
            is_ltr=False,
            max_beam_fork=max_beam_fork,
        )

        st.session_state.best_hypo = best_hypo
        st.session_state.all_hypos = all_hypos

        spinner_holder.empty()

    if "best_hypo" in st.session_state and st.session_state.best_hypo:
        if st.session_state.best_hypo:
            st.markdown(
                f"""
                <div style='background-color: rgb(38 39 48); color: white !important; padding: 4px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ecb252;'>
                    <div style='direction: rtl; text-align: left; font-size: 1.2em; color: rgb(240, 177, 73); padding: 10px;'>
                        {st.session_state.best_hypo} 
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Show All Hypotheses"):
                sorted_hypos = sorted(
                    st.session_state.all_hypos, key=lambda x: x["score"], reverse=True
                )
                for hypo in sorted_hypos:
                    st.markdown(
                        f"""
                        <div style='direction: rtl; display: flex; justify-content: space-between; align-items: center; background-color: #34495E; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                            <span style='color: rgb(233 172 72);'>{'[Score: {:.2f}]'.format(hypo['score'])}</span>
                            <span style='color: #ECF0F1; font-size: 1.1em; text-align: right;'>{hypo['hypothesis']}</span>
                        </div>
                        """,
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


if __name__ == "__main__":
    main()


# poetry run streamlit run translation_engine/app/main.py
