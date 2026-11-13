from pathlib import Path
import re
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Student Wellbeing Assessment
# Educational demonstration using synthetic training data.
# Not a clinical or diagnostic assessment.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Your Wellbeing Check-in",
    page_icon="💛",
    layout="centered"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "3_ML_Model"
    / "Model_Outputs"
    / "logistic_regression.joblib"
)

LIKERT_OPTIONS = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]
LIKERT_MAP = {label: number for number, label in enumerate(LIKERT_OPTIONS, start=1)}

QUESTIONS = {
    "Emotional wellbeing": [
        ("Q01", "I feel happy and positive about my life.", False),
        ("Q02", "I feel worried or anxious about things happening around me.", True),
        ("Q03", "I sometimes feel alone, even when I am surrounded by people.", True),
    ],
    "Academic wellbeing": [
        ("Q04", "I feel overwhelmed by schoolwork, homework or exams.", True),
        ("Q05", "I worry about disappointing others with my academic performance.", True),
        ("Q06", "I have enough time for my studies, hobbies and myself.", False),
        ("Q07", "I feel motivated and able to concentrate on my studies.", False),
    ],
    "Family and relationships": [
        ("Q08", "I feel comfortable talking to my family about things that bother me.", False),
        ("Q09", "I feel that my family understands me and my feelings.", False),
        ("Q10", "I feel pressured by the expectations my family has of me.", True),
    ],
    "Friendships and social wellbeing": [
        ("Q11", "I feel accepted and included by my friends or classmates.", False),
        ("Q12", "I sometimes do things I don't want to do just to fit in with others.", True),
        ("Q13", "I feel safe and respected when I interact with other students.", False),
    ],
    "Physical wellbeing": [
        ("Q14", "I wake up feeling refreshed and well-rested.", False),
        ("Q15", "I have enough energy to enjoy my daily activities.", False),
    ],
    "Self-image and confidence": [
        ("Q16", "I feel confident about who I am.", False),
        ("Q17", "I compare myself with others and sometimes feel less confident.", True),
    ],
    "Digital wellbeing": [
        ("Q18", "I feel good about myself when I use social media.", False),
        ("Q19", "I find it difficult to put my phone away, even when I want to.", True),
        ("Q20", "I make enough time for activities that don't involve screens.", False),
    ],
}

BAND_DESCRIPTIONS = {
    "Generally positive indicators": (
        "Your responses show several positive indicators across the areas in this check-in. "
        "Keep making time for the people and activities that help you feel good."
    ),
    "Some areas may need a check-in": (
        "Your responses suggest that some parts of everyday life may deserve a little more "
        "attention. Consider talking with someone you trust about what has been on your mind."
    ),
    "More support may be helpful": (
        "Your responses suggest that reaching out for support could be helpful. You do not "
        "have to work through difficult feelings alone. Consider speaking with a trusted "
        "adult, parent or guardian, teacher, or school counsellor."
    ),
}

# Friendly visual style: playful but still appropriate for a school project.
st.markdown(
    """
    <style>
    .stApp { background: #fffaf5; }
    .block-container { max-width: 790px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #29324b; }
    .hero {
        background: linear-gradient(120deg, #fce7f3 0%, #e0f2fe 100%);
        border: 1px solid #f4d8e8;
        border-radius: 22px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: #78536d; font-size: .82rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
    }
    .question-card {
        background: white; border: 1px solid #eee3eb;
        border-radius: 20px; padding: 1.35rem; margin: .8rem 0 1rem 0;
    }
    div.stButton > button {
        border-radius: 13px; min-height: 2.8rem; font-weight: 650;
    }
    div[data-testid="stProgressBar"] > div > div { background-color: #a78bfa; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state keeps the three screens together during this browser session.
defaults = {
    "page": 1,
    "full_name": "",
    "age": 15,
    "school_class": "Class 9",
    "gender": "Prefer not to say",
    "sharing": "Keep my results private",
    "parent_email": "",
    "teacher_email": "",
    "question_index": 0,
    "answers": {},
    "show_results": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page_number):
    st.session_state.page = page_number
    st.rerun()


def valid_email(value):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def get_flat_questions():
    return [
        (dimension, question_id, statement, is_negative)
        for dimension, items in QUESTIONS.items()
        for question_id, statement, is_negative in items
    ]


flat_questions = get_flat_questions()
total_questions = len(flat_questions)

st.title("💛 Your Wellbeing Check-in")
st.caption("A little space to check in with yourself. No grades. No judgement.")

st.progress(st.session_state.page / 3)
st.caption(f"PAGE {st.session_state.page} OF 3")

# ---------------- PAGE 1: ABOUT YOU ----------------
if st.session_state.page == 1:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Hey there 👋</div>
          <h2>Let's make this about you.</h2>
          <p>Life is more than school, marks and expectations. Your feelings,
          friendships, family, confidence, health and everyday experiences matter too.</p>
          <p>Take a few minutes to reflect on how things have been going lately.
          There are no right or wrong answers. Just be honest with yourself. 💛</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("First, a little about you")
    st.write("What should we call you?")

    full_name = st.text_input(
        "Your full name",
        value=st.session_state.full_name,
        placeholder="Enter your full name",
    )
    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox(
            "Your age",
            [14, 15, 16],
            index=[14, 15, 16].index(st.session_state.age),
        )
    with col2:
        school_class = st.selectbox(
            "Your class",
            ["Class 9", "Class 10"],
            index=["Class 9", "Class 10"].index(st.session_state.school_class),
        )
    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Girl", "Boy", "Another identity"],
        index=["Prefer not to say", "Girl", "Boy", "Another identity"].index(
            st.session_state.gender
        ),
    )

    st.info(
        "This is a student project demonstration, not a medical or psychological test. "
        "The model uses synthetic training data and cannot diagnose or predict a health condition."
    )
    with st.expander("A quick note before you start"):
        st.write(
            "Your answers are used to create the check-in summary on the next screen. "
            "This demo does not save your answers to a database or send emails. "
            "If something is worrying you or you do not feel safe, please speak with a trusted adult."
        )

    if st.button("Continue →", type="primary", use_container_width=True):
        if not full_name.strip():
            st.error("Please enter your full name to continue.")
        else:
            st.session_state.full_name = full_name.strip()
            st.session_state.age = age
            st.session_state.school_class = school_class
            st.session_state.gender = gender
            go_to(2)

# ---------------- PAGE 2: SHARING PREFERENCES ----------------
elif st.session_state.page == 2:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Your choice, always</div>
          <h2>Who would you like by your side?</h2>
          <p>Sometimes sharing how we're feeling can make things easier.
          You get to choose whether you'd like to share your results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Choose your sharing preference")
    sharing_options = [
        "Keep my results private",
        "Share with my parent or guardian",
        "Share with my teacher or school counsellor",
        "Share with both",
    ]
    st.radio(
        "Would you like to share your wellbeing results with someone you trust?",
        sharing_options,
        key="sharing",
    )

    sharing = st.session_state.sharing
    if sharing in ("Share with my parent or guardian", "Share with both"):
        st.text_input(
            "Parent/guardian email address",
            key="parent_email",
            placeholder="name@example.com",
        )
    if sharing in ("Share with my teacher or school counsellor", "Share with both"):
        st.text_input(
            "Teacher/counsellor email address",
            key="teacher_email",
            placeholder="name@school.org",
        )

    if sharing == "Keep my results private":
        st.success("Your results will stay private in this demo.")
    else:
        st.warning(
            "This is a prototype only. It will not send an email or share results. "
            "The email fields demonstrate how the sharing option would work. "
            "Please use this only with appropriate school review, safeguarding, "
            "parent/guardian consent and student assent."
        )

    left, right = st.columns(2)
    with left:
        if st.button("← Back", use_container_width=True):
            go_to(1)
    with right:
        if st.button("Let's begin →", type="primary", use_container_width=True):
            errors = []
            if sharing in ("Share with my parent or guardian", "Share with both"):
                if not valid_email(st.session_state.parent_email):
                    errors.append("Enter a valid parent/guardian email address.")
            if sharing in ("Share with my teacher or school counsellor", "Share with both"):
                if not valid_email(st.session_state.teacher_email):
                    errors.append("Enter a valid teacher/counsellor email address.")
            if errors:
                for error in errors:
                    st.error(error)
            else:
                go_to(3)

# ---------------- PAGE 3: QUESTIONNAIRE + RESULTS ----------------
elif st.session_state.page == 3:
    if not st.session_state.show_results:
        st.markdown(
            f"""
            <div class="hero">
              <div class="eyebrow">Your check-in starts here ✨</div>
              <h2>You've got this, {st.session_state.full_name.split()[0]}.</h2>
              <p>Think about how you've been feeling lately. Pick the answer that feels
              closest to you. We'll take it one statement at a time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_index = st.session_state.question_index
        dimension, question_id, statement, is_negative = flat_questions[current_index]
        st.progress((current_index + 1) / total_questions)
        st.caption(f"QUESTION {current_index + 1} OF {total_questions}  ·  {dimension}")

        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        st.subheader(statement)
        st.caption("Choose one response to keep going.")
        st.markdown("</div>", unsafe_allow_html=True)

        # A fresh key for each question preserves each answer when navigating back.
        selected = st.radio(
            "Your response",
            LIKERT_OPTIONS,
            key=f"response_{question_id}",
            label_visibility="collapsed",
        )

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button(
                "← Previous",
                use_container_width=True,
                disabled=(current_index == 0),
            ):
                st.session_state.answers[question_id] = selected
                st.session_state.question_index -= 1
                st.rerun()
        with col_next:
            button_label = "See my results →" if current_index == total_questions - 1 else "Next →"
            if st.button(button_label, type="primary", use_container_width=True):
                st.session_state.answers[question_id] = selected
                if current_index < total_questions - 1:
                    st.session_state.question_index += 1
                    st.rerun()
                else:
                    st.session_state.show_results = True
                    st.rerun()

    else:
        # Calculate dimension averages, reverse-coding negatively worded items.
        dimension_scores = {}
        for dimension, items in QUESTIONS.items():
            scores = []
            for question_id, _statement, is_negative in items:
                response = st.session_state.answers[question_id]
                score = LIKERT_MAP[response]
                if is_negative:
                    score = 6 - score
                scores.append(score)
            dimension_scores[dimension] = sum(scores) / len(scores)

        overall_score = sum(dimension_scores.values()) / len(dimension_scores)

        st.markdown(
            f"""
            <div class="hero">
              <div class="eyebrow">That's a wrap, {st.session_state.full_name.split()[0]} 💛</div>
              <h2>Thanks for checking in with yourself.</h2>
              <p>Here's a reflection of the answers you shared. Remember, this is a
              conversation starter, not a label or a judgement.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Your average reflection score", f"{overall_score:.2f} / 5")
        st.caption(
            "This is an illustrative average across the seven areas. "
            "It is not a clinical score or diagnosis."
        )

        score_df = pd.DataFrame(
            {
                "Wellbeing area": list(dimension_scores.keys()),
                "Average score": list(dimension_scores.values()),
            }
        ).set_index("Wellbeing area")
        st.bar_chart(score_df, x_label="Wellbeing area", y_label="Average score (1–5)")

        strongest = max(dimension_scores, key=dimension_scores.get)
        lowest = min(dimension_scores, key=dimension_scores.get)
        st.subheader("A couple of things to notice")
        st.success(f"An area with more positive responses: **{strongest}**")
        st.info(f"An area you may want to give a little attention: **{lowest}**")

        st.subheader("Your model's demonstration output")
        try:
            if not MODEL_PATH.exists():
                st.warning(
                    "The reflection summary is ready, but the model file was not found. "
                    "Check 3_ML_Model/Model_Outputs/logistic_regression.joblib."
                )
            else:
                model = joblib.load(MODEL_PATH)
                model_input = pd.DataFrame(
                    [[LIKERT_MAP[st.session_state.answers[f"Q{i:02d}"]]
                      for i in range(1, 21)]],
                    columns=[f"Q{i:02d}" for i in range(1, 21)],
                )
                predicted_band = model.predict(model_input)[0]
                st.markdown(f"### {predicted_band}")
                st.write(
                    BAND_DESCRIPTIONS.get(
                        predicted_band,
                        "This is an illustrative model output based on synthetic training data.",
                    )
                )
                st.caption(
                    "The model was trained on synthetic, rule-labelled data. "
                    "This category is for demonstrating a machine-learning workflow only; "
                    "it does not establish or predict a student's real-world wellbeing."
                )
        except Exception as exc:
            st.warning(f"The reflection summary is ready, but the model could not run: {exc}")

        st.subheader("A few gentle reminders")
        st.write(
            "- Make time for something you enjoy.\n"
            "- If something has been bothering you, consider talking to someone you trust.\n"
            "- Make room for rest, movement, friends and screen-free time.\n"
            "- Asking for support is a positive step, not a weakness."
        )

        st.caption(
            "Privacy note: this prototype does not save answers or send emails. "
            "Do not use it to collect real student information without appropriate "
            "school review, safeguarding arrangements, parent/guardian consent and student assent."
        )

        if st.button("Start a new check-in", use_container_width=True):
            for key in list(defaults.keys()):
                if key in st.session_state:
                    del st.session_state[key]
            for key in list(st.session_state.keys()):
                if key.startswith("response_Q"):
                    del st.session_state[key]
            st.rerun()
