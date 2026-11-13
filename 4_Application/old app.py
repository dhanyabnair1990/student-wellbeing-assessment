from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Student Wellbeing Assessment
# Educational demonstration using synthetic training data.
# Not a clinical or diagnostic assessment.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Student Wellbeing Check-in",
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

st.title("💛 Student Wellbeing Check-in")
st.caption("A friendly reflection on how things have been going lately")

st.info(
    "This is an educational project demonstration, not a medical or psychological test. "
    "It cannot diagnose a condition or replace advice from a qualified professional. "
    "The machine-learning model was trained on synthetic data, so its category is illustrative only."
)

with st.expander("Before you begin", expanded=True):
    st.write(
        "There are no right or wrong answers. Choose the response that feels closest to "
        "how you have been feeling recently. You may skip any statement. Your answers are "
        "not saved or sent by this demo."
    )
    st.write(
        "If something is worrying you or you do not feel safe, please speak with a trusted "
        "adult or a qualified support person."
    )

st.subheader("1. A little about you")
age = st.selectbox("Your age", [14, 15, 16], index=1)
school_class = st.selectbox("Your class", ["Class 9", "Class 10"])
gender = st.selectbox(
    "Gender (optional)",
    ["Prefer not to say", "Girl", "Boy", "Another identity"],
    index=0,
)
st.caption("These details are for the demonstration only. They are not used by the model.")

st.subheader("2. Sharing preferences")
sharing = st.radio(
    "Who would you like to share your results with?",
    [
        "Keep my results private",
        "Parent or guardian",
        "Teacher or school counsellor",
        "Parent/guardian and teacher/counsellor",
    ],
    index=0,
)
if sharing != "Keep my results private":
    st.warning(
        "This is only a demonstration. The app does not send emails, notify anyone, "
        "or share your answers. Please do not enter anyone's email address here."
    )

st.subheader("3. Your wellbeing check-in")
st.write("Select one response for each statement, or choose “Skip this statement”.")

answers = {}
for dimension, items in QUESTIONS.items():
    with st.expander(dimension, expanded=True):
        for question_id, statement, is_negative in items:
            response = st.radio(
                statement,
                ["Skip this statement"] + LIKERT_OPTIONS,
                index=0,
                key=question_id,
                horizontal=False,
            )
            answers[question_id] = response

if st.button("View my check-in results", type="primary", use_container_width=True):
    answered = {
        qid: response
        for qid, response in answers.items()
        if response != "Skip this statement"
    }

    if not answered:
        st.error("Please answer at least one statement to see your reflection.")
        st.stop()

    # Convert answers to 1–5 values and reverse-code negatively worded items
    # for dimension scores. Skipped items are excluded from these averages.
    dimension_scores = {}
    for dimension, items in QUESTIONS.items():
        scores = []
        for question_id, _statement, is_negative in items:
            response = answers[question_id]
            if response == "Skip this statement":
                continue
            score = LIKERT_MAP[response]
            if is_negative:
                score = 6 - score
            scores.append(score)

        if scores:
            dimension_scores[dimension] = sum(scores) / len(scores)

    st.success("Your check-in is ready.")
    st.subheader("Your wellbeing overview")

    if dimension_scores:
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        st.metric("Average reflection score", f"{overall_score:.2f} / 5")
        st.caption(
            "This is a simple average of the dimensions you answered, with negatively "
            "worded statements reverse-scored. It is not a clinical score."
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
        st.write(f"**An area with positive responses:** {strongest}")
        st.write(f"**An area you may wish to reflect on:** {lowest}")

    # The trained model requires all 20 input features. Do not silently
    # impute skipped answers; show a model result only for complete responses.
    if len(answered) == 20:
        if not MODEL_PATH.exists():
            st.warning(
                "Your reflection scores are available, but the trained model file "
                "could not be found. Check that logistic_regression.joblib is inside "
                "3_ML_Model/Model_Outputs."
            )
        else:
            try:
                model = joblib.load(MODEL_PATH)
                model_input = pd.DataFrame(
                    [[LIKERT_MAP[answers[f"Q{i:02d}"]] for i in range(1, 21)]],
                    columns=[f"Q{i:02d}" for i in range(1, 21)],
                )
                predicted_band = model.predict(model_input)[0]
                st.subheader("Illustrative model category")
                st.markdown(f"### {predicted_band}")
                st.write(BAND_DESCRIPTIONS.get(
                    predicted_band,
                    "This is an illustrative model output based on synthetic training data."
                ))
                st.caption(
                    "This category is generated by a model trained on synthetic, "
                    "rule-labelled data. It should not be treated as a real assessment "
                    "of your wellbeing."
                )
            except Exception as exc:
                st.warning(
                    "Your reflection scores are available, but the model could not be "
                    f"loaded or run. Technical detail: {exc}"
                )
    else:
        st.info(
            f"You skipped {20 - len(answered)} statement(s). Your reflection scores are "
            "shown above, but the machine-learning category is not displayed because "
            "the model requires all 20 responses. Skipped answers have not been guessed."
        )

    st.subheader("A few gentle reminders")
    st.write(
        "- Make time for an activity you enjoy.\n"
        "- Try talking to someone you trust if something has been bothering you.\n"
        "- Take breaks from screens and make room for rest, movement and hobbies.\n"
        "- Asking for support is a positive step, not a weakness."
    )

    st.divider()
    st.caption(
        "Privacy note: This demo does not save responses or send them to anyone. "
        "Do not use it to collect real student information without appropriate school "
        "review, safeguarding arrangements, and parent/guardian consent and student assent."
    )
