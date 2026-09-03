import os
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="StayPrice | Airbnb Price Estimate",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "airbnb_price_model.pkl")

MODEL_MAX_PRICE = 500

# Representative locations used internally by the model.
# Users never need to see or enter coordinates.
BOROUGH_COORDS = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}

LANDMARKS = {
    "Midtown": (40.7549, -73.9840),
    "Lower Manhattan": (40.7128, -74.0060),
    "Central Park": (40.7812, -73.9665),
}


# Common NYC neighbourhood coordinates.
# For any neighbourhood not listed here, the app safely falls back
# to the selected borough's representative location.
NEIGHBOURHOOD_COORDS = {
    "Chelsea": (40.7465, -74.0014),
    "Clinton Hill": (40.6895, -73.9650),
    "East Harlem": (40.7957, -73.9415),
    "East Village": (40.7265, -73.9815),
    "Financial District": (40.7075, -74.0113),
    "Flatiron District": (40.7397, -73.9903),
    "Gramercy": (40.7370, -73.9850),
    "Greenpoint": (40.7300, -73.9540),
    "Harlem": (40.8116, -73.9465),
    "Hell's Kitchen": (40.7640, -73.9910),
    "Kips Bay": (40.7420, -73.9800),
    "Little Italy": (40.7191, -73.9973),
    "Lower East Side": (40.7178, -73.9850),
    "Midtown": (40.7549, -73.9840),
    "Midtown East": (40.7585, -73.9710),
    "Midtown South": (40.7500, -73.9870),
    "Morningside Heights": (40.8100, -73.9625),
    "Murray Hill": (40.7487, -73.9787),
    "NoHo": (40.7233, -73.9920),
    "Nolita": (40.7223, -73.9947),
    "SoHo": (40.7233, -74.0000),
    "Stuyvesant Town": (40.7310, -73.9780),
    "Tribeca": (40.7163, -74.0086),
    "Two Bridges": (40.7110, -73.9900),
    "Upper East Side": (40.7736, -73.9566),
    "Upper West Side": (40.7870, -73.9754),
    "Washington Heights": (40.8400, -73.9400),
    "West Village": (40.7340, -74.0060),
    "Williamsburg": (40.7081, -73.9571),
    "Bedford-Stuyvesant": (40.6872, -73.9418),
    "Bushwick": (40.6944, -73.9213),
    "Crown Heights": (40.6694, -73.9442),
    "Downtown Brooklyn": (40.6960, -73.9845),
    "Park Slope": (40.6723, -73.9770),
    "Prospect Heights": (40.6775, -73.9750),
    "Brooklyn Heights": (40.6960, -73.9930),
    "DUMBO": (40.7033, -73.9880),
    "Carroll Gardens": (40.6784, -73.9990),
    "Fort Greene": (40.6915, -73.9740),
    "Greenwood Heights": (40.6520, -74.0050),
    "Red Hook": (40.6760, -74.0120),
    "Astoria": (40.7644, -73.9235),
    "Long Island City": (40.7447, -73.9485),
    "Flushing": (40.7675, -73.8330),
    "Jackson Heights": (40.7557, -73.8831),
    "Sunnyside": (40.7433, -73.9196),
    "Woodside": (40.7454, -73.9026),
    "Elmhurst": (40.7370, -73.8770),
    "Forest Hills": (40.7196, -73.8448),
    "Jamaica": (40.7027, -73.7890),
    "Richmond Hill": (40.6940, -73.8310),
    "Rockaway Beach": (40.5860, -73.8110),
    "Ridgewood": (40.7040, -73.9030),
    "Morrisania": (40.8290, -73.9020),
    "Mott Haven": (40.8090, -73.9220),
    "Fordham": (40.8610, -73.8900),
    "Kingsbridge": (40.8810, -73.9040),
    "Riverdale": (40.8950, -73.9120),
    "Concourse": (40.8250, -73.9220),
    "Port Morris": (40.8010, -73.9100),
    "St. George": (40.6437, -74.0736),
    "Stapleton": (40.6260, -74.0770),
    "Tompkinsville": (40.6360, -74.0740),
}


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_options(_model):
    preprocessor = _model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]

    return {
        "borough": encoder.categories_[0].tolist(),
        "neighbourhood": encoder.categories_[1].tolist(),
        "room_type": encoder.categories_[2].tolist(),
    }


# ============================================================
# STYLING
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --yellow: #FFD21F;
            --black: #0B0B0B;
            --dark: #151515;
            --border: #2C2C2C;
            --muted: #8E8E8E;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: #0B0B0B;
            color: #FFFFFF;
        }

        [data-testid="stHeader"] {
            background: #0B0B0B !important;
            height: 3.5rem;
        }

        .block-container {
            max-width: 820px;
            padding-top: 5.2rem !important;
            padding-bottom: 4rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2.5rem;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        .logo-box {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: var(--yellow);
            color: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }

        .logo-name {
            font-weight: 800;
            font-size: 1rem;
        }

        .logo-sub {
            color: #777;
            font-size: 0.67rem;
            margin-top: 2px;
        }

        .location-pill {
            border: 1px solid #2C2C2C;
            background: #151515;
            border-radius: 999px;
            color: #C8C8C8;
            font-size: 0.68rem;
            padding: 0.45rem 0.75rem;
            font-weight: 600;
        }

        .welcome {
            margin-bottom: 2.2rem;
        }

        .hello {
            color: var(--yellow);
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .welcome h1 {
            font-size: 2.7rem;
            line-height: 1.08;
            letter-spacing: -1.7px;
            margin: 0;
            font-weight: 800;
            color: #FFF;
        }

        .welcome p {
            max-width: 610px;
            color: #969696;
            font-size: 0.91rem;
            line-height: 1.65;
            margin-top: 0.85rem;
        }

        .question {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 1.7rem 0 0.45rem;
        }

        .number {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--yellow);
            color: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 800;
        }

        .question-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #FFF;
        }

        .hint {
            color: #686868;
            font-size: 0.69rem;
            margin: 0 0 0.8rem 2.25rem;
        }

        .card {
            background: #151515;
            border: 1px solid var(--border);
            border-radius: 17px;
            padding: 1rem 1rem 0.75rem;
        }

        label {
            color: #BDBDBD !important;
            font-size: 0.73rem !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] > div {
            background: #1D1D1D !important;
            border: 1px solid #363636 !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] * {
            color: #FFF !important;
        }

        div[data-testid="stNumberInput"] input {
            color: #FFF !important;
        }

        div[data-testid="stNumberInput"] button {
            color: var(--yellow) !important;
        }

        .tiny {
            color: #666;
            font-size: 0.66rem;
            line-height: 1.5;
            margin: 0.35rem 0 0.1rem;
        }

        .cta {
            margin-top: 1.5rem;
        }

        .stButton > button {
            min-height: 52px !important;
            border-radius: 12px !important;
            background: var(--yellow) !important;
            color: #000 !important;
            border: none !important;
            font-size: 0.9rem !important;
            font-weight: 800 !important;
        }

        .stButton > button:hover {
            background: #FFE15A !important;
        }

        .result {
            margin-top: 2rem;
            background: var(--yellow);
            color: #000;
            border-radius: 22px;
            padding: 2rem 1.2rem;
            text-align: center;
        }

        .result-label {
            font-size: 0.66rem;
            letter-spacing: 1.3px;
            font-weight: 800;
            color: #514800;
        }

        .price {
            font-size: 4.2rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -2.8px;
            margin: 0.4rem 0;
        }

        .per-night {
            color: #403A00;
            font-size: 0.82rem;
        }

        .result-text {
            color: #373100;
            font-size: 0.74rem;
            margin-top: 0.8rem;
        }

        .footer {
            color: #4E4E4E;
            text-align: center;
            font-size: 0.65rem;
            line-height: 1.5;
            margin-top: 2.5rem;
        }

        div[data-testid="stExpander"] {
            background: #111;
            border: 1px solid #292929;
            border-radius: 14px;
            margin-top: 1.3rem;
        }

        @media (max-width: 700px) {
            .welcome h1 {
                font-size: 2.1rem;
            }

            .price {
                font-size: 3.3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def question(number, title, hint):
    st.markdown(
        f"""
        <div class="question">
            <div class="number">{number}</div>
            <div class="question-title">{title}</div>
        </div>
        <div class="hint">{hint}</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FRIENDLY INPUT -> MODEL INPUT CONVERSION
# ============================================================
def get_coordinates(borough, neighbourhood):
    """
    Coordinates are kept behind the scenes.
    If a neighbourhood has a known representative coordinate,
    use it. Otherwise use the selected borough's representative point.
    """
    if neighbourhood in NEIGHBOURHOOD_COORDS:
        return NEIGHBOURHOOD_COORDS[neighbourhood]

    return BOROUGH_COORDS.get(borough, BOROUGH_COORDS["Manhattan"])


def euclidean_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def calculate_distances(latitude, longitude):
    return {
        "distance_midtown": euclidean_distance(
            latitude, longitude, *LANDMARKS["Midtown"]
        ),
        "distance_lower_manhattan": euclidean_distance(
            latitude, longitude, *LANDMARKS["Lower Manhattan"]
        ),
        "distance_central_park": euclidean_distance(
            latitude, longitude, *LANDMARKS["Central Park"]
        ),
    }


def convert_listing_experience(experience):
    """
    Convert a simple user-friendly description into values
    expected by the trained model.
    """
    if experience == "I'm just getting started":
        return 0, 0.0, 1, 180

    if experience == "I have a few reviews":
        return 15, 1.5, 1, 220

    return 60, 3.0, 1, 250


def build_features(
    borough,
    neighbourhood,
    room_type,
    bedrooms,
    beds,
    baths,
    minimum_nights,
    experience,
    rating,
):
    latitude, longitude = get_coordinates(borough, neighbourhood)

    number_of_reviews, reviews_per_month, host_listings, availability = (
        convert_listing_experience(experience)
    )

    distances = calculate_distances(latitude, longitude)

    return pd.DataFrame(
        [
            {
                "latitude": latitude,
                "longitude": longitude,
                "minimum_nights": minimum_nights,
                "number_of_reviews": number_of_reviews,
                "reviews_per_month": reviews_per_month,
                "calculated_host_listings_count": host_listings,
                "availability_365": availability,
                "rating": rating,
                "bedrooms": bedrooms,
                "beds": beds,
                "baths": baths,
                "distance_midtown": distances["distance_midtown"],
                "distance_lower_manhattan": distances["distance_lower_manhattan"],
                "distance_central_park": distances["distance_central_park"],
                "neighbourhood_group": borough,
                "neighbourhood": neighbourhood,
                "room_type": room_type,
            }
        ]
    )


def show_result(price):
    if price < 100:
        message = "Your place looks like a more budget-friendly option."
    elif price < 250:
        message = "This is a moderate nightly estimate for the details you entered."
    else:
        message = "This is a higher-end nightly estimate for the details you entered."

    st.markdown(
        f"""
        <div class="result">
            <div class="result-label">ESTIMATED NIGHTLY PRICE</div>
            <div class="price">${price:,.0f}</div>
            <div class="per-night">per night</div>
            <div class="result-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if price > MODEL_MAX_PRICE:
        st.warning(
            "This estimate is above the price range used to train the predictor, "
            "so treat it as a rough estimate."
        )


# ============================================================
# MAIN APP
# ============================================================
def main():
    inject_css()

    try:
        model = load_model()
        options = get_options(model)
    except Exception as exc:
        st.error("Sorry, we couldn't start the price estimator.")
        st.caption("Please make sure the trained model is in the project's models folder.")
        st.stop()

    # Header
    st.markdown(
        """
        <div class="nav">
            <div class="logo">
                <div class="logo-box">🏠</div>
                <div>
                    <div class="logo-name">StayPrice</div>
                    <div class="logo-sub">Airbnb price estimate</div>
                </div>
            </div>
            <div class="location-pill">New York City</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Welcome
    st.markdown(
        """
        <div class="welcome">
            <div class="hello">WELCOME 👋</div>
            <h1>What could your place<br>cost per night?</h1>
            <p>
                Answer a few simple questions about your property and we'll
                give you an estimated nightly price.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # 1. LOCATION
    # --------------------------------------------------------
    question(
        "1",
        "Where is your place?",
        "Just choose the area. You don't need to know your exact coordinates.",
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    borough = st.selectbox(
        "Borough",
        options["borough"],
        help="Choose the New York City borough where your place is located.",
    )

    neighbourhood = st.selectbox(
        "Neighbourhood",
        options["neighbourhood"],
        help="Choose the neighbourhood closest to your property.",
    )

    st.markdown(
        '<div class="tiny">We use your area to understand how location may affect the price.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 2. PROPERTY
    # --------------------------------------------------------
    question(
        "2",
        "What is your place like?",
        "Tell us the basics. These are the details guests usually care about.",
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    room_type = st.selectbox(
        "What are you offering?",
        options["room_type"],
        help="Choose the type of accommodation guests will book.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

    with c2:
        beds = st.number_input(
            "Beds",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
        )

    with c3:
        baths = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 3. LISTING
    # --------------------------------------------------------
    question(
        "3",
        "A little about your listing",
        "Don't worry if you're new. We provide simple options for you.",
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    experience = st.selectbox(
        "How established is your listing?",
        [
            "I'm just getting started",
            "I have a few reviews",
            "I have lots of reviews",
        ],
    )

    rating = st.slider(
        "Guest rating",
        min_value=0.0,
        max_value=5.0,
        value=4.5,
        step=0.1,
        help="If you don't have reviews yet, leave this at 4.5.",
    )

    minimum_nights = st.number_input(
        "Minimum stay",
        min_value=1,
        max_value=365,
        value=2,
        step=1,
        help="For example, enter 2 if guests must stay at least two nights.",
    )

    st.markdown(
        '<div class="tiny">New to Airbnb? The suggested values are perfectly fine for a first estimate.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------
    st.markdown('<div class="cta">', unsafe_allow_html=True)

    predict = st.button(
        "✨ Estimate my nightly price",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if predict:
        features = build_features(
            borough=borough,
            neighbourhood=neighbourhood,
            room_type=room_type,
            bedrooms=bedrooms,
            beds=beds,
            baths=baths,
            minimum_nights=minimum_nights,
            experience=experience,
            rating=rating,
        )

        with st.spinner("Calculating your estimate..."):
            try:
                prediction = float(
                    np.asarray(model.predict(features)).ravel()[0]
                )
                prediction = max(0.0, prediction)
            except Exception:
                st.error(
                    "We couldn't calculate the estimate. Please check your selections and try again."
                )
                st.stop()

        show_result(prediction)

        st.markdown(
            '<div class="tiny" style="text-align:center; margin-top:0.7rem;">Change any answer above to compare another property.</div>',
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # SIMPLE EXPLANATION
    # --------------------------------------------------------
    with st.expander("How is this estimate made?"):
        st.write(
            "The estimate is based on patterns found in thousands of New York City "
            "Airbnb listings. Your location, type of place, bedrooms, beds, bathrooms, "
            "guest rating and listing details all help us estimate a nightly price."
        )
        st.write(
            "This is only an estimate. Actual Airbnb prices can change because of "
            "season, dates, demand, events and other factors."
        )

    st.markdown(
        """
        <div class="footer">
            StayPrice provides estimates for informational purposes only.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
