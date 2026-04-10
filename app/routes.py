from flask import Blueprint, render_template, request
from models.predictor import CareerPredictor
from models.scheme_matcher import SchemeMatcher

bp = Blueprint("main", __name__)
predictor = CareerPredictor()
scheme_matcher = SchemeMatcher()

@bp.route("/", methods=["GET", "POST"])
def home():
    results = None
    error = None

    if request.method == "POST":
        skills = request.form.get("skills", "").strip()
        education = request.form.get("education", "").strip()
        location = request.form.get("communication", "").strip()
        location = request.form.get("location", "").strip() 

        if not skills or not education or not location:
            error = "Please fill all fields"
        else:
            user_profile = {
                "skills": skills,
                "education": education,
                "location": location
            }

            df = predictor.recommend(user_profile)

            if df.empty:
                error = "No suitable careers found for your profile"
            else:
                df["match_percent"] = (df["similarity"] * 100).round(1)
                results = df.to_dict("records")

                for r in results:
                    r["schemes"] = scheme_matcher.get_schemes_for_job(
                        r["job_title"]
                    )

    return render_template("index.html", results=results, error=error)
