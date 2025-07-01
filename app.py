import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from rapidfuzz import process #fuzzy loading


#Load environment variables
load_dotenv()

#Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Initialize Slack app
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

#Load Chroma vector DB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_store"))
collection = chroma_client.get_or_create_collection("hr_docs")

#Embed a query using OpenAI
def embed_query(query):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding

# aps keywords in a query to specific section filenames
SECTION_MAP = {
    "introduction": "section_1_0_introduction.txt",
    "message from the ceo": "section_1_0_message_from_the_ceo.txt",
    "message from the founders": "section_1_1_message_from_the_founders.txt",
    "rouxbe overview": "section_1_2_rouxbe_overview.txt",
    "a word about this handbook": "section_1_3_a_word_about_this_handbook.txt",
    "organizational chart": "section_1_4_organizational_chart.txt",
    "at-will employment": "section_2_1_atwill_employment.txt",
    "employment": "section_2_1_atwill_employment.txt",
    "new employee orientation": "section_2_2_new_employee_orientation.txt",
    "orientation": "section_2_2_new_employee_orientation.txt",
    "equal employment opportunity": "section_2_3_equal_employment_opportunity.txt",
    "equal": "section_2_3_equal_employment_opportunity.txt",
    "reasonable accommodation policy": "section_2_4_reasonable_accomodation_policy.txt",
    "accommodation": "section_2_4_reasonable_accomodation_policy.txt",
    "non-harassment": "section_2_5_nonharassment.txt",
    "harassment": "section_2_5_nonharassment.txt",
    "talk to us": "section_2_6_talk_to_us.txt",
    "categories of employment": "section_2_7_categories_of_employment.txt",
    "immigration reform and control act": "section_2_8_immigration_reform_and_control_act.txt",
    "your pay and progress": "section_3_0_your_pay_and_progress.txt",
    "pay": "section_3_0_your_pay_and_progress.txt",
    "recording your time": "section_3_1_recording_your_time.txt",
    "time": "section_3_1_recording_your_time.txt",
    "pay day": "section_3_2_pay_day.txt",
    "internal job postings": "section_3_3_internal_job_postings.txt",
    "job posting": "section_3_3_internal_job_postings.txt",
    "performance reviews and compensation": "section_3_4_our_approach_to_performance_reviews_and_compensation.txt",
    "performance review": "section_3_4_our_approach_to_performance_reviews_and_compensation.txt",
    "overtime": "section_3_5_working_together_to_meet_our_goals_overtime_policy.txt",
    "benefits": "section_4_0_employee_benefits_and_overview.txt",
    "medical insurance": "section_4_1_medical_and_dental_insurance.txt",
    "dental insurance": "section_4_1_medical_and_dental_insurance.txt",
    "holiday": "section_4_2_celebrating_together_our_holiday_policy.txt",
    "paid time off": "section_4_3_unlimited_paid_time_off_pto.txt",
    "PTO": "section_4_3_unlimited_paid_time_off_pto.txt",
    "military leave": "section_4_4_military_leave.txt",
    "military leave policy": "section_4_5_honoring_our_military_team_members_leave_policy.txt",
    "medical leave": "section_4_6_family_and_medical_leave_policy.txt",
    "family leave": "section_4_6_family_and_medical_leave_policy.txt",
    "maternity leave": "section_4_7_maternity_leave.txt",
    "domestic violence leave": "section_4_8_domestic_violence_leave.txt",
    "parental leave": "section_4_9_parental_leave.txt",
    "personal leave": "section_4_10_personal_leave.txt",
    "continuation of benefits": "section_4_11_continuation_of_benefits.txt",
    "workers compensation": "section_4_12_workers_compensation.txt",
    "401k retirement plan": "section_4_13_401k_qualified_retirement_plan.txt",
    "401k": "section_4_14_effect_of_termination_on_the_401k_plan.txt",
    "termination": "section_4_14_effect_of_termination_on_the_401k_plan.txt",
    "stock option": "section_4_15_stock_option_plan.txt",
    "long term disability": "section_4_16_long_term_disability.txt",
    "internet reimbursement policy": "section_4_17_mobile_device_and_internet_reimbursement_policy.txt",
    "growth champion initiative": "section_4_18_growth_champion_initiative.txt",
    "on the job": "section_5_0_on_the_job.txt",
    "remote ": "section_5_0_remote_work.txt",
    "confidentiality": "section_5_1_confidentiality.txt",
    "customer records": "section_5_3_customer_records.txt",
    "attendance": "section_5_4_attendance.txt",
    "workweek": "section_5_5_workweek_and_work_hours.txt",
    "hours": "section_5_5_workweek_and_work_hours.txt",
    "meal period policy": "section_5_6_nourishing_breaks_our_meal_period_policy.txt",
    "lunch": "section_5_6_nourishing_breaks_our_meal_period_policy.txt",
    "standards": "section_5_7_standards_of_conduct.txt",
    "conduct": "section_5_7_standards_of_conduct.txt",
    "personnel files": "section_5_8_access_to_personnel_files.txt",
    "solicitation": "section_5_9_solicitation_and_distribution.txt",
    "personal data": "section_5_10_changes_in_personal_data.txt",
    "conflict of interest": "section_5_11_conflict_of_interestcode_of_ethics.txt",
    "property": "section_5_12_personal_property.txt",
    "weather": "section_5_13_severe_weather.txt",
    "internet monitoring": "section_5_14_email_voicemail_and_internet_monitoring.txt",
    "dress policy": "section_5_15_dress_policy.txt",
    "dress code": "section_5_15_dress_policy.txt",
    "clothing": "section_5_15_dress_policy.txt",
    "verification": "section_5_16_employment_verification.txt",
    "media": "section_5_17_contact_with_the_media.txt",
    "leaving the company": "section_5_18_if_you_must_leave_us.txt",
    "quitting": "section_5_18_if_you_must_leave_us.txt",
    "reimbursement ": "section_5_19_streamlined_expense_reimbursement_policy.txt",
    "hr training": "section_5_20_hr_training.txt",
    "safety": "section_6_0_safety_in_the_workplace.txt",
    "responsibility": "section_6_1_each_employees_responsibility.txt",
    "responsibilities": "section_6_1_each_employees_responsibility.txt",
    "workplace violence": "section_6_2_workplace_violence.txt",
    "weapons": "section_6_3_concealed_weapons.txt",
    "drug": "section_6_4_drug_and_alcohol_policy.txt",
    "alcohol": "section_6_4_drug_and_alcohol_policy.txt",
    "privacy commitment": "section_6_5_privacy_committment.txt",
    "bullying": "section_6_6_bullying.txt",
    "development": "section_6_7_professional_development.txt",
    "whistleblower": "section_6_8_whitleblower_protection_policy.txt",
    "complaint": "section_6_9_complaint_resolution.txt",
    "technology usage": "section_7_0_technology_resources_usage_policy.txt",
    "mobile email device": "section_7_1_mobile_email_device_and_use_of_unmanaged_company_device.txt",
    "piia": "section_8_0_piia_proprietary_information_and_inventions_agreement.txt",
    "handbook receipt": "section_9_0_receipt_of_handbook_and_employment-at-will_statement.txt"
}

def get_best_matching_section(query, threshold=70): #fuzzy match for similar keywords
    """Fuzzy match the query to SECTION_MAP keys."""
    best_match = process.extractOne(query.lower(), SECTION_MAP.keys())
    if best_match and best_match[1] >= threshold:
        return SECTION_MAP[best_match[0]]
    return None

def get_answer(query):
    matched_section = get_best_matching_section(query)

    if matched_section:
        #Use exact section text from hr_docs_by_section
        with open(f"hr_docs_by_section/{matched_section}", "r", encoding="utf-8") as f:
            context = f.read()

        prompt = f"""
You are an HR assistant. Use the following HR policy section to answer the employee's question.

Context:
{context}

Question:
{query}

Answer:"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content

    else:
        #No fuzzy match — fallback to vector similarity search
        query_embedding = embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
        )
        docs = results["documents"][0]
        context = "\n\n".join(docs)

        prompt = f"""
You are an HR assistant. Use the following HR policy content to answer the employee's question.

Context:
{context}

Question:
{query}

Answer:"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content


#Handle Slack mentions
@app.event("app_mention")
def handle_message_events(body, say):
    print("📩 EVENT RECEIVED:", body)

    user_query = body["event"]["text"]
    user_query = user_query.replace(f"<@{body['event']['user']}>", "").strip()

    say("🤖 Thinking...")
    try:
        answer = get_answer(user_query)
        say(answer)
    except Exception as e:
        say(f"⚠️ Error: {str(e)}")

#Run the Slack bot
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
