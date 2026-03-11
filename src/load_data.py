import pandas as pd

data_path = "../data/"

student_info = pd.read_csv(data_path + "studentInfo.csv")
student_assessment = pd.read_csv(data_path + "studentAssessment.csv")
assessments = pd.read_csv(data_path + "assessments.csv")
student_registration = pd.read_csv(data_path + "studentRegistration.csv")
student_vle = pd.read_csv(data_path + "studentVle.csv")
vle = pd.read_csv(data_path + "vle.csv")
courses = pd.read_csv(data_path + "courses.csv")

print("studentInfo:", student_info.shape)
print("studentAssessment:", student_assessment.shape)
print("assessments:", assessments.shape)
print("studentRegistration:", student_registration.shape)
print("studentVle:", student_vle.shape)
print("vle:", vle.shape)
print("courses:", courses.shape)

print("\nstudentInfo columns:")
print(student_info.columns)

print("\nOutcome distribution:")
print(student_info["final_result"].value_counts())

student_info["dropout"] = student_info["final_result"].apply(
    lambda x: 1 if x == "Withdrawn" else 0
)

print("\nDropout distribution:")
print(student_info["dropout"].value_counts())
