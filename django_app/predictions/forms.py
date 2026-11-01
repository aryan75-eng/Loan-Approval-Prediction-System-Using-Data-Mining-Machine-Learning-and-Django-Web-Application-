"""
predictions/forms.py — PredictionForm with 11 typed and validated fields.
"""

from django import forms


GENDER_CHOICES = [("Male", "Male"), ("Female", "Female")]
MARRIED_CHOICES = [("Yes", "Yes"), ("No", "No")]
DEPENDENTS_CHOICES = [("0", "0"), ("1", "1"), ("2", "2"), ("3+", "3+")]
EDUCATION_CHOICES = [("Graduate", "Graduate"), ("Not Graduate", "Not Graduate")]
SELF_EMPLOYED_CHOICES = [("Yes", "Yes"), ("No", "No")]
PROPERTY_CHOICES = [("Urban", "Urban"), ("Semiurban", "Semiurban"), ("Rural", "Rural")]


class PredictionForm(forms.Form):
    # ── Numeric Fields ────────────────────────────────────────────────────────
    ApplicantIncome = forms.FloatField(
        label="Applicant Monthly Income (₹)",
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 5000"}),
    )
    CoapplicantIncome = forms.FloatField(
        label="Co-Applicant Monthly Income (₹)",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 1500"}),
    )
    LoanAmount = forms.FloatField(
        label="Loan Amount (in thousands)",
        min_value=1,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 128"}),
    )
    Loan_Amount_Term = forms.FloatField(
        label="Loan Term (months)",
        min_value=12,
        max_value=480,
        initial=360,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 360"}),
    )
    Credit_History = forms.ChoiceField(
        label="Credit History",
        choices=[("1.0", "Meets Guidelines (1)"), ("0.0", "Does Not Meet (0)")],
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    # ── Categorical Fields ────────────────────────────────────────────────────
    Gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    Married = forms.ChoiceField(
        choices=MARRIED_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    Dependents = forms.ChoiceField(
        choices=DEPENDENTS_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    Education = forms.ChoiceField(
        choices=EDUCATION_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    Self_Employed = forms.ChoiceField(
        label="Self Employed",
        choices=SELF_EMPLOYED_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    Property_Area = forms.ChoiceField(
        label="Property Area",
        choices=PROPERTY_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    def clean_Credit_History(self):
        """Convert string choice to float for the ML pipeline."""
        return float(self.cleaned_data["Credit_History"])

    def clean_ApplicantIncome(self):
        val = self.cleaned_data["ApplicantIncome"]
        if val > 1_000_000:
            raise forms.ValidationError("Income value seems unusually high. Please double-check.")
        return val

    def clean_LoanAmount(self):
        val = self.cleaned_data["LoanAmount"]
        if val > 10_000:
            raise forms.ValidationError("Loan amount exceeds 10,000 thousand. Please verify.")
        return val

    def to_prediction_dict(self) -> dict:
        """Return cleaned data as a dict ready for ml_service.predict()."""
        d = self.cleaned_data.copy()
        return d
