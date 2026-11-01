"""
accounts/forms.py — Registration form with password confirmation.
"""

from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User


class RegistrationForm(forms.ModelForm):
    """User registration form with password confirmation."""

    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Create a password",
            "autocomplete": "new-password",
        }),
        min_length=8,
        help_text="Minimum 8 characters.",
    )
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Repeat password",
            "autocomplete": "new-password",
        }),
    )

    class Meta:
        model = User
        fields = ["username", "email", "first_name", "last_name"]
        widgets = {
            "username":   forms.TextInput(attrs={"class": "form-control", "placeholder": "Username"}),
            "email":      forms.EmailInput(attrs={"class": "form-control", "placeholder": "Email address"}),
            "first_name": forms.TextInput(attrs={"class": "form-control", "placeholder": "First name"}),
            "last_name":  forms.TextInput(attrs={"class": "form-control", "placeholder": "Last name"}),
        }

    def clean_email(self):
        email = self.cleaned_data.get("email", "").strip()
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get("password1")
        p2 = cleaned.get("password2")
        if p1 and p2 and p1 != p2:
            self.add_error("password2", "Passwords do not match.")
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user


class StyledAuthForm(AuthenticationForm):
    """Login form with Bootstrap-styled widgets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update({
            "class": "form-control",
            "placeholder": "Username",
            "autofocus": True,
        })
        self.fields["password"].widget.attrs.update({
            "class": "form-control",
            "placeholder": "Password",
        })
