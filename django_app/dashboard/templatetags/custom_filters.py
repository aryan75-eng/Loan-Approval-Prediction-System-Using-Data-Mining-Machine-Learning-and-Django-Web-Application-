"""
custom_filters.py — Shared template filters used across all apps.

Load with: {% load custom_filters %}
"""

from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """
    Access a dictionary value by key in templates.
    Usage: {{ my_dict|get_item:key_var }}
    """
    if isinstance(dictionary, dict):
        val = dictionary.get(key, "")
        return val if val is not None else ""
    return ""


@register.filter
def split(value, delimiter=","):
    """
    Split a string by delimiter.
    Usage: {{ "a,b,c"|split:"," }}
    """
    return value.split(delimiter)


@register.filter
def percentage(value):
    """Convert a float 0–1 to a percentage string."""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return value


@register.filter
def abs_value(value):
    """Return absolute value."""
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return value
